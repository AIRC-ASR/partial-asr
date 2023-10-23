import os
import sys
from tqdm import tqdm
import fire
import gradio as gr
import torch
import csv
import json
import transformers
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from transformers import GenerationConfig,AutoModelForCausalLM,AutoTokenizer
from transformers import WhisperProcessor

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
from evaluate import load
from datasets import load_dataset

# TODO: SHOWS HOW TO DO INFERENCE ON PEFT MODELS: https://huggingface.co/blog/peft

PARTIAL_AMOUNT = "25-percent-removed"
DS_SPLIT = "test-clean"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

print("DEVICE", device)
def main(
    use_lora = True,
    load_8bit: bool = True,
    base_model: str = "FreedomIntelligence/phoenix-inst-chat-7b",
    lora_weights: str = f"models/{PARTIAL_AMOUNT}/checkpoint-7500",
    prompt_template: str = "phoenix",  # The prompt template to use, will default to alpaca.
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='FreedomIntelligence/phoenix-inst-chat-7b'"
    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        cache_dir="cache/transformers"
    )
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir="cache/transformers",
        )
        if use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=True, r=8, lora_alpha=16, lora_dropout=0.05
            )
            model = get_peft_model(model, peft_config)

    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float32,
            cache_dir="cache/transformers",
        )
        if use_lora:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map="auto",
                torch_dtype=torch.float16,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        if use_lora:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    
    def evaluate(
        instruction,
        input=None,
        temperature=0.7,
        top_p=0.75,
        top_k=40,
        num_beams=5,
        max_new_tokens=1024,
        stream_output=False,
        prompt_template: str = "phoenix",
        **kwargs,
    ):
        prompter = Prompter(prompt_template)
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            # temperature=temperature,
            # top_p=top_p,
            # top_k=top_k,
            do_sample=False,
            num_beams=num_beams,
            **kwargs,
        )
        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break
                    yield prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        yield prompter.get_response(output)

    def find_model_predictions(prompts, instruction):
        predictions = []
        processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")
        # with open(f"final-predictions/{PARTIAL_AMOUNT}/{DS_SPLIT}-checkpoint-2000.json", "r") as pred_file:
        #     predictions_formatted = json.load(pred_file)
        #     predictions = [pred["prediction"] for pred in predictions_formatted]
        i = 0
        for prompt in tqdm(prompts):
            if i < len(predictions):
                i += 1
                continue
            for model_output in evaluate(instruction, prompt):
                model_output = processor.tokenizer._normalize(model_output.replace('</s>',''))
                print(model_output)
                if i % 500 == 0:
                    predictions_formatted = [{"prediction": prediction} for prediction in predictions]
                    with open(f"final-predictions/{PARTIAL_AMOUNT}/{DS_SPLIT}-checkpoint-{i}.json", "w") as pred_file:
                        json.dump(predictions_formatted, pred_file, indent=2)
                predictions.append(model_output)

            i += 1

        return predictions

    def get_prompts_references_instruction(partial_amount, ds_split):
        ds = load_dataset("json", data_files=f"llama2-data/{partial_amount}/{ds_split}.json", cache_dir="cache/datasets")["train"]
        prompts = ds["input"]
        references = ds["output"]
        instruction = ds["instruction"][0]

        return prompts, references, instruction

    prompts, references, instruction = get_prompts_references_instruction(PARTIAL_AMOUNT, DS_SPLIT)
    predictions = find_model_predictions(prompts, instruction)
    predictions_formatted = [{"prediction": prediction} for prediction in predictions]
    with open(f"final-predictions/{PARTIAL_AMOUNT}/{DS_SPLIT}.json", "w") as pred_file:
        json.dump(predictions_formatted, pred_file, indent=2)
    # wer = load("wer")
    # wer_ = 100 * wer.compute(predictions=[pred["prediction"] for pred in predictions_formatted], references=references)
    # print(f"WER: {wer_:.2f}%")



if __name__ == "__main__":
    main()
