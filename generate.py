import os
import sys
from tqdm import tqdm
import fire
import gradio as gr
import torch
import csv
import json
import transformers
from peft import PeftModel
from transformers import GenerationConfig,AutoModelForCausalLM,AutoTokenizer
from transformers import WhisperProcessor

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
from evaluate import load
from datasets import load_dataset

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

DS_SIZE = 2939

def read_transcripts_and_references(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    
    print("LEN", len(data))
    print("DATA", data[0])
    transcripts = [example["transcript"] for example in data]
    references = [example["reference"] for example in data]

    return transcripts, references


def read_librispeech_ground_truth():
  dataset = load_dataset(
    "librispeech_asr",
    "other",
    split="other",
    # cache_dir="/gpfs/u/home/NLUG/NLUGcbsm/scratch/cache/dataset"
  )
  ground_truths = []

  for example in dataset:
    ground_truth = example["text"]
    ground_truths.append(ground_truth)

  return ground_truths

def main(
    use_lora= True,
    test_file='nacgec.test.input',
    des_file ='../submission.txt',
    load_8bit: bool = False,
    base_model: str = "FreedomIntelligence/phoenix-inst-chat-7b",
    lora_weights: str = "./saved_models/no_added_errors/clean/1000/5_best_ipa2/checkpoint-210",
    prompt_template: str = "phoenix",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
    
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='FreedomIntelligence/phoenix-inst-chat-7b'"
    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        cache_dir="/gpfs/u/home/NLUG/NLUGcbsm/scratch/cache/transformers"
    )
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float32,
            device_map="auto",
            cache_dir="/gpfs/u/home/NLUG/NLUGcbsm/scratch/cache/transformers",
        )
        if use_lora:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float32,
            cache_dir="/gpfs/u/home/NLUG/NLUGcbsm/scratch/cache/transformers",
        )
        if use_lora:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
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
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)
    
    def evaluate(
        instruction,
        input=None,
        temperature=0.7,
        # temperature=0.7,
        top_p=0.75,
        # top_k=40,
        # do_sample=True,
        num_beams=4,
        max_new_tokens=512,
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
                    # new_tokens = len(output) - len(input_ids[0])
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
        # yield output
        yield prompter.get_response(output)

    def find_model_predictions(transcripts, instruction):
        predictions = []
        processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")

        for transcript in tqdm(transcripts[:DS_SIZE]):
            for model_output in evaluate(instruction, transcript):
                model_output = processor.tokenizer._normalize(model_output.replace('</s>',''))

                # print(instruction, model_output)
                predictions.append(model_output)

        return predictions


    # with open(test_file,'r',encoding='utf8')as fr:
    #     lines = fr.readlines()
    import re
    NUM_HYPOTHESES = 3
    instruction = f"Perform error correction on the top{NUM_HYPOTHESES} outputs generated by an Automatic Speech Recognition (ASR) system. The ASR hypotheses, listed in order of their ASR posterior score, are as follows:"
    
    wer = load("wer")

    # librispeech_test_other_references = read_librispeech_ground_truth()
    # librispeech_test_other_transcripts, librispeech_test_other_references = read_transcripts_and_references("./pseudo_data/transcripts/whisper-small.en/training_data/other/1000/transcripts.json")
    # librispeech_test_other_transcripts, librispeech_test_other_references = read_transcripts_and_references("./pseudo_data/transcripts/whisper-small.en/training_data/other/1000/5_best/transcripts.json")
    librispeech_test_other_transcripts1, librispeech_test_other_references1 = read_transcripts_and_references("pseudo_data/transcripts/whisper-small.en/training_data/other/1000/5_best/transcripts_ipa1.json")
    librispeech_test_other_transcripts2, librispeech_test_other_references2 = read_transcripts_and_references("pseudo_data/transcripts/whisper-small.en/training_data/other/1000/5_best/transcripts_ipa2.json")
    librispeech_test_other_transcripts3, librispeech_test_other_references3 = read_transcripts_and_references("pseudo_data/transcripts/whisper-small.en/training_data/other/1000/5_best/transcripts_ipa3.json")

    librispeech_test_other_transcripts = librispeech_test_other_transcripts1 + librispeech_test_other_transcripts2 + librispeech_test_other_transcripts3
    librispeech_test_other_references = librispeech_test_other_references1 + librispeech_test_other_references2 + librispeech_test_other_references3
    librispeech_test_other_predictions = find_model_predictions(librispeech_test_other_transcripts, instruction)
    # for i, x in enumerate(librispeech_test_other_predictions):
    # #   if i == DS_SIZE: break
    #   print("CORRECTION", x)
    #   print("REFERENCE", librispeech_test_other_references[i])
    # for i, y in enumerate(librispeech_test_other_references):
    #   print("REF", y)
    #   if i == 10: break
    librispeech_test_other_wer = 100 * wer.compute(
        references=librispeech_test_other_references[:DS_SIZE],
        predictions=librispeech_test_other_predictions[:DS_SIZE]
        # references=librispeech_test_other_references[:DS_SIZE],
        # predictions=librispeech_test_other_transcripts[:DS_SIZE]
    )
    print(f"LibriSpeech: Test Other WER: {librispeech_test_other_wer:.3f}%")

if __name__ == "__main__":
    main()
