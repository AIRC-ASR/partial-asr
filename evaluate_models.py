import torch
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from evaluate import load
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer
from tqdm import tqdm
from transformers import WhisperProcessor

# Get the test other and test clean datasets for original audio
# Get just the outputs in their own list for all of them
original_test_other_ds = load_dataset("json", data_files="llama2-data/original/test-other.jsonl")["train"]
original_test_clean_ds = load_dataset("json", data_files="llama2-data/original/test-clean.jsonl")["train"]
original_test_other_prompts = original_test_other_ds["text"]
original_test_clean_prompts = original_test_clean_ds["text"]
original_test_other_refs = original_test_other_ds["output"]
original_test_clean_refs = original_test_clean_ds["output"]

# # Get the test other and test clean datasets for 5% audio removed
# five_percent_removed_test_other_ds = load_dataset("json", data_files="llama2-data/5-percent-removed/test-other.jsonl")["train"]
# five_percent_removed_test_clean_ds = load_dataset("json", data_files="llama2-data/5-percent-removed/test-clean.jsonl")["train"]
# five_percent_removed_test_other_outputs = five_percent_removed_test_other_ds["output"]
# five_percent_removed_test_clean_outputs = five_percent_removed_test_clean_ds["output"]

# # Get the test other and test clean datasets for 10% audio removed
# ten_percent_removed_test_other_ds = load_dataset("json", data_files="llama2-data/10-percent-removed/test-other.jsonl")["train"]
# ten_percent_removed_test_clean_ds = load_dataset("json", data_files="llama2-data/10-percent-removed/test-clean.jsonl")["train"]
# ten_percent_removed_test_other_outputs = ten_percent_removed_test_other_ds["output"]
# ten_percent_removed_test_clean_outputs = ten_percent_removed_test_clean_ds["output"]

# # Get the test other and test clean datasets for 15% audio removed
# fifteen_percent_removed_test_other_ds = load_dataset("json", data_files="llama2-data/15-percent-removed/test-other.jsonl")["train"]
# fifteen_percent_removed_test_clean_ds = load_dataset("json", data_files="llama2-data/15-percent-removed/test-clean.jsonl")["train"]
# fifteen_percent_removed_test_other_outputs = fifteen_percent_removed_test_other_ds["output"]
# fifteen_percent_removed_test_clean_outputs = fifteen_percent_removed_test_clean_ds["output"]

# # Get the test other and test clean datasets for 20% audio removed
# twenty_percent_removed_test_other_ds = load_dataset("json", data_files="llama2-data/20-percent-removed/test-other.jsonl")["train"]
# twenty_percent_removed_test_clean_ds = load_dataset("json", data_files="llama2-data/20-percent-removed/test-clean.jsonl")["train"]
# twenty_percent_removed_test_other_outputs = twenty_percent_removed_test_other_ds["output"]
# twenty_percent_removed_test_clean_outputs = twenty_percent_removed_test_clean_ds["output"]

# # Get the test other and test clean datasets for 25% audio removed
# twenty_five_percent_removed_test_other_ds = load_dataset("json", data_files="llama2-data/25-percent-removed/test-other.jsonl")["train"]
# twenty_five_percent_removed_test_clean_ds = load_dataset("json", data_files="llama2-data/25-percent-removed/test-clean.jsonl")["train"]
# twenty_five_percent_removed_test_other_outputs = twenty_five_percent_removed_test_other_ds["output"]
# twenty_five_percent_removed_test_clean_outputs = twenty_five_percent_removed_test_clean_ds["output"]

wer = load("wer")
cer = load("cer")

# Get the baseline model

# Get the instruction tuned models for original audio file
original_config = PeftConfig.from_pretrained("omarc/llama2-partial-asr-original")
original_model = AutoModelForCausalLM.from_pretrained("TinyPixel/Llama-2-7B-bf16-sharded", load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, device_map={'': 3})
print("DEV MAP", original_model.hf_device_map)
original_model = PeftModel.from_pretrained(original_model, "omarc/llama2-partial-asr-original", config=original_config)
# original_model = torch.compile(original_model)
original_model.eval()

# # Get the instruction tuned models for 5% removed audio file
# five_percent_config = PeftConfig.from_pretrained("omarc/llama2-partial-asr-5-percent-removed")
# five_percent_model = AutoModelForCausalLM.from_pretrained("TinyPixel/Llama-2-7B-bf16-sharded", quantization_config=bnb_config, trust_remote_code=True)
# five_percent_model = PeftModel.from_pretrained(five_percent_model, "omarc/llama2-partial-asr-5-percent-removed")
# five_percent_model.config.use_cache = False
# five_percent_model.eval()

# # Get the instruction tuned models for 10% removed audio file
# ten_percent_config = PeftConfig.from_pretrained("omarc/llama2-partial-asr-10-percent-removed")
# ten_percent_model = AutoModelForCausalLM.from_pretrained("TinyPixel/Llama-2-7B-bf16-sharded", quantization_config=bnb_config, trust_remote_code=True)
# ten_percent_model = PeftModel.from_pretrained(ten_percent_model, "omarc/llama2-partial-asr-10-percent-removed")
# ten_percent_model.config.use_cache = False
# ten_percent_model.eval()

# # Get the instruction tuned models for 15% removed audio file
# fifteen_percent_config = PeftConfig.from_pretrained("omarc/llama2-partial-asr-15-percent-removed")
# fifteen_percent_model = AutoModelForCausalLM.from_pretrained("TinyPixel/Llama-2-7B-bf16-sharded", quantization_config=bnb_config, trust_remote_code=True)
# fifteen_percent_model = PeftModel.from_pretrained(fifteen_percent_model, "omarc/llama2-partial-asr-15-percent-removed")
# fifteen_percent_model.config.use_cache = False
# fifteen_percent_model.eval()

# # Get the instruction tuned models for 20% removed audio file
# twenty_percent_config = PeftConfig.from_pretrained("omarc/llama2-partial-asr-20-percent-removed")
# twenty_percent_model = AutoModelForCausalLM.from_pretrained("TinyPixel/Llama-2-7B-bf16-sharded", quantization_config=bnb_config, trust_remote_code=True)
# twenty_percent_model = PeftModel.from_pretrained(twenty_percent_model, "omarc/llama2-partial-asr-20-percent-removed")
# twenty_percent_model.config.use_cache = False
# twenty_percent_model.eval()

# # Get the instruction tuned models for 25% removed audio file
# twenty_five_percent_config = PeftConfig.from_pretrained("omarc/llama2-partial-asr-25-percent-removed")
# twenty_five_percent_model = AutoModelForCausalLM.from_pretrained("TinyPixel/Llama-2-7B-bf16-sharded", quantization_config=bnb_config, trust_remote_code=True)
# twenty_five_percent_model = PeftModel.from_pretrained(twenty_five_percent_model, "omarc/llama2-partial-asr-25-percent-removed")
# twenty_five_percent_model.config.use_cache = False
# twenty_five_percent_model.eval()

tokenizer = AutoTokenizer.from_pretrained("TinyPixel/Llama-2-7B-bf16-sharded", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

from transformers import GenerationConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(
    prompt,
    model,
    tokenizer,
    input=None,
    temperature=0.7,
    num_beams=5,
    max_new_tokens=128,
    **kwargs,
):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        num_beams=num_beams,
        **kwargs,
    )
    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": False,
        "max_new_tokens": max_new_tokens,
    }

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
            use_cache=True,
            num_return_sequences=1,
            no_repeat_ngram_size = 2

        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    output = output.replace("<s>", "").replace("</s>", "")
    if "###Assistant" in output:
      output = output.split("###Assistant")[1]

    yield output


def find_model_predictions(prompts, model, tokenizer):
  predictions = []
  processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")

  for prompt in tqdm(prompts):
    for model_output in evaluate(prompt, model, tokenizer):
      model_output = processor.tokenizer._normalize(model_output)
      print(model_output)
      predictions.append(model_output)

  return predictions

original_test_other_preds = find_model_predictions(original_test_other_prompts, original_model, tokenizer)
original_test_other_wer = 100 * wer.compute(predictions=original_test_other_preds, references=original_test_other_refs)
original_test_other_cer = 100 * cer.compute(predictions=original_test_other_preds, references=original_test_other_refs)

results_file_path = "original_results.txt"
with open(results_file_path, "w") as results_file:
  results_file.write(f"LibriSpeech Test Other WER (original): {original_test_other_wer:.2f}%")
  results_file.write(f"LibriSpeech Test Other CER (original): {original_test_other_cer:.2f}%")
