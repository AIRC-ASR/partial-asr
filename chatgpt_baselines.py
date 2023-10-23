import os
import json
import openai
import string
import evaluate
from tqdm import tqdm

# MICHAEL's
# openai.api_key = "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"
openai.api_key = "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"
openai.organization = "org-lC3wy4kACAHLYDh84LxeQaY4"

PARTIAL_AMOUNT = "25-percent-removed"
DS_SPLIT = "test-clean"

with open(f"llama2-data/{PARTIAL_AMOUNT}/{DS_SPLIT}.json", "r") as f:
  data = json.load(f)

inputs = [data_point["instruction"] + data_point["input"] for data_point in data]
outputs = [data_point["output"] for data_point in data]
messages = [{"role": "user", "content": input_} for input_ in inputs]
translator = str.maketrans('', '', string.punctuation)
NUM_RETRIES = 3

predictions = []
for message in tqdm(messages):
  for _ in range(NUM_RETRIES):
    try:
      response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."}] + [message],
        request_timeout=30
      )
      if response and response['choices'][0]['message']['content']:
        break
    except Exception as e:
      print("ERROR:", str(e))
  
  if response:
    prediction = response['choices'][0]['message']['content'].lower().strip().translate(translator)
  else:
    print("ERROR: Request failed after 3 retries. Setting prediction to empty string.")
    prediction = ""

  print(prediction)

  predictions.append({"prediction": prediction})

with open("chatgpt-predictions/original/test-clean-predictions.json", "w") as f:
  json.dump(predictions, f, indent=2)

assert(len(predictions) == len(outputs))
# Find WER and CER, by comparing completions to outputs
# cer = evaluate.load("cer")
# wer = evaluate.load("wer")

# with open(f"chatgpt-predictions/{PARTIAL_AMOUNT}/{DS_SPLIT}-predictions.json", "r") as f:
#   predictions = json.load(f)

# wer_ = 100 * wer.compute(predictions=[pred["prediction"] for pred in predictions], references=outputs)
# print(f"WER: {wer_:.2f}%")

# cer_ = 100 * cer.compute(predictions=[pred["prediction"] for pred in predictions], references=outputs)
# print(f"CER: {cer_:.2f}%")

