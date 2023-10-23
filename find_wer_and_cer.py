import json
import evaluate

PARTIAL_AMOUNT = "25-percent-removed"
DS_SPLIT = "test-clean"

cer = evaluate.load("cer")
wer = evaluate.load("wer")

with open(f"llama2-data/{PARTIAL_AMOUNT}/{DS_SPLIT}.json", "r") as f:
  data = json.load(f)

with open(f"final-predictions/{PARTIAL_AMOUNT}/{DS_SPLIT}.json", "r") as f:
  predictions = json.load(f)

outputs = [data_point["output"] for data_point in data]

assert(len(predictions) == len(outputs))
wer_ = 100 * wer.compute(predictions=[pred["prediction"] for pred in predictions], references=outputs)
print(f"WER: {wer_:.2f}%")

cer_ = 100 * cer.compute(predictions=[pred["prediction"] for pred in predictions], references=outputs)
print(f"CER: {cer_:.2f}%")
