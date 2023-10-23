import re
import json
from evaluate import load

wer = load("wer")
cer = load("cer")

def find_top_predictions_and_references(file_path):
  pattern = r'<hypothesis1>(.*?)</hypothesis1>'
  with open(file_path, "r") as f:
    data = json.load(f)

  predictions = []
  references = []
  for data_point in data:
    matches = re.findall(pattern, data_point["input"], re.DOTALL)
    predictions.append(matches[0])
    references.append(data_point["output"])

  return predictions, references

def find_wer_and_cer(predictions, references):
  assert(len(predictions) == len(references))
  wer_ = 100 * wer.compute(predictions=predictions, references=references)
  cer_ = 100 * cer.compute(predictions=predictions, references=references)

  return wer_, cer_

partial_amounts = ["5-percent-removed", "10-percent-removed", "15-percent-removed", "20-percent-removed", "25-percent-removed"]
for partial_amount in partial_amounts:
  test_other_predictions, test_other_references = find_top_predictions_and_references(f"llama2-data/{partial_amount}/test-other.json")
  test_clean_predictions, test_clean_references = find_top_predictions_and_references(f"llama2-data/{partial_amount}/test-clean.json")

  test_other_wer, test_other_cer = find_wer_and_cer(test_other_predictions, test_other_references)
  test_clean_wer, test_clean_cer = find_wer_and_cer(test_clean_predictions, test_clean_references)

  print(f"LibriSpeech Test Other WER ({partial_amount}): {test_other_wer:.2f}%")
  print(f"LibriSpeech Test Clean WER ({partial_amount}): {test_clean_wer:.2f}%")

  print(f"LibriSpeech Test Other CER ({partial_amount}): {test_other_cer:.2f}%")
  print(f"LibriSpeech Test Clean CER ({partial_amount}): {test_clean_cer:.2f}%")
