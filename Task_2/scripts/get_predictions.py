import json
import os
import torch
import sys
from transformers import MarianMTModel, MarianTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utilities.data_cleaning_utils import clean_text

# Paths
model_path = "models/FinetuningModel"
input_file = "data/references/validation/es_ES.jsonl"
output_file = "data/predictions/Pruebas/validation/es_ES.jsonl"

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path).to(device)

def translate(texts):
    """Translates a batch of texts after cleaning them."""
    if isinstance(texts, str):  # Convert single string to list
        texts = [texts]

    # Apply text cleaning
    cleaned_texts = [clean_text(text, 'configs/data_config.json') for text in texts]

    inputs = tokenizer(cleaned_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=7,
        length_penalty=1.2,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

def process_and_save():
    """Processes the input file and saves translated outputs."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        total_lines = sum(1 for _ in infile)
        infile.seek(0)

        for idx, line in enumerate(infile, start=1):
            try:
                data = json.loads(line.strip())
                source_text = data.get("source", "")

                if not source_text:
                    print(f"Warning: No 'source' text found in entry {idx}")
                    continue

                print(f"Processing {idx}/{total_lines}: {source_text}")
                translated_text = translate(source_text)[0]  # Extract single translation from list

                prediction_entry = {
                    "id": data["id"],
                    "source_language": "English",
                    "target_language": "Spanish",
                    "text": source_text,
                    "prediction": translated_text
                }

                outfile.write(json.dumps(prediction_entry, ensure_ascii=False) + "\n")
                outfile.flush()

            except json.JSONDecodeError:
                print(f"Error decoding JSON on line {idx}: {line.strip()}")
            except Exception as e:
                print(f"Unexpected error on line {idx}: {e}")

    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    process_and_save()
