import json
import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Paths
model_path = "models/T5SmallFinetuningModel"  # Ajusta según tu directorio
input_file = "data/references/test/es_ES.jsonl"
output_file = "data/predictions/T5Model/validation/es_ES.jsonl"

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo y tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

def translate(text):
    # Añadir el prefijo requerido por T5
    input_text = f"translate English to Spanish: {text}"
    
    # Tokenización
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=256, 
        truncation=True,
        padding='max_length'
    ).to(device)
    
    # Generación
    with torch.no_grad():  # No necesitamos gradientes para inferencia
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=5,  # Beam search
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=2,  # Evitar repeticiones
            temperature=0.7  # Añadir algo de variabilidad
        )
    
    # Decodificar y retornar
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def process_and_save():
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Procesar archivo
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
        
        # Contar total de líneas para progreso
        total_lines = sum(1 for _ in infile)
        infile.seek(0)
        
        # Procesar cada línea
        for idx, line in enumerate(infile, start=1):
            data = json.loads(line.strip())
            print(f"Processing {idx}/{total_lines}: {data['source']}")
            
            try:
                translated_text = translate(data["source"])
                
                prediction_entry = {
                    "id": data["id"],
                    "source_language": "English",
                    "target_language": "Spanish",
                    "text": data["source"],
                    "prediction": translated_text
                }
                
                outfile.write(json.dumps(prediction_entry, ensure_ascii=False) + "\n")
                outfile.flush()  # Asegurar escritura inmediata
                
            except Exception as e:
                print(f"Error processing line {idx}: {e}")
                continue
    
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    # Liberar memoria CUDA antes de empezar
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    process_and_save()