import json
import csv
import os
import glob
import ast
import pandas as pd

def extract_q_value(uri):
    """Extraer identificador de una URL."""
    return uri.split("/")[-1] if "http" in uri else uri

def convert_jsonl_to_csv(jsonl_file):
    """Convertir archivo JSONL a CSV."""
    base_name = os.path.basename(jsonl_file).replace(".jsonl", "")
    csv_file = f"data/processed/wikidata/{base_name}.csv"
    
    with open(jsonl_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    csv_data = []
    for line in lines:
        data = json.loads(line.strip())
        
        for entity_id, labels in data.items():
            label_es = labels.get("es", "")
            label_en = labels.get("en", "")
            
            if label_es and label_en:
                csv_data.append({
                    "id": entity_id,
                    "es": label_es,
                    "en": label_en
                })

    with open(csv_file, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=csv_data[0].keys(), delimiter=";")
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"CSV file '{csv_file}' has been created.")

def safe_json_loads(json_str):
    """Manejar comillas simples dentro de JSON."""
    try:
        if isinstance(json_str, dict):
            return json_str
        
        return ast.literal_eval(json_str)
    except (ValueError, SyntaxError):
        try:
            return json.loads(json_str.replace("'", '"'))
        except json.JSONDecodeError:
            print(f"Error al decodificar JSON: {json_str}")
            return {}

def expand_wikidata_labels(input_file, output_file):
    """Expandir etiquetas de Wikidata."""
    df = pd.read_csv(input_file, encoding='utf-8', sep=';')
    df['wikidata_labels'] = df['wikidata_labels'].apply(safe_json_loads)

    expanded_rows = []
    for _, row in df.iterrows():
        wikidata = row['wikidata_labels']
        for key, value in wikidata.items():
            expanded_rows.append({
                'id': key, 
                'es': value.get('es'), 
                'en': value.get('en')
            })

    new_df = pd.DataFrame(expanded_rows)
    new_df.to_csv(output_file, sep=';', encoding='utf-8', index=False)

def combine_csv_files(wikidata_path, train_file, output_file):
    """Combinar archivos CSV de Wikidata y entrenamiento."""
    csv_files = glob.glob(wikidata_path)
    csv_files.append(train_file)

    all_dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, sep=';', encoding='utf-8')
            all_dfs.append(df)
            print(f"Archivo leído: {csv_file}")
        except Exception as e:
            print(f"Error al leer el archivo {csv_file}: {e}")

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df.drop_duplicates(subset=['id'], keep='first')
    final_df.to_csv(output_file, sep=';', encoding='utf-8', index=False)
    print(f"Archivo combinado guardado en: {output_file}")

def main():
    # Convertir archivos JSONL a CSV
    jsonl_files = glob.glob("data/processed/wikidata/*.jsonl")
    for jsonl_file in jsonl_files:
        convert_jsonl_to_csv(jsonl_file)

    # Expandir etiquetas de Wikidata
    expand_wikidata_labels(
        'data/processed/train/train_with_labels.csv', 
        'data/processed/train/entities_traductions.csv'
    )

    # Combinar archivos CSV
    combine_csv_files(
        "/home/daniel-linux/EA-MT-Task/data/processed/wikidata/*.csv",
        "/home/daniel-linux/EA-MT-Task/data/processed/train/entities_traductions.csv",
        "/home/daniel-linux/EA-MT-Task/data/processed/train/concatenate_train_final.csv"
    )

if __name__ == "__main__":
    main()