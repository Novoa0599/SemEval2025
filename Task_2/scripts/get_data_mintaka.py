import re
import json

# Ruta del archivo
file_path = "data/raw/train/es/mintaka_train.json"

# Leer el archivo JSON
with open(file_path, 'r', encoding='utf-8') as file:
    data = file.read()

# Buscar todos los identificadores Qxxxxxx
matches = re.findall(r'Q\d+', data)

# Mostrar los resultados únicos
unique_matches = set(matches)  # Eliminar duplicados
print(f"Identificadores encontrados ({len(unique_matches)}):", unique_matches)

# Opcional: guardar los resultados en un archivo de texto
output_path = "data/processed/mintaka/q_identifiers.txt"
with open(output_path, 'w', encoding='utf-8') as output_file:
    output_file.write("\n".join(sorted(unique_matches)))

print(f"Identificadores guardados en: {output_path}")
