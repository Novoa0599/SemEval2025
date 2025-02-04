import sys
import json
import time
import random
import os
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm

class WikidataIDScraper:
    def __init__(self, output_dir='data/processed/wikidata'):
        self.output_dir = output_dir
        self.ensure_directories()

    def ensure_directories(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def get_results(self, query):
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        return sparql.query().convert()

    def save_ids(self, results, filename, processed_ids):
        with open(filename, 'a', encoding='utf-8') as f:
            for binding in results["results"]["bindings"]:
                entity_id = binding["entity"]["value"].split('/')[-1]
                if entity_id not in processed_ids:
                    processed_ids.add(entity_id)
                    f.write(f"{entity_id}\n")

    def load_processed_ids(self, id_file):
        try:
            with open(id_file, 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f)
        except FileNotFoundError:
            return set()

    def get_query_for_category(self, type_value, offset, limit=1000):
        type_with_prefix = f"wd:{type_value}" if not type_value.startswith("wd:") else type_value
        
        return f"""
        SELECT DISTINCT ?entity WHERE {{
          ?entity wdt:P31/wdt:P279* ?type.
          VALUES ?type {{ {type_with_prefix} }}
          ?entity rdfs:label ?label_es . 
          FILTER(LANG(?label_es) = "es")
        }}
        ORDER BY ?entity
        LIMIT {limit} OFFSET {offset}
        """

    def get_count_for_category(self, type_value):
        try:
            type_with_prefix = f"wd:{type_value}" if not type_value.startswith("wd:") else type_value
            query = f"""
            SELECT (COUNT(DISTINCT ?entity) AS ?count)
            WHERE {{
              ?entity wdt:P31/wdt:P279* ?type.
              VALUES ?type {{ {type_with_prefix} }}
              ?entity rdfs:label ?label_es . 
              FILTER(LANG(?label_es) = "es")
            }}
            """
            results = self.get_results(query)
            return int(results["results"]["bindings"][0]["count"]["value"])
        except Exception as e:
            print(f"Error getting count for {type_value}: {e}")
            return 100000

    def process_category(self, type_value, total_count):
        limit = 500
        offset = 0
        max_retries = 1
        filename = os.path.join(self.output_dir, f'wikidata_ids_{type_value}.txt')
        
        processed_ids = self.load_processed_ids(filename)
        
        with tqdm(total=total_count, desc=f"Procesando IDs de {type_value}") as pbar:
            while offset < total_count:
                retry_count = 0
                success = False
                
                while retry_count < max_retries and not success:
                    try:
                        query = self.get_query_for_category(type_value, offset, limit)
                        results = self.get_results(query)
                        bindings = results["results"]["bindings"]
                        
                        if not bindings:
                            print(f"No se encontraron más resultados después del offset {offset}")
                            pbar.update(limit)
                            offset += limit
                            break
                        
                        self.save_ids(results, filename, processed_ids)
                        
                        success = True
                        pbar.update(len(bindings))
                        offset += limit
                        
                        time.sleep(random.uniform(2, 5))
                        
                    except Exception as e:
                        retry_count += 1
                        print(f"\nError: {e}. Intento {retry_count} de {max_retries}")
                        time.sleep(random.uniform(5, 10))
                
                if not success:
                    print(f"\nError persistente después de {max_retries} intentos.")
                    pbar.update(limit)
                    offset += limit

# ENTITY_TYPES_WITH_IDS = {
#     "Musical work": "Q2188189 - Q105543609 -",    # Obra musical
#     "Artwork": "Q838948",          # Obra de arte
#     "Food": "Q2095 Q25403900",               # Alimento 
#     "Animal": "Q729",              # Animal x
#     "Plant": "Q756",               # Planta -
#     "Book": "Q571",                # Libro
#     "Book series": "Q1667921",     # Serie de libros -
#     "Fictional entity": "Q14897293", # Entidad ficticia -
#     "Landmark": "Q570116 - Q2319498",          # Punto de referencia
#     "Movie": "Q11424",             # Película
#     "Place of worship": "Q24398318", # Lugar de culto -
#     "Natural place": "Q1286517",    # Lugar natural -
#     "TV series": "Q5398426",       # Serie de televisión
#     "Person": "Q5",                # Ser humano
# }

def main():
    scraper = WikidataIDScraper()
    
    # Lista de tipos a procesar
    types = ["Q2095", "Q25403900"]
    
    for type_value in types:
        try:
            total_count = scraper.get_count_for_category(type_value)
            print(f"Categoría {type_value}: {total_count} entidades totales")
            
            scraper.process_category(type_value, total_count)
        except Exception as e:
            print(f"Error al procesar la categoría {type_value}: {e}")

if __name__ == "__main__":
    main()