import requests
import json
import os
import sys
import logging
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class WikidataLabelScraper:
    def __init__(self, max_requests=10, per_seconds=60):
        self.session = self._create_requests_session()
        self.rate_limiter = WikidataRateLimiter(max_requests, per_seconds)
        self.logger = self._setup_logging()

    def _create_requests_session(self):
        """Create a robust requests session with retry mechanism."""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        return session

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        return logging.getLogger(__name__)

    def _load_processed_ids(self, output_file):
        """Load already processed IDs from the output file."""
        processed_ids = set()
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        processed_ids.update(data.keys())
                    except json.JSONDecodeError:
                        continue
        return processed_ids

    def _execute_sparql_query(self, entity_ids):
        """Execute SPARQL query with rate limiting and robust error handling."""
        self.rate_limiter.wait()
        
        try:
            entity_values = " ".join([f"wd:{eid}" for eid in entity_ids])
            
            query = f"""
            SELECT ?entity ?label_es ?label_en WHERE {{
             VALUES ?entity {{ {entity_values} }}
             OPTIONAL {{ ?entity rdfs:label ?label_es . FILTER(LANG(?label_es) = "es") }}
             OPTIONAL {{ ?entity rdfs:label ?label_en . FILTER(LANG(?label_en) = "en") }}
            }}
            """
            
            headers = {
                'Accept': 'application/sparql-results+json',
                'User-Agent': 'WikidataLabelScraper/1.1'
            }
            
            response = self.session.get(
                "https://query.wikidata.org/sparql", 
                params={'query': query}, 
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            return response.json()['results']['bindings']
        
        except requests.RequestException as e:
            self.logger.error(f"SPARQL query failed: {e}")
            return []

    def process_entity_list(self, entity_list):
        """Process a list of entity IDs and save their labels."""
        for id in entity_list:
            try:
                id_file = f'data/processed/wikidata/wikidata_ids_{id}.txt'
                with open(id_file, 'r', encoding='utf-8') as f:
                    entity_ids = [line.strip() for line in f]

                os.makedirs('data/processed/wikidata', exist_ok=True)
                output_file = f'data/processed/wikidata/entity_labels_{id}.jsonl'

                processed_ids = self._load_processed_ids(output_file)
                self.logger.info(f"Loaded {len(processed_ids)} already processed IDs for {id}.")

                entity_ids = [eid for eid in entity_ids if eid not in processed_ids]

                if not entity_ids:
                    self.logger.info(f"All IDs for {id} have already been processed. Skipping.")
                    continue

                batch_size = 150
                for i in range(0, len(entity_ids), batch_size):
                    batch_ids = entity_ids[i:i+batch_size]
                    results = self._execute_sparql_query(batch_ids)
                    
                    mode = 'a' if os.path.exists(output_file) else 'w'
                    with open(output_file, mode, encoding='utf-8') as f:
                        for result in results:
                            entity = result['entity']['value'].split('/')[-1]
                            labels = {
                                'es': result.get('label_es', {}).get('value'),
                                'en': result.get('label_en', {}).get('value')
                            }
                            f.write(json.dumps({entity: labels}, ensure_ascii=False) + '\n')
                    
                    self.logger.info(f"Processed batch {i//batch_size + 1} for {id}")
                    time.sleep(2)

            except Exception as e:
                self.logger.error(f"Unexpected error while processing {id}: {e}")

class WikidataRateLimiter:
    def __init__(self, max_requests=10, per_seconds=60):
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self.requests_made = []

    def wait(self):
        current_time = time.time()
        
        self.requests_made = [
            req for req in self.requests_made 
            if current_time - req < self.per_seconds
        ]

        if len(self.requests_made) >= self.max_requests:
            sleep_time = self.per_seconds - (current_time - self.requests_made[0])
            time.sleep(sleep_time)

        self.requests_made.append(current_time)

def main():
    scraper = WikidataLabelScraper()
    entity_list = ['Q2095', 'Q25403900']  # Example entity types
    scraper.process_entity_list(entity_list)

if __name__ == "__main__":
    main()