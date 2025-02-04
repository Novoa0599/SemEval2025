import sys
import os
import pandas as pd
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utilities.data_cleaning_utils import clean_text

def custom_cleaning(dataframe, output_file_path, text_columns):
    """
    Cleans specified text columns in a DataFrame and saves the cleaned version.

    Args:
        dataframe (pd.DataFrame): Input data.
        output_file_path (str): Path to save cleaned file.
        text_columns (list): List of columns to clean.
    """
    logger.debug(f"Starting data cleaning. Initial Shape: {dataframe.shape}")

    # Remove duplicates based on 'id' if present
    if 'id' in dataframe.columns:
        original_shape = dataframe.shape
        dataframe = dataframe.drop_duplicates(subset='id', keep='first')
        logger.debug(f"Removed duplicates based on 'id'. New Shape: {dataframe.shape} (Was {original_shape})")
    else:
        logger.warning("No 'id' column found. Ensure your DataFrame has unique identifiers.")

    # Clean specified text columns
    for col in text_columns:
        if col in dataframe.columns:
            language = 'english' if col in ['en', 'source'] else 'spanish'
            
            # Progress bar
            pbar = tqdm(total=len(dataframe), desc=f'Cleaning {col}')
            
            cleaned_texts = []
            for text in dataframe[col]:
                cleaned_text = clean_text(text, 'configs/data_config.json', language=language) if isinstance(text, str) else text
                cleaned_texts.append(cleaned_text)
                pbar.update(1)

            dataframe[col] = cleaned_texts
            pbar.close()

    # Save cleaned DataFrame
    dataframe.to_csv(output_file_path, index=False, sep=';')
    logger.debug(f"Data cleaned and saved to {output_file_path}. Final Shape: {dataframe.shape}")
    
    return dataframe

def main():
    # Paths
    input_train_final = "data/processed/train/concatenate_train_final.csv"
    output_train_final = "data/processed/train/entities_cleaned.csv"
    
    input_train_labels = "data/processed/train/train_with_labels.csv"
    output_train_labels = "data/processed/train/train_with_labels_cleaned.csv"

    try:
        # Clean `concatenate_train_final.csv`
        df_train_final = pd.read_csv(input_train_final, sep=';')
        custom_cleaning(df_train_final, output_train_final, text_columns=['en', 'es'])

        # Clean `train_with_labels.csv`
        df_train_labels = pd.read_csv(input_train_labels, sep=';')
        df_train_labels = df_train_labels[['id','source_locale','target_locale','source','target']]
        custom_cleaning(df_train_labels, output_train_labels, text_columns=['source', 'target'])

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    main()