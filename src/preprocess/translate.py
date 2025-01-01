import os
import pandas as pd
from google.cloud import translate_v2 as translate
from tqdm import tqdm
from ..utils.logging_setup import logger
import time

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'YOUR PATH TO GOOGLE CLOUD KEY.json'

client = translate.Client()

def translate_batch(texts, target_language='pl', max_retries=5, base_wait_time=1):
    """Translates a batch of texts using Google Translate API with retry logic."""
    attempt = 0
    while attempt < max_retries:
        try:
            result = client.translate(texts, target_language=target_language)
            return [res['translatedText'] for res in result]
        except Exception as e:
            attempt += 1
            wait_time = base_wait_time * (2 ** (attempt - 1))
            if attempt < max_retries:
                logger.warning(f"Translation attempt {attempt} failed: {e}. Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Translation failed after {max_retries} attempts: {e}")
                raise

def process_csv(input_csv, output_csv, batch_size=50, char_limit=5000):
    """Process the CSV, checking which rows need translation and translating them."""
    if os.path.exists(output_csv):
        translated_df = pd.read_csv(output_csv)
        last_translated_idx = translated_df['text'].notna().sum()
    else:
        translated_df = pd.DataFrame()
        last_translated_idx = 0
    
    df = pd.read_csv(input_csv)
    df = df[df['text'].notna()]  # Remove rows where 'text' is NaN

    if last_translated_idx == 0:
        logger.info("Starting fresh translation.")
    else:
        logger.info(f"Resuming translation from row {last_translated_idx}.")

    for start_idx in tqdm(range(last_translated_idx, len(df), batch_size), desc="Total Translation Progress", unit="batch"):
        texts = df['text'][start_idx:start_idx+batch_size].tolist()

        total_chars = sum(len(text) for text in texts)
        
        while total_chars > char_limit and batch_size > 1:
            batch_size -= 1
            texts = df['text'][start_idx:start_idx+batch_size].tolist()
            total_chars = sum(len(text) for text in texts)

        translations = translate_batch(texts, target_language='pl')

        df.loc[start_idx:start_idx+batch_size-1, 'text'] = translations

        df.iloc[start_idx:start_idx+batch_size, :].to_csv(output_csv, mode='a', header=(start_idx == 0), index=False)

    logger.info("Translation complete!")

if __name__ == "__main__":
    input_csv = "data/civil_comments_train.csv"
    output_csv = "data/translated/translated_civil_comments_google.csv"

    process_csv(input_csv, output_csv)
