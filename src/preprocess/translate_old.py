from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import pandas as pd
import sys
import os
from tqdm import tqdm  # Import tqdm for progress bars
from ..utils.logging_setup import logger
import time

def translate_batch(df, model, tokenizer, start_idx=0, batch_size=50):
    """
    Translate a batch of rows starting from `start_idx`.
    Args:
        df (DataFrame): The CSV data to translate.
        model: The translation model.
        tokenizer: The translation tokenizer.
        start_idx (int): Index to start translating from.
        batch_size (int): Number of rows to translate in a batch.
    Returns:
        DataFrame: DataFrame with translated rows.
    """
    end_idx = min(start_idx + batch_size, len(df))
    batch_df = df.iloc[start_idx:end_idx]
    
    translations = []
    for text in tqdm(batch_df['text'], desc=f"Translating rows {start_idx+1} to {end_idx}", unit="row"):
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        translated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["pl_PL"])
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        translations.append(translated_text)
        time.sleep(0)  # Prevent rate-limiting, adjust as needed
    
    batch_df['translated_text'] = translations
    return batch_df

def process_csv(input_csv, output_csv, model, tokenizer, batch_size=50):
    """
    Process the CSV, checking which rows need translation and translating them.
    """
    # Check if the output file already exists and read it if so
    if os.path.exists(output_csv):
        translated_df = pd.read_csv(output_csv)
        # Find the last index of the translated text column
        last_translated_idx = translated_df['translated_text'].notna().sum()
    else:
        translated_df = pd.DataFrame()  # If no output file, start fresh
        last_translated_idx = 0
    
    # Load the CSV to translate
    df = pd.read_csv(input_csv)

    # If no rows have been translated yet, we start from the beginning
    if last_translated_idx == 0:
        logger.info("Starting fresh translation.")
    else:
        logger.info(f"Resuming translation from row {last_translated_idx}.")

    # Translate in batches
    for start_idx in tqdm(range(last_translated_idx, len(df), batch_size), desc="Total Translation Progress", unit="batch"):
        batch_df = translate_batch(df, model, tokenizer, start_idx=start_idx, batch_size=batch_size)
        translated_df = pd.concat([translated_df, batch_df], ignore_index=True)

        # Save the partially translated CSV
        translated_df.to_csv(output_csv, index=False)
        logger.info(f"Saved translated rows up to index {start_idx + batch_size}.")
    
    logger.info("Translation complete!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python translate.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]

    # Load the model and tokenizer
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    # Specify the CSV files
    input_csv = "data/civil_comments_train.csv"  # Input file with comments in English
    output_csv = "data/translated/civil_comments_train_pl.csv"  # Output file for translated comments

    # Start the translation process
    process_csv(input_csv, output_csv, model, tokenizer)
