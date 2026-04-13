import pandas as pd
from pathlib import Path

# Define paths for raw data and output files
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'

RAW_CSV = DATA_DIR / 'Diabetes_and_LifeStyle_Dataset.csv'
TEST_OUT = DATA_DIR / 'test.csv'
TRAIN_OUT = DATA_DIR / 'train.csv'

# Function to load the raw CSV, split it into train and test sets, and save them as separate CSV files
def load_and_split(test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(RAW_CSV)
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['diabetes_stage'])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)
    print(f"Saved: train ({len(train)} rows), test ({len(test)} rows)")
    return train, test

#Only runs if you run the prepare_data.py script directly, not when imported as a module
if __name__ == '__main__':
    train, test = load_and_split()
