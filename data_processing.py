import pandas as pd
from sklearn.utils import resample
from datasets import Dataset

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, encoding='latin-1')
    test_df = pd.read_csv(test_path, encoding='latin-1')
    return train_df, test_df

def preprocess_data(train_df, test_df):
    train_df = train_df[["text", "sentiment"]]
    test_df = test_df[["text", "sentiment"]]
    
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    train_df["sentiment"] = train_df["sentiment"].map(label_map)
    test_df["sentiment"] = test_df["sentiment"].map(label_map)
    
    train_df = train_df.drop_duplicates()
    
    max_count = train_df["sentiment"].value_counts().max()
    dfs = [
        resample(train_df[train_df["sentiment"] == label], replace=True,
                 n_samples=max_count, random_state=42)
        for label in train_df["sentiment"].unique()
    ]
    
    train_df_balanced = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_dataset = Dataset.from_pandas(train_df_balanced)
    test_dataset = Dataset.from_pandas(test_df)
    
    return train_dataset, test_dataset
