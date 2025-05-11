# %%
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils import resample
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import Dataset
import numpy as np

# %%
# Load your datasets (make sure these CSVs are in your project directory)
train_df = pd.read_csv(r"D:\Zewail city\ZC 3rd Year\2nd Semester\Natural language processing\NLP_project\NLP_project\train.csv", encoding='latin-1')
test_df = pd.read_csv(r"D:\Zewail city\ZC 3rd Year\2nd Semester\Natural language processing\NLP_project\NLP_project\test.csv", encoding='latin-1')


# %%
# Keep only the relevant columns
train_df = train_df[["text", "sentiment"]]
test_df = test_df[["text", "sentiment"]]

# %%
# Check for nulls
print("Train nulls:\n", train_df.isnull().sum())
print("Test nulls:\n", test_df.isnull().sum())

# %%
# Encode labels
label_map = {"negative": 0, "neutral": 1, "positive": 2}
train_df["sentiment"] = train_df["sentiment"].map(label_map)
test_df["sentiment"] = test_df["sentiment"].map(label_map)

# %%
# Optional: visualize label distribution
print(train_df['sentiment'].value_counts(normalize=True))
train_df['sentiment'].value_counts(normalize=True).plot(kind='bar')

# %%
# Remove duplicates
train_df = train_df.drop_duplicates()

# %%
# Balance the training set using upsampling
max_count = train_df["sentiment"].value_counts().max()
dfs = [
    resample(train_df[train_df["sentiment"] == label], replace=True,
             n_samples=max_count, random_state=42)
    for label in train_df["sentiment"].unique()
]

train_df_balanced = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)

# %%
# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df_balanced)
test_dataset = Dataset.from_pandas(test_df)

# %%
# Tokenization
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    batch["text"] = [str(text) for text in batch["text"]]
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# %%
# Rename label column and convert to PyTorch format
train_dataset = train_dataset.rename_column("sentiment", "labels")
test_dataset = test_dataset.rename_column("sentiment", "labels")
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# %%
# Load model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

# %%
# Define metrics
def compute_metrics(eval_pred: tuple) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# %%
# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    run_name="sentiment-classification",
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir='./logs',
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_strategy="epoch"
)

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=2,
    early_stopping_threshold=0.01
)

#%%
print(train_dataset[0]["labels"], type(train_dataset[0]["labels"]))
print(train_dataset[0]["labels"].dtype)


# %%
# Define data collator to fix label type issues
class DataCollatorWithCorrectLabels(DataCollatorWithPadding):
    def __call__(self, features):  # <-- correct method name
        batch = super().__call__(features)
        batch["labels"] = batch["labels"].long()
        return batch


collator = DataCollatorWithCorrectLabels(tokenizer=tokenizer)


# %%
# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    callbacks=[early_stopping_callback],
    data_collator=collator
)

# %%
# Train the model
trainer.train()

# %%
# Evaluate the model
results = trainer.evaluate()
print("Evaluation Results:", results)

#%%
# Save model and tokenizer
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")