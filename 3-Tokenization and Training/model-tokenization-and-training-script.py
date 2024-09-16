import pandas as pd
import torch
import os
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from tqdm.auto import tqdm
from typing import Dict, List, Optional

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Get the data file path from the user
print("Step 1: Enter the path to your CSV file.")
while True:
    file_path = input("Please enter the full path to your CSV file: ").strip()
    if os.path.isfile(file_path) and file_path.lower().endswith('.csv'):
        break
    else:
        print("Invalid file path or not a CSV file. Please try again.")

# Step 2: Load and preprocess the data
print("Step 2: Loading and preprocessing data...")
try:
    data = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {data.shape}")
except Exception as e:
    print(f"Error loading the file: {e}")
    exit(1)

# Initialize tokenizer
print("Initializing tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Function to process and tokenize data
def process_data(data):
    data = data.rename(columns={'text': 'cleaned_text'})
    data['sentiment'] = data['sentiment'].map({1: 0, 2: 1})
    texts = data['cleaned_text'].tolist()
    labels = data['sentiment'].tolist()
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    return Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })

# Process data
print("Processing data...")
dataset = process_data(data)
print("Data processing complete.")

# Step 3: Set up model
print("Step 3: Setting up model...")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.to(device)

# Step 4: Set training parameters
print("Step 4: Setting training parameters.")
batch_size = 32  # Adjust this based on your GPU memory

# Step 5: Set up training arguments
print("Step 5: Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,  # Maximum number of epochs
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1,
    fp16=True,  # Enable mixed precision training
    dataloader_num_workers=0,  # Set to 0 to avoid potential multiprocessing issues
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Custom Trainer class to implement minimum epochs and early stopping
class CustomTrainer(Trainer):
    def __init__(self, *args, min_epochs=5, early_stopping_patience=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_epochs = min_epochs
        self.early_stopping_patience = early_stopping_patience
        self.best_metric = float('inf')
        self.no_improvement_count = 0

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = super().training_step(model, inputs)
        if self.state.epoch >= self.min_epochs:
            self._check_early_stopping()
        return loss

    def _check_early_stopping(self):
        eval_metric = self.evaluate()['eval_loss']
        if eval_metric < self.best_metric:
            self.best_metric = eval_metric
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.early_stopping_patience:
                self.control.should_training_stop = True
                print(f"Early stopping triggered at epoch {self.state.epoch}")

# Step 6: Initialize Trainer
print("Step 6: Initializing Trainer...")
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,  # Using same dataset for evaluation (not ideal, but workable for now)
    min_epochs=5,
    early_stopping_patience=2
)

# Step 7: Train the model
print("Step 7: Training model...")
try:
    trainer.train()
except Exception as e:
    print(f"An error occurred during training: {e}")
    exit(1)

# Step 8: Save the model
print("Step 8: Saving model...")
model_save_path = "./distilbert_sentiment_model"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Training complete. Model and tokenizer saved to {model_save_path}")