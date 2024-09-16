import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import pandas as pd

def load_model_and_tokenizer(model_dir):
    """Load the model and tokenizer from the specified directory."""
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def preprocess_data(examples, tokenizer, text_column, label_column):
    """Tokenize and encode the text data."""
    encodings = tokenizer(examples[text_column], truncation=True, padding='max_length', max_length=128)
    encodings['labels'] = [label - 1 for label in examples[label_column]]  # Convert 1,2 to 0,1
    return encodings

def compute_metrics(pred_labels, true_labels):
    """Compute evaluation metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
    accuracy = accuracy_score(true_labels, pred_labels)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_model(model, dataloader, device):
    """Evaluate the model on the given dataloader."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return compute_metrics(all_predictions, all_labels)

def main():
    # Get model directory from user
    while True:
        model_dir = input("Enter the directory path of your trained model: ").strip()
        if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
            break
        else:
            print("Invalid directory or model not found. Please try again.")

    # Get test dataset path from user
    while True:
        test_data_path = input("Enter the path to your test CSV file: ").strip()
        if os.path.isfile(test_data_path) and test_data_path.lower().endswith('.csv'):
            break
        else:
            print("Invalid file path or not a CSV file. Please try again.")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_dir)
    model = model.to(device)
    print("Model and tokenizer loaded successfully.")

    # Load and preprocess the test dataset
    print("Loading and preprocessing test data...")
    df = pd.read_csv(test_data_path)
    print(f"Columns in the CSV file: {df.columns.tolist()}")
    
    # Determine column names
    text_column = 'cleaned_text' if 'cleaned_text' in df.columns else 'text'
    label_column = 'sentiment'
    
    if text_column not in df.columns or label_column not in df.columns:
        print(f"Error: Required columns not found. Need '{text_column}' and '{label_column}'.")
        return

    # Print unique labels
    unique_labels = df[label_column].unique()
    print(f"Unique labels in the dataset: {unique_labels}")
    num_labels = len(unique_labels)
    print(f"Number of unique labels: {num_labels}")

    # Ensure model is configured for binary classification
    if model.num_labels != 2:
        print("Adjusting model for binary classification...")
        model.num_labels = 2
        model.classifier = torch.nn.Linear(model.config.dim, 2)
        model.to(device)

    test_dataset = load_dataset('csv', data_files=test_data_path)['train']
    test_dataset = test_dataset.map(
        lambda examples: preprocess_data(examples, tokenizer, text_column, label_column), 
        batched=True,
        remove_columns=test_dataset.column_names
    )
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Create dataloader
    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluate the model
    print("Evaluating model on test dataset...")
    metrics = evaluate_model(model, dataloader, device)

    # Print results
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

if __name__ == "__main__":
    main()