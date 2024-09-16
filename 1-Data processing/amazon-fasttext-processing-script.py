import bz2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import re

def read_fasttext_file(file_path):
    data = []
    with bz2.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                label = parts[0].replace('__label__', '')
                text = parts[1]
                data.append({'sentiment': int(label), 'text': text})
    return pd.DataFrame(data)

def get_file_path(file_type):
    while True:
        file_path = input(f"Enter the path to the {file_type} file: ").strip()
        if os.path.exists(file_path):
            return file_path
        else:
            print(f"File not found. Please enter a valid path for the {file_type} file.")

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Get the current working directory
current_dir = os.getcwd()

# Get file paths from user
train_file_path = get_file_path("training")
test_file_path = get_file_path("test")

# Read the training data
train_data = read_fasttext_file(train_file_path)

# Read the test data
test_data = read_fasttext_file(test_file_path)

# Display basic information
print("Training data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

# Show the first few rows of training data
print("\nSample of training data:")
print(train_data.head())

# Display summary statistics
print("\nSummary statistics:")
print(train_data.describe())

# Plot sentiment distribution
plt.figure(figsize=(10, 6))
train_data['sentiment'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Sentiments in Training Data')
plt.xlabel('Sentiment (1: Negative, 2: Positive)')
plt.ylabel('Count')

# Save the plot
plot_file = os.path.join(current_dir, 'sentiment_distribution.png')
plt.savefig(plot_file)
plt.close()
print(f"Sentiment distribution plot saved: {plot_file}")

# Clean the text
train_data['cleaned_text'] = train_data['text'].apply(clean_text)
test_data['cleaned_text'] = test_data['text'].apply(clean_text)

# Basic text statistics
train_data['text_length'] = train_data['cleaned_text'].str.len()
print("\nText length statistics:")
print(train_data['text_length'].describe())

# Split training data into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_data['cleaned_text'], train_data['sentiment'], test_size=0.2, random_state=42
)

# Create DataFrames for train and validation sets
train_df = pd.DataFrame({'text': train_texts, 'sentiment': train_labels})
val_df = pd.DataFrame({'text': val_texts, 'sentiment': val_labels})

print("\nSplit sizes:")
print(f"Training set: {len(train_df)}")
print(f"Validation set: {len(val_df)}")
print(f"Test set: {len(test_data)}")

# Save processed data
train_file = os.path.join(current_dir, 'processed_train_data.csv')
val_file = os.path.join(current_dir, 'processed_val_data.csv')
test_file = os.path.join(current_dir, 'processed_test_data.csv')

train_df.to_csv(train_file, index=False)
val_df.to_csv(val_file, index=False)
test_data[['cleaned_text', 'sentiment']].to_csv(test_file, index=False)

print("\nProcessed data saved to CSV files:")
print(f"Training data: {train_file}")
print(f"Validation data: {val_file}")
print(f"Test data: {test_file}")

print("\nScript execution completed.")

# Verify file creation
for file in [train_file, val_file, test_file, plot_file]:
    if os.path.exists(file):
        print(f"File created successfully: {file}")
    else:
        print(f"Error: File not created: {file}")