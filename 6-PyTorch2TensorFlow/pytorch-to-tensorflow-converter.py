import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
PYTORCH_MODEL_PATH = r"C:\path\to\"
TENSORFLOW_OUTPUT_PATH = r"C:\path\to\"

def load_pytorch_model(model_path):
    logging.info(f"Loading PyTorch model from {model_path}")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logging.info("PyTorch model and tokenizer loaded successfully")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading PyTorch model: {e}")
        raise

def convert_to_tensorflow(pytorch_model, tokenizer):
    logging.info("Converting PyTorch model to TensorFlow")
    try:
        tf_model = TFAutoModelForSequenceClassification.from_pretrained(PYTORCH_MODEL_PATH, from_pt=True)
        logging.info("Model converted to TensorFlow successfully")
        return tf_model
    except Exception as e:
        logging.error(f"Error converting model to TensorFlow: {e}")
        raise

def save_tensorflow_model(tf_model, tokenizer, output_path):
    logging.info(f"Saving TensorFlow model to {output_path}")
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        tf_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        logging.info("TensorFlow model and tokenizer saved successfully")
    except Exception as e:
        logging.error(f"Error saving TensorFlow model: {e}")
        raise

def main():
    try:
        # Load PyTorch model
        pytorch_model, tokenizer = load_pytorch_model(PYTORCH_MODEL_PATH)

        # Convert to TensorFlow
        tf_model = convert_to_tensorflow(pytorch_model, tokenizer)

        # Save TensorFlow model
        save_tensorflow_model(tf_model, tokenizer, TENSORFLOW_OUTPUT_PATH)

        logging.info("Conversion process completed successfully")

    except Exception as e:
        logging.error(f"An error occurred during the conversion process: {e}")

if __name__ == "__main__":
    main()
