import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os

def load_model_and_tokenizer(model_dir):
    """Load the model and tokenizer from the specified directory."""
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def predict_sentiment(text, model, tokenizer, device):
    """Predict sentiment for the given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move input tensors to the correct device
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = "positive" if probabilities[0][1] > probabilities[0][0] else "negative"
    confidence = probabilities[0][1].item() if sentiment == "positive" else probabilities[0][0].item()
    
    return sentiment, confidence

def main():
    # Get model directory from user
    while True:
        model_dir = input("Enter the directory path of your trained model: ").strip()
        if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
            break
        else:
            print("Invalid directory or model not found. Please try again.")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_dir)
    model = model.to(device)  # Move model to the correct device
    print("Model and tokenizer loaded successfully.")

    print("\nSentiment Analysis Model Tester")
    print("Enter 'quit' to exit the program.")

    while True:
        text = input("\nEnter text for sentiment analysis: ").strip()
        if text.lower() == 'quit':
            break

        try:
            sentiment, confidence = predict_sentiment(text, model, tokenizer, device)
            print(f"Sentiment: {sentiment}")
            print(f"Confidence: {confidence:.2f}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try again with a different input.")

    print("Thank you for using the Sentiment Analysis Model Tester!")

if __name__ == "__main__":
    main()
