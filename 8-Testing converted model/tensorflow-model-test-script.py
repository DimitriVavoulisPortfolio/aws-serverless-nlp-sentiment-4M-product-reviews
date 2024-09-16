from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import numpy as np

# Set the model path
model_path = r"C:\path\to\"

# Load the model and tokenizer
model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Maximum sequence length (you may need to adjust this based on your model's configuration)
MAX_LENGTH = 128

def preprocess_text(text):
    # Tokenize and encode the text
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf'
    )
    return encoded['input_ids'], encoded['attention_mask']

def predict_sentiment(text):
    # Preprocess the input text
    input_ids, attention_mask = preprocess_text(text)
    
    # Make prediction
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # Convert logits to probabilities
    probabilities = tf.nn.softmax(logits, axis=-1)
    
    # Get the predicted class (1 for negative, 2 for positive)
    predicted_class = tf.argmax(probabilities, axis=-1).numpy()[0] + 1
    
    # Get the confidence score
    confidence = tf.reduce_max(probabilities, axis=-1).numpy()[0]
    
    return predicted_class, confidence

def main():
    print("Sentiment Analysis Model Tester")
    print("Enter 'quit' to exit the program.")
    
    while True:
        user_input = input("\nEnter a product review to analyze: ")
        
        if user_input.lower() == 'quit':
            print("Exiting the program. Goodbye!")
            break
        
        predicted_class, confidence = predict_sentiment(user_input)
        
        sentiment = "Positive" if predicted_class == 2 else "Negative"
        print(f"Predicted sentiment: {sentiment}")
        print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()
