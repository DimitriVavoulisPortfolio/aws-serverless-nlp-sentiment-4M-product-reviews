# Sentiment Analysis Model for Product Reviews

## Project Overview

This project is a production-ready DistilBERT Sentiment Analysis model for product reviews desgined as a low cost market research tool with the nuiance of an actual market researcher, it was trained with a dataset of over 4 million Amazon product reviews to work for a wide variety of products in a highly scalable manner. This project comes with designs for a serverless live demo in AWS. 

### Key Features

- Pre-trained DistilBERT model for sentiment analysis on product reviews
- Designed for low compute costs to maximize scalability
- Conversion from PyTorch to TensorFlow
- Data processing and tokenization pipeline
- Comprehensive testing of both original and converted models
- Designs of an AWS serverless architecture for a live demo with future deployment(TBD)

## Project Structure

1. **1-Data processing**: Script for data preparation
2. **2-CSV files(Production test)**: CSV file used for production testing(Full dataset https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)
3. **3-Tokenization and Training**: Tokenization process and initial model training
4. **4-distilbert_sentiment_model**: Original PyTorch DistilBERT model
5. **5-Testing model**: Scripts for testing the PyTorch model, both a write in and a CSV input script 
6. **6-PyTorch2TensorFlow**: Conversion script from PyTorch to TensorFlow
7. **7-tensorflow_model**: Converted TensorFlow model
8. **8-Testing converted model**: Scripts for testing the TensorFlow model, includes a write in one, and 2 CSV input ones, one for testing the model, one for production usage

## Documentation

- **MODEL.md**: https://github.com/DimitriVavoulisPortfolio/aws-serverless-nlp-sentiment-4M-product-reviews/blob/main/MODEL.md
- **PROCESS.md**: https://github.com/DimitriVavoulisPortfolio/aws-serverless-nlp-sentiment-4M-product-reviews/blob/main/PROCESS.md
- **AWS-IMPLEMENTATION-AND-COST-REPORT.md**: https://github.com/DimitriVavoulisPortfolio/aws-serverless-nlp-sentiment-4M-product-reviews/blob/main/MODEL.md

## Model Performance

- **4M-dataset-sentiment_distribution.png**: Visualization of sentiment distribution in the dataset
- **400K-dataset-testing.png**: Results of model testing on a 400K sample dataset
- **Accuracy: 0.9973**
- **Precision: 0.9968**
- **Recall: 0.9979**
- **F1: 0.9973**
 
## Quick Start Guide

1. Clone the repository:
   ```
   git clone https://github.com/DimitriVavoulisPortfolio/aws-serverless-nlp-sentiment-4M-product-reviews.git
   cd aws-serverless-nlp-sentiment-4M-product-reviews
   ```

2. Install dependencies:
   ```
   pip install tensorflow transformers torch pandas numpy scikit-learn onnx onnx-tf 
   ```

3. To test the PyTorch model:
   ```
   python 5-Testing\ model/test_pytorch_model.py
   ```

4. To test the TensorFlow model:
   ```
   python 8-Testingconverted\ model/test_tensorflow_model.py
   ```

## Future Work

- Implement AWS deployment
- Create API for real-time sentiment prediction
- Optimize model performance and size 

## License

This project is licensed under the Apache-2.0 license - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please open an issue in this repository or contact [Dimitri Vavoulis](mailto:dimitrivavoulis3@gmail.com).
