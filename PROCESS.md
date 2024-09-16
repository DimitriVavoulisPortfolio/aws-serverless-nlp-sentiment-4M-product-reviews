# PROCESS.md - Development Process Documentation

## Project Overview

This document outlines the development process and planned deployment for our serverless sentiment analysis system, designed to analyze product reviews using AWS services. The system includes both back-end processing and a front-end user interface.

## Development Stages

### 1. Data Preparation and Model Training

#### Data Collection and Preprocessing
- Collected 2.8 million Amazon product reviews
- Cleaned and preprocessed the text data
- Split the dataset into training and testing sets

#### Model Training
- Chose DistilBERT as the base model for its balance of performance and efficiency
- Implemented and trained the model using PyTorch on the full dataset
- Achieved high accuracy (99.73%) on the test set

### 2. Model Conversion

#### PyTorch to TensorFlow Conversion
- Developed a script to convert the trained PyTorch model to TensorFlow format
- Ensured consistency in tokenization and inference results between PyTorch and TensorFlow versions

### 3. Planned AWS Deployment Design

#### Back-End Architecture

##### Simplified Single-Lambda Design
- Design incorporates a single Lambda function for efficient processing and simple management
- Plan to deploy only the TensorFlow model due to its smaller library footprint in the Lambda environment

##### Model Size Estimation and Storage Strategy

###### TensorFlow Model Size Estimate
- DistilBERT base model: ~250 MB
- TensorFlow library and dependencies: ~100 MB
- Tokenizer and additional files: ~5-10 MB
- Total estimated size: ~355-360 MB

###### Planned Storage Strategy
Given that the estimated total size exceeds the Lambda layer limit of 250 MB, the following strategy is proposed:

1. **S3 Storage**: 
   - Store the entire model in an S3 bucket
   - This approach will allow for easier updates and version control of the model

2. **Planned Lambda Function Implementation**:
   - Design the Lambda function to download the model from S3 upon cold start
   - Plan to implement a caching mechanism to keep the model in memory for subsequent invocations within the same Lambda instance

3. **Performance Considerations**:
   - Initial cold start will have additional latency due to downloading the model from S3
   - Subsequent invocations will benefit from the in-memory cached model
   - Consider using Provisioned Concurrency to keep warm instances with loaded models for critical low-latency scenarios

4. **Cost Implications**:
   - Anticipate small additional cost for S3 storage and data transfer
   - Potential increase in Lambda execution time and memory usage to accommodate model loading

5. **Security**:
   - Plan to implement proper encryption of the model in S3 (e.g., using SSE-S3 or SSE-KMS)
   - Will set up appropriate IAM roles for Lambda to access the S3 bucket

##### Planned Lambda Function Design
- Develop a Python-based Lambda function that will:
  1. Download the model from S3 (if not already cached)
  2. Receive input data, which can be either:
     a. An individual product review text
     b. A CSV file containing multiple reviews
  3. Preprocess the text (for individual reviews) or read and preprocess the CSV file
  4. Perform sentiment analysis using the TensorFlow model
  5. Return the sentiment prediction(s)
- Implement logic to detect input type (individual review or CSV file) and process accordingly
- For CSV files, add batch processing capability to handle multiple reviews efficiently

#### Front-End Architecture

1. **CloudFront Distribution**:
   - Set up a CloudFront distribution to serve the web application
   - Enables low-latency access to the user interface from various geographic locations
   - Provides HTTPS support for secure communication

2. **S3 Bucket for Static Web Hosting**:
   - Host the front-end web application files (HTML, CSS, JavaScript) in an S3 bucket
   - Configure the bucket for static website hosting
   - Set up appropriate bucket policies to allow CloudFront access

3. **API Gateway**:
   - Create an API Gateway to expose the Lambda function as a RESTful API
   - Set up appropriate routes for handling individual reviews and CSV file uploads
   - Implement CORS (Cross-Origin Resource Sharing) to allow requests from the CloudFront distribution

4. **Integration Flow**:
   - User accesses the web application via CloudFront URL
   - Front-end application sends requests to API Gateway
   - API Gateway triggers the Lambda function
   - Lambda function processes the request and returns results
   - Results are displayed to the user through the web interface

## Anticipated Challenges and Proposed Solutions

1. **Large Model Size**
   - Challenge: The model size exceeds Lambda's deployment package limit
   - Proposed Solution: Store the model in S3 and implement a downloading and caching mechanism in the Lambda function

2. **Cold Start Latency**
   - Challenge: Initial invocations of the Lambda function may be slow due to model loading from S3
   - Proposed Solution: Implement model caching within the Lambda function to reduce cold start times for subsequent invocations

3. **Handling Multiple Input Types**
   - Challenge: Supporting both individual reviews and CSV files in the same Lambda function
   - Proposed Solution: Implement input type detection and separate processing paths for each input type

4. **Efficient CSV Processing**
   - Challenge: Processing large CSV files within Lambda execution limits
   - Proposed Solution: Implement batch processing for CSV files to optimize memory usage and execution time

5. **Cost Optimization**
   - Challenge: Balancing performance with AWS costs
   - Proposed Solution: Fine-tune Lambda memory allocation and optimize code for faster execution

6. **Front-End Performance**
   - Challenge: Ensuring fast load times for the web application
   - Proposed Solution: Optimize front-end assets and leverage CloudFront caching

## Future Steps

1. Implement the designed AWS Lambda function and test thoroughly
2. Set up the S3 bucket for model storage and configure appropriate security measures
3. Develop the front-end web application
4. Create and configure the API Gateway to expose the Lambda function
5. Set up CloudFront distribution and S3 bucket for web hosting
6. Integrate front-end with API Gateway
7. Conduct end-to-end testing of the entire system
8. Perform security review and implement necessary measures
9. Optimize performance and conduct load testing
10. Develop a monitoring and logging strategy for the deployed system
11. Create a CI/CD pipeline for automated testing and deployment
12. Explore techniques for further model compression to reduce Lambda cold start times
13. Consider implementing A/B testing capability for future model updates
14. Investigate using Amazon SageMaker for model hosting if Lambda constraints become limiting
15. Enhance CSV processing capabilities, potentially integrating with AWS Glue for larger datasets
16. Plan for implementing result storage in a database for historical analysis and reporting

