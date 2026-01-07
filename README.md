# Detection of Attachment Style Through Textual Conversations

This project analyzes textual conversations to identify and classify an individual’s attachment style (anxious, avoidant, or secure) using natural language processing and machine learning techniques.

## Overview
The system processes conversational text data, extracts meaningful linguistic features, and applies supervised machine learning to predict psychological attachment styles based on communication patterns.

## Attachment Styles
The model classifies text into the following categories:
- Anxious
- Avoidant
- Secure

## Dataset
- Input: Textual conversation data
- Labels: Attachment style categories
- Dataset file: `attachment_style.xlsx`

The data is cleaned, preprocessed, and encoded before model training.

## Methodology
- Text preprocessing and normalization  
- Feature extraction using TF-IDF  
- Label encoding  
- Train–test split  
- Model training and prediction  
- Evaluation using classification report and confusion matrix  

## Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  

## How to Run
1. Install dependencies:
pip install -r requirements.txt

2. Run the model:


## Output
- Precision, recall, and F1-score for each attachment style
- Confusion matrix visualization
- Overall classification accuracy

## Applications
- Behavioral and psychological analysis  
- Mental health research support  
- Conversational AI personalization  

## Author
Ambica Natraj
