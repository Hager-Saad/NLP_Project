#  Sentiment Analysis using DistilBERT
This project implements a sentiment analysis pipeline using **DistilBERT** and Hugging Face Transformers. It classifies text into **negative**, **neutral**, or **positive** sentiments. The model is fine-tuned on a cleaned and balanced version of the [Kaggle Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset), and served via a Flask API.

# Dataset
https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset

# Requirements
Before running the project, ensure you have the following packages installed:
Flask==2.2.3
torch==2.0.1
transformers==4.31.0
torchmetrics==0.11.0
scikit-learn==1.2.2
numpy==1.24.2
pandas==1.5.3
gunicorn==20.1.0




