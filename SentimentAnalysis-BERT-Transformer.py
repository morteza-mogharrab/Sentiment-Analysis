from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def read_reviews(file_path):
    """Read reviews from a file."""
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as fh:
            return fh.readlines()
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path, header=None, encoding='latin1', error_bad_lines=False)
        df.fillna('', inplace=True)
        return df.iloc[:, 0].tolist()  # Assuming the review text is in the first column
    else:
        raise ValueError("Unsupported file format. Only .txt and .csv files are supported.")

def analyze_sentiment(reviews, model, tokenizer, device):
    """Analyze sentiment of reviews using a pre-trained sentiment analysis model."""
    labels = ["Negative", "Neutral", "Positive"]
    values = [0, 0, 0]

    # Tokenize all reviews at once
    inputs = tokenizer(reviews, return_tensors='pt', truncation=True, padding=True)
    inputs = inputs.to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted sentiment labels
    _, predicted_labels = torch.max(outputs.logits, dim=1)

    # Count sentiment labels
    for label in predicted_labels:
        values[label.item()] += 1

    return labels, values

def analyze_subjectivity(reviews):
    """Analyze subjectivity of reviews."""
    labels = ["Objective", "Subjective"]
    values = [0, 0]
    for review in reviews:
        sentiment = TextBlob(review)
        subjectivity = round(sentiment.subjectivity)
        values[subjectivity] += 1
    return labels, values

def plot_pie(labels, values, colors, explode, title):
    """Plot a pie chart."""
    plt.figure(figsize=(10, 10))
    plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140, explode=explode)
    plt.title(title)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    file_path = "Comments.txt"  # Example file path

    # Load pre-trained sentiment analysis model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('textattack/bert-base-uncased-imdb')
    model = BertForSequenceClassification.from_pretrained('textattack/bert-base-uncased-imdb')
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    try:
        reviews = read_reviews(file_path)

        # Analyze sentiment
        sentiment_labels, sentiment_values = analyze_sentiment(reviews, model, tokenizer, device)
        print("Sentiment summarized counts:", sentiment_values)
        # Colors: golden, silver, light gray
        plot_pie(sentiment_labels, sentiment_values, ["#A30000", "#D2D2D2", "#009596"], (0.2, 0, 0), "Sentiment Analysis")

        # Analyze subjectivity
        subjectivity_labels, subjectivity_values = analyze_subjectivity(reviews)
        print("Subjectivity summarized counts:", subjectivity_values)
        # Colors: golden, silver
        plot_pie(subjectivity_labels, subjectivity_values, ["#EC7A08", "#8A8D90"], (0.2, 0), "Subjectivity Analysis")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")
