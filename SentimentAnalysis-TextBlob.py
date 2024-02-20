from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_reviews(file_path):
    """Read reviews from a file."""
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as fh:
            return fh.readlines()
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path, header=None, encoding='latin1', on_bad_lines='skip')
        # Replace NaN values with empty strings
        df.fillna('', inplace=True)
        return df.iloc[:, 0].tolist()  # Assuming the review text is in the first column
    else:
        raise ValueError("Unsupported file format. Only .txt and .csv files are supported.")


def analyze_sentiment(reviews):
    """Analyze sentiment of reviews."""
    labels = ["Negative", "Neutral", "Positive"]
    values = [0, 0, 0]
    for review in reviews:
        sentiment = TextBlob(review)
        polarity = sentiment.polarity
        if polarity < -0.1:
            values[0] += 1  # Negative
        elif polarity > 0.1:
            values[2] += 1  # Positive
        else:
            values[1] += 1  # Neutral
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
    try:
        reviews = read_reviews(file_path)

        # Analyze sentiment
        sentiment_labels, sentiment_values = analyze_sentiment(reviews)
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
