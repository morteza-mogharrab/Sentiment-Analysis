import matplotlib.pyplot as plt
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

def analyze_sentiment_vader(reviews):
    """Analyze sentiment of reviews using VADER."""
    analyzer = SentimentIntensityAnalyzer()
    labels = ["Negative", "Neutral", "Positive"]
    values = [0, 0, 0]
    for review in reviews:
        sentiment = analyzer.polarity_scores(review)
        if sentiment['compound'] <= -0.05:
            values[0] += 1  # Negative
        elif sentiment['compound'] >= 0.05:
            values[2] += 1  # Positive
        else:
            values[1] += 1  # Neutral
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

        # Analyze sentiment using VADER
        sentiment_labels, sentiment_values = analyze_sentiment_vader(reviews)
        print("Sentiment summarized counts:", sentiment_values)
        # Colors: golden, silver, light gray
        plot_pie(sentiment_labels, sentiment_values, ["#A30000", "#D2D2D2", "#009596"], (0.2, 0, 0), "Sentiment Analysis")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")
