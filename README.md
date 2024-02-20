# Sentiment Analysis Project ReadMe

## Overview

This project aims to analyze the sentiment of Amazon food reviews using various natural language processing (NLP) techniques. Three different sentiment analysis methods are employed: TextBlob, BERT transformers, and VADER.

## Data Source

The data used for sentiment analysis is sourced from the Amazon food review dataset available on Kaggle. You can access the dataset [here](https://www.kaggle.com/datasets/satyabrat35/amazon-food-review-dataset).

## Sentiment Analysis Methods

### TextBlob

TextBlob is a simple NLP library that provides tools for text processing tasks such as part-of-speech tagging, noun phrase extraction, and sentiment analysis.

#### Code Implementation:

```python
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Functions for reading reviews, analyzing sentiment, and plotting results
...
