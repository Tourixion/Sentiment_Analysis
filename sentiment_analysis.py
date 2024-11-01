import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Read the data
df = pd.read_csv('data/reviews.csv')

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get compound sentiment score
def get_sentiment(text):
    try:
        return analyzer.polarity_scores(str(text))['compound']
    except:
        return 0

# Apply sentiment analysis to reviews
df['sentiment_score'] = df['review'].apply(get_sentiment)

# Add sentiment category
def get_sentiment_category(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment_category'] = df['sentiment_score'].apply(get_sentiment_category)

# Save detailed results
df.to_csv('results/sentiment_scores.csv', index=False)

# Create visualization
plt.figure(figsize=(12, 6))

# Create subplot for sentiment distribution
plt.subplot(1, 2, 1)
sns.histplot(data=df, x='sentiment_score', bins=50)
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Count')

# Create subplot for sentiment categories
plt.subplot(1, 2, 2)
sentiment_counts = df['sentiment_category'].value_counts()
colors = ['#2ecc71', '#95a5a6', '#e74c3c']
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90)
plt.title('Sentiment Categories Distribution')

plt.tight_layout()
plt.savefig('results/sentiment_summary.png')

# Print summary statistics
summary = df['sentiment_category'].value_counts().to_frame()
summary['percentage'] = (summary['sentiment_category'] / len(df) * 100).round(2)
print("\nSentiment Analysis Summary:")
print(summary)

# Print average sentiment score
print(f"\nAverage Sentiment Score: {df['sentiment_score'].mean():.3f}")
