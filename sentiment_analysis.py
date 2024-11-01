import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import os
import base64
from io import StringIO
import csv

# GitHub Gist API configuration
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
GIST_ID = os.environ.get('GIST_ID')
RESULTS_GIST_ID = os.environ.get('RESULTS_GIST_ID')

headers = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}

def fetch_gist_content(gist_id):
    """Fetch content from a GitHub Gist"""
    response = requests.get(f'https://api.github.com/gists/{gist_id}', headers=headers)
    response.raise_for_status()
    
    gist_data = response.json()
    first_file = list(gist_data['files'].values())[0]
    return first_file.get('content') or requests.get(first_file['raw_url']).text

def process_reviews(content):
    """Process raw review content into a DataFrame"""
    # Split content into lines and clean them
    lines = content.strip().split('\n')
    cleaned_reviews = []
    
    current_review = []
    for line in lines:
        line = line.strip()
        if line:
            # Check if this is a new review (by checking if it starts with a quote)
            if line.startswith('"') and current_review:
                # Save previous review
                cleaned_reviews.append(' '.join(current_review))
                current_review = []
            
            # Remove any surrounding quotes
            line = line.strip('"')
            current_review.append(line)
    
    # Don't forget the last review
    if current_review:
        cleaned_reviews.append(' '.join(current_review))
    
    # Create DataFrame
    return pd.DataFrame({'review': cleaned_reviews})

print("Fetching data from Gist...")
content = fetch_gist_content(GIST_ID)

print("Processing reviews...")
df = process_reviews(content)
print(f"Successfully loaded {len(df)} reviews")

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    """Get sentiment score for a piece of text"""
    try:
        # Handle emoji by removing them (or you could keep them if VADER handles them well)
        text = ''.join(char for char in str(text) if ord(char) < 0x10000)
        return analyzer.polarity_scores(text)['compound']
    except Exception as e:
        print(f"Warning: Error processing text: {str(e)[:100]}")
        return 0

print("Calculating sentiment scores...")
df['sentiment_score'] = df['review'].apply(get_sentiment)

def get_sentiment_category(score):
    """Categorize sentiment scores"""
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment_category'] = df['sentiment_score'].apply(get_sentiment_category)

# Create visualizations
print("Generating visualizations...")
plt.style.use('seaborn')
fig = plt.figure(figsize=(15, 8))

# Sentiment distribution
plt.subplot(1, 2, 1)
sns.histplot(data=df, x='sentiment_score', bins=30, color='skyblue')
plt.title('Distribution of Sentiment Scores', pad=20)
plt.xlabel('Sentiment Score')
plt.ylabel('Number of Reviews')

# Add a vertical line at 0
plt.axvline(x=0, color='red', linestyle='--', alpha=0.3)

# Sentiment categories
plt.subplot(1, 2, 2)
sentiment_counts = df['sentiment_category'].value_counts()
colors = ['#2ecc71', '#95a5a6', '#e74c3c']
wedges, texts, autotexts = plt.pie(sentiment_counts, 
                                  labels=sentiment_counts.index, 
                                  autopct='%1.1f%%',
                                  colors=colors, 
                                  startangle=90)
plt.title('Sentiment Distribution', pad=20)

# Make the percentage labels easier to read
plt.setp(autotexts, size=9, weight="bold")
plt.setp(texts, size=10)

plt.tight_layout()

# Save plot to bytes
from io import BytesIO
plot_buffer = BytesIO()
plt.savefig(plot_buffer, format='png', dpi=300, bbox_inches='tight')
plot_buffer.seek(0)
plot_base64 = base64.b64encode(plot_buffer.getvalue()).decode()

# Create HTML report
html_report = f"""
<html>
<head>
    <title>Museum Reviews Sentiment Analysis</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1, h2 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background-color: white; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f8f9fa; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-box {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }}
        .visualization {{ margin: 30px 0; text-align: center; }}
        .reviews-sample {{ margin-top: 30px; }}
        .review-card {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Museum Reviews Sentiment Analysis</h1>
        
        <div class="stats-grid">
            <div class="stat-box">
                <h3>Total Reviews</h3>
                <p>{len(df)}</p>
            </div>
            <div class="stat-box">
                <h3>Average Sentiment</h3>
                <p>{df['sentiment_score'].mean():.3f}</p>
            </div>
            <div class="stat-box">
                <h3>Positive Reviews</h3>
                <p>{(df['sentiment_category'] == 'Positive').sum()} ({(df['sentiment_category'] == 'Positive').mean()*100:.1f}%)</p>
            </div>
        </div>

        <div class="visualization">
            <h2>Sentiment Analysis Visualization</h2>
            <img src="data:image/png;base64,{plot_base64}" alt="Sentiment Analysis Visualization">
        </div>

        <div class="reviews-sample">
            <h2>Sample Reviews by Sentiment</h2>
            {
            ''.join(
                f'<div class="review-card"><strong>{cat}</strong> (Score: {row["sentiment_score"]:.3f})<br>{row["review"]}</div>'
                for cat in ['Positive', 'Neutral', 'Negative']
                for row in [df[df['sentiment_category'] == cat].iloc[0]] if not df[df['sentiment_category'] == cat].empty
            )
            }
        </div>
    </div>
</body>
</html>
"""

# Update results Gist
print("Updating results Gist...")
def update_gist(gist_id, filename, content):
    """Update a GitHub Gist with new content"""
    try:
        payload = {
            'files': {
                filename: {
                    'content': content
                }
            }
        }
        response = requests.patch(
            f'https://api.github.com/gists/{gist_id}',
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error updating gist: {e}")
        raise

update_gist(RESULTS_GIST_ID, 'museum_reviews_analysis.html', html_report)

# Save summary as JSON
summary = {
    'total_reviews': len(df),
    'average_sentiment': float(df['sentiment_score'].mean()),
    'sentiment_distribution': sentiment_counts.to_dict(),
    'timestamp': pd.Timestamp.now().isoformat()
}

update_gist(RESULTS_GIST_ID, 'sentiment_summary.json', json.dumps(summary, indent=2))

print("Analysis complete! Results have been updated to the Gist.")
