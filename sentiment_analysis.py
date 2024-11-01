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
    # Get the first file's content
    first_file = list(gist_data['files'].values())[0]
    if 'content' in first_file:
        return first_file['content']
    else:
        raw_url = first_file['raw_url']
        raw_response = requests.get(raw_url)
        raw_response.raise_for_status()
        return raw_response.text

def update_gist(gist_id, filename, content):
    """Update a GitHub Gist with new content"""
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

# Fetch data from Gist
print("Fetching data from Gist...")
csv_content = fetch_gist_content(GIST_ID)
df = pd.read_csv(StringIO(csv_content))

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

print(f"Processing {len(df)} reviews...")

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

# Create visualizations
plt.figure(figsize=(12, 6))

# Sentiment distribution
plt.subplot(1, 2, 1)
sns.histplot(data=df, x='sentiment_score', bins=50)
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Count')

# Sentiment categories
plt.subplot(1, 2, 2)
sentiment_counts = df['sentiment_category'].value_counts()
colors = ['#2ecc71', '#95a5a6', '#e74c3c']
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90)
plt.title('Sentiment Categories Distribution')

plt.tight_layout()

# Save plot to bytes
from io import BytesIO
plot_buffer = BytesIO()
plt.savefig(plot_buffer, format='png')
plot_buffer.seek(0)
plot_base64 = base64.b64encode(plot_buffer.getvalue()).decode()

# Create HTML report with embedded image
html_report = f"""
<html>
<head>
    <title>Sentiment Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Sentiment Analysis Report</h1>
        <h2>Summary Statistics</h2>
        <table>
            <tr>
                <th>Category</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
            {''.join(f"<tr><td>{cat}</td><td>{count}</td><td>{count/len(df)*100:.1f}%</td></tr>" 
                     for cat, count in sentiment_counts.items())}
        </table>
        
        <h2>Average Sentiment Score: {df['sentiment_score'].mean():.3f}</h2>
        
        <h2>Visualization</h2>
        <img src="data:image/png;base64,{plot_base64}" alt="Sentiment Analysis Visualization">
        
        <h2>Detailed Results</h2>
        {df[['review', 'sentiment_score', 'sentiment_category']].to_html()}
    </div>
</body>
</html>
"""

# Update results Gist
print("Updating results Gist...")
update_gist(RESULTS_GIST_ID, 'sentiment_analysis_report.html', html_report)

# Save summary as JSON
summary = {
    'total_reviews': len(df),
    'average_sentiment': float(df['sentiment_score'].mean()),
    'sentiment_distribution': sentiment_counts.to_dict(),
    'timestamp': pd.Timestamp.now().isoformat()
}

update_gist(RESULTS_GIST_ID, 'sentiment_summary.json', json.dumps(summary, indent=2))

print("Analysis complete! Results have been updated to the Gist.")
