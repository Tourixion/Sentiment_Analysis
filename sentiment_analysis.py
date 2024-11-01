import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import os
import base64
from io import StringIO, BytesIO
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
    # Split by newlines but preserve quoted content
    reviews = []
    current_review = []
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            if current_review:
                reviews.append(' '.join(current_review))
                current_review = []
            continue
            
        # Handle quoted content
        if line.startswith('"') and not line.endswith('"'):
            # Start of a quoted review
            if current_review:
                reviews.append(' '.join(current_review))
            current_review = [line.strip('"')]
        elif line.endswith('"') and not line.startswith('"'):
            # End of a quoted review
            current_review.append(line.strip('"'))
            reviews.append(' '.join(current_review))
            current_review = []
        elif line.startswith('"') and line.endswith('"'):
            # Single-line quoted review
            if current_review:
                reviews.append(' '.join(current_review))
            reviews.append(line.strip('"'))
            current_review = []
        else:
            # Part of a multi-line review or unquoted review
            if line:
                current_review.append(line)
    
    # Don't forget the last review
    if current_review:
        reviews.append(' '.join(current_review))
    
    # Create DataFrame with index
    return pd.DataFrame({'review': reviews, 'original_order': range(len(reviews))})

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
        text = ''.join(char for char in str(text) if ord(char) < 0x10000)
        scores = analyzer.polarity_scores(text)
        return scores['compound'], scores['pos'], scores['neu'], scores['neg']
    except Exception as e:
        print(f"Warning: Error processing text: {str(e)[:100]}")
        return 0, 0, 0, 0

print("Calculating sentiment scores...")
# Apply sentiment analysis and create separate columns for each score
df['sentiment_score'], df['positive_score'], df['neutral_score'], df['negative_score'] = zip(*df['review'].apply(get_sentiment))

def get_sentiment_category(score):
    """Categorize sentiment scores"""
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment_category'] = df['sentiment_score'].apply(get_sentiment_category)

# Sort by original order
df = df.sort_values('original_order')

# Create visualizations
print("Generating visualizations...")
fig = plt.figure(figsize=(15, 8))

# Sentiment distribution
plt.subplot(1, 2, 1)
plt.hist(df['sentiment_score'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Number of Reviews')
plt.axvline(x=0, color='red', linestyle='--', alpha=0.3)
plt.grid(True, alpha=0.3)

# Sentiment categories
plt.subplot(1, 2, 2)
sentiment_counts = df['sentiment_category'].value_counts()
colors = ['#2ecc71', '#95a5a6', '#e74c3c']
plt.pie(sentiment_counts, 
        labels=sentiment_counts.index, 
        autopct='%1.1f%%',
        colors=colors, 
        startangle=90)
plt.title('Sentiment Distribution')

plt.tight_layout()

# Save plot to bytes
plot_buffer = BytesIO()
plt.savefig(plot_buffer, format='png', dpi=300, bbox_inches='tight')
plot_buffer.seek(0)
plot_base64 = base64.b64encode(plot_buffer.getvalue()).decode()

# Create detailed results CSV
results_csv = df.to_csv(index=False)

# Create HTML report with table
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
        .table-container {{ max-height: 500px; overflow-y: auto; margin: 20px 0; }}
        .download-links {{ margin: 20px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }}
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
            <img src="data:image/png;base64,{plot_base64}" alt="Sentiment Analysis Visualization" style="max-width: 100%; height: auto;">
        </div>

        <h2>Complete Results Table</h2>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Order</th>
                        <th>Review</th>
                        <th>Sentiment Score</th>
                        <th>Category</th>
                        <th>Positive</th>
                        <th>Neutral</th>
                        <th>Negative</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(
                        f"<tr><td>{i+1}</td><td>{row['review'][:100]}...</td><td>{row['sentiment_score']:.3f}</td>" +
                        f"<td>{row['sentiment_category']}</td><td>{row['positive_score']:.3f}</td>" +
                        f"<td>{row['neutral_score']:.3f}</td><td>{row['negative_score']:.3f}</td></tr>"
                        for i, row in df.iterrows()
                    )}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""

# Update results Gist
print("Updating results Gist...")
def update_gist(gist_id, files_content):
    """Update a GitHub Gist with multiple files"""
    try:
        payload = {
            'files': {
                filename: {'content': content}
                for filename, content in files_content.items()
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

# Update both HTML report and CSV in the same gist
update_gist(RESULTS_GIST_ID, {
    'museum_reviews_analysis.html': html_report,
    'sentiment_results.csv': results_csv,
    'sentiment_summary.json': json.dumps({
        'total_reviews': len(df),
        'average_sentiment': float(df['sentiment_score'].mean()),
        'sentiment_distribution': sentiment_counts.to_dict(),
        'timestamp': pd.Timestamp.now().isoformat()
    }, indent=2)
})

print("Analysis complete! Results have been updated to the Gist.")
print(f"Total reviews processed: {len(df)}")
