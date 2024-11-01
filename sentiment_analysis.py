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

# GitHub Gist API configuration
GITHUB_TOKEN = os.environ.get('PERSONAL_ACCESS_TOKEN')  # Changed to match your existing token name
INPUT_GIST_IDS = [os.environ.get('GIST_ID_1'), os.environ.get('GIST_ID_2')]
OUTPUT_GIST_IDS = [os.environ.get('RESULTS_GIST_ID_1'), os.environ.get('RESULTS_GIST_ID_2')]

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
    return pd.read_csv(StringIO(content))

print("Fetching data from Gists...")
# Fetch and combine data from both input gists
dfs = []
for i, gist_id in enumerate(INPUT_GIST_IDS, 1):
    print(f"Processing input gist {i}...")
    content = fetch_gist_content(gist_id)
    df = process_reviews(content)
    dfs.append(df)

# Combine all reviews while maintaining order
df = pd.concat(dfs, ignore_index=True)
print(f"Successfully loaded {len(df)} total reviews")

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    """Get sentiment score for a piece of text"""
    try:
        scores = analyzer.polarity_scores(str(text))
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

# Split results for two output gists
mid_point = len(df) // 2
df1 = df.iloc[:mid_point]
df2 = df.iloc[mid_point:]

def create_html_report(df_part, total_reviews, total_sentiment_counts):
    """Create HTML report for a subset of reviews"""
    return f"""
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
        .table-container {{ max-height: 500px; overflow-y: auto; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Museum Reviews Sentiment Analysis</h1>
        
        <div class="stats-grid">
            <div class="stat-box">
                <h3>Reviews in This Part</h3>
                <p>{len(df_part)} of {total_reviews}</p>
            </div>
            <div class="stat-box">
                <h3>Average Sentiment</h3>
                <p>{df_part['sentiment_score'].mean():.3f}</p>
            </div>
            <div class="stat-box">
                <h3>Overall Positive Reviews</h3>
                <p>{total_sentiment_counts['Positive']} ({total_sentiment_counts['Positive']/total_reviews*100:.1f}%)</p>
            </div>
        </div>

        <div class="visualization">
            <h2>Overall Sentiment Analysis</h2>
            <img src="data:image/png;base64,{plot_base64}" alt="Sentiment Analysis Visualization" style="max-width: 100%; height: auto;">
        </div>

        <h2>Results Table</h2>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
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
                        f"<tr><td>{row['review']}</td><td>{row['sentiment_score']:.3f}</td>" +
                        f"<td>{row['sentiment_category']}</td><td>{row['positive_score']:.3f}</td>" +
                        f"<td>{row['neutral_score']:.3f}</td><td>{row['negative_score']:.3f}</td></tr>"
                        for _, row in df_part.iterrows()
                    )}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""

# Create and update results in both output gists
print("Updating results Gists...")
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

# Get overall statistics for both reports
total_reviews = len(df)
total_sentiment_counts = df['sentiment_category'].value_counts().to_dict()

# Update first output gist
update_gist(OUTPUT_GIST_IDS[0], {
    'museum_reviews_analysis_part1.html': create_html_report(df1, total_reviews, total_sentiment_counts),
    'sentiment_results_part1.csv': df1.to_csv(index=False),
    'sentiment_summary_part1.json': json.dumps({
        'total_reviews': len(df1),
        'average_sentiment': float(df1['sentiment_score'].mean()),
        'sentiment_distribution': df1['sentiment_category'].value_counts().to_dict(),
        'timestamp': pd.Timestamp.now().isoformat()
    }, indent=2)
})

# Update second output gist
update_gist(OUTPUT_GIST_IDS[1], {
    'museum_reviews_analysis_part2.html': create_html_report(df2, total_reviews, total_sentiment_counts),
    'sentiment_results_part2.csv': df2.to_csv(index=False),
    'sentiment_summary_part2.json': json.dumps({
        'total_reviews': len(df2),
        'average_sentiment': float(df2['sentiment_score'].mean()),
        'sentiment_distribution': df2['sentiment_category'].value_counts().to_dict(),
        'timestamp': pd.Timestamp.now().isoformat()
    }, indent=2)
})

print("Analysis complete! Results have been updated to both Gists.")
print(f"Total reviews processed: {len(df)}")
print("\nSentiment Distribution:")
print(df['sentiment_category'].value_counts())
