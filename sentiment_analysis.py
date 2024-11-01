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
GITHUB_TOKEN = os.environ.get('PERSONAL_ACCESS_TOKEN')
INPUT_GIST_IDS = [os.environ.get('GIST_ID_1'), os.environ.get('GIST_ID_2')]
OUTPUT_GIST_IDS = [
    os.environ.get('RESULTS_GIST_ID_1'),  # HTML report and visualizations
    os.environ.get('RESULTS_GIST_ID_2'),  # First part of CSV data
    os.environ.get('RESULTS_GIST_ID_3'),  # Second part of CSV data
    os.environ.get('RESULTS_GIST_ID_4')   # Third part of CSV data
]

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

# Create HTML report
html_report = f"""
<html>
<head>
    <title>Museum Reviews Sentiment Analysis</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1, h2 {{ color: #2c3e50; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-box {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }}
        .visualization {{ margin: 30px 0; text-align: center; }}
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

        <div class="download-links">
            <h2>Download Links</h2>
            <p>The complete dataset has been split into three parts for easier downloading:</p>
            <ul>
                <li>Part 1: Gist {OUTPUT_GIST_IDS[1]}</li>
                <li>Part 2: Gist {OUTPUT_GIST_IDS[2]}</li>
                <li>Part 3: Gist {OUTPUT_GIST_IDS[3]}</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

# Split data into three parts for CSV output
total_rows = len(df)
chunk_size = total_rows // 3
remainder = total_rows % 3

splits = [
    df.iloc[:chunk_size + (1 if remainder > 0 else 0)],
    df.iloc[chunk_size + (1 if remainder > 0 else 0):2*chunk_size + (2 if remainder > 1 else 1 if remainder > 0 else 0)],
    df.iloc[2*chunk_size + (2 if remainder > 1 else 1 if remainder > 0 else 0):]
]

print(f"Splitting data into chunks of approximately {chunk_size} rows each")

# Update results Gists
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

# Update visualization gist
update_gist(OUTPUT_GIST_IDS[0], {
    'museum_reviews_analysis.html': html_report,
    'sentiment_summary.json': json.dumps({
        'total_reviews': len(df),
        'average_sentiment': float(df['sentiment_score'].mean()),
        'sentiment_distribution': sentiment_counts.to_dict(),
        'timestamp': pd.Timestamp.now().isoformat()
    }, indent=2)
})

# Update CSV data gists
for i, split_df in enumerate(splits, 1):
    update_gist(OUTPUT_GIST_IDS[i], {
        f'sentiment_results_part{i}.csv': split_df.to_csv(index=False)
    })

print("Analysis complete! Results have been updated to all Gists.")
print(f"Total reviews processed: {len(df)}")
print("\nSplit sizes:")
for i, split_df in enumerate(splits, 1):
    print(f"Part {i}: {len(split_df)} reviews")
print("\nSentiment Distribution:")
print(df['sentiment_category'].value_counts())
