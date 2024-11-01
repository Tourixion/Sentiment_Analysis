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

# Input gists
INPUT_GIST_1 = os.environ.get('GIST_ID_2')
INPUT_GIST_2 = os.environ.get('GIST_ID_1')

# Output gists for GIST_ID_1
OUTPUT_GIST_1 = os.environ.get('RESULTS_GIST_ID_1')  # HTML/viz for input 1
OUTPUT_GIST_2 = os.environ.get('RESULTS_GIST_ID_2')  # CSV part 1
OUTPUT_GIST_3 = os.environ.get('RESULTS_GIST_ID_3')  # CSV part 2
OUTPUT_GIST_4 = os.environ.get('RESULTS_GIST_ID_4')  # CSV part 3

# Output gists for GIST_ID_2
OUTPUT_GIST_5 = os.environ.get('RESULTS_GIST_ID_5')  # HTML/viz for input 2
OUTPUT_GIST_6 = os.environ.get('RESULTS_GIST_ID_6')  # CSV for input 2

headers = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}

def fetch_gist_content(gist_id):
    """Fetch content from a GitHub Gist"""
    print(f"Fetching content from gist ID: {gist_id}")
    response = requests.get(f'https://api.github.com/gists/{gist_id}', headers=headers)
    response.raise_for_status()
    
    gist_data = response.json()
    all_dfs = []
    
    for file_info in gist_data['files'].values():
        content = file_info.get('content') or requests.get(file_info['raw_url']).text
        df = pd.read_csv(StringIO(content))
        all_dfs.append(df)
        print(f"Successfully read {len(df)} reviews from file {file_info['filename']}")
    
    return pd.concat(all_dfs, ignore_index=True)

def create_visualizations(df):
    """Create and return visualization as base64 string"""
    plt.figure(figsize=(15, 8))

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
    plt.close()
    plot_buffer.seek(0)
    return base64.b64encode(plot_buffer.getvalue()).decode()

def create_html_report(df, plot_base64, input_number, output_gists):
    """Create HTML report"""
    return f"""
<html>
<head>
    <title>Museum Reviews Sentiment Analysis - Input {input_number}</title>
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
        <h1>Museum Reviews Sentiment Analysis - Input {input_number}</h1>
        
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
            <p>The CSV data can be found in the following gists:</p>
            <ul>
                {' '.join(f'<li>Part {i+1}: Gist {gist_id}</li>' for i, gist_id in enumerate(output_gists))}
            </ul>
        </div>
    </div>
</body>
</html>
"""

def process_sentiment(df):
    """Process sentiment analysis on dataframe"""
    analyzer = SentimentIntensityAnalyzer()
    
    def get_sentiment(text):
        try:
            scores = analyzer.polarity_scores(str(text))
            return scores['compound'], scores['pos'], scores['neu'], scores['neg']
        except Exception as e:
            print(f"Warning: Error processing text: {str(e)[:100]}")
            return 0, 0, 0, 0

    def get_sentiment_category(score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    print("Calculating sentiment scores...")
    df['sentiment_score'], df['positive_score'], df['neutral_score'], df['negative_score'] = zip(*df['review'].apply(get_sentiment))
    df['sentiment_category'] = df['sentiment_score'].apply(get_sentiment_category)
    return df

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

def process_input_gist(input_gist, report_gist, csv_gists, input_number):
    """Process a single input gist and its corresponding output gists"""
    print(f"\nProcessing input gist {input_number}...")
    
    # Fetch and process data
    df = fetch_gist_content(input_gist)
    df = process_sentiment(df)
    
    # Create visualization
    plot_base64 = create_visualizations(df)
    
    # Create and save HTML report
    html_report = create_html_report(df, plot_base64, input_number, csv_gists)
    sentiment_counts = df['sentiment_category'].value_counts()
    
    update_gist(report_gist, {
        f'museum_reviews_analysis_input{input_number}.html': html_report,
        f'sentiment_summary_input{input_number}.json': json.dumps({
            'total_reviews': len(df),
            'average_sentiment': float(df['sentiment_score'].mean()),
            'sentiment_distribution': sentiment_counts.to_dict(),
            'timestamp': pd.Timestamp.now().isoformat()
        }, indent=2)
    })
    
    # Split and save CSV data
    if len(csv_gists) == 1:
        # Single CSV output
        update_gist(csv_gists[0], {
            f'sentiment_results_input{input_number}.csv': df.to_csv(index=False)
        })
    else:
        # Multiple CSV outputs
        chunk_size = len(df) // len(csv_gists)
        for i, gist_id in enumerate(csv_gists):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < len(csv_gists) - 1 else len(df)
            chunk_df = df.iloc[start_idx:end_idx]
            update_gist(gist_id, {
                f'sentiment_results_input{input_number}_part{i+1}.csv': chunk_df.to_csv(index=False)
            })
    
    print(f"Completed processing input gist {input_number}")
    return len(df)

# Process first input gist
if INPUT_GIST_1:
    total_reviews_1 = process_input_gist(
        INPUT_GIST_1,
        OUTPUT_GIST_1,
        [OUTPUT_GIST_2, OUTPUT_GIST_3, OUTPUT_GIST_4],
        1
    )
    print(f"\nProcessed {total_reviews_1} reviews from input gist 1")

# Process second input gist
if INPUT_GIST_2:
    total_reviews_2 = process_input_gist(
        INPUT_GIST_2,
        OUTPUT_GIST_5,
        [OUTPUT_GIST_6],
        2
    )
    print(f"\nProcessed {total_reviews_2} reviews from input gist 2")

print("\nAnalysis complete!")
