import pandas as pd
from transformers import pipeline
import os

def analyze_sentiment_distilbert(news_data):
    # Initialize the sentiment analysis pipeline with distilbert
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    results = []
    for index, row in news_data.iterrows():
        news_sentence = row['Sentence']
        # Analyze the sentiment of the news sentence
        result = sentiment_pipeline(news_sentence)
        results.append(result)
        print(f"Processed news {index + 1}/{len(news_data)}")
    
    return results

def analyze_sentiment_bert_baseline(news_data):
    # Initialize the sentiment analysis pipeline with bert-base-uncased
    sentiment_pipeline = pipeline("sentiment-analysis", model="bert-base-uncased")
    
    results = []
    for index, row in news_data.iterrows():
        news_sentence = row['Sentence']
        # Analyze the sentiment of the news sentence
        result = sentiment_pipeline(news_sentence)
        results.append(result)
        print(f"Processed news {index + 1}/{len(news_data)}")
    
    return results

def main():
    # Read the CSV file
    file_path = 'data/efsa_sentiment_classification.csv'
    news_data = pd.read_csv(file_path)
    
    # Analyze the sentiment of the news data using the baseline model
    sentiment_results = analyze_sentiment_bert_baseline(news_data)
    
    # Create a new DataFrame with the results
    results_df = pd.DataFrame({
        'news_sentence': news_data['Sentence'],
        'sentiment_analysis': sentiment_results
    })
    
    # Ensure the results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Save the results to a new CSV file
    results_df.to_csv('results/sentiment_analysis_results_bert_baseline.csv', index=False)

if __name__ == "__main__":
    main()