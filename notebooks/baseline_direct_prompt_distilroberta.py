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

from sklearn.metrics import accuracy_score, classification_report

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

def analyze_sentiment_distilbert_baseline(news_data):
    # Initialize the sentiment analysis pipeline with distilbert-base-uncased
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased")
    
    results = []
    for index, row in news_data.iterrows():
        news_sentence = row['Sentence']
        # Analyze the sentiment of the news sentence
        result = sentiment_pipeline(news_sentence)
        results.append(result)
        print(f"Processed news {index + 1}/{len(news_data)}")
    
    return results

def analyze_sentiment_3_class(text):
    # This model is specifically fine-tuned for financial news and provides three labels: positive, negative, neutral.
    model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
    results = sentiment_pipeline(text)
    return results

def run_distilroberta_financial_baseline():
    # Read the CSV file
    file_path = 'data/efsa_sentiment_classification.csv'
    news_data = pd.read_csv(file_path)

    # Run 3-class sentiment analysis
    print("Running DistilRoBERTa financial news sentiment analysis baseline...")
    results = analyze_sentiment_3_class(news_data['Sentence'].tolist())
    
    # Extract predicted labels and confidence scores
    news_data['predicted_sentiment'] = [res['label'] for res in results]
    news_data['confidence'] = [res['score'] for res in results]

    # Map model output to dataset labels for accuracy calculation
    # The model output is 'positive', 'negative', 'neutral'. We need to map it to 'POS', 'NEG', 'NEU'
    label_mapping = {'positive': 'POS', 'negative': 'NEG', 'neutral': 'NEU'}
    news_data['mapped_sentiment'] = news_data['predicted_sentiment'].apply(lambda x: label_mapping.get(x, ''))

    # Calculate and print accuracy
    accuracy = accuracy_score(news_data['Sentiment'], news_data['mapped_sentiment'])
    print(f"Accuracy (DistilRoBERTa baseline): {accuracy:.4f}\n")

    # Generate and print the classification report
    print("Classification Report (DistilRoBERTa baseline):")
    report = classification_report(news_data['Sentiment'], news_data['mapped_sentiment'])
    print(report)

    # Save the classification report to a file
    report_dict = classification_report(news_data['Sentiment'], news_data['mapped_sentiment'], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_output_path = 'results/classification_report_summary_distilroberta_baseline.csv'
    report_df.to_csv(report_output_path, index=True)
    print(f"Classification report saved to {report_output_path}")

    # Save the results to a new CSV file
    output_path = 'results/sentiment_analysis_distilroberta_baseline_results.csv'
    news_data[['Sentence', 'Sentiment', 'predicted_sentiment', 'confidence']].to_csv(output_path, index=False)
    print(f"Sentiment analysis results saved to {output_path}")

def main():
    run_distilroberta_financial_baseline()

if __name__ == "__main__":
    main()