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

def main():
    # Read the CSV file
    file_path = 'data/efsa_sentiment_classification.csv'
    news_data = pd.read_csv(file_path)

    # Run BERT baseline sentiment analysis
    results = analyze_sentiment_bert_baseline(news_data)
    
    # Extract predicted labels and confidence scores
    news_data['predicted_sentiment_bert'] = [res[0]['label'] for res in results]
    news_data['confidence_bert'] = [res[0]['score'] for res in results]

    # Map model output to dataset labels for accuracy calculation
    news_data['mapped_sentiment_bert'] = news_data['predicted_sentiment_bert'].apply(lambda x: 'POS' if x.upper() == 'POSITIVE' else 'NEG')

    # Calculate and print accuracy
    accuracy = accuracy_score(news_data['Sentiment'], news_data['mapped_sentiment_bert'])
    print(f"Accuracy (BERT baseline): {accuracy:.4f}\n")

    # Generate and print the classification report
    print("Classification Report (BERT baseline):")
    report = classification_report(news_data['Sentiment'], news_data['mapped_sentiment_bert'])
    print(report)

    # Save the classification report to a file
    report_dict = classification_report(news_data['Sentiment'], news_data['mapped_sentiment_bert'], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_output_path = 'results/classification_report_summary_bert_baseline.csv'
    report_df.to_csv(report_output_path, index=True)
    print(f"Classification report for BERT baseline saved to {report_output_path}")

    # Save the results to a new CSV file
    output_path = 'results/sentiment_analysis_results_bert_baseline.csv'
    news_data[['Sentence', 'Sentiment', 'predicted_sentiment_bert', 'confidence_bert']].to_csv(output_path, index=False)
    print(f"BERT baseline sentiment analysis results saved to {output_path}")

    # Run DistilBERT baseline sentiment analysis
    results_distilbert = analyze_sentiment_distilbert_baseline(news_data)
    
    # Extract predicted labels and confidence scores
    news_data['predicted_sentiment_distilbert'] = [res[0]['label'] for res in results_distilbert]
    news_data['confidence_distilbert'] = [res[0]['score'] for res in results_distilbert]

    # Map model output to dataset labels for accuracy calculation
    news_data['mapped_sentiment_distilbert'] = news_data['predicted_sentiment_distilbert'].apply(lambda x: 'POS' if x.upper() == 'POSITIVE' else 'NEG')

    # Calculate and print accuracy
    accuracy_distilbert = accuracy_score(news_data['Sentiment'], news_data['mapped_sentiment_distilbert'])
    print(f"\nAccuracy (DistilBERT baseline): {accuracy_distilbert:.4f}\n")

    # Generate and print the classification report
    print("Classification Report (DistilBERT baseline):")
    report_distilbert = classification_report(news_data['Sentiment'], news_data['mapped_sentiment_distilbert'])
    print(report_distilbert)

    # Save the classification report to a file
    report_dict_distilbert = classification_report(news_data['Sentiment'], news_data['mapped_sentiment_distilbert'], output_dict=True)
    report_df_distilbert = pd.DataFrame(report_dict_distilbert).transpose()
    report_output_path_distilbert = 'results/classification_report_summary_distilbert_baseline.csv'
    report_df_distilbert.to_csv(report_output_path_distilbert, index=True)
    print(f"Classification report for DistilBERT baseline saved to {report_output_path_distilbert}")

    # Save the results to a new CSV file
    output_path_distilbert = 'results/sentiment_analysis_results_distilbert_baseline.csv'
    news_data[['Sentence', 'Sentiment', 'predicted_sentiment_distilbert', 'confidence_distilbert']].to_csv(output_path_distilbert, index=False)
    print(f"DistilBERT baseline sentiment analysis results saved to {output_path_distilbert}")

if __name__ == "__main__":
    main()