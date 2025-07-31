import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # Load data
    df = pd.read_csv('data/efsa_sentiment_classification.csv')
    df = df.rename(columns={'Sentence': 'text', 'Sentiment': 'label'})

    # Map labels to integers
    labels = sorted(df['label'].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    df['label'] = df['label'].map(label2id)

    # Split data to get the same test set
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    eval_dataset = Dataset.from_pandas(eval_df)

    # Load tokenizer and model
    model_path = './results/fine_tuned_model/best_model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # Dummy TrainingArguments
    training_args = TrainingArguments(
        output_dir='./results/evaluation',
        per_device_eval_batch_size=16,
    )

    # Trainer for evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Evaluate
    results = trainer.evaluate()
    print("Evaluation Results:")
    print(results)

    # Detailed classification report
    predictions = trainer.predict(eval_dataset)
    preds = predictions.predictions.argmax(-1)
    report = classification_report(eval_dataset['label'], preds, target_names=labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("\nClassification Report:")
    print(pd.DataFrame(report).transpose())

    # Save evaluation results and classification report to CSV
    results_df = pd.DataFrame([results])
    results_output_path = 'results/evaluation_metrics.csv'
    results_df.to_csv(results_output_path, index=False)
    print(f"\nEvaluation metrics saved to {results_output_path}")

    report_output_path = 'results/classification_report_summary.csv'
    report_df.to_csv(report_output_path, index=True)
    print(f"Classification report saved to {report_output_path}")

    # Save predictions to CSV
    eval_df['predicted_label_id'] = preds
    eval_df['predicted_label'] = eval_df['predicted_label_id'].map(id2label)
    eval_df['label'] = eval_df['label'].map(id2label) # Map original labels back to string

    output_df = eval_df[['text', 'label', 'predicted_label']]
    output_df.rename(columns={'text': 'Sentence', 'label': 'Original_Sentiment', 'predicted_label': 'Predicted_Sentiment'}, inplace=True)

    output_path = 'results/fine_tuned_model_predictions.csv'
    output_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")

if __name__ == "__main__":
    main()