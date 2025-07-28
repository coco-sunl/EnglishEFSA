import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

    # Split data
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # Model
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results/fine_tuned_model',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,

    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train
    trainer.train()

    # Save the best model
    trainer.save_model('./results/fine_tuned_model/best_model')
    tokenizer.save_pretrained('./results/fine_tuned_model/best_model')

if __name__ == "__main__":
    main()