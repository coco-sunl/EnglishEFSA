# Event-Level Financial Sentiment Analysis (EFSA)

A five-hop Chain-of-Thought framework for extracting granular, event-level sentiment from English financial texts using LoRA-tuned Mistral-7B-Instruct.

[Read the Paper](https://github.com/coco-sunl/EnglishEFSA/blob/main/EFSA_Paper.pdf)

## Overview

Traditional financial sentiment analysis assigns sentiment to entire documents, obscuring the distinct impact of individual corporate events. This project extends the EFSA framework (Chen et al., 2024) to English financial texts by extracting structured quintuples of the form:

**(Company, Industry, Coarse-Grained Event, Fine-Grained Event, Sentiment)**

For example, given the headline *"AstraZeneca Explores Potential Deal With Acerta for Cancer Drug"*, the system extracts:
> (AstraZeneca, Pharmaceuticals, Business Operations, Initiating Cooperation, POS)

## Dataset

- **Source:** FiQA-2018 Challenge dataset (Task 1) — 1,173 instances of financial news headlines and tweets
- **Enhancements:**
  - Mapped continuous sentiment scores to discrete labels (POS, NEG, NEU)
  - Enriched with GICS industry classifications via the yfinance API
  - Mapped to a hierarchical event taxonomy (7 coarse-grained, 34 fine-grained event classes)
- **Split:** 822 train / 234 test / 117 validation

## Approach

### Baseline Models
Evaluated CNN, RNN, LSTM, DistilBERT, BERT-base, and DistilRoBERTa as sentiment classification baselines.

### Five-Hop Chain-of-Thought (CoT)
A sequential reasoning pipeline using Mistral-7B-Instruct that mirrors financial analyst workflows:

1. **Company Extraction** — identify mentioned companies
2. **Industry Classification** — classify into one of 74 GICS sectors
3. **Coarse-Event Classification** — assign one of 7 event superclasses
4. **Fine-Event Classification** — assign one of 34 event subclasses
5. **Sentiment Classification** — predict POS / NEG / NEU

Evaluated across four configurations: Zero-Shot, Few-Shot, LoRA-Tuned, and LoRA-Tuned + Zero-Shot.

### LoRA Fine-Tuning
- Base model: `Mistral-7B-Instruct-v0.3`
- 4-bit quantization, r=8, alpha=16, dropout=0.1
- Targets: `q_proj` and `v_proj` layers

## Results

| Task | LoRA-Tuned (Weighted F1) |
|---|---|
| Sentiment | **0.97** |
| Company | 0.95 |
| Industry | 0.85 |
| Coarse Event | 0.82 |
| Fine Event | 0.72 |
| Complete EFSA | 0.39 |

The LoRA-tuned model substantially outperforms fine-tuned DistilRoBERTa (0.77 F1) on sentiment classification. Complete EFSA performance (0.39 F1) reflects error propagation across the sequential reasoning chain.

## Key Findings

- LoRA fine-tuning is highly effective for financial domain adaptation with limited data
- Zero-shot prompting outperforms few-shot for instruction-following models like Mistral-7B
- Each additional reasoning hop introduces ~15% performance degradation due to error propagation

## Tech Stack

- Python, PyTorch, HuggingFace Transformers
- Mistral-7B-Instruct-v0.3, DistilRoBERTa
- PEFT (LoRA), BitsAndBytes (4-bit quantization)
- scikit-learn, yfinance

## Authors

Coco Sun · Kyle Yu · Haoyue Zhang  
UC Berkeley School of Information
