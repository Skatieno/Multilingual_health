# Multilingual Health QA — mT5 Fine-Tuning

Fine-tuning mT5 on health question-answering across 8 African language subsets using LoRA. Built for the Zindi African Language Health challenge.

## Project Overview

This project fine-tunes Google's mT5 model to answer health questions in four African languages (Akan, Amharic, Luganda, Kiswahili) and four regional English variants (Ethiopia, Ghana, Kenya, Uganda). Training uses LoRA (Low-Rank Adaptation) to make fine-tuning feasible on a single GPU.

---

## Repository Structure

```
├── notebooks/
│   ├── multilingual-health-eda.ipynb   # EDA: language distribution, text length, tokenization analysis
│   └── multilingual-health.ipynb       # Full training + evaluation + submission pipeline
├── data/                               # Train.csv, Test.csv, SampleSubmission.csv
├── requirements.txt
└── README.md
```

---

## Setup

### On Kaggle (recommended)

1. Upload both notebooks to Kaggle
2. Attach the dataset: **+ Add Input** → search `african-language-health` by sherylotieno
3. Set **Accelerator → GPU T4** in Settings
4. Run `multilingual-health-eda.ipynb` first, then `multilingual-health.ipynb`

### On Google Colab

1. Click a Colab badge above
2. Set **Runtime → Change runtime type → GPU**
3. Upload `Train.csv` and `Test.csv`, then update `DATA_DIR` at the top of the notebook
4. Run all cells top to bottom

The notebooks install all dependencies automatically:

```
transformers==4.46.0  peft==0.13.0  rouge_score  datasets
```

---

## Experiments

All experiments run inside `multilingual-health.ipynb`.

| #   | Description                             | ROUGE-1    | ROUGE-L    | Zindi Score |
| --- | --------------------------------------- | ---------- | ---------- | ----------- |
| 1   | Zero-shot mT5-small (no fine-tuning)    | 0.0027     | 0.0024     | 0.000676    |
| 2   | LoRA fine-tune — r=16, 2 epochs         | 0.1146     | 0.1032     | —           |
| 3   | + Repetition penalty & n-gram blocking  | **0.1666** | **0.1352** | 0.145043    |
| 4   | + Amharic 3× oversampling               | 0.1583     | 0.1295     | —           |
| 5   | Tokenization diagnostic (no retraining) | —          | —          | —           |
| 6   | mT5-base, LoRA r=32, Adafactor, 1 epoch | 0.1516     | 0.1277     | 0.211963    |

**Best validation result:** Experiment 3 — LoRA fine-tune + decoding fix  
**Best Zindi score:** Experiment 6 — mT5-base

---

## Key Findings

- **Decoding mattered most.** Adding `repetition_penalty=1.3` and `no_repeat_ngram_size=3` with zero retraining pushed ROUGE-1 from 0.1146 to 0.1666 — the biggest single gain in the project.
- **Amharic is a tokenization problem.** Amharic needs ~3.5 tokens/word vs ~1.6 for English due to poor Ge'ez script coverage in mT5's vocabulary. Oversampling helped slightly but didn't close the gap.
- **Bigger model needs more time.** mT5-base only got 1 training epoch (GPU limits), which wasn't enough to beat the 2-epoch mT5-small run.

---

## Reproducing Results

To reproduce Experiment 3 (best validation ROUGE):

1. Run `multilingual-health.ipynb` top to bottom on a T4 GPU
2. All seeds are fixed at `random_state=42`
3. Expected runtime: ~35–45 minutes for Experiments 1–3

## Limitations

- Amharic remains the weakest language across all experiments
- mT5-base was only trained for 1 epoch due to GPU time constraints
- ROUGE measures word overlap only — outputs should not be used as medical advice without expert review

```

```
