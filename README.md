# Domain-specific Pretraining and Transformer Performance: Evidence from Classifying Digital Pragmatics in Arabic-English Code-switching

This repository contains the code for a comparative transformer-based study of **emoji pragmatic function classification in Arabic-English code-switched tweets**.

The implementation fine-tunes three transformer models:

- **BERT** (`bert-base-uncased`) as a baseline
- **MARBERT** (`UBC-NLP/MARBERT`) for Arabic social-media text
- **XLM-RoBERTa** (`xlm-roberta-base`) for multilingual text

The task is **pragmatic function classification**, not emotion classification. Each tweet is assigned one of nine pragmatic-function labels.

## Dataset

The experiments use three Excel files:

```text
data/
├── CS_training_dataset.xlsx
├── CS_valid_dataset.xlsx
└── CS_eval_dataset.xlsx
```

The expected split sizes are:

| Split | Tweets |
|---|---:|
| Training | 8,000 |
| Validation | 1,847 |
| Test | 1,848 |
| **Total** | **11,695** |

The dataset files are not included in this repository unless their redistribution is authorized.

## Pragmatic-function labels

The nine classification labels are:

| ID | Label |
|---:|---|
| 0 | HUM |
| 1 | SAR |
| 2 | HAP |
| 3 | LOV |
| 4 | SAD |
| 5 | AGR |
| 6 | PRY |
| 7 | PRD |
| 8 | FER |

## Requirements

Python 3.10 or later is recommended.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

A CUDA-enabled PyTorch installation is recommended for practical training speed. The code automatically uses the GPU when CUDA is available and otherwise falls back to CPU.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── emoji_pragmatic_function_classification.py
├── emoji_pragmatic_function_classification.ipynb
└── data/
    ├── CS_training_dataset.xlsx
    ├── CS_valid_dataset.xlsx
    └── CS_eval_dataset.xlsx
```

## Running the Python script

Place the three dataset files in the `data/` directory and run:

```bash
python emoji_pragmatic_function_classification.py
```

The script creates a `results/` directory containing model checkpoints, validation results, independent-test results, classification reports, and the final model comparison.

## Running the notebook

Open:

```text
emoji_pragmatic_function_classification.ipynb
```

in Jupyter Notebook, JupyterLab, or Google Colab. If using Colab, upload the `data/` directory or adjust the `DATA_DIR` setting to the location of the dataset files.

## Experimental procedure

For each model, the tweets are tokenized with the corresponding pretrained tokenizer using a maximum sequence length of 128 tokens. Models are fine-tuned for five epochs with a learning rate of `2e-5`, batch size of 16, and weight decay of `0.01`.

The best checkpoint is selected using validation **macro F1**. Accuracy, macro precision, macro recall, macro F1, and weighted F1 are calculated during evaluation.

BERT is treated as the baseline and is evaluated on the validation set only. MARBERT and XLM-RoBERTa are evaluated on the independent test set and are included in the final model comparison.

## Reproducibility

The random seed is fixed at `42`. Dataset-size and label-ID assertions are included in the code to detect unexpected changes to the experimental data.

## Outputs

For each model, the script saves the following where applicable:

- validation epoch results
- trained model and tokenizer
- test results
- classification report
- final comparison of MARBERT and XLM-RoBERTa

The main comparison file is:

```text
results/final_model_comparison.xlsx
```

## Citation

If you use this code or the associated dataset in academic work, please cite the corresponding study.

## License

The source code is released under the **MIT License**. Dataset access and reuse are subject to the terms specified by the data provider or the associated study.
