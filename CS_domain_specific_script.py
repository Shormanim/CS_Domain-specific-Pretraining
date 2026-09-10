import os
import gc
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

DATA_DIR = "data"
OUTPUT_DIR = "results"
TRAIN_FILE = os.path.join(DATA_DIR, "CS_training_dataset.xlsx")
VALIDATION_FILE = os.path.join(DATA_DIR, "CS_valid_dataset.xlsx")
TEST_FILE = os.path.join(DATA_DIR, "CS_eval_dataset.xlsx")
MAX_LENGTH = 128
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
WEIGHT_DECAY = 0.01
SEED = 42
LABEL_NAMES = ["HUM", "SAR", "HAP", "LOV", "SAD", "AGR", "PRY", "PRD", "FER"]
NUM_LABELS = len(LABEL_NAMES)

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

train_df = pd.read_excel(TRAIN_FILE)
validation_df = pd.read_excel(VALIDATION_FILE)
test_df = pd.read_excel(TEST_FILE)

assert len(train_df) == 8000
assert len(validation_df) == 1847
assert len(test_df) == 1848

for name, dataframe in [("training", train_df), ("validation", validation_df), ("test", test_df)]:
    missing = {"tweet_text", "label_id"}.difference(dataframe.columns)
    assert not missing, f"Missing required columns in {name}: {sorted(missing)}"

expected_label_ids = list(range(NUM_LABELS))
assert sorted(train_df["label_id"].unique()) == expected_label_ids
assert sorted(validation_df["label_id"].unique()) == expected_label_ids
assert sorted(test_df["label_id"].unique()) == expected_label_ids

train_dataset = Dataset.from_pandas(train_df[["tweet_text", "label_id"]], preserve_index=False)
validation_dataset = Dataset.from_pandas(validation_df[["tweet_text", "label_id"]], preserve_index=False)
test_dataset = Dataset.from_pandas(test_df[["tweet_text", "label_id"]], preserve_index=False)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": recall_score(labels, predictions, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
    }

def train_and_evaluate(model_name, model_output, results_output, run_test=True):
    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)
    os.makedirs(model_output, exist_ok=True)
    os.makedirs(results_output, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize(batch):
        return tokenizer(batch["tweet_text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    tokenized_train = train_dataset.map(tokenize, batched=True)
    tokenized_validation = validation_dataset.map(tokenize, batched=True)
    tokenized_test = test_dataset.map(tokenize, batched=True)
    for dataset_name in ["tokenized_train", "tokenized_validation", "tokenized_test"]:
        dataset = locals()[dataset_name].rename_column("label_id", "labels").remove_columns(["tweet_text"])
        dataset.set_format("torch")
        locals()[dataset_name] = dataset
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS, id2label=dict(enumerate(LABEL_NAMES)), label2id={label: i for i, label in enumerate(LABEL_NAMES)})
    training_args = TrainingArguments(output_dir=results_output, num_train_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, per_device_train_batch_size=TRAIN_BATCH_SIZE, per_device_eval_batch_size=EVAL_BATCH_SIZE, weight_decay=WEIGHT_DECAY, eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True, fp16=torch.cuda.is_available(), seed=SEED, report_to="none")
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_train, eval_dataset=tokenized_validation, processing_class=tokenizer, compute_metrics=compute_metrics)
    trainer.train()
    validation_results = trainer.evaluate(tokenized_validation)
    history = pd.DataFrame(trainer.state.log_history)
    validation_history = history[history["eval_loss"].notna()][["epoch", "eval_loss", "eval_accuracy", "eval_precision", "eval_recall", "eval_macro_f1", "eval_weighted_f1"]].copy()
    training_history = history[history["loss"].notna()][["epoch", "loss"]].copy()
    epoch_results = pd.merge(training_history, validation_history, on="epoch", how="inner").rename(columns={"epoch":"Epoch", "loss":"Training Loss", "eval_loss":"Validation Loss", "eval_accuracy":"Accuracy", "eval_precision":"Precision", "eval_recall":"Recall", "eval_macro_f1":"Macro F1", "eval_weighted_f1":"Weighted F1"})
    print("\nVALIDATION RESULTS\n" + "-" * 70)
    print(epoch_results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    epoch_results.to_excel(os.path.join(model_output, "validation_epoch_results.xlsx"), index=False)
    if not run_test:
        pd.DataFrame([{
            "Model": model_name,
            "Validation Accuracy": validation_results["eval_accuracy"],
            "Validation Precision": validation_results["eval_precision"],
            "Validation Recall": validation_results["eval_recall"],
            "Validation Macro F1": validation_results["eval_macro_f1"],
            "Validation Weighted F1": validation_results["eval_weighted_f1"],
        }]).to_excel(os.path.join(model_output, "baseline_validation_results.xlsx"), index=False)
        trainer.save_model(model_output)
        tokenizer.save_pretrained(model_output)
        del trainer, model, tokenizer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return None, epoch_results
    test_results = trainer.evaluate(tokenized_test)
    prediction = trainer.predict(tokenized_test)
    y_pred = np.argmax(prediction.predictions, axis=1)
    y_true = prediction.label_ids
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4, zero_division=0)
    print("\nINDEPENDENT TEST RESULTS\n" + "-" * 70)
    print(f"Accuracy:    {test_results['eval_accuracy']:.6f}")
    print(f"Precision:   {test_results['eval_precision']:.6f}")
    print(f"Recall:      {test_results['eval_recall']:.6f}")
    print(f"Macro F1:    {test_results['eval_macro_f1']:.6f}")
    print(f"Weighted F1: {test_results['eval_weighted_f1']:.6f}")
    print(f"Loss:        {test_results['eval_loss']:.6f}")
    print("\nCLASSIFICATION REPORT\n" + "-" * 70)
    print(report)
    trainer.save_model(model_output)
    tokenizer.save_pretrained(model_output)
    pd.DataFrame([{
        "Model": model_name, "Test Loss": test_results["eval_loss"], "Accuracy": test_results["eval_accuracy"], "Precision": test_results["eval_precision"], "Recall": test_results["eval_recall"], "Macro F1": test_results["eval_macro_f1"], "Weighted F1": test_results["eval_weighted_f1"]
    }]).to_excel(os.path.join(model_output, "test_results.xlsx"), index=False)
    with open(os.path.join(model_output, "classification_report.txt"), "w", encoding="utf-8") as f: f.write(report)
    final_test_results = {"Model": model_name, "Test Loss": test_results["eval_loss"], "Accuracy": test_results["eval_accuracy"], "Precision": test_results["eval_precision"], "Recall": test_results["eval_recall"], "Macro F1": test_results["eval_macro_f1"], "Weighted F1": test_results["eval_weighted_f1"]}
    del trainer, model, tokenizer
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return final_test_results, epoch_results

bert_results, bert_history = train_and_evaluate("bert-base-uncased", os.path.join(OUTPUT_DIR, "BERT_final_model"), os.path.join(OUTPUT_DIR, "BERT_results"), run_test=False)
marbert_results, marbert_history = train_and_evaluate("UBC-NLP/MARBERT", os.path.join(OUTPUT_DIR, "MARBERT_final_model"), os.path.join(OUTPUT_DIR, "MARBERT_results"), run_test=True)
xlmr_results, xlmr_history = train_and_evaluate("xlm-roberta-base", os.path.join(OUTPUT_DIR, "XLMR_final_model"), os.path.join(OUTPUT_DIR, "XLMR_results"), run_test=True)
comparison = pd.DataFrame([marbert_results, xlmr_results])
print("\n" + "=" * 70)
print("FINAL INDEPENDENT TEST COMPARISON")
print("=" * 70)
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
comparison.to_excel(os.path.join(OUTPUT_DIR, "final_model_comparison.xlsx"), index=False)
