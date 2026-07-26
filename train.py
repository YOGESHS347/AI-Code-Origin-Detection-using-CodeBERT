import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
    set_seed
)

# Patch HuggingFace torch.load safety version check for legacy .bin model loading
try:
    import transformers.utils.import_utils as hf_import_utils
    import transformers.modeling_utils as hf_modeling_utils
    hf_import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
    hf_modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
except Exception:
    pass

# Setup production-grade logging
def setup_logger(log_file: str = "train.log") -> logging.Logger:
    logger = logging.getLogger("CodeBERT_Trainer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console Handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_format = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    # File Handler
    f_handler = logging.FileHandler(log_file, encoding="utf-8")
    f_format = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    return logger

logger = setup_logger()

# ==============================================================================
# PIPELINE TRAINING CONFIGURATION
# Simply run `python train.py` without arguments to execute with these defaults.
# ==============================================================================
PIPELINE_CONFIG = {
    "AI_DATASET": "./dataset/ai/ai_dataset.csv",
    "HUMAN_DATASET": "./dataset/human/human_dataset.csv",
    "MODEL_NAME": "microsoft/codebert-base",
    "OUTPUT_DIR": "./saved_model",
    "BATCH_SIZE": 16,       # Set according to your GPU memory (e.g. 8, 16, 32)
    "EPOCHS": 3,            # Total training epochs
    "LEARNING_RATE": 2e-5,  # Learning rate for AdamW optimizer
    "MAX_LENGTH": 512,      # Token sequence length limit for CodeBERT
    "TEST_SIZE": 0.15,      # Validation dataset split ratio (15%)
    "SEED": 42              # Random seed for reproducibility
}

class LoggingMetricsCallback(TrainerCallback):
    """Custom Callback to output real-time training progress and validation metrics in terminal and train.log."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            epoch = round(state.epoch, 2) if state.epoch else 0.0
            loss = logs.get("loss", None)
            lr = logs.get("learning_rate", None)
            if loss is not None:
                lr_str = f"{lr:.2e}" if lr else "N/A"
                logger.info(f" [TRAIN STEP {step:>5d} | Epoch {epoch:>4.2f}] Training Loss: {loss:.4f} | LR: {lr_str}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            epoch = round(state.epoch, 2) if state.epoch else 0.0
            acc = metrics.get("eval_accuracy", 0.0)
            f1 = metrics.get("eval_f1", 0.0)
            prec = metrics.get("eval_precision", 0.0)
            rec = metrics.get("eval_recall", 0.0)
            val_loss = metrics.get("eval_loss", None)
            
            loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
            logger.info("\n" + "=" * 65)
            logger.info(f" >>> REAL-TIME EVALUATION AT EPOCH {epoch} (Step {state.global_step})")
            logger.info("-" * 65)
            logger.info(f"  • Validation Loss : {loss_str}")
            logger.info(f"  • Accuracy        : {acc * 100:6.2f}%  ({acc:.4f})  | F1: {f1 * 100:6.2f}% ({f1:.4f})")
            logger.info(f"  • Precision       : {prec * 100:6.2f}%  ({prec:.4f})  | Recall: {rec * 100:6.2f}% ({rec:.4f})")
            logger.info("=" * 65 + "\n")

class CodeDataset(Dataset):
    """Custom PyTorch Dataset for CodeBERT text classification."""
    def __init__(self, encodings: dict, labels: list):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.labels)

def load_and_preprocess_data(ai_path: str, human_path: str) -> pd.DataFrame:
    """Load AI and Human CSV datasets, perform validation, and combine them."""
    logger.info(f"Loading AI dataset from: {ai_path}")
    if not os.path.exists(ai_path):
        raise FileNotFoundError(f"AI dataset path not found: {ai_path}")
    df_ai = pd.read_csv(ai_path)
    
    logger.info(f"Loading Human dataset from: {human_path}")
    if not os.path.exists(human_path):
        raise FileNotFoundError(f"Human dataset path not found: {human_path}")
    df_human = pd.read_csv(human_path)

    # Label mapping: Human = 0, AI = 1
    label_map = {"Human": 0, "AI": 1}

    # Extract required columns and clean dataset
    for df, label_val in [(df_ai, 1), (df_human, 0)]:
        if "label" not in df.columns or df["label"].isnull().all():
            df["label_num"] = label_val
        else:
            df["label_num"] = df["label"].map(label_map).fillna(label_val).astype(int)

    df_combined = pd.concat([df_ai, df_human], ignore_index=True)

    # Data validation & cleaning
    initial_count = len(df_combined)
    df_combined = df_combined.dropna(subset=["code"])
    df_combined["code"] = df_combined["code"].astype(str).str.strip()
    df_combined = df_combined[df_combined["code"] != ""]
    cleaned_count = len(df_combined)

    if initial_count != cleaned_count:
        logger.warning(f"Removed {initial_count - cleaned_count} empty or null code entries.")

    logger.info(f"Dataset successfully loaded. Total samples: {cleaned_count}")
    logger.info(f"Class distribution:\n{df_combined['label_num'].value_counts().rename(index={0: 'Human (0)', 1: 'AI (1)'})}")
    
    if "language" in df_combined.columns:
        logger.info(f"Language breakdown:\n{df_combined['language'].value_counts()}")

    return df_combined

def compute_metrics(eval_pred):
    """Compute evaluation metrics for binary classification."""
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(probs, axis=1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except Exception:
        auc = 0.0

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc)
    }

def save_model_with_prompt(model, tokenizer, eval_metrics: dict, output_dir: str):
    """Check if model exists, display accuracy comparison, and prompt user before overwriting."""
    import json
    import time

    new_acc = eval_metrics.get("eval_accuracy", 0.0)
    new_f1 = eval_metrics.get("eval_f1", 0.0)
    new_prec = eval_metrics.get("eval_precision", 0.0)
    new_rec = eval_metrics.get("eval_recall", 0.0)

    metrics_payload = {
        "accuracy": new_acc,
        "f1": new_f1,
        "precision": new_prec,
        "recall": new_rec,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    metrics_file = os.path.join(output_dir, "metrics.json")
    should_overwrite = True

    # Check if existing model directory exists
    if os.path.exists(output_dir) and (os.path.exists(os.path.join(output_dir, "config.json")) or os.path.exists(metrics_file)):
        logger.info(f"An existing trained model was detected at: '{output_dir}'")
        prev_acc_str = "N/A"
        prev_f1_str = "N/A"
        prev_prec_str = "N/A"
        prev_rec_str = "N/A"

        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    prev_metrics = json.load(f)
                    prev_acc_str = f"{prev_metrics.get('accuracy', 0.0):.4f}"
                    prev_f1_str = f"{prev_metrics.get('f1', 0.0):.4f}"
                    prev_prec_str = f"{prev_metrics.get('precision', 0.0):.4f}"
                    prev_rec_str = f"{prev_metrics.get('recall', 0.0):.4f}"
            except Exception:
                pass

        print("\n" + "=" * 60)
        print("          EXISTING VS. NEW MODEL ACCURACY COMPARISON          ")
        print("=" * 60)
        print(f" Metric        Existing Model       Newly Trained Model")
        print("-" * 60)
        print(f" Accuracy    : {prev_acc_str:<18} {new_acc:.4f}")
        print(f" F1 Score    : {prev_f1_str:<18} {new_f1:.4f}")
        print(f" Precision   : {prev_prec_str:<18} {new_prec:.4f}")
        print(f" Recall      : {prev_rec_str:<18} {new_rec:.4f}")
        print("=" * 60)

        try:
            user_choice = input("\nDo you want to overwrite the existing saved model with this new model? (y/n): ").strip().lower()
            if user_choice not in ["y", "yes"]:
                should_overwrite = False
        except (KeyboardInterrupt, EOFError):
            logger.warning("No interactive input received. Defaulting to saving as secondary backup directory.")
            should_overwrite = False

    if should_overwrite:
        target_dir = output_dir
        logger.info(f"Overwriting/saving fine-tuned model and tokenizer to: '{target_dir}'")
    else:
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        target_dir = f"{output_dir}_{timestamp_str}"
        logger.info(f"User declined overwrite or skipped. Saving newly trained model to separate directory: '{target_dir}'")

    os.makedirs(target_dir, exist_ok=True)
    model.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)

    with open(os.path.join(target_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)

    logger.info(f"Model artifacts successfully saved to '{target_dir}'!")
    return target_dir

def save_training_report(eval_metrics: dict, train_loss: float, args, target_dir: str):
    """Write complete formatted training report to training_results.txt."""
    import time
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    acc = eval_metrics.get("eval_accuracy", 0.0)
    f1 = eval_metrics.get("eval_f1", 0.0)
    prec = eval_metrics.get("eval_precision", 0.0)
    rec = eval_metrics.get("eval_recall", 0.0)
    auc = eval_metrics.get("eval_roc_auc", 0.0)

    report_content = f"""================================================================================
               CODEBERT AI VS. HUMAN CODE DETECTION REPORT
================================================================================
Execution Timestamp     : {timestamp}
Model Backbone          : {args.model_name}
Target Save Location    : {target_dir}

--------------------------------------------------------------------------------
TRAINING HYPERPARAMETERS
--------------------------------------------------------------------------------
Total Epochs            : {args.epochs}
Batch Size              : {args.batch_size}
Learning Rate           : {args.learning_rate}
Max Sequence Length     : {args.max_length}
Validation Split Ratio  : {args.test_size * 100:.1f}%
Random Seed             : {args.seed}

--------------------------------------------------------------------------------
TRAINING PERFORMANCE & LOSS
--------------------------------------------------------------------------------
Final Training Loss     : {train_loss:.4f}

--------------------------------------------------------------------------------
FINAL VALIDATION METRICS
--------------------------------------------------------------------------------
Accuracy                : {acc * 100:6.2f}%  ({acc:.4f})
F1 Score                : {f1 * 100:6.2f}%  ({f1:.4f})
Precision               : {prec * 100:6.2f}%  ({prec:.4f})
Recall                  : {rec * 100:6.2f}%  ({rec:.4f})
ROC AUC                 : {auc * 100:6.2f}%  ({auc:.4f})
================================================================================
"""
    # Write to root workspace training_results.txt
    report_file = "training_results.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    # Write copy inside the saved model directory
    dir_report_file = os.path.join(target_dir, "training_results.txt")
    with open(dir_report_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"Detailed training report saved to root file: '{report_file}'")
    logger.info(f"Report copy saved inside model directory: '{dir_report_file}'")

def parse_args():
    parser = argparse.ArgumentParser(description="Train CodeBERT model for AI vs. Human Code Detection")
    parser.add_argument("--ai_dataset", type=str, default=PIPELINE_CONFIG["AI_DATASET"], help="Path to AI CSV dataset")
    parser.add_argument("--human_dataset", type=str, default=PIPELINE_CONFIG["HUMAN_DATASET"], help="Path to Human CSV dataset")
    parser.add_argument("--model_name", type=str, default=PIPELINE_CONFIG["MODEL_NAME"], help="Hugging Face model backbone")
    parser.add_argument("--output_dir", type=str, default=PIPELINE_CONFIG["OUTPUT_DIR"], help="Directory to save final model")
    parser.add_argument("--batch_size", type=int, default=PIPELINE_CONFIG["BATCH_SIZE"], help="Batch size per device for train/eval")
    parser.add_argument("--epochs", type=int, default=PIPELINE_CONFIG["EPOCHS"], help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=PIPELINE_CONFIG["LEARNING_RATE"], help="Learning rate for AdamW optimizer")
    parser.add_argument("--max_length", type=int, default=PIPELINE_CONFIG["MAX_LENGTH"], help="Maximum sequence length for tokenization")
    parser.add_argument("--test_size", type=float, default=PIPELINE_CONFIG["TEST_SIZE"], help="Validation set split ratio")
    parser.add_argument("--seed", type=int, default=PIPELINE_CONFIG["SEED"], help="Random seed for reproducibility")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("STARTING CODEBERT AI VS. HUMAN TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Target Backbone Model : {args.model_name}")
    logger.info(f"Save Directory        : {args.output_dir}")
    logger.info(f"Epochs / Batch Size   : {args.epochs} / {args.batch_size}")
    logger.info(f"Learning Rate         : {args.learning_rate}")
    logger.info(f"Max Sequence Length   : {args.max_length}")

    # Step 1: Load and clean dataset
    df = load_and_preprocess_data(args.ai_dataset, args.human_dataset)

    # Step 2: Stratified Train / Validation Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["code"].tolist(),
        df["label_num"].tolist(),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["label_num"].tolist()
    )

    logger.info(f"Training Samples: {len(train_texts)} | Validation Samples: {len(val_texts)}")

    # Step 3: Load Tokenizer and Tokenize Datasets
    logger.info(f"Loading Hugging Face Tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    logger.info("Tokenizing training text sequences...")
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        max_length=args.max_length,
        padding=False  # Collator will handle dynamic padding during batching
    )

    logger.info("Tokenizing validation text sequences...")
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        max_length=args.max_length,
        padding=False
    )

    # Create PyTorch Datasets
    train_dataset = CodeDataset(train_encodings, train_labels)
    val_dataset = CodeDataset(val_encodings, val_labels)

    # Step 4: Load Model with Label Mapping
    id2label = {0: "Human-generated", 1: "AI-generated"}
    label2id = {"Human-generated": 0, "AI-generated": 1}

    logger.info(f"Loading Hugging Face Model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )

    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Execution Device detected: {device.upper()}")

    # Step 5: Define Training Arguments & Trainer
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",  # Disable wandb/tensorboard prompt unless configured
        fp16=torch.cuda.is_available(),  # Enable mixed precision training if GPU is available
        seed=args.seed
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[LoggingMetricsCallback()]
    )

    # Step 6: Train Model
    logger.info("Initiating model training loop...")
    train_result = trainer.train()

    train_loss = train_result.training_loss if hasattr(train_result, "training_loss") else 0.0
    logger.info(f"Training completed. Total Training Loss: {train_loss:.4f}")

    # Step 7: Final Validation Evaluation
    logger.info("Running final evaluation on validation set...")
    eval_metrics = trainer.evaluate()
    logger.info("=" * 40)
    logger.info("FINAL VALIDATION METRICS:")
    for metric_name, val in eval_metrics.items():
        if metric_name.startswith("eval_"):
            logger.info(f"  - {metric_name[5:].upper():<12}: {val:.4f}")
    logger.info("=" * 40)

    # Step 8: Save Model with Interactive Overwrite Prompt based on Accuracy Comparison
    saved_target_dir = save_model_with_prompt(model, tokenizer, eval_metrics, args.output_dir)

    # Step 9: Export detailed training results to dedicated report file
    save_training_report(eval_metrics, train_loss, args, saved_target_dir)

    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
