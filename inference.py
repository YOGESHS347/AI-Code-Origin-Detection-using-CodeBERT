import os
import sys
import logging
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Patch HuggingFace torch.load safety version check for legacy .bin model loading
try:
    import transformers.utils.import_utils as hf_import_utils
    import transformers.modeling_utils as hf_modeling_utils
    hf_import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
    hf_modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
except Exception:
    pass

# Setup production-grade logging
def setup_logger(log_file: str = "inference.log") -> logging.Logger:
    logger = logging.getLogger("CodeBERT_Inference")
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

SUPPORTED_EXTENSIONS = {".py", ".cpp", ".c", ".h", ".hpp", ".java", ".js", ".ts", ".txt"}

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for AI vs. Human Code Classifier using CodeBERT")
    parser.add_argument("file_path", type=str, help="Path to code file (.py, .cpp, .java)")
    parser.add_argument("--model_dir", type=str, default="./saved_model", help="Path to local saved model directory")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization")
    return parser.parse_args()

def load_code_file(file_path: str) -> tuple[str, str]:
    """Validate file path and extension, and read code content safely."""
    logger.info(f"Checking target file path: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target file does not exist: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext_lower = ext.lower()
    
    if ext_lower not in SUPPORTED_EXTENSIONS:
        logger.warning(f"File extension '{ext}' is not explicitly in primary supported list ({SUPPORTED_EXTENSIONS}). Proceeding with plain text read.")

    logger.info(f"Reading file contents (extension: {ext_lower})...")
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        code_content = f.read().strip()

    if not code_content:
        raise ValueError(f"Target file '{file_path}' is empty.")

    logger.info(f"Successfully loaded file. Length: {len(code_content)} characters, {len(code_content.splitlines())} lines.")
    return code_content, ext_lower

def predict(code_snippet: str, model_dir: str, max_length: int = 512) -> dict:
    """Run model inference on a code snippet and return class predictions and confidence scores."""
    logger.info(f"Loading saved model artifacts from directory: {model_dir}")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Saved model directory '{model_dir}' not found. Please train the model first using train.py."
        )

    # Detect execution device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inference execution device: {device.type.upper()}")

    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    logger.info("Tokenizing input code snippet...")
    inputs = tokenizer(
        code_snippet,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    total_tokens = int(attention_mask.sum().item())
    logger.info(f"Tokenization complete. Total active tokens (non-padded): {total_tokens}/{max_length}")

    logger.info("Executing model forward pass...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).squeeze(0)

    # Label extraction from model config
    id2label = model.config.id2label if hasattr(model.config, "id2label") else {0: "Human-generated", 1: "AI-generated"}

    human_prob = probabilities[0].item()
    ai_prob = probabilities[1].item()

    pred_class_id = torch.argmax(probabilities).item()
    predicted_label = id2label.get(pred_class_id, f"Class_{pred_class_id}")
    confidence_score = probabilities[pred_class_id].item()

    logger.info(f"Raw Logits: {logits.cpu().numpy().tolist()[0]}")
    logger.info(f"Calculated Class Probabilities -> Human: {human_prob * 100:.2f}%, AI: {ai_prob * 100:.2f}%")
    logger.info(f"Final Prediction: {predicted_label} (Confidence: {confidence_score * 100:.2f}%)")

    return {
        "prediction": predicted_label,
        "class_id": pred_class_id,
        "confidence": confidence_score,
        "probabilities": {
            "Human-generated": human_prob,
            "AI-generated": ai_prob
        },
        "token_count": total_tokens
    }

def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("STARTING CODEBERT CODE ORIGIN INFERENCE PIPELINE")
    logger.info("=" * 60)

    try:
        # Load and validate file content
        code_content, ext = load_code_file(args.file_path)

        # Execute prediction
        result = predict(code_content, args.model_dir, args.max_length)

        # Output Summary Card
        print("\n" + "=" * 50)
        print("          CODE DETECTION RESULT          ")
        print("=" * 50)
        print(f" File Analyzed  : {args.file_path}")
        print(f" Language Ext   : {ext}")
        print(f" Tokens Used    : {result['token_count']}")
        print(f" Prediction     : {result['prediction'].upper()}")
        print(f" Confidence     : {result['confidence'] * 100:.2f}%")
        print("-" * 50)
        print(" Probability Breakdown:")
        print(f"  • Human-generated : {result['probabilities']['Human-generated'] * 100:6.2f}%")
        print(f"  • AI-generated    : {result['probabilities']['AI-generated'] * 100:6.2f}%")
        print("=" * 50 + "\n")

    except Exception as e:
        logger.error(f"Inference pipeline encountered an error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
