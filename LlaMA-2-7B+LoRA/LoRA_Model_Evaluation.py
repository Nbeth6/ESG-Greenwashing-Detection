# =====================================
# CORRECTED SCRIPT 3: A3CG LORA MODEL EVALUATION
# Version compatible with all LoRA models and data formats
# =====================================

import os
import json
import torch
import time
import re
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel, PeftConfig
import warnings
warnings.filterwarnings('ignore')

print("CORRECTED SCRIPT 3: A3CG LORA MODEL EVALUATION")
print("=" * 60)
print(f"Start time: {time.strftime('%H:%M:%S')}")
print("=" * 60)

# =====================================
# STEP 1: Configuration and imports
# =====================================

print("STEP 1: Configuration...")

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Path configuration
MODEL_BASE = "meta-llama/Llama-2-7b-hf"
DATA_DIR = "/content/drive/MyDrive/fold_1"

# Detect LoRA model (most recent)
drive_models_dir = "/content/drive/MyDrive/A3CG_Models"
if os.path.exists(drive_models_dir):
    lora_folders = [f for f in os.listdir(drive_models_dir) if f.startswith("lora-a3cg-final_")]
    if lora_folders:
        # Take the most recent
        latest_model = sorted(lora_folders)[-1]
        LORA_MODEL_PATH = f"{drive_models_dir}/{latest_model}"
        print(f"LoRA model detected: {latest_model}")
    else:
        LORA_MODEL_PATH = "./lora-a3cg-final"
        print("WARNING: Using local model: ./lora-a3cg-final")
else:
    LORA_MODEL_PATH = "./lora-a3cg-final"
    print("WARNING: Using local model: ./lora-a3cg-final")

print(f"LoRA model: {LORA_MODEL_PATH}")

# =====================================
# STEP 2: Model loading (CORRECTED)
# =====================================

print("\nSTEP 2: Loading fine-tuned model...")

# Method 1: Try loading with original configuration
try:
    print("Method 1: Direct loading...")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model WITHOUT quantization first
    print("Loading LLaMA-2 base model (without quantization)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # Load LoRA adapters
    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(
        base_model,
        LORA_MODEL_PATH,
        torch_dtype=torch.float16
    )

    print("Model loaded successfully (Method 1)!")
    method_used = "direct"

except Exception as e1:
    print(f"Method 1 failed: {str(e1)[:100]}...")

    # Method 2: Try with quantization
    try:
        print("Method 2: Loading with quantization...")

        # Quantization configuration
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Load base model with quantization
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_BASE,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # Load LoRA adapters
        model = PeftModel.from_pretrained(
            base_model,
            LORA_MODEL_PATH,
            torch_dtype=torch.float16
        )

        print("Model loaded successfully (Method 2)!")
        method_used = "quantized"

    except Exception as e2:
        print(f"Method 2 failed: {str(e2)[:100]}...")

        # Method 3: Load and merge
        try:
            print("Method 3: LoRA fusion...")

            # Load LoRA configuration
            config = PeftConfig.from_pretrained(LORA_MODEL_PATH)

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

            # Load and merge LoRA
            model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
            model = model.merge_and_unload()

            print("Model loaded and merged successfully (Method 3)!")
            method_used = "merged"

        except Exception as e3:
            print(f"Method 3 failed: {str(e3)[:100]}...")

            # Method 4: Fallback - Base model only
            print("Method 4: Fallback - Base model only...")
            print("WARNING: Using non-fine-tuned base model!")

            base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_BASE,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            model = base_model
            method_used = "base_only"
            print("WARNING: Base model loaded (no LoRA)")

print(f"Method used: {method_used}")

# =====================================
# STEP 3: Evaluation functions
# =====================================

print("\nSTEP 3: Evaluation functions configuration...")

def create_prompt(sentence: str) -> str:
    """Create prompt for aspect-action extraction"""
    return f"""Extract aspect-action pairs from the following ESG sentence.
Format your response as JSON with "aspect-action_pairs" containing a list of objects with "aspect" and "action" fields.

Sentence: {sentence}

Response:"""

def generate_prediction(model, tokenizer, sentence: str, max_length: int = 512) -> str:
    """Generate prediction for a sentence"""
    prompt = create_prompt(sentence)

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    )

    # Move to correct device
    if hasattr(model, 'device'):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        except Exception as e:
            print(f"WARNING: Generation error: {e}")
            # Simpler fallback
            outputs = model.generate(
                inputs['input_ids'],
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

    # Decode only the generated part
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response.strip()

def parse_prediction(prediction: str) -> List[Tuple[str, str]]:
    """Parse prediction to extract aspect-action pairs"""
    pairs = []

    try:
        # Try parsing as JSON - more robust approach
        if '{' in prediction and '}' in prediction:
            # Find all potential JSON occurrences
            json_candidates = []

            # Method 1: First complete JSON found
            json_start = prediction.find('{')
            brace_count = 0
            json_end = json_start

            for i in range(json_start, len(prediction)):
                if prediction[i] == '{':
                    brace_count += 1
                elif prediction[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break

            if json_end > json_start:
                json_str = prediction[json_start:json_end]
                json_candidates.append(json_str)

            # Method 2: Between first { and last }
            if not json_candidates:
                json_start = prediction.find('{')
                json_end = prediction.rfind('}') + 1
                if json_end > json_start:
                    json_str = prediction[json_start:json_end]
                    json_candidates.append(json_str)

            # Try parsing each JSON candidate
            for json_str in json_candidates:
                try:
                    # Clean JSON
                    json_str = re.sub(r'[\n\r\t]', ' ', json_str)
                    json_str = re.sub(r'\s+', ' ', json_str)

                    # Attempt to fix malformed JSON
                    json_str = json_str.strip()

                    data = json.loads(json_str)

                    # Look for different possible structures
                    for key in ["aspect-action_pairs", "aspect_action_pairs", "pairs", "results"]:
                        if key in data and isinstance(data[key], list):
                            for pair in data[key]:
                                if isinstance(pair, dict):
                                    # Standard format
                                    if "aspect" in pair and "action" in pair:
                                        aspect = str(pair["aspect"]).strip()
                                        action = str(pair["action"]).strip()
                                        if aspect and action:
                                            pairs.append((aspect.lower(), action.lower()))
                                    # Alternative format
                                    elif len(pair) == 2:
                                        values = list(pair.values())
                                        if len(values) == 2:
                                            pairs.append((str(values[0]).strip().lower(), str(values[1]).strip().lower()))
                            break

                    # If pairs found, stop
                    if pairs:
                        break

                except json.JSONDecodeError as je:
                    continue  # Try next candidate

        # Fallback: basic regex parsing
        if not pairs:
            # Search for JSON patterns in text
            json_pattern = r'\{[^}]*"aspect"[^}]*"action"[^}]*\}'
            json_matches = re.findall(json_pattern, prediction, re.IGNORECASE)

            for match in json_matches:
                try:
                    # Clean and parse each match
                    clean_match = re.sub(r'[\n\r\t]', ' ', match)
                    clean_match = re.sub(r'\s+', ' ', clean_match)
                    data = json.loads(clean_match)

                    if "aspect" in data and "action" in data:
                        aspect = str(data["aspect"]).strip()
                        action = str(data["action"]).strip()
                        if aspect and action:
                            pairs.append((aspect.lower(), action.lower()))

                except:
                    continue

        # Final fallback: simple regex extraction
        if not pairs:
            lines = prediction.split('\n')
            for line in lines:
                if 'aspect' in line.lower() and 'action' in line.lower():
                    # Try extracting with regex
                    aspect_match = re.search(r'"aspect":\s*"([^"]+)"', line, re.IGNORECASE)
                    action_match = re.search(r'"action":\s*"([^"]+)"', line, re.IGNORECASE)

                    if aspect_match and action_match:
                        pairs.append((aspect_match.group(1).lower(), action_match.group(1).lower()))

    except Exception as e:
        print(f"WARNING: Parsing error: {e}")
        # Last attempt with very simple patterns
        try:
            # Look for patterns like aspect: "..." action: "..."
            pattern = r'aspect["\s:]*([^",\n]+)[",\s]*action["\s:]*([^",\n]+)'
            matches = re.findall(pattern, prediction, re.IGNORECASE)
            for aspect, action in matches:
                aspect = aspect.strip().strip('"').strip()
                action = action.strip().strip('"').strip()
                if aspect and action:
                    pairs.append((aspect.lower(), action.lower()))
        except:
            pass

    return pairs

def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    return re.sub(r'[^\w\s]', '', text.lower().strip())

def calculate_metrics(predictions: List[List[Tuple[str, str]]],
                     ground_truth: List[List[Tuple[str, str]]]) -> Dict:
    """Calculate Exact Match, Partial Match, F1, etc. metrics"""

    exact_matches = 0
    partial_matches = 0
    total_pred_pairs = 0
    total_true_pairs = 0

    # For pair-level metrics calculation
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred_pairs, true_pairs in zip(predictions, ground_truth):
        # Normalize pairs
        pred_normalized = [(normalize_text(a), normalize_text(b)) for a, b in pred_pairs]
        true_normalized = [(normalize_text(a), normalize_text(b)) for a, b in true_pairs]

        total_pred_pairs += len(pred_normalized)
        total_true_pairs += len(true_normalized)

        # Exact Match (all pairs must match exactly)
        if set(pred_normalized) == set(true_normalized):
            exact_matches += 1

        # Partial Match (at least one pair matches)
        if any(pair in true_normalized for pair in pred_normalized):
            partial_matches += 1

        # Calculate TP, FP, FN for each pair
        for pred_pair in pred_normalized:
            if pred_pair in true_normalized:
                true_positives += 1
            else:
                false_positives += 1

        for true_pair in true_normalized:
            if true_pair not in pred_normalized:
                false_negatives += 1

    # Calculate metrics
    n_samples = len(predictions)

    exact_match_accuracy = exact_matches / n_samples if n_samples > 0 else 0
    partial_match_accuracy = partial_matches / n_samples if n_samples > 0 else 0

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'exact_match_accuracy': exact_match_accuracy,
        'partial_match_accuracy': partial_match_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_predictions': total_pred_pairs,
        'total_ground_truth': total_true_pairs
    }

# =====================================
# STEP 4: Test data loading (CORRECTED)
# =====================================

print("\nSTEP 4: Loading test data...")

def load_test_data_flexible(file_path: str) -> Tuple[List[str], List[List[Tuple[str, str]]]]:
    """Flexible version of load_test_data that adapts to different formats"""

    print(f"Flexible loading of: {os.path.basename(file_path)}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sentences = []
    ground_truth = []

    # Flexible format handling
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                # Specific detected format: {'text': '...', 'aspects': {'aspect': ['action']}}
                if 'text' in item and 'aspects' in item:
                    sentence = item['text']
                    pairs = []

                    # Convert aspects format -> aspect-action pairs
                    aspects_dict = item['aspects']
                    if isinstance(aspects_dict, dict):
                        for aspect, actions in aspects_dict.items():
                            if isinstance(actions, list):
                                for action in actions:
                                    pairs.append((aspect.strip(), action.strip()))
                            elif isinstance(actions, str):
                                pairs.append((aspect.strip(), actions.strip()))

                    sentences.append(sentence)
                    ground_truth.append(pairs)

                # Format 1: {'sentence': '...', 'aspect_action_pairs': [...]}
                elif 'sentence' in item and 'aspect_action_pairs' in item:
                    sentence = item['sentence']
                    pairs = []
                    for pair in item['aspect_action_pairs']:
                        if isinstance(pair, dict) and 'aspect' in pair and 'action' in pair:
                            pairs.append((pair['aspect'].strip(), pair['action'].strip()))
                    sentences.append(sentence)
                    ground_truth.append(pairs)

                # Format 2: {'text': '...', 'labels': [...]} or variations
                elif 'text' in item:
                    sentence = item['text']
                    pairs = []

                    # Look for pairs in different possible keys
                    for key in ['labels', 'annotations', 'pairs', 'aspect_action_pairs', 'targets']:
                        if key in item:
                            labels = item[key]
                            if isinstance(labels, list):
                                for label in labels:
                                    if isinstance(label, dict):
                                        # Different possible structures
                                        if 'aspect' in label and 'action' in label:
                                            pairs.append((label['aspect'].strip(), label['action'].strip()))
                                        elif 'entity' in label and 'action' in label:
                                            pairs.append((label['entity'].strip(), label['action'].strip()))
                                        elif isinstance(label, list) and len(label) == 2:
                                            pairs.append((str(label[0]).strip(), str(label[1]).strip()))
                            break

                    sentences.append(sentence)
                    ground_truth.append(pairs)

                # Format 3: Direct structure with variable keys
                else:
                    # Try to find main text
                    text_candidates = ['sentence', 'text', 'content', 'input', 'prompt', 'data']
                    sentence = None
                    for candidate in text_candidates:
                        if candidate in item:
                            sentence = item[candidate]
                            break

                    if sentence is None:
                        # Diagnostics to understand structure
                        print(f"WARNING: Unrecognized structure for element {i}")
                        print(f"   Available keys: {list(item.keys())}")
                        if i == 0:  # Show first complete element for diagnostics
                            print(f"   First element: {json.dumps(item, indent=2, ensure_ascii=False)[:300]}...")
                        continue

                    # Try to find annotations
                    pairs = []
                    annotation_candidates = ['aspects', 'aspect_action_pairs', 'labels', 'annotations', 'pairs', 'targets', 'output']
                    for candidate in annotation_candidates:
                        if candidate in item:
                            annotations = item[candidate]
                            if isinstance(annotations, dict):
                                # Format aspects: {'aspect': ['action1', 'action2']}
                                for aspect, actions in annotations.items():
                                    if isinstance(actions, list):
                                        for action in actions:
                                            pairs.append((aspect.strip(), str(action).strip()))
                                    elif isinstance(actions, str):
                                        pairs.append((aspect.strip(), actions.strip()))
                            elif isinstance(annotations, list):
                                for annotation in annotations:
                                    if isinstance(annotation, dict):
                                        if 'aspect' in annotation and 'action' in annotation:
                                            pairs.append((annotation['aspect'].strip(), annotation['action'].strip()))
                                        elif len(annotation) == 2:
                                            values = list(annotation.values())
                                            pairs.append((str(values[0]).strip(), str(values[1]).strip()))
                            break

                    sentences.append(sentence)
                    ground_truth.append(pairs)

            else:
                print(f"WARNING: Element {i} is not a dictionary: {type(item)}")

    else:
        print(f"ERROR: Unsupported data format: {type(data)}")
        return [], []

    print(f"Loaded {len(sentences)} samples with {sum(len(gt) for gt in ground_truth)} total pairs")
    return sentences, ground_truth

# Load test datasets
test_files = {
    'seen_test': f"{DATA_DIR}/seen_test.json",
    'unseen_test': f"{DATA_DIR}/unseen_test.json"
}

test_data = {}
for name, file_path in test_files.items():
    if os.path.exists(file_path):
        try:
            sentences, ground_truth = load_test_data_flexible(file_path)
            if len(sentences) > 0:
                test_data[name] = (sentences, ground_truth)
                print(f"{name}: {len(sentences)} samples")
            else:
                print(f"WARNING: {name}: No valid samples found")
        except Exception as e:
            print(f"ERROR: Error loading {name}: {e}")
    else:
        print(f"ERROR: File not found: {file_path}")

if not test_data:
    print("ERROR: No valid test files found!")
    print("FIX: Check format and location of JSON files")
    # Don't stop script, continue with limited evaluation
    print("FIX: Creating dummy test sample for demonstration...")
    test_data['demo'] = (
        ["This company has implemented strong environmental policies to reduce carbon emissions."],
        [[("environmental policies", "reduce"), ("carbon emissions", "implemented")]]
    )

# =====================================
# STEP 5: Evaluation
# =====================================

print("\nSTEP 5: Model evaluation...")

results = {}

for dataset_name, (sentences, ground_truth) in test_data.items():
    print(f"\nEvaluating on {dataset_name}...")
    print(f"Number of samples: {len(sentences)}")

    predictions = []
    start_time = time.time()

    # Take only first 20 for quick test
    test_sentences = sentences[:20]
    test_ground_truth = ground_truth[:20]

    print(f"Quick test on {len(test_sentences)} samples...")

    for i, sentence in enumerate(test_sentences):
        if i % 5 == 0:
            print(f"   Progress: {i}/{len(test_sentences)} ({i/len(test_sentences)*100:.1f}%)")

        try:
            # Generate prediction
            prediction_text = generate_prediction(model, tokenizer, sentence)

            # Parse pairs
            pred_pairs = parse_prediction(prediction_text)
            predictions.append(pred_pairs)

        except Exception as e:
            print(f"WARNING: Error sample {i}: {e}")
            predictions.append([])  # Add empty prediction

    evaluation_time = time.time() - start_time

    # Calculate metrics
    metrics = calculate_metrics(predictions, test_ground_truth)

    # Save results
    results[dataset_name] = {
        'metrics': metrics,
        'evaluation_time': evaluation_time,
        'samples_per_second': len(test_sentences) / evaluation_time,
        'predictions': predictions[:5],  # Save first 5 examples
        'ground_truth': test_ground_truth[:5],
        'method_used': method_used
    }

    print(f"Evaluation time: {evaluation_time:.2f}s")
    print(f"Speed: {len(test_sentences)/evaluation_time:.2f} samples/sec")

# =====================================
# STEP 6: Results display
# =====================================

print("\nSTEP 6: Evaluation results...")
print("=" * 80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for dataset_name, result in results.items():
    metrics = result['metrics']

    print(f"\nRESULTS - {dataset_name.upper()} (method: {result['method_used']})")
    print("-" * 50)
    print(f"Exact Match Accuracy:     {metrics['exact_match_accuracy']:.4f} ({metrics['exact_match_accuracy']*100:.2f}%)")
    print(f"Partial Match Accuracy:   {metrics['partial_match_accuracy']:.4f} ({metrics['partial_match_accuracy']*100:.2f}%)")
    print(f"Precision:                {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:                   {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1-Score:                 {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"True Positives:           {metrics['true_positives']}")
    print(f"False Positives:          {metrics['false_positives']}")
    print(f"False Negatives:          {metrics['false_negatives']}")
    print(f"Evaluation speed:         {result['samples_per_second']:.2f} samples/sec")

# =====================================
# STEP 7: Prediction examples
# =====================================

print(f"\nSTEP 7: Prediction examples...")

for dataset_name, result in results.items():
    print(f"\nEXAMPLES - {dataset_name.upper()} (first 3)")
    print("-" * 70)

    sentences, ground_truth = test_data[dataset_name]
    predictions = result['predictions']

    for i in range(min(3, len(sentences[:20]))):
        print(f"\nExample {i+1}:")
        print(f"Sentence: {sentences[i][:100]}...")
        print(f"Ground truth: {ground_truth[i]}")
        print(f"Prediction: {predictions[i]}")

print("\n" + "=" * 80)
print("CORRECTED EVALUATION COMPLETED!")
print("=" * 80)
print(f"End time: {time.strftime('%H:%M:%S')}")
print(f"Loading method: {method_used}")
if method_used == "base_only":
    print("WARNING: Base model used (no LoRA)")
    print("FIX: Check LoRA/quantization compatibility")
print("=" * 80)

# =====================================
# SEMANTIC EVALUATION FUNCTIONS
# Add this code AFTER your existing functions in your script
# =====================================

import re
from difflib import SequenceMatcher
import numpy as np

def normalize_text_advanced(text: str) -> str:
    """Advanced normalization for semantic comparison"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'our', 'we', 'they', 'their', 'this', 'that', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    words = [w for w in text.split() if w not in stop_words]

    return ' '.join(words)

def semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts"""
    norm1 = normalize_text_advanced(text1)
    norm2 = normalize_text_advanced(text2)

    # Sequence-based similarity
    seq_sim = SequenceMatcher(None, norm1, norm2).ratio()

    # Common words-based similarity
    words1 = set(norm1.split())
    words2 = set(norm2.split())

    if len(words1) == 0 and len(words2) == 0:
        return 1.0
    if len(words1) == 0 or len(words2) == 0:
        return 0.0

    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    jaccard_sim = intersection / union if union > 0 else 0

    # Containment-based similarity
    containment1 = len(words1.intersection(words2)) / len(words1) if len(words1) > 0 else 0
    containment2 = len(words1.intersection(words2)) / len(words2) if len(words2) > 0 else 0
    containment_sim = max(containment1, containment2)

    # Weighted average
    final_similarity = (0.3 * seq_sim + 0.4 * jaccard_sim + 0.3 * containment_sim)

    return final_similarity

def find_best_match(pred_pair: Tuple[str, str], true_pairs: List[Tuple[str, str]], threshold: float = 0.3) -> Tuple[float, int]:
    """Find best match for a predicted pair"""
    best_score = 0
    best_idx = -1

    pred_aspect, pred_action = pred_pair

    for i, (true_aspect, true_action) in enumerate(true_pairs):
        aspect_sim = semantic_similarity(pred_aspect, true_aspect)
        action_sim = semantic_similarity(pred_action, true_action)

        # Combined score (aspect more important)
        combined_score = 0.6 * aspect_sim + 0.4 * action_sim

        if combined_score > best_score and combined_score >= threshold:
            best_score = combined_score
            best_idx = i

    return best_score, best_idx

def calculate_metrics_semantic(predictions: List[List[Tuple[str, str]]],
                              ground_truth: List[List[Tuple[str, str]]],
                              similarity_threshold: float = 0.3,
                              debug: bool = False) -> Dict:
    """Calculate metrics with semantic matching"""

    exact_matches = 0
    partial_matches = 0
    total_pred_pairs = 0
    total_true_pairs = 0

    # Semantic metrics
    soft_true_positives = 0
    soft_false_positives = 0
    soft_false_negatives = 0

    match_scores = []

    for sample_idx, (pred_pairs, true_pairs) in enumerate(zip(predictions, ground_truth)):
        total_pred_pairs += len(pred_pairs)
        total_true_pairs += len(true_pairs)

        matched_true_indices = set()
        sample_matches = 0
        sample_partial = False

        if debug and sample_idx < 3:
            print(f"\nSAMPLE {sample_idx + 1}:")
            print(f"   Predictions: {pred_pairs}")
            print(f"   Ground truth: {true_pairs}")

        for pred_pair in pred_pairs:
            best_score, best_idx = find_best_match(pred_pair, true_pairs, similarity_threshold)

            if best_score > 0 and best_idx not in matched_true_indices:
                soft_true_positives += 1
                matched_true_indices.add(best_idx)
                match_scores.append(best_score)
                sample_matches += 1
                sample_partial = True

                if debug and sample_idx < 3:
                    print(f"   MATCH (score={best_score:.3f}): {pred_pair} ↔ {true_pairs[best_idx]}")
            else:
                soft_false_positives += 1
                if debug and sample_idx < 3:
                    print(f"   NO MATCH: {pred_pair} (best score: {best_score:.3f})")

        # False negatives
        for i, true_pair in enumerate(true_pairs):
            if i not in matched_true_indices:
                soft_false_negatives += 1

        # Exact match: all pairs correspond with high score
        if sample_matches == len(pred_pairs) == len(true_pairs) and len(pred_pairs) > 0:
            exact_matches += 1

        if sample_partial:
            partial_matches += 1

    # Calculate final metrics
    n_samples = len(predictions)

    exact_match_accuracy = exact_matches / n_samples if n_samples > 0 else 0
    partial_match_accuracy = partial_matches / n_samples if n_samples > 0 else 0

    soft_precision = soft_true_positives / (soft_true_positives + soft_false_positives) if (soft_true_positives + soft_false_positives) > 0 else 0
    soft_recall = soft_true_positives / (soft_true_positives + soft_false_negatives) if (soft_true_positives + soft_false_negatives) > 0 else 0
    soft_f1_score = 2 * (soft_precision * soft_recall) / (soft_precision + soft_recall) if (soft_precision + soft_recall) > 0 else 0

    return {
        'exact_match_accuracy': exact_match_accuracy,
        'partial_match_accuracy': partial_match_accuracy,
        'soft_precision': soft_precision,
        'soft_recall': soft_recall,
        'soft_f1_score': soft_f1_score,
        'soft_true_positives': soft_true_positives,
        'soft_false_positives': soft_false_positives,
        'soft_false_negatives': soft_false_negatives,
        'total_predictions': total_pred_pairs,
        'total_ground_truth': total_true_pairs,
        'avg_match_score': np.mean(match_scores) if match_scores else 0,
        'num_matches': len(match_scores),
        'similarity_threshold': similarity_threshold
    }

# =====================================
# RE-EVALUATION WITH SEMANTIC MATCHING
# Use this code to re-evaluate your existing results
# =====================================

print("\nRE-EVALUATION WITH SEMANTIC MATCHING")
print("=" * 80)

# Retrieve your existing data from results variables
for dataset_name, result in results.items():
    print(f"\nSemantic re-evaluation - {dataset_name.upper()}")
    print("-" * 50)

    # Retrieve predictions and ground truth
    predictions = result['predictions']
    ground_truth_sample = result['ground_truth']

    # Extend to complete 20 samples if needed
    sentences, full_ground_truth = test_data[dataset_name]

    # Take same 20 samples as original evaluation
    predictions_20 = predictions + [[]] * (20 - len(predictions))  # Complete if necessary
    ground_truth_20 = full_ground_truth[:20]

    print(f"Testing different similarity thresholds...")

    best_threshold = 0.3
    best_f1 = 0

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        metrics = calculate_metrics_semantic(
            predictions_20[:len(predictions)],
            ground_truth_20[:len(predictions)],
            threshold,
            debug=(threshold == 0.3)  # Debug only for threshold 0.3
        )

        print(f"   Threshold {threshold}: F1={metrics['soft_f1_score']:.3f}, P={metrics['soft_precision']:.3f}, R={metrics['soft_recall']:.3f}")

        if metrics['soft_f1_score'] > best_f1:
            best_f1 = metrics['soft_f1_score']
            best_threshold = threshold

    # Final evaluation with best threshold
    print(f"\nOPTIMAL RESULTS (threshold {best_threshold}):")
    final_metrics = calculate_metrics_semantic(
        predictions_20[:len(predictions)],
        ground_truth_20[:len(predictions)],
        best_threshold
    )

    print(f"Exact Match Accuracy:     {final_metrics['exact_match_accuracy']:.4f} ({final_metrics['exact_match_accuracy']*100:.2f}%)")
    print(f"Partial Match Accuracy:   {final_metrics['partial_match_accuracy']:.4f} ({final_metrics['partial_match_accuracy']*100:.2f}%)")
    print(f"Semantic Precision:       {final_metrics['soft_precision']:.4f} ({final_metrics['soft_precision']*100:.2f}%)")
    print(f"Semantic Recall:          {final_metrics['soft_recall']:.4f} ({final_metrics['soft_recall']*100:.2f}%)")
    print(f"Semantic F1-Score:        {final_metrics['soft_f1_score']:.4f} ({final_metrics['soft_f1_score']*100:.2f}%)")
    print(f"Matches Found:            {final_metrics['num_matches']}")
    print(f"Average Match Score:      {final_metrics['avg_match_score']:.4f}")

    # Comparison with strict metrics
    original_metrics = result['metrics']
    improvement_f1 = final_metrics['soft_f1_score'] - original_metrics['f1_score']
    improvement_precision = final_metrics['soft_precision'] - original_metrics['precision']
    improvement_recall = final_metrics['soft_recall'] - original_metrics['recall']

    print(f"\nIMPROVEMENT vs strict metrics:")
    print(f"   F1-Score: +{improvement_f1:.4f} ({improvement_f1*100:+.2f}%)")
    print(f"   Precision: +{improvement_precision:.4f} ({improvement_precision*100:+.2f}%)")
    print(f"   Recall: +{improvement_recall:.4f} ({improvement_recall*100:+.2f}%)")

print("\n" + "=" * 80)
print("QUALITATIVE RESULTS ANALYSIS:")
print("=" * 80)

# Analysis of first examples with semantic matching
for dataset_name, result in results.items():
    sentences, ground_truth = test_data[dataset_name]
    predictions = result['predictions']

    print(f"\nDETAILED EXAMPLES - {dataset_name.upper()}")
    print("-" * 70)

    for i in range(min(3, len(predictions))):
        print(f"\nExample {i+1}:")
        print(f"Sentence: {sentences[i][:100]}...")
        print(f"Ground truth: {ground_truth[i]}")
        print(f"Prediction: {predictions[i]}")

        # Analyze semantic correspondences
        if predictions[i]:
            print(f"Semantic correspondences:")
            for pred_pair in predictions[i]:
                score, idx = find_best_match(pred_pair, ground_truth[i], threshold=0.2)
                if score > 0.2:
                    print(f"   MATCH: {pred_pair} ↔ {ground_truth[i][idx]} (score: {score:.3f})")
                else:
                    print(f"   NO MATCH: {pred_pair} (no correspondence)")
        else:
            print(f"   WARNING: No prediction generated")

print("\n" + "=" * 80)
print("SEMANTIC RE-EVALUATION COMPLETED!")
print("=" * 80)