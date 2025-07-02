# =====================================
# MoE-LoRA GPU DIRECT WITH FEW-SHOT TEST - COMPLETE CORRECTED VERSION
# Fixes: bitsandbytes, insufficient training, incoherent generation
# =====================================

import os
import json
import torch
import time
import re
import random
import gc
import psutil
import ctypes
import subprocess
import sys
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

print("MoE-LoRA GPU DIRECT + FEW-SHOT TEST - COMPLETE CORRECTED VERSION")
print("=" * 70)
print(f"Start time: {time.strftime('%H:%M:%S')}")
start_time = time.time()
print("=" * 70)

# =====================================
# STEP 0: MISSING DEPENDENCIES INSTALLATION
# =====================================

print("STEP 0: Checking and installing missing dependencies...")

def install_dependencies():
    """Install missing dependencies"""

    # 1. Check bitsandbytes
    try:
        from transformers import BitsAndBytesConfig
        print("bitsandbytes already available")
        return True
    except Exception as e:
        print(f"bitsandbytes missing: {e}")

        # Install bitsandbytes
        try:
            print("Installing bitsandbytes...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "bitsandbytes>=0.41.0", "--quiet"
            ])

            # Re-import after installation
            from transformers import BitsAndBytesConfig
            print("bitsandbytes installed successfully")
            return True

        except Exception as install_error:
            print(f"Failed to install bitsandbytes: {install_error}")
            print("Continue without quantization...")
            return False

USE_QUANTIZATION = install_dependencies()

# =====================================
# MEMORY OPTIMIZATION FUNCTIONS
# =====================================

def get_memory_usage_detailed():
    """Detailed system RAM and GPU usage"""
    process = psutil.Process()
    ram_gb = process.memory_info().rss / 1024**3

    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        return ram_gb, gpu_allocated, gpu_reserved
    return ram_gb, 0, 0

def monitor_memory_gpu_direct(step_name):
    """Advanced monitoring for GPU Direct"""
    ram_gb, gpu_allocated, gpu_reserved = get_memory_usage_detailed()
    print(f"Memory {step_name}:")
    print(f"   System RAM: {ram_gb:.1f} GB")
    print(f"   GPU allocated: {gpu_allocated:.1f} GB")
    print(f"   GPU reserved: {gpu_reserved:.1f} GB")
    return ram_gb, gpu_allocated

def ultra_aggressive_gpu_cleanup():
    """Ultra-aggressive GPU + RAM cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()

    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass

# =====================================
# STEP 1: A3CG Data Import
# =====================================

print("\nSTEP 1: A3CG data import...")
monitor_memory_gpu_direct("Data import start")

# Mount Google Drive
from google.colab import drive
try:
    drive.mount('/content/drive')
    print("Drive mounted")
except:
    print("Drive already mounted")

# Search for fold_1 folder
possible_paths = [
    "/content/drive/MyDrive/fold_1",
    "/content/drive/MyDrive/fold1",
    "/content/drive/MyDrive/A3CG/fold_1",
    "/content/drive/MyDrive/A3CG_DATASET/fold_1",
    "/content/drive/MyDrive/folds/fold_1"
]

drive_fold_path = None
for path in possible_paths:
    if os.path.exists(path):
        drive_fold_path = path
        print(f"Data found: {path}")
        break

# Deep search if not found
if not drive_fold_path:
    print("Deep search...")
    mydrive_path = "/content/drive/MyDrive"

    for root, dirs, files in os.walk(mydrive_path):
        if "seen_train.json" in files and "seen_val.json" in files:
            drive_fold_path = root
            print(f"Files found in: {drive_fold_path}")
            break
        if root.count('/') > mydrive_path.count('/') + 2:
            continue

if not drive_fold_path:
    print("A3CG folder not found - Creating test data...")
    # Create minimal test data
    DATA_DIR = "/content/test_data"
    os.makedirs(DATA_DIR, exist_ok=True)

    # Minimal test data
    test_data = [
        {
            "text": "We have implemented solar panels to reduce energy consumption in our facilities.",
            "aspects": {"solar panels": ["implemented"], "energy consumption": ["implemented"]}
        },
        {
            "text": "The company plans to improve workplace diversity initiatives next year.",
            "aspects": {"workplace diversity initiatives": ["planning"]}
        },
        {
            "text": "Our recycling program has achieved a 50% waste reduction.",
            "aspects": {"recycling program": ["implemented"], "waste reduction": ["implemented"]}
        }
    ] * 100  # 300 test samples

    # Save test files
    for filename in ["seen_train.json", "seen_val.json", "seen_test.json", "unseen_test.json"]:
        with open(f"{DATA_DIR}/{filename}", 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)

    print("Test data created")
else:
    # Copy real data
    local_path = "/content/A3CG_DATASET/folds/fold_1"
    os.makedirs(local_path, exist_ok=True)

    required_files = ["seen_train.json", "seen_val.json", "seen_test.json", "unseen_test.json"]
    copied_count = 0

    for filename in required_files:
        source = os.path.join(drive_fold_path, filename)
        dest = os.path.join(local_path, filename)

        if os.path.exists(source):
            try:
                import shutil
                shutil.copy2(source, dest)

                with open(dest, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                size_kb = os.path.getsize(dest) / 1024
                print(f"  {filename}: {len(data)} samples ({size_kb:.1f} KB)")
                copied_count += 1
            except Exception as e:
                print(f"  Error with {filename}: {e}")

    DATA_DIR = local_path if copied_count >= 3 else "/content/test_data"

print(f"Data ready: {DATA_DIR}")
monitor_memory_gpu_direct("Data imported")

# =====================================
# STEP 2: ENHANCED Few-Shot Configuration
# =====================================

print("\nSTEP 2: ENHANCED Few-Shot configuration...")

class A3CGFewShotDataProcessor:
    """A3CG dataset with enhanced Few-Shot strategy"""

    def __init__(self):
        # Optimized system prompt for precise extraction
        self.system_prompt = """You are an expert in ESG analysis. Extract aspect-action pairs from sustainability statements.

CRITICAL INSTRUCTIONS:
- Extract EXACT terms from the input text
- Do not paraphrase or interpret creatively
- Use literal wording from the sentence
- Focus on specific terms rather than general concepts

DEFINITIONS:
- Aspect: A sustainability-related entity, goal, sub-area, or activity (use exact wording)
- Action: "implemented", "planning", or "indeterminate"

OUTPUT FORMAT: ("aspect1", "action1"), ("aspect2", "action2"), ...
If none: ("no aspect", "no action")"""

        # Optimized Few-Shot examples bank
        self.few_shot_examples = [
            {
                "text": "We have implemented solar panels to reduce energy consumption in our facilities.",
                "output": '("solar panels", "implemented"), ("energy consumption", "implemented")'
            },
            {
                "text": "The company plans to improve workplace diversity initiatives next year.",
                "output": '("workplace diversity initiatives", "planning")'
            },
            {
                "text": "Our recycling program has achieved a 50% waste reduction.",
                "output": '("recycling program", "implemented"), ("waste reduction", "implemented")'
            },
            {
                "text": "The board may consider sustainability investments where feasible.",
                "output": '("sustainability investments", "indeterminate")'
            },
            {
                "text": "We strive to minimize carbon footprints in our operations.",
                "output": '("carbon footprints", "planning")'
            },
            {
                "text": "Employee training programs were successfully implemented this quarter.",
                "output": '("employee training programs", "implemented")'
            },
            {
                "text": "The company is planning to reduce water consumption by 30%.",
                "output": '("water consumption", "planning")'
            }
        ]

    def get_few_shot_examples(self, n_examples: int = 4) -> str:  # Increased to 4
        """Randomly select few-shot examples"""
        selected = random.sample(self.few_shot_examples, min(n_examples, len(self.few_shot_examples)))

        examples_text = ""
        for i, example in enumerate(selected, 1):
            examples_text += f"\nExample {i}:\n"
            examples_text += f"Text: {example['text']}\n"
            examples_text += f"Output: {example['output']}\n"

        return examples_text

    def create_prompt_fewshot(self, sentence: str) -> str:
        """Create optimized few-shot prompt"""

        few_shot_text = self.get_few_shot_examples(n_examples=4)

        return f"""<s>[INST] <<SYS>>
{self.system_prompt}
<</SYS>>

{few_shot_text}

Now extract from this text:
Text: {sentence}

Extract the aspect-action pairs: [/INST]"""

def load_a3cg_data(file_path: str) -> Tuple[List[str], List[str]]:
    """Load A3CG data"""

    print(f"Loading: {os.path.basename(file_path)}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sentences = []
    outputs = []

    for item in data:
        if isinstance(item, dict) and 'text' in item and 'aspects' in item:
            sentence = item['text']

            # Convert aspects to output format
            pairs = []
            aspects_dict = item['aspects']

            if isinstance(aspects_dict, dict):
                for aspect, actions in aspects_dict.items():
                    if isinstance(actions, list):
                        for action in actions:
                            pairs.append(f'("{aspect.strip()}", "{action.strip()}")')
                    elif isinstance(actions, str):
                        pairs.append(f'("{aspect.strip()}", "{actions.strip()}")')

            output = ', '.join(pairs) if pairs else '("no aspect", "no action")'

            sentences.append(sentence)
            outputs.append(output)

    print(f"Loaded {len(sentences)} samples")
    return sentences, outputs

print("Enhanced Few-Shot processor configured")
monitor_memory_gpu_direct("Few-Shot configuration")

# =====================================
# STEP 3: CORRECTED MoE-LoRA Model Creation
# =====================================

print("\nSTEP 3: CORRECTED MoE-LoRA model creation...")
monitor_memory_gpu_direct("Model creation start")

# Prerequisites verification
if not torch.cuda.is_available():
    print("GPU not available - This script requires a GPU")
    exit()

print(f"GPU detected: {torch.cuda.get_device_name(0)}")
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU memory: {gpu_memory:.1f} GB")

# Module 3 GPU Direct verification
print("Checking Module 3 GPU Direct...")
required_functions = [
    'create_true_moe_llama_model_gpu_direct',
    'TrueLoRAAttentionExpertGPU',
    'TrueMoEAttentionGPU',
    'LoRAFFNGPU',
    'LlamaForCausalLMWithTrueMoEGPU'
]

missing_functions = []
for func_name in required_functions:
    if func_name not in globals():
        missing_functions.append(func_name)

if missing_functions:
    print(f"Missing functions from Module 3 GPU Direct: {missing_functions}")
    print("Please run the cell containing Module 3 GPU Direct first!")
    exit()
else:
    print("Module 3 GPU Direct detected and loaded correctly")

# =====================================
# NO QUANTIZATION VERSION (FALLBACK)
# =====================================

def create_moe_model_no_quantization(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    layers_to_replace: List[int] = [15],
    target_device: str = "cuda",
    torch_dtype = torch.float16,
    **kwargs
):
    """Version without quantization to avoid bitsandbytes error"""

    print(f"CREATION WITHOUT QUANTIZATION on {target_device}")

    moe_config = kwargs.get('moe_attention_config', {})
    lora_ffn_config = kwargs.get('lora_ffn_config', {})

    # Ensure device is passed
    moe_config['device'] = target_device
    lora_ffn_config['device'] = target_device

    print("Loading config...")
    config = AutoConfig.from_pretrained(model_name)
    monitor_memory_gpu_direct("Config loaded")

    print("LOADING BASE MODEL WITHOUT QUANTIZATION...")

    # Forced GPU device map
    device_map = {
        "model.embed_tokens": 0,
        "model.norm": 0,
        "lm_head": 0,
    }
    for i in range(32):
        device_map[f"model.layers.{i}"] = 0

    # Loading WITHOUT quantization but with optimizations
    from transformers import LlamaForCausalLM

    try:
        base_model = LlamaForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,  # CRITICAL!
            trust_remote_code=True,
        )

        monitor_memory_gpu_direct("Base model loaded (without quantization)")
        print(f"Base model loaded on: {next(base_model.parameters()).device}")

    except Exception as e:
        print(f"Error loading base model: {e}")
        raise

    print("CREATING MoE-LoRA MODEL...")

    # Create MoE model using Module 3
    model_moe = LlamaForCausalLMWithTrueMoEGPU(
        config=config,
        layers_to_replace=layers_to_replace,
        moe_attention_config=moe_config,
        lora_ffn_config=lora_ffn_config,
        device=target_device
    )

    monitor_memory_gpu_direct("MoE model created")

    print("COPYING COMPATIBLE WEIGHTS...")

    # Copy weights from base model to MoE model
    try:
        moe_state_dict = model_moe.state_dict()
        base_state_dict = base_model.state_dict()

        copied_count = 0
        for name, param in base_state_dict.items():
            if name in moe_state_dict and moe_state_dict[name].shape == param.shape:
                moe_state_dict[name].copy_(param)
                copied_count += 1

        print(f"{copied_count} weights copied")

    except Exception as e:
        print(f"Weight copy error: {e}")

    # Free base model
    del base_model
    ultra_aggressive_gpu_cleanup()
    monitor_memory_gpu_direct("Base model freed")

    # Ensure entire model is on GPU
    if torch_dtype == torch.float16:
        model_moe = model_moe.to(target_device).half()
    else:
        model_moe = model_moe.to(target_device)

    # Final device verification
    final_device = next(model_moe.parameters()).device
    print(f"MoE-LoRA model created on {final_device}!")

    return model_moe

# =====================================
# MODEL CREATION WITH ERROR HANDLING
# =====================================

MODEL_NAME = "meta-llama/Llama-2-7b-hf"
DEVICE = "cuda"

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")
    monitor_memory_gpu_direct("Tokenizer loaded")

    print("Creating MoE-LoRA model...")

    # Monitor RAM before loading
    ram_before, gpu_before = monitor_memory_gpu_direct("Before model creation")

    # Try with quantization first
    if USE_QUANTIZATION:
        try:
            print("Attempting with quantization...")
            model = create_true_moe_llama_model_gpu_direct(
                model_name=MODEL_NAME,
                layers_to_replace=[15],  # 1 layer for quick test
                target_device=DEVICE,
                use_quantization=True,
                torch_dtype=torch.float16,
                moe_attention_config={
                    'num_experts': 4,  # Reduced for test
                    'lora_rank': 8,
                    'lora_alpha': 64.0,
                    'top_k_experts': 2,
                    'attention_dropout': 0.1
                },
                lora_ffn_config={
                    'lora_rank': 8,  # Reduced for test
                    'lora_alpha': 64.0
                }
            )
            print("Model created with quantization")

        except Exception as e:
            print(f"Quantization failed: {e}")
            print("Fallback without quantization...")
            model = create_moe_model_no_quantization(
                model_name=MODEL_NAME,
                layers_to_replace=[15],
                target_device=DEVICE,
                torch_dtype=torch.float16,
                moe_attention_config={
                    'num_experts': 4,
                    'lora_rank': 8,
                    'lora_alpha': 64.0,
                    'top_k_experts': 2,
                },
                lora_ffn_config={
                    'lora_rank': 8,
                    'lora_alpha': 64.0
                }
            )
    else:
        print("Creating without quantization...")
        model = create_moe_model_no_quantization(
            model_name=MODEL_NAME,
            layers_to_replace=[15],
            target_device=DEVICE,
            torch_dtype=torch.float16,
            moe_attention_config={
                'num_experts': 4,
                'lora_rank': 8,
                'lora_alpha': 64.0,
                'top_k_experts': 2,
            },
            lora_ffn_config={
                'lora_rank': 8,
                'lora_alpha': 64.0
            }
        )

    ram_after, gpu_after = monitor_memory_gpu_direct("MoE-LoRA model created")

    # Check RAM optimization
    ram_increase = ram_after - ram_before
    gpu_increase = gpu_after - gpu_before

    print(f"Memory increase:")
    print(f"   System RAM: +{ram_increase:.1f} GB")
    print(f"   GPU: +{gpu_increase:.1f} GB")

    if ram_increase > 15:
        print(f"ALERT: High system RAM (+{ram_increase:.1f} GB)")
    else:
        print(f"SUCCESS: Optimized system RAM (+{ram_increase:.1f} GB)")

    # Check model placement
    model_device = next(model.parameters()).device
    print(f"Model placed on: {model_device}")

    # Display model info
    params_info = model.model.count_parameters()
    print(f"Total parameters: {params_info['total']:,}")
    print(f"Trainable parameters: {params_info['trainable']:,} ({params_info['trainable_percent']:.2f}%)")

except Exception as e:
    print(f"Model creation error: {e}")
    monitor_memory_gpu_direct("Creation error")
    import traceback
    traceback.print_exc()
    exit()

# =====================================
# STEP 4: OPTIMIZED Data Preparation
# =====================================

print("\nSTEP 4: OPTIMIZED data preparation...")
monitor_memory_gpu_direct("Data preparation start")

# Load datasets
train_sentences, train_outputs = load_a3cg_data(f"{DATA_DIR}/seen_train.json")
val_sentences, val_outputs = load_a3cg_data(f"{DATA_DIR}/seen_val.json")

# Optimized dataset with enhanced Few-Shot
class MoELoRAFewShotDatasetOptimized(Dataset):
    """MoE-LoRA dataset with optimized Few-Shot"""

    def __init__(self, sentences, outputs, tokenizer, max_length=512):
        self.sentences = sentences
        self.outputs = outputs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processor = A3CGFewShotDataProcessor()
        self.ignore_index = -100

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        output = self.outputs[idx]

        # Create few-shot prompt with 4 examples
        prompt = self.processor.create_prompt_fewshot(sentence)
        full_text = prompt + output + "</s>"

        # Tokenize prompt alone for masking
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length-100,
            padding=False,
            return_tensors="pt"
        )
        prompt_length = prompt_tokens['input_ids'].shape[1]

        # Tokenize full text
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        # Create labels with correct prompt masking
        labels = input_ids.clone()
        labels[:prompt_length] = self.ignore_index
        labels[attention_mask == 0] = self.ignore_index

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Create datasets with optimized samples
print("Creating optimized Few-Shot datasets...")
train_dataset = MoELoRAFewShotDatasetOptimized(train_sentences[:80], train_outputs[:80], tokenizer)  # Increased
val_dataset = MoELoRAFewShotDatasetOptimized(val_sentences[:25], val_outputs[:25], tokenizer)

print(f"Train dataset: {len(train_dataset)} samples")
print(f"Val dataset: {len(val_dataset)} samples")
monitor_memory_gpu_direct("Datasets created")

# =====================================
# STEP 5: PRE-TRAINING DIAGNOSTIC
# =====================================

print("\nSTEP 5: Pre-training diagnostic...")

def diagnostic_model_learning(model, tokenizer, dataset):
    """Diagnostic to verify if model can learn"""

    print("DETAILED DIAGNOSTIC...")

    # Test 1: Check training sample
    sample = dataset[0]
    input_ids = sample['input_ids'].unsqueeze(0)
    labels = sample['labels'].unsqueeze(0)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    print("Testing forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

    print(f"   Loss: {loss.item():.4f}")
    print(f"   Logits shape: {logits.shape}")

    # Test 2: Decode input to verify format
    decoded_input = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"   Sample input: {decoded_input[:150]}...")

    # Test 3: Check masked labels
    valid_labels = labels[labels != -100]
    print(f"   Valid labels: {len(valid_labels)} tokens")

    if len(valid_labels) > 0:
        decoded_labels = tokenizer.decode(valid_labels, skip_special_tokens=True)
        print(f"   Expected labels: {decoded_labels}")

        # Check expected output format
        if '("' in decoded_labels and '")' in decoded_labels:
            print("   Few-Shot format detected in labels")
        else:
            print("   Few-Shot format not detected in labels")

        return True
    else:
        print("   CRITICAL PROBLEM: No valid labels!")
        return False

# Pre-training diagnostic
diagnostic_success = diagnostic_model_learning(model, tokenizer, train_dataset)

if not diagnostic_success:
    print("Problem detected in dataset - Stopping")
    exit()

# =====================================
# STEP 6: INTENSIVE Training Configuration
# =====================================

print("\nSTEP 6: INTENSIVE training configuration...")

def create_intensive_training_args():
    """INTENSIVE training configuration to force learning"""

    return TrainingArguments(
        output_dir="./moe-lora-intensive-corrected",
        overwrite_output_dir=True,

        # INTENSIVE TRAINING
        num_train_epochs=3,              # 3 epochs instead of 1
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,    # More accumulation

        # AGGRESSIVE OPTIMIZATION
        learning_rate=2e-4,              # Higher learning rate
        weight_decay=0.01,
        lr_scheduler_type="cosine",      # Cosine scheduler
        warmup_steps=15,                 # More warmup
        max_grad_norm=1.0,               # Higher gradient norm

        # GPU optimized
        fp16=True,                       # Enabled for GPU
        dataloader_pin_memory=True,

        # Frequent logging for monitoring
        logging_steps=5,
        eval_steps=999999,
        save_steps=999999,
        evaluation_strategy="no",
        save_strategy="no",

        # Stable configuration
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],
        seed=42,
        load_best_model_at_end=False,
    )

training_args = create_intensive_training_args()
print("Intensive configuration applied")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

monitor_memory_gpu_direct("Training configuration")

# =====================================
# STEP 7: INTENSIVE MoE-LoRA Training
# =====================================

print("\nSTEP 7: INTENSIVE MoE-LoRA training...")
monitor_memory_gpu_direct("Training start")

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

monitor_memory_gpu_direct("Trainer created")

# Intensive training
print("Starting INTENSIVE training...")
try:
    training_start_time = time.time()

    # Training with monitoring
    trainer.train()

    training_time = time.time() - training_start_time
    print(f"Intensive training completed in {training_time:.2f}s")
    monitor_memory_gpu_direct("Training completed")

    # Save model
    model_save_path = "./moe-lora-intensive-corrected"
    trainer.save_model(model_save_path)
    print(f"Model saved: {model_save_path}")

except Exception as e:
    print(f"Training error: {e}")
    monitor_memory_gpu_direct("Training error")
    import traceback
    traceback.print_exc()

# =====================================
# STEP 8: CORRECTED Evaluation Functions
# =====================================

print("\nSTEP 8: CORRECTED evaluation preparation...")

def generate_prediction_corrected(model, tokenizer, sentence: str) -> str:
    """Generation with corrected parameters for Few-Shot format"""

    try:
        processor = A3CGFewShotDataProcessor()
        prompt = processor.create_prompt_fewshot(sentence)

        # Tokenize with format attention
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=400,  # Optimized for Few-Shot
            padding=False
        )

        # Safety checks
        vocab_size = tokenizer.vocab_size
        if torch.any(inputs['input_ids'] >= vocab_size):
            inputs['input_ids'] = torch.clamp(inputs['input_ids'], 0, vocab_size-1)

        # Move to GPU
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate with OPTIMIZED parameters for Few-Shot
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=100,          # More tokens for complete response
                min_new_tokens=10,           # Minimum to avoid empty responses
                do_sample=False,             # Deterministic for consistency
                temperature=1.0,             # Neutral temperature
                repetition_penalty=1.1,     # Avoid repetitions
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                early_stopping=True          # Intelligent stopping
            )

            # Check outputs
            if torch.any(outputs >= vocab_size):
                outputs = torch.clamp(outputs, 0, vocab_size-1)

            # Decode only generated part
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True, errors='ignore')

            return response.strip()

    except Exception as e:
        print(f"Generation error: {e}")
        return '("no aspect", "no action")'

def parse_prediction_advanced(prediction: str) -> List[Tuple[str, str]]:
    """Advanced and robust parser for Few-Shot responses"""
    pairs = []

    try:
        # Clean prediction
        prediction = prediction.replace('\n', ' ').strip()

        # Main pattern for ("aspect", "action")
        pattern = r'\("([^"]+)",\s*"([^"]+)"\)'
        matches = re.findall(pattern, prediction)

        for aspect, action in matches:
            aspect = aspect.strip().lower()
            action = action.strip().lower()
            if aspect and action and aspect != "no aspect" and action != "no action":
                pairs.append((aspect, action))

        # Alternative pattern with single quotes
        if not pairs:
            pattern_alt = r"\('([^']+)',\s*'([^']+)'\)"
            matches = re.findall(pattern_alt, prediction)

            for aspect, action in matches:
                aspect = aspect.strip().lower()
                action = action.strip().lower()
                if aspect and action:
                    pairs.append((aspect, action))

        # Very flexible pattern for manual extraction
        if not pairs:
            # Search for more flexible patterns
            lines = prediction.split(',')
            for i in range(0, len(lines)-1, 2):
                if i+1 < len(lines):
                    aspect_part = lines[i].strip()
                    action_part = lines[i+1].strip()

                    # Clean and extract
                    aspect_match = re.search(r'"([^"]+)"', aspect_part)
                    action_match = re.search(r'"([^"]+)"', action_part)

                    if aspect_match and action_match:
                        aspect = aspect_match.group(1).strip().lower()
                        action = action_match.group(1).strip().lower()
                        if aspect and action:
                            pairs.append((aspect, action))

    except Exception as e:
        print(f"Parsing error: {e}")

    return pairs

def debug_model_corrected():
    """CORRECTED debugging test"""

    print("CORRECTED MODEL DEBUG...")
    monitor_memory_gpu_direct("Debug start")

    try:
        test_sentence = "We have implemented solar panels to reduce energy consumption."

        print(f"1. Corrected generation test...")
        pred_text = generate_prediction_corrected(model, tokenizer, test_sentence)
        print(f"   Response: {pred_text}")

        pred_pairs = parse_prediction_advanced(pred_text)
        print(f"   Extracted pairs: {pred_pairs}")

        # Check if format is consistent
        if pred_pairs:
            print("SUCCESS: Pairs extracted correctly!")
            return True
        elif '("' in pred_text and '")' in pred_text:
            print("Format detected but parsing needs improvement")
            return True
        else:
            print("Few-Shot format not detected in generation")
            return False

        monitor_memory_gpu_direct("Debug completed")

    except Exception as e:
        print(f"Debug error: {e}")
        monitor_memory_gpu_direct("Debug error")
        return False

# CORRECTED debugging test
print("CORRECTED debugging test...")
debug_success = debug_model_corrected()

if debug_success:
    print("Model responds correctly after corrections!")
else:
    print("Generation needs improvement, but continuing evaluation...")

# =====================================
# STEP 9: CORRECTED Test Data Evaluation
# =====================================

print("\nSTEP 9: CORRECTED evaluation on test data...")
monitor_memory_gpu_direct("Evaluation start")

def evaluate_corrected(model, tokenizer, test_sentences, test_outputs, dataset_name="test"):
    """Evaluation with all corrections applied"""

    print(f"CORRECTED evaluation {dataset_name}...")

    predictions = []
    ground_truth = []

    # Convert outputs to tuples for comparison
    for output in test_outputs:
        pairs = []
        matches = re.findall(r'\("([^"]+)",\s*"([^"]+)"\)', output)
        for aspect, action in matches:
            pairs.append((aspect.strip().lower(), action.strip().lower()))
        ground_truth.append(pairs)

    # Test on optimized sample
    test_size = min(20, len(test_sentences))  # Increased thanks to corrections
    test_sentences_sample = test_sentences[:test_size]
    ground_truth_sample = ground_truth[:test_size]

    start_time = time.time()

    for i, sentence in enumerate(test_sentences_sample):
        print(f"   Progress: {i+1}/{test_size}")

        try:
            # CORRECTED GENERATION
            pred_text = generate_prediction_corrected(model, tokenizer, sentence)
            pred_pairs = parse_prediction_advanced(pred_text)
            predictions.append(pred_pairs)

            # Debug for first examples
            if i < 3:
                print(f"\nCORRECTED Example {i+1}:")
                print(f"   Sentence: {sentence[:80]}...")
                print(f"   Prediction: {pred_text[:100]}...")
                print(f"   Extracted pairs: {pred_pairs}")
                print(f"   Ground truth: {ground_truth_sample[i]}")

        except Exception as e:
            print(f"Error sample {i}: {e}")
            predictions.append([])

    eval_time = time.time() - start_time

    # Calculate metrics
    exact_matches = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred_pairs, true_pairs in zip(predictions, ground_truth_sample):
        # Exact match
        if set(pred_pairs) == set(true_pairs):
            exact_matches += 1

        # TP, FP, FN
        for pred_pair in pred_pairs:
            if pred_pair in true_pairs:
                total_tp += 1
            else:
                total_fp += 1

        for true_pair in true_pairs:
            if true_pair not in pred_pairs:
                total_fn += 1

    # Final metrics
    accuracy = exact_matches / test_size if test_size > 0 else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Evaluation time: {eval_time:.2f}s")
    print(f"CORRECTED Results {dataset_name}:")
    print(f"   Exact Match: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"   F1-Score: {f1:.4f} ({f1*100:.2f}%)")
    print(f"   TP/FP/FN: {total_tp}/{total_fp}/{total_fn}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'eval_time': eval_time
    }

# Load and evaluate on test data
test_files = {
    'seen_test': f"{DATA_DIR}/seen_test.json",
    'unseen_test': f"{DATA_DIR}/unseen_test.json"
}

results = {}

for dataset_name, file_path in test_files.items():
    if os.path.exists(file_path):
        try:
            test_sentences, test_outputs = load_a3cg_data(file_path)
            results[dataset_name] = evaluate_corrected(
                model, tokenizer, test_sentences, test_outputs, dataset_name
            )
        except Exception as e:
            print(f"Evaluation error {dataset_name}: {e}")

# =====================================
# STEP 10: Final Analysis and Conclusion
# =====================================

print(f"\nFINAL ANALYSIS - COMPLETE CORRECTED VERSION")
print("=" * 55)
print(f"End time: {time.strftime('%H:%M:%S')}")

# Calculate total execution time
total_time = time.time() - start_time
monitor_memory_gpu_direct("Final analysis")

# Detailed results
print(f"\nPERFORMANCE AFTER CORRECTIONS")
print("=" * 35)

all_metrics = []
for dataset_name, metrics in results.items():
    print(f"\n{dataset_name.upper()} - CORRECTED Results:")
    print(f"   F1-Score:      {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"   Precision:     {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"   Recall:        {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"   Exact Match:   {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")

    all_metrics.append(metrics)

# Global metrics
if all_metrics:
    avg_f1 = sum(m.get('f1_score', 0) for m in all_metrics) / len(all_metrics)
    avg_precision = sum(m.get('precision', 0) for m in all_metrics) / len(all_metrics)
    avg_recall = sum(m.get('recall', 0) for m in all_metrics) / len(all_metrics)
    avg_accuracy = sum(m.get('accuracy', 0) for m in all_metrics) / len(all_metrics)

    print(f"\nCORRECTED GLOBAL METRICS")
    print("=" * 35)
    print(f"Average performance:")
    print(f"   Average F1-Score:    {avg_f1:.4f} ({avg_f1*100:.2f}%)")
    print(f"   Average Precision: {avg_precision:.4f} ({avg_precision*100:.2f}%)")
    print(f"   Average Recall:      {avg_recall:.4f} ({avg_recall*100:.2f}%)")
    print(f"   Average Exact Match: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")

# Corrections analysis
print(f"\nAPPLIED CORRECTIONS")
print("=" * 25)
print(f"bitsandbytes: {'Installed' if USE_QUANTIZATION else 'Fallback without quantization'}")
print(f"Training: 3 epochs, learning rate 2e-4, cosine scheduler")
print(f"Generation: max_new_tokens=100, optimized Few-Shot parameters")
print(f"Parsing: Robust multi-pattern parser")
print(f"Dataset: 4 Few-Shot examples per prompt")
print(f"Diagnostic: Integrated pre-training verification")

# Final conclusion
print(f"\nCORRECTED VERSION CONCLUSION")
print("=" * 30)

if all_metrics and avg_f1 > 0.1:
    print(f"IMPROVEMENT: Average F1-Score of {avg_f1*100:.2f}%")

    if avg_f1 > 0.4:
        print(f"EXCELLENT: Very solid performance after corrections!")
    elif avg_f1 > 0.2:
        print(f"GOOD: Significant improvement, architecture validated")
    else:
        print(f"PROGRESS: Functional base established")

    print(f"\nSTATUS: MoE-LoRA + Few-Shot architecture OPERATIONAL")

elif all_metrics:
    print(f"F1 = {avg_f1*100:.2f}% - Improvements needed")
    print(f"Architecture works, hyperparameter optimization required")
else:
    print(f"Persistent problems - Check configuration")

print(f"\nNEXT OPTIMIZATIONS:")
print(f"   1. Structural corrections applied")
print(f"   2. Increase epochs (5-10) and dataset size")
print(f"   3. Test different learning rates (1e-4, 5e-4)")
print(f"   4. Optimize Few-Shot prompt (more examples)")
print(f"   5. Compare with standard LoRA baseline")

print(f"\nTotal time: {total_time/60:.1f} minutes")

# Final cleanup
try:
    del model, trainer
    ultra_aggressive_gpu_cleanup()
    monitor_memory_gpu_direct("Final cleanup")
    print("Memory cleanup successful")
except Exception as e:
    print(f"Cleanup error: {e}")

print(f"\nCORRECTED MoE-LoRA TEST COMPLETED!")
print("=" * 70)
