# =====================================
# MODIFIED SCRIPT 2 - LORA TRAINING WITH FEW-SHOT EXAMPLES
# Enhanced version for exact predictions like Claude 3.5
# =====================================

print("MODIFIED SCRIPT 2: A3CG LORA TRAINING WITH FEW-SHOT")
print("=" * 70)

# STEP 1: Critical package verification
print("STEP 1: Critical package verification...")

# Check NumPy first
try:
    import numpy as np
    numpy_version = np.__version__
    print(f"  numpy: {numpy_version}")
    if numpy_version.startswith('2.'):
        print("  WARNING: NumPy 2.x detected - may cause issues")
        print("  Installing NumPy 1.x...")
        import subprocess
        import sys
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'numpy<2.0'], check=True)
        print("  NumPy 1.x installed - RESTART THE RUNTIME!")
        exit("Restart runtime after NumPy correction")
except Exception as e:
    print(f"  numpy error: {e}")
    exit("NumPy required - install with: !pip install 'numpy<2.0'")

try:
    import torch
    print(f"  torch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
except Exception as e:
    print(f"  torch error: {e}")
    exit()

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, Trainer, DataCollatorForLanguageModeling, TrainerCallback
    print(f"  transformers: Import OK")
except Exception as e:
    print(f"  transformers error: {e}")
    exit()

try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    print(f"  peft: Import OK")
except Exception as e:
    print(f"  peft error: {e}")
    exit()

try:
    import bitsandbytes
    print(f"  bitsandbytes: Import OK")
except Exception as e:
    print(f"  bitsandbytes error: {e}")

# Additional imports
import json
import time
import os
import re
import random
from datasets import Dataset
from typing import Dict, List, Tuple

print("All critical imports successful!")

# STEP 2: Google Drive mounting
print("\nSTEP 2: Google Drive mounting...")
from google.colab import drive
try:
    drive.mount('/content/drive')
    print("Drive mounted")
except Exception as e:
    print(f"Drive error: {e}")

# STEP 3: Data verification
print("\nSTEP 3: Data verification...")
dataset_base = "/content/A3CG_DATASET/folds/fold_1"

required_files = ["seen_train.json", "seen_val.json", "seen_test.json", "unseen_test.json"]
files_ok = True

for filename in required_files:
    filepath = os.path.join(dataset_base, filename)
    if os.path.exists(filepath):
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  {filename}: {size_kb:.1f} KB")
    else:
        print(f"  {filename}: Missing")
        files_ok = False

if not files_ok:
    print("WARNING: Data files missing!")
    print("Run the data import script first")
    exit()

# STEP 4: Simplified monitoring callback
print("\nSTEP 4: Monitoring configuration...")

class MemoryMonitorCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()
        self.last_check = time.time()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        current_time = time.time()
        step = state.global_step

        if step % 10 == 0:
            elapsed = current_time - self.start_time
            step_time = current_time - self.last_check

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"Step {step}: GPU {allocated:.1f}/{reserved:.1f} GB, Time: {elapsed/60:.1f}min")

                # Auto cleanup
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                if reserved / gpu_total > 0.9:
                    torch.cuda.empty_cache()
                    print("Automatic cleanup")
            else:
                print(f"Step {step}: Time: {elapsed/60:.1f}min")

            self.last_check = current_time

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = state.epoch
        elapsed = time.time() - self.start_time
        print(f"Epoch {epoch} completed in {elapsed/60:.1f} minutes")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print("Callback configured")

# STEP 5: Model configuration
print("\nSTEP 5: Model configuration...")

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model_name = "meta-llama/Llama-2-7b-hf"

print(f"Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="left"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading LLaMA-2 7B model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# LoRA preparation
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Model configured with LoRA")

# STEP 6: NEW - Few-Shot data processor
print("\nSTEP 6: Few-Shot processor configuration...")

class A3CGFewShotDataProcessor:
    def __init__(self):
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

        # FEW-SHOT EXAMPLES BANK (extracted from actual training data)
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
                "text": "We are committed to enhancing our environmental management systems.",
                "output": '("environmental management systems", "planning")'
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
                "text": "All staff complete an e-learning programme from an accredited training provider.",
                "output": '("e-learning programme", "implemented"), ("training provider", "implemented"), ("staff", "implemented")'
            },
            {
                "text": "We strive to minimize carbon footprints in our operations.",
                "output": '("carbon footprints", "planning")'
            },
            {
                "text": "Recycled items are processed to create new products.",
                "output": '("recycled items", "indeterminate")'
            },
            {
                "text": "Our properties are designed to achieve energy reduction targets.",
                "output": '("energy reduction", "implemented")'
            },
            {
                "text": "Safeguarding stakeholder data is of paramount importance.",
                "output": '("stakeholder data", "planning")'
            }
        ]

    def get_few_shot_examples(self, n_examples: int = 3) -> str:
        """Randomly selects n examples for few-shot"""
        selected = random.sample(self.few_shot_examples, min(n_examples, len(self.few_shot_examples)))

        examples_text = ""
        for i, example in enumerate(selected, 1):
            examples_text += f"\nExample {i}:\n"
            examples_text += f"Text: {example['text']}\n"
            examples_text += f"Output: {example['output']}\n"

        return examples_text

    def create_prompt(self, text: str, aspects_dict: Dict = None) -> str:
        # Generate few-shot examples for EACH training sample
        few_shot_text = self.get_few_shot_examples(n_examples=3)

        if aspects_dict:
            # TRAINING: Include few-shot + expected response
            output_pairs = []
            for aspect, actions in aspects_dict.items():
                for action in actions:
                    output_pairs.append(f'("{aspect}", "{action}")')

            expected_output = ', '.join(output_pairs) if output_pairs else '("no aspect", "no action")'

            return f"""<s>[INST] <<SYS>>
{self.system_prompt}
<</SYS>>

{few_shot_text}

Now extract from this text:
Text: {text}

Extract the aspect-action pairs: [/INST] {expected_output}</s>"""

        else:
            # INFERENCE: Same format with few-shot examples
            return f"""<s>[INST] <<SYS>>
{self.system_prompt}
<</SYS>>

{few_shot_text}

Now extract from this text:
Text: {text}

Extract the aspect-action pairs: [/INST]"""

    def load_data(self, file_path: str) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def prepare_dataset(self, data: List[Dict]) -> Dataset:
        prompts = []

        print(f"  Preparing {len(data)} samples with few-shot...")

        for i, item in enumerate(data):
            if i % 100 == 0:
                print(f"    Progress: {i}/{len(data)} ({i/len(data)*100:.1f}%)")

            prompt = self.create_prompt(item['text'], item.get('aspects', {}))
            prompts.append(prompt)

        print(f"  {len(prompts)} few-shot prompts generated")
        return Dataset.from_dict({"text": prompts})

print("Few-Shot processor configured")

# STEP 7: Data preparation
print("\nSTEP 7: Loading and preparing data with Few-Shot...")

processor = A3CGFewShotDataProcessor()

print("Loading JSON files...")
train_data = processor.load_data(f"{dataset_base}/seen_train.json")
val_data = processor.load_data(f"{dataset_base}/seen_val.json")

print(f"Train: {len(train_data)} samples")
print(f"Validation: {len(val_data)} samples")

# Prepare datasets with few-shot
print("Generating few-shot prompts for training...")
train_dataset = processor.prepare_dataset(train_data)
val_dataset = processor.prepare_dataset(val_data)

# Display example few-shot prompt
print("\nEXAMPLE GENERATED FEW-SHOT PROMPT:")
print("=" * 50)
sample_prompt = train_dataset[0]['text']
print(sample_prompt[:800] + "..." if len(sample_prompt) > 800 else sample_prompt)
print("=" * 50)

# Tokenization function
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=1024,  # Increased for few-shot examples
        return_tensors=None
    )
    return tokenized

print("Tokenizing few-shot data...")
try:
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Train few-shot tokenization"
    )
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Validation few-shot tokenization"
    )
    print("Few-shot tokenization successful")
except Exception as e:
    print(f"Tokenization error: {e}")
    print("Using alternative method...")

    # Alternative method for few-shot
    def manual_tokenize_fewshot(dataset, name):
        texts = [item['text'] for item in dataset]
        tokenized_texts = []

        for i, text in enumerate(texts):
            if i % 50 == 0:
                print(f"  Tokenizing {name}: {i}/{len(texts)}")

            tokens = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=1024,  # Longer for few-shot
                return_tensors=None
            )
            tokenized_texts.append(tokens)

        input_ids = [t['input_ids'] for t in tokenized_texts]
        attention_mask = [t['attention_mask'] for t in tokenized_texts]

        return Dataset.from_dict({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids
        })

    train_dataset = manual_tokenize_fewshot(train_data, "train")
    val_dataset = manual_tokenize_fewshot(val_data, "validation")
    print("Alternative few-shot tokenization successful")

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("Few-shot data prepared")

# STEP 8: Few-Shot training configuration
print("\nSTEP 8: Few-shot training configuration...")

# NEW DIRECTORIES to avoid confusion with old model
training_args = TrainingArguments(
    output_dir="./lora-a3cg-fewshot",  # NEW NAME
    num_train_epochs=2,
    per_device_train_batch_size=4,  # Reduced due to longer prompts
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Increased to compensate
    optim="paged_adamw_32bit",
    save_steps=200,
    logging_steps=25,
    learning_rate=3e-5,  # Slightly reduced for few-shot
    weight_decay=0.01,
    fp16=False,
    bf16=True,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    group_by_length=True,
    lr_scheduler_type="cosine",
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
)

# Trainer with callback
memory_callback = MemoryMonitorCallback()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.add_callback(memory_callback)

print("Few-shot trainer configured")

# STEP 9: Final verification
print("\nSTEP 9: Final few-shot verification...")

if torch.cuda.is_available():
    gpu_allocated = torch.cuda.memory_allocated() / 1e9
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {gpu_allocated:.1f}/{gpu_total:.1f} GB")
else:
    print("WARNING: GPU not available - training on CPU")

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Max prompt length: 1024 tokens (including few-shot)")

# STEP 10: Few-Shot Training
print("\nSTEP 10: STARTING FEW-SHOT TRAINING...")
print("=" * 70)
print(f"Start time: {time.strftime('%H:%M:%S')}")
print(f"Config: Batch {training_args.per_device_train_batch_size}, Epochs {training_args.num_train_epochs}")
print(f"LoRA: Rank {lora_config.r}, Alpha {lora_config.lora_alpha}")
print(f"Few-Shot: 3 examples per training prompt")
print("=" * 70)

start_time = time.time()

try:
    trainer.train()

    training_time = time.time() - start_time
    print(f"\nFEW-SHOT TRAINING COMPLETED!")
    print(f"Total time: {training_time/3600:.1f}h ({training_time/60:.1f}min)")

except Exception as e:
    print(f"\nERROR DURING TRAINING: {e}")
    print("Automatic cleanup...")
    torch.cuda.empty_cache()
    raise e

# STEP 11: Few-Shot saving
print("\nSTEP 11: Saving few-shot model...")
try:
    trainer.model.save_pretrained("./lora-a3cg-fewshot-final")  # NEW NAME
    tokenizer.save_pretrained("./lora-a3cg-fewshot-final")
    print("Few-shot model saved in: ./lora-a3cg-fewshot-final")
except Exception as e:
    print(f"Save error: {e}")

# AUTOMATIC BACKUP TO GOOGLE DRIVE - FEW-SHOT VERSION
print("\nAUTOMATIC FEW-SHOT BACKUP TO GOOGLE DRIVE...")
print("=" * 70)

import os
import shutil
import zipfile
from datetime import datetime

# Few-shot backup configuration
model_path = "./lora-a3cg-fewshot-final"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

try:
    # 1. Check model exists
    if not os.path.exists(model_path):
        print("Few-shot model not found for backup!")
        raise Exception("Model not found")

    # 2. Create FEWSHOT folder in Drive
    drive_folder = f"/content/drive/MyDrive/A3CG_Models_FewShot"  # NEW FOLDER
    os.makedirs(drive_folder, exist_ok=True)
    print(f"Few-shot folder created: {drive_folder}")

    # 3. Direct copy of few-shot folder
    drive_model_path = f"{drive_folder}/lora-a3cg-fewshot_{timestamp}"  # NEW NAME

    print("Copying few-shot model to Drive...")
    shutil.copytree(model_path, drive_model_path)

    # Verify copy
    files_copied = os.listdir(drive_model_path)
    total_size = 0

    print(f"Few-shot model copied to: {drive_model_path}")
    print("Files copied:")

    for file in files_copied:
        file_path = os.path.join(drive_model_path, file)
        size_mb = os.path.getsize(file_path) / (1024*1024)
        total_size += size_mb
        print(f"   {file} ({size_mb:.1f} MB)")

    print(f"Total size: {total_size:.1f} MB")

    # 4. ZIP backup archive for few-shot
    print("\nCreating few-shot ZIP archive...")

    zip_filename = f"lora-a3cg-fewshot-backup_{timestamp}.zip"
    zip_path = f"{drive_folder}/{zip_filename}"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, model_path)
                zipf.write(file_path, arcname)

    zip_size = os.path.getsize(zip_path) / (1024*1024)
    print(f"Few-shot archive created: {zip_filename} ({zip_size:.1f} MB)")

    # 5. Few-shot metadata
    metadata_file = f"{drive_folder}/fewshot_model_info_{timestamp}.txt"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(f"A3CG LORA FEW-SHOT MODEL - AUTOMATIC BACKUP\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"Creation date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Base model: LLaMA-2 7B\n")
        f.write(f"Technique: LoRA + Few-Shot Examples\n")
        f.write(f"Task: A3CG aspect-action extraction\n")
        f.write(f"LoRA config: Rank 32, Alpha 64\n")
        f.write(f"Few-Shot: 3 examples per training prompt\n")
        f.write(f"Epochs: 2\n")
        f.write(f"Max Length: 1024 tokens\n")
        f.write(f"Model size: {total_size:.1f} MB\n")
        f.write(f"Drive location: {drive_model_path}\n")
        f.write(f"Backup archive: {zip_filename}\n")
        f.write(f"\nIMPROVEMENTS vs PREVIOUS VERSION:\n")
        f.write(f"• Few-shot examples integrated in training\n")
        f.write(f"• Literal predictions (non-creative)\n")
        f.write(f"• Consistent format with Claude 3.5\n")
        f.write(f"• Better exact match expected\n")
        f.write(f"\nUsage:\n")
        f.write(f"1. Same loading process as old model\n")
        f.write(f"2. Use new few-shot evaluation script\n")
        f.write(f"3. Compare with previous performances\n")

    print(f"Few-shot metadata created: fewshot_model_info_{timestamp}.txt")

    # 6. Few-shot restore script
    restore_script = f"{drive_folder}/restore_fewshot_model_{timestamp}.py"
    with open(restore_script, 'w', encoding='utf-8') as f:
        f.write(f'''# Few-shot model restore script
# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

import shutil
import os

def restore_fewshot_model():
    """Restores LoRA few-shot model from Google Drive"""

    # Few-shot paths
    drive_model = "{drive_model_path}"
    local_model = "/content/lora-a3cg-fewshot-final"

    print("Restoring LoRA few-shot model...")

    # Verifications
    if not os.path.exists("/content/drive"):
        print("ERROR: Google Drive not mounted!")
        return False

    if not os.path.exists(drive_model):
        print(f"ERROR: Few-shot model not found: {{drive_model}}")
        return False

    # Restoration
    if os.path.exists(local_model):
        shutil.rmtree(local_model)

    shutil.copytree(drive_model, local_model)

    print(f"Few-shot model restored: {{local_model}}")

    # Verification
    files = os.listdir(local_model)
    print("Restored files:")
    for file in files:
        print(f"   {{file}}")

    return True

if __name__ == "__main__":
    restore_fewshot_model()
''')

    print(f"Few-shot restore script created")

    # 7. Final few-shot summary
    print("\n" + "="*70)
    print("FEW-SHOT BACKUP SUCCESSFUL!")
    print("="*70)
    print(f"Folder: A3CG_Models_FewShot/lora-a3cg-fewshot_{timestamp}")
    print(f"Archive: lora-a3cg-fewshot-backup_{timestamp}.zip")
    print(f"Metadata: fewshot_model_info_{timestamp}.txt")
    print(f"Script: restore_fewshot_model_{timestamp}.py")
    print()
    print("NEW FEW-SHOT MODEL READY!")
    print("Expected improvement: 0% -> 15-25% F1 exact match")
    print("Comparable to Claude 3.5 performance (42% F1)")
    print("="*70)

except Exception as e:
    print(f"ERROR during few-shot backup: {e}")

    # Emergency backup
    try:
        emergency_backup = "/content/drive/MyDrive/lora-fewshot-emergency"
        shutil.copytree(model_path, emergency_backup)
        print(f"Emergency backup: {emergency_backup}")
    except:
        print("Emergency backup failed!")

# STEP 12: Final cleanup
print("\nSTEP 12: Final cleanup...")
torch.cuda.empty_cache()

print("\nFEW-SHOT SCRIPT 2 COMPLETED!")
print("=" * 70)
print(f"End time: {time.strftime('%H:%M:%S')}")
print(f"Local model: ./lora-a3cg-fewshot-final")
print(f"Drive model: A3CG_Models_FewShot/lora-a3cg-fewshot_{timestamp}")
print(f"FEW-SHOT MODEL READY FOR EVALUATION!")
print(f"Expected performance: 15-25% F1 exact match")
print("=" * 70)

# NEXT STEPS INSTRUCTIONS
print("\nNEXT STEPS:")
print("=" * 40)
print("1. Run evaluation script on the new model")
print("2. Compare with previous results (0% -> ?%)")
print("3. Analyze exact match improvements")
print("4. Document results for your paper")
print("5. If satisfactory, test on other folds")