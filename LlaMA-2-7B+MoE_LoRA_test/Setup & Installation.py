# =============================================================================
# SETUP & INSTALLATION
# =============================================================================

# Environment check
import sys
import torch
import pkg_resources

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Dependency installation
!pip install -q transformers==4.35.0
!pip install -q datasets==2.14.0
!pip install -q accelerate==0.24.0
!pip install -q peft==0.6.0
!pip install -q wandb
!pip install -q scikit-learn
!pip install -q matplotlib seaborn
!pip install -q spacy
!python -m spacy download en_core_web_sm

# Hugging Face authentication (for LLaMA)
from huggingface_hub import notebook_login
notebook_login()  # Enter your HF token here

# Warning configuration
import warnings
warnings.filterwarnings('ignore')

print("Setup complete!")
