# Get your token from https://huggingface.co/settings/tokens
from huggingface_hub import login

# Option 1: Interactive login (recommended)
# login()

# Option 2: Direct token login (less secure)
login("your_huggingface_token")

# Verification
from huggingface_hub import whoami
try:
    user_info = whoami()
    print(f"Connected as: {user_info['name']}")
except Exception as e:
    print(f"Authentication error: {e}")