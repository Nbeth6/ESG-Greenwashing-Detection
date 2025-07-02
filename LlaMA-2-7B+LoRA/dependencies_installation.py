# =====================================
# CLEAN INSTALLATION SCRIPT WITH FIXED NUMPY
# =====================================

print("SCRIPT 1: CLEANUP AND INSTALLATION")
print("=" * 60)

# STEP 1: Full cleanup (including NumPy)
print("STEP 1: Full package cleanup...")
!pip uninstall -y -q transformers trl accelerate peft bitsandbytes datasets torch torchvision torchaudio numpy

print("Cleanup completed.")

# STEP 2: Installation in correct order with NumPy 1.x
print("\nSTEP 2: Installing with compatible NumPy...")

print("  - Installing NumPy 1.x...")
!pip install -q "numpy<2.0"

print("  - Installing PyTorch...")
!pip install -q torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

print("  - Installing Transformers...")
!pip install -q transformers==4.35.2

print("  - Installing PEFT...")
!pip install -q peft==0.6.2

print("  - Installing BitsAndBytes...")
!pip install -q bitsandbytes==0.41.2

print("  - Installing Accelerate...")
!pip install -q accelerate==0.24.1

print("  - Installing Datasets...")
!pip install -q datasets==2.14.6

print("  - Installing additional packages...")
!pip install -q scikit-learn

print("Installation completed.")

# STEP 3: Check critical versions
print("\nSTEP 3: Checking installed package versions...")

import subprocess
import sys

def check_package_version(package_name):
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'show', package_name],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            version_line = [line for line in lines if line.startswith('Version:')]
            if version_line:
                version = version_line[0].split(': ')[1]
                print(f"  OK {package_name}: {version}")
                return version
        print(f"  NOT FOUND {package_name}")
        return None
    except Exception as e:
        print(f"  ERROR {package_name}: {e}")
        return None

# Verify versions
critical_packages = {
    'numpy': 'numpy<2.0',
    'torch': '2.0.1',
    'transformers': '4.35.2',
    'peft': '0.6.2',
    'datasets': '2.14.6'
}

all_good = True
for package, expected in critical_packages.items():
    version = check_package_version(package)
    if package == 'numpy' and version:
        if version.startswith('2.'):
            print(f"  WARNING: NumPy 2.x detected ({version}) - may cause issues")
            all_good = False
        else:
            print(f"  NumPy 1.x OK ({version})")

# STEP 4: Basic import test
print("\nSTEP 4: Basic import test...")

test_script = '''
try:
    import numpy as np
    print(f"  OK numpy: {np.__version__}")
except Exception as e:
    print(f"  ERROR numpy: {e}")

try:
    import torch
    print(f"  OK torch: {torch.__version__}")
except Exception as e:
    print(f"  ERROR torch: {e}")
'''

exec(test_script)

# STEP 5: Final instructions
print("\n" + "=" * 60)
print("INSTALLATION SUMMARY:")
print("=" * 60)

if all_good:
    print("Installation successful with compatible versions.")
    print("\nNEXT STEPS:")
    print("1. RESTART THE RUNTIME (MANDATORY)")
    print("   Runtime > Restart session")
    print("2. Wait 30 seconds after restart")
    print("3. Run the corrected SCRIPT 2")

    print("\nWHY THIS RESTART IS CRITICAL:")
    print("   - Prevents NumPy 1.x/2.x conflicts in memory")
    print("   - Clears Python cache")
    print("   - Ensures correct versions are loaded")

else:
    print("Issues detected during installation.")
    print("Rerun this script or check errors above.")

print("\nSCRIPT 2 READY TO RUN AFTER RESTART.")
print("=" * 60)
print("END OF SCRIPT 1")
print("=" * 60)