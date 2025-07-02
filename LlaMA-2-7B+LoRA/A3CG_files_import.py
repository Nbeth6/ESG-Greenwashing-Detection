# =====================================
# SIMPLE IMPORT OF A3CG FILES
# =====================================

from google.colab import drive
import os
import shutil
import json

print("IMPORTING A3CG FILES")
print("=" * 40)

# Mount Google Drive
print("Mounting Google Drive...")
try:
    drive.mount('/content/drive')
    print("Drive mounted successfully")
except Exception as e:
    print(f"Error: {e}")
    exit()

# Search for the fold_1 folder
print("\nSearching for the 'fold_1' folder...")

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
        print(f"Found: {path}")
        break

# Deeper search if not found
if not drive_fold_path:
    print("Performing deeper search...")
    mydrive_path = "/content/drive/MyDrive"

    for root, dirs, files in os.walk(mydrive_path):
        if "seen_train.json" in files and "seen_val.json" in files:
            drive_fold_path = root
            print(f"Files found in: {drive_fold_path}")
            break

        # Limit search depth
        if root.count('/') > mydrive_path.count('/') + 2:
            continue

if not drive_fold_path:
    print("Folder 'fold_1' not found.")
    print("\nPlease ensure the 'fold_1' folder is uploaded to your Google Drive.")
    exit()

# List contents of the folder
print(f"\nContents of: {drive_fold_path}")
files_in_folder = os.listdir(drive_fold_path)
required_files = ["seen_train.json", "seen_val.json", "seen_test.json", "unseen_test.json"]

for file in files_in_folder:
    if file.endswith('.json'):
        file_path = os.path.join(drive_fold_path, file)
        size_kb = os.path.getsize(file_path) / 1024
        status = "[OK]" if file in required_files else "[INFO]"
        print(f"  {status} {file} ({size_kb:.1f} KB)")

# Create local structure
local_path = "/content/A3CG_DATASET/folds/fold_1"
os.makedirs(local_path, exist_ok=True)
print(f"\nDirectory created: {local_path}")

# Copy files
print("\nCopying files...")
copied_count = 0

for filename in required_files:
    source = os.path.join(drive_fold_path, filename)
    dest = os.path.join(local_path, filename)

    if os.path.exists(source):
        try:
            shutil.copy2(source, dest)
            size_kb = os.path.getsize(dest) / 1024

            # Check for valid JSON
            with open(dest, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"  [OK] {filename}: {len(data)} samples ({size_kb:.1f} KB)")
            copied_count += 1

        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")
    else:
        print(f"  [MISSING] {filename} not found")

# Final result
print(f"\nRESULT:")
print(f"  Files copied: {copied_count}/4")

if copied_count >= 3:
    print(f"SUCCESS! Files are ready.")
    print(f"Path: {local_path}")

    # Show final structure
    print(f"\nFinal Directory Structure:")
    for root, dirs, files in os.walk("/content/A3CG_DATASET"):
        level = root.replace("/content/A3CG_DATASET", '').count(os.sep)
        indent = '  ' * level
        folder_name = os.path.basename(root) or "A3CG_DATASET"
        print(f'{indent}- {folder_name}/')

        sub_indent = '  ' * (level + 1)
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                size_kb = os.path.getsize(file_path) / 1024
                print(f'{sub_indent}- {file} ({size_kb:.1f} KB)')

    print(f"\nUse this path in your code:")
    print(f"/content/A3CG_DATASET/folds/fold_1/")

else:
    print(f"FAILURE: Only {copied_count}/4 files were copied.")
    print("Please verify that all required JSON files are in your Google Drive folder.")