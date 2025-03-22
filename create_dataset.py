import os
import random
import json
from collections import defaultdict

# Path to the root dataset directory
root_dir = "normalized2_code_data"
#root_dir = "tester"
#root_dir = "normalized_atcoder_data"

# Define the number of similar and dissimilar pairs you want
# num_similar_pairs = 12000
# num_dissimilar_pairs = 12000
num_similar_pairs = 25000
num_dissimilar_pairs = 25000
def load_llvm_ir(file_path):
    """Load LLVM IR code from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def find_llvm_ir_files(directory):
    """
    Recursively finds all .ll files in subdirectories and groups them by their deepest subfolder.
    Returns:
    - `subfolder_groups`: {subfolder_path: list_of_files}
    - `language_distribution`: {language: num_files} (to ensure balance)
    """
    subfolder_groups = defaultdict(list)
    language_distribution = defaultdict(int)

    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".ll"):
                file_path = os.path.join(root, f)

                # Categorize by subdirectory
                subfolder_groups[root].append(file_path)

                # Identify language and count occurrences
                if "_c.ll" in f:
                    language_distribution["C"] += 1
                elif "_cpp.ll" in f:
                    language_distribution["C++"] += 1
                elif ".ll" in f:
                    language_distribution["Java"] += 1

    return subfolder_groups, language_distribution

# Step 1: Get all .ll files grouped by their subdirectory and track language distribution
subfolder_groups, language_distribution = find_llvm_ir_files(root_dir)

# Step 2: Create function pairs
def create_pairs():
    pairs = []
    subfolders = list(subfolder_groups.keys())

    # Step 2a: Generate similar pairs (only within the same subdirectory)
    similar_pairs = []
    for subfolder, files in subfolder_groups.items():
        if len(files) > 1:
            for i in range(len(files)):
                for j in range(i + 1, len(files)):  # Pair each file with others in the same subfolder
                    similar_pairs.append((load_llvm_ir(files[i]), load_llvm_ir(files[j]), 1))

    # Ensure the required number of similar pairs (randomly sample if more exist)
    similar_pairs = random.sample(similar_pairs, min(num_similar_pairs, len(similar_pairs)))
    pairs.extend(similar_pairs)

    # Step 2b: Generate dissimilar pairs (only from different subdirectories)
    dissimilar_pairs = []
    
    while len(dissimilar_pairs) < num_dissimilar_pairs:
        sub1, sub2 = random.sample(subfolders, 2)  # Pick two different subdirectories
        file1 = random.choice(subfolder_groups[sub1])
        file2 = random.choice(subfolder_groups[sub2])
        dissimilar_pairs.append((load_llvm_ir(file1), load_llvm_ir(file2), 0))

    # Ensure the required number of dissimilar pairs (randomly sample if more exist)
    dissimilar_pairs = random.sample(dissimilar_pairs, min(num_dissimilar_pairs, len(dissimilar_pairs)))
    pairs.extend(dissimilar_pairs)

    return pairs

dataset_pairs = create_pairs()

# Step 3: Save the dataset as JSON
#dataset_path = "normal_training.json"
#dataset_path = "tester1.json"
#dataset_path = "normalize_atcoder_test.json"
dataset_path = "normalize1_atcoder_test.json"

with open(dataset_path, "w") as f:
    json.dump(dataset_pairs, f, indent=4)

# Print statistics
print(f"Dataset saved to {dataset_path}")
print(f"Total pairs created: {len(dataset_pairs)} (Similar: {num_similar_pairs}, Dissimilar: {num_dissimilar_pairs})")
print("Language Distribution:")
for lang, count in language_distribution.items():
    print(f"- {lang}: {count} files")
