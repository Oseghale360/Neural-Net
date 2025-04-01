import os
import json
import random
import torch
import itertools
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity

# === CONFIG === #
ROOT_DIR = "normalized_code_data"
MODEL_PATH = "./starcoderbase_IR2"
MAX_LENGTH = 2048
POSITIVE_TARGET = 12500
NEGATIVE_TARGET = 18750  # Balanced ratio: 1.5 negatives for every positive
SOFT_NEG_SHARE = 0.33
MEDIUM_NEG_SHARE = 0.47
HARD_NEG_SHARE = 0.20
#OUTPUT_PATH = "final_similarity_dataset_balanced2.json"
OUTPUT_PATH = "final_similarity_dataset_balanced3.json"

# === LOAD MODEL === #
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH).eval().cuda()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === GROUP FILES BY LEAF DIRECTORY === #
def group_by_leaf_dir(root):
    grouped = {}
    for dirpath, _, files in os.walk(root):
        ll_files = [f for f in files if f.endswith(".ll")]
        if ll_files:
            key = os.path.relpath(dirpath, root)
            grouped[key] = [os.path.join(dirpath, f) for f in ll_files]
    return grouped

# === LOAD FILE CONTENT === #
def load_code(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# === GET EMBEDDINGS === #
def get_embedding(code):
    inputs = tokenizer(code, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH).to('cuda')
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.last_hidden_state
    mask = inputs['attention_mask'].unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1)
    return (summed / counts).squeeze(0).cpu()

# === BUILD POSITIVE PAIRS === #
def generate_positive_pairs(grouped):
    pairs = []
    for files in grouped.values():
        if len(files) >= 2:
            for f1, f2 in itertools.combinations(files, 2):
                pairs.append((f1, f2))
    return pairs

# === COMPUTE ALL EMBEDDINGS FIRST === #
def compute_all_embeddings(file_list):
    emb = {}
    for f in tqdm(file_list, desc="Embedding Files"):
        try:
            code = load_code(f)
            emb[f] = get_embedding(code)
        except Exception as e:
            print(f"[Error] {f}: {e}")
    print(f"Successfully generated embeddings for {len(emb)} out of {len(file_list)} files")
    return emb

# === SCORE PAIRS === #
def score_pairs(pairs, embeddings):
    scored = []
    for f1, f2 in tqdm(pairs, desc="Scoring Similar Pairs"):
        if f1 in embeddings and f2 in embeddings:
            sim = cosine_similarity(embeddings[f1].unsqueeze(0), embeddings[f2].unsqueeze(0)).item()
            scored.append((f1, f2, sim))
    return scored

# === GENERATE NEGATIVES WITH RANGES === #
def generate_negative_bins(grouped, embeddings):
    all_files = list(embeddings.keys())
    group_map = {f: g for g, fs in grouped.items() for f in fs}
    bins = {"soft": [], "medium": [], "hard": []}

    for anchor in tqdm(all_files, desc="Mining All Negatives"):
        anchor_emb = embeddings[anchor]
        anchor_group = group_map[anchor]
        for f in all_files:
            if group_map[f] != anchor_group:
                sim = cosine_similarity(anchor_emb.unsqueeze(0), embeddings[f].unsqueeze(0)).item()
                if sim <= 0.5:
                    bins["soft"].append((anchor, f, sim))
                elif 0.5 < sim <= 0.8:
                    bins["medium"].append((anchor, f, sim))
                else:
                    bins["hard"].append((anchor, f, sim))
    return bins

# === MAIN === #
grouped_files = group_by_leaf_dir(ROOT_DIR)
all_files = [f for group in grouped_files.values() for f in group]
embeddings = compute_all_embeddings(all_files)

# 1. Positive pairs
raw_pos_pairs = generate_positive_pairs(grouped_files)
scored_pos = score_pairs(raw_pos_pairs, embeddings)
easy_pos = [(f1, f2) for f1, f2, s in scored_pos if s >= 0.9]
hard_pos = [(f1, f2) for f1, f2, s in scored_pos if s < 0.9]
selected_pos = easy_pos[:7500] + hard_pos[:POSITIVE_TARGET - 7500]
print(f"Selected {len(selected_pos)} positive pairs")

# 2. Negatives by similarity bins
neg_bins = generate_negative_bins(grouped_files, embeddings)
num_soft = int(NEGATIVE_TARGET * SOFT_NEG_SHARE)
num_medium = int(NEGATIVE_TARGET * MEDIUM_NEG_SHARE)
num_hard = NEGATIVE_TARGET - num_soft - num_medium

soft_neg = [(f1, f2) for f1, f2, _ in sorted(neg_bins["soft"], key=lambda x: -x[2])[:num_soft]]
medium_neg = [(f1, f2) for f1, f2, _ in sorted(neg_bins["medium"], key=lambda x: -x[2])[:num_medium]]
hard_neg = [(f1, f2) for f1, f2, _ in sorted(neg_bins["hard"], key=lambda x: -x[2])[:num_hard]]
print(f"Selected {len(soft_neg)} soft, {len(medium_neg)} medium, {len(hard_neg)} hard negatives")

# === Build final dataset === #
final_dataset = []
for f1, f2 in selected_pos:
    final_dataset.append({"code1": load_code(f1), "code2": load_code(f2), "label": 1})
for f1, f2 in soft_neg + medium_neg + hard_neg:
    final_dataset.append({"code1": load_code(f1), "code2": load_code(f2), "label": 0})

random.shuffle(final_dataset)

with open(OUTPUT_PATH, "w") as f:
    json.dump(final_dataset, f, indent=2)

print(f"Final dataset of {len(final_dataset)} pairs saved to {OUTPUT_PATH}")
