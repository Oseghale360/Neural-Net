# import os
# import glob
# import torch
# from datasets import Dataset
# from transformers import (
#     AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
#     DataCollatorForLanguageModeling
# )
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# # Ensure only the first GPU is used (Modify if multi-GPU is required)
# #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# # Detect available device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Specify folders containing LLVM-IR (.ll) files
# folders = [ "postgresql", "tensorflow"]

# # Collect all .ll files from directories
# ll_files = []
# for folder in folders:
#     ll_files.extend(glob.glob(f"{folder}/*.ll"))

# print(f"Found {len(ll_files)} .ll files.")

# # Function to load .ll files into dataset
# def load_ll_data(file_list):
#     data = []
#     for file_path in file_list:
#         with open(file_path, "r", encoding="utf-8") as f:
#             content = f.read()
#             data.append({"text": content})
#     return data

# # Convert data to Hugging Face dataset
# dataset = Dataset.from_list(load_ll_data(ll_files))

# # Split dataset into train (80%) and test (20%)
# dataset = dataset.train_test_split(test_size=0.2)

# # Load tokenizer
# model_name = "bigcode/starcoder"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

# # Tokenization function
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# # Tokenize dataset
# tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# # Load model and move to GPU
# model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# # Enable gradient checkpointing for memory efficiency
# model.gradient_checkpointing_enable()

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir="./starcoder_finetuned",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=5e-5,
#     per_device_train_batch_size=2,  # Adjust for GPU memory
#     per_device_eval_batch_size=2,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     save_total_limit=2,
#     logging_steps=50,
#     gradient_accumulation_steps=4,  # Helps with smaller batch sizes
#     fp16=True,  # Enable mixed precision training for faster training
# )

# # Data collator for CausalLM (no MLM)
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# # Move batch to correct device
# def move_to_device(batch, device):
#     return {key: val.to(device) for key, val in batch.items()}

# # Define evaluation metrics
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = logits.argmax(-1)
#     # Convert tensors to numpy arrays
#     predictions = predictions.reshape(-1)
#     labels = labels.reshape(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
#     accuracy = accuracy_score(labels, predictions)
#     return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# # Use DataParallel if multiple GPUs are available
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
#     model = torch.nn.DataParallel(model)

# # Define Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["test"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )

# # Start training
# trainer.train()

# # Save trained model
# model.save_pretrained("./starcoder_finetuned")
# tokenizer.save_pretrained("./starcoder_finetuned")

# print("Training complete. Model saved to './starcoder_finetuned'.")
import os
import glob
import torch
import math
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)

# Ensure efficient CUDA memory usage
torch.backends.cuda.matmul.allow_tf32 = True  # Reduce memory overhead
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Detect available device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Fix tokenizer parallelism warning
#os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Define folders containing LLVM-IR (.ll) files
folders = ["postgresql", "apache", "firefox", "blender", "imagemagick", "linux-module", "linux-vmlinux", "tensorflow", "openblas","gcc", "mplayer"]
#folders = ["postgresql"]
# Collect all .ll files
ll_files = []
for folder in folders:
    ll_files.extend(glob.glob(f"{folder}/*.ll"))

print(f"Found {len(ll_files)} .ll files.")

# Function to load .ll files into dataset
def load_ll_data(file_list):
    data = []
    for file_path in file_list:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            data.append({"text": content})
    return data

# Convert data to Hugging Face dataset
dataset = Dataset.from_list(load_ll_data(ll_files))

# Split dataset into train (80%) and test (20%)
#dataset = dataset.train_test_split(test_size=0.2)

# Load tokenizer
model_name = "bigcode/starcoderbase-1b"
#model_name = "./starcoder_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"], load_from_cache_file=False)

# Load model with automatic memory management
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"  # Avoids memory fragmentation by auto-placing layers
)

# Enable gradient checkpointing before moving model to device
#model.to(device)
model.gradient_checkpointing_enable()

# Define training arguments with memory optimizations
training_args = TrainingArguments(
    output_dir="./starcoderbase_finetuned",
    #evaluation_strategy="epoch",
    evaluation_strategy="no",
    #save_strategy="epoch",
    save_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=2,  # Reduce batch size to prevent OOM
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_steps=50,
    gradient_accumulation_steps=16,  # Helps with small batch sizes
    fp16=True,  # Enable mixed precision training
    #bf16 = True,
    optim="adamw_torch",
)

# Data collator for CausalLM (no MLM)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define evaluation metrics (Use Perplexity)
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     loss_fct = torch.nn.CrossEntropyLoss()
#     loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
#     perplexity = math.exp(loss.item())
#     return {"perplexity": perplexity}

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    #train_dataset=tokenized_datasets["train"],
    #eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
)

torch.cuda.empty_cache()

# Start training
trainer.train()

# Save trained model
model.save_pretrained("./starcoderbase_finetuned")
tokenizer.save_pretrained("./starcoderbase_finetuned")

print("Training complete. Model saved to './starcoderbase_finetuned'.")
