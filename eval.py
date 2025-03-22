import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import orjson  # Faster than json

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("./starcoderbase_finetuned")
#tokenizer = AutoTokenizer.from_pretrained("./starcoderbase_IR")

def tokenize_code(code):
    """Tokenize LLVM IR code."""
    return tokenizer(
        code, return_tensors="pt", truncation=True, padding="max_length", max_length=512
    )

# Define Dataset Class
class CodeSimilarityDataset(Dataset):
    def __init__(self, file_path, chunk_size=50000):
        """
        Load JSON as a list instead of line-by-line.
        - file_path: Path to JSON file
        - chunk_size: Number of pairs to load at a time
        """
        self.file_path = file_path
        self.chunk_size = chunk_size

        # Read JSON file using orjson (faster than json)
        with open(self.file_path, "rb") as f:
            self.all_pairs = orjson.loads(f.read())  # Load entire JSON list

        random.shuffle(self.all_pairs)  # Shuffle data
        self.current_index = 0
        self.load_next_chunk()

    def load_next_chunk(self):
        """Load the next chunk of data into memory."""
        start = self.current_index
        end = min(start + self.chunk_size, len(self.all_pairs))
        self.pairs = self.all_pairs[start:end]
        self.current_index = end  # Move index forward

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if idx >= len(self.pairs):  # Load new chunk when needed
            self.load_next_chunk()

        code1, code2, label = self.pairs[idx]
        tokenized1 = tokenize_code(code1)
        tokenized2 = tokenize_code(code2)

        return (
            tokenized1["input_ids"].squeeze(0),
            tokenized1["attention_mask"].squeeze(0),
            tokenized2["input_ids"].squeeze(0),
            tokenized2["attention_mask"].squeeze(0),
            torch.tensor(label, dtype=torch.float),
        )

# Collate Function for Efficient Batching
def collate_fn(batch):
    input_ids1, attention_mask1, input_ids2, attention_mask2, labels = zip(*batch)
    return (
        torch.stack(input_ids1), torch.stack(attention_mask1),
        torch.stack(input_ids2), torch.stack(attention_mask2),
        torch.tensor(labels, dtype=torch.float)
    )

# Define Siamese Network (same as during training)
class SiameseNetwork(nn.Module):
    def __init__(self, model_name="./starcoderbase_finetuned", embedding_size=128):
    #def __init__(self, model_name="./starcoderbase_IR", embedding_size=128):
        super(SiameseNetwork, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.fc = nn.Linear(hidden_size, embedding_size)

    def forward_once(self, input_ids, attention_mask):
        """Pass one input through the encoder."""
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = output.last_hidden_state.mean(dim=1)
        return self.fc(embeddings)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        embed1 = self.forward_once(input_ids1, attention_mask1)
        embed2 = self.forward_once(input_ids2, attention_mask2)
        return embed1, embed2

# Cosine Similarity Loss (same as during training)
class CosineSimilarityLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(CosineSimilarityLoss, self).__init__()
        self.margin = margin
        self.cosine = nn.CosineSimilarity(dim=1)

    def forward(self, embed1, embed2, label):
        similarity = self.cosine(embed1, embed2)
        loss = (1 - label) * (similarity ** 2) + label * F.relu(self.margin - similarity) ** 2
        return loss.mean()

# Evaluation Function with Metrics
def evaluate_model_with_metrics(model, test_loader, loss_fn, device, threshold=0.7):
    model.eval()
    all_labels = []
    all_predictions = []
    total_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids1, attention_mask1, input_ids2, attention_mask2, label = batch
            input_ids1, attention_mask1, input_ids2, attention_mask2, label = (
                input_ids1.to(device), attention_mask1.to(device),
                input_ids2.to(device), attention_mask2.to(device),
                label.to(device)
            )

            with autocast():  # Enable mixed precision if needed
                embed1, embed2 = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
                loss = loss_fn(embed1, embed2, label)

            # Calculate Cosine Similarity
            similarity = F.cosine_similarity(embed1, embed2)

            # Apply threshold to get binary predictions (1: similar, 0: dissimilar)
            predictions = (similarity > threshold).float()

            # Collect true labels and predictions
            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

            total_loss += loss.item()
    
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
    # Calculate Accuracy, Precision, Recall, and F1 Score
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    print(f"Average Loss: {total_loss / len(test_loader)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Print Confusion Matrix Components
    print("\nConfusion Matrix:")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data (Test set)
#test_dataset = CodeSimilarityDataset("eval2_ir_pairs.json", chunk_size=50000)
#test_dataset = CodeSimilarityDataset("atcoder_test_small.json", chunk_size=50000)
test_dataset = CodeSimilarityDataset("normalize_atcoder_test.json", chunk_size=50000)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

# Initialize Model and Load Pre-trained Weights
model = SiameseNetwork(embedding_size=128).to(device)

# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)  # Multi-GPU support

# Load model weights
model.load_state_dict(torch.load("siamese_model2.pth", map_location=device))

# Loss function
loss_fn = CosineSimilarityLoss().to(device)

# Evaluate the model
evaluate_model_with_metrics(model, test_loader, loss_fn, device)
