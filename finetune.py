import orjson  # Faster than json
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from torch.cuda.amp import GradScaler, autocast

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("./starcoderbase_finetuned")

def tokenize_code(code):
    """Tokenize LLVM IR code."""
    return tokenizer(
        code, return_tensors="pt", truncation=True, padding="max_length", max_length=512
    )

# Define Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self, model_name="./starcoderbase_finetuned", embedding_size=128):
        super(SiameseNetwork, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # Get correct hidden size
        self.fc = nn.Linear(hidden_size, embedding_size)  # Project embeddings to lower dimension

    def forward_once(self, input_ids, attention_mask):
        """Pass one input through the encoder."""
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = output.last_hidden_state.mean(dim=1)  # Average pooling
        return self.fc(embeddings)  # Project to lower-dimensional space

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        """Compute embeddings for both inputs."""
        embed1 = self.forward_once(input_ids1, attention_mask1)
        embed2 = self.forward_once(input_ids2, attention_mask2)
        return embed1, embed2

# Define Cosine Similarity Loss
class CosineSimilarityLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(CosineSimilarityLoss, self).__init__()
        self.margin = margin
        self.cosine = nn.CosineSimilarity(dim=1)

    def forward(self, embed1, embed2, label):
        similarity = self.cosine(embed1, embed2)  # Compute cosine similarity
        loss = (1 - label) * (similarity ** 2) + label * F.relu(self.margin - similarity) ** 2
        return loss.mean()

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

# Load Data
train_dataset = CodeSimilarityDataset("llvm_ir_pairs.json", chunk_size=50000)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)

# Initialize Model and Enable Multi-GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork(embedding_size=128)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)  # Multi-GPU support

model = model.to(device)

# Initialize Loss and Optimizer
loss_fn = CosineSimilarityLoss().to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
scaler = GradScaler()  # Mixed precision scaler

# Training Loop
num_epochs = 9
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids1, attention_mask1, input_ids2, attention_mask2, label = batch
        input_ids1, attention_mask1, input_ids2, attention_mask2, label = (
            input_ids1.to(device), attention_mask1.to(device),
            input_ids2.to(device), attention_mask2.to(device),
            label.to(device)
        )

        optimizer.zero_grad()

        with autocast():  # Enable mixed precision
            embed1, embed2 = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            loss = loss_fn(embed1, embed2, label)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")

print("Training complete!")
model_save_path = "siamese_model2.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved successfully at {model_save_path}!")
