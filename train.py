import torch
import torch.nn as nn
from torch.nn import functional as F
from model import TinyGPT
import os
from tqdm import tqdm

# --- SPEED OPTIMIZATION ---
# Enable faster matrix multiplications (TF32 on Nvidia, equivalent on ROCm)
torch.set_float32_matmul_precision('high')

# --- HYPERPARAMETERS (Tuned for Radeon 780M iGPU) ---
batch_size = 32        
block_size = 256       
max_iters = 5000       
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50        
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ----------------------------------------------------

print(f"--- Training on {device} ---")
torch.manual_seed(1337)

# 1. Load Data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 2. Build the Tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(f"Vocab Size: {vocab_size}")
print(f"Total Tokens: {len(text):,}")

# 3. Train/Val Split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# 4. Batch Loader
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# 5. Helper to estimate loss (Only used at the end now)
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 6. Initialize Model
class GPTConfig:
    block_size = block_size
    vocab_size = vocab_size
    n_layer = n_layer
    n_head = n_head
    n_embd = n_embd
    dropout = dropout
    bias = False 

config = GPTConfig()
model = TinyGPT(config)
m = model.to(device)

print(f"Model Parameters: {sum(p.numel() for p in m.parameters())/1e6:.2f} M")

# OPTIMIZATION: Torch Compile
# Disabled by default as it can be slower on some iGPU setups.
# Uncomment if you want to try it (wait 60s for startup).
# model = torch.compile(model)

# 7. Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --- TRAINING LOOP (SPEED MODE) ---
print("Starting training (No validation until the end)...")

pbar = tqdm(range(max_iters), desc="Training Steps", unit="step")

for iter in pbar:

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss on this batch only (Instant)
    logits, loss = model(xb, yb)
    
    # Update progress bar with the current training loss
    # This is "free" because we calculated 'loss' for backprop anyway
    if iter % 10 == 0:
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    # Backprop
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# --- FINAL VALIDATION ---
print("\nTraining Complete. Running final validation...")
final_stats = estimate_loss()
print(f"Final Train Loss: {final_stats['train']:.4f}")
print(f"Final Val Loss:   {final_stats['val']:.4f}")

# --- SAVE MODEL ---
print("Saving model...")
# Handle saving compiled models safely
torch.save(model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict(), "tiny_shakespeare.pth")