import torch
from model import TinyGPT


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "tiny_shakespeare.pth"


class GPTConfig:
    block_size = 256
    vocab_size = 65 
    n_layer = 6
    n_head = 6
    n_embd = 384
    dropout = 0.0 # No dropout for inference
    bias = False 

# 1. Load the Model
print(f"Loading {model_path} to {device}...")
config = GPTConfig()
model = TinyGPT(config)

# Handle the case where we saved a compiled model vs raw model
state_dict = torch.load(model_path, map_location=device)
# Fix keys if they have the "_orig_mod." prefix from compilation
state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

model.to(device)
model.eval()

# 2. Helper to decode

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
decode = lambda l: ''.join([itos[i] for i in l])
encode = lambda s: [stoi[c] for c in s]

# 3. Generate
print("\n--- GENERATING SHAKESPEARE ---\n")
start_str = "\n" 
context = torch.tensor(encode(start_str), dtype=torch.long, device=device).unsqueeze(0)

# Generate 500 tokens
output = model.generate(context, max_new_tokens=500)
print(decode(output[0].tolist()))