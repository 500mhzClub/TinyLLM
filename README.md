# tinyllm

building a gpt-2 style transformer from scratch. running this on a framework laptop with the radeon 780m igpu, so it uses rocm for acceleration.

the goal is just to make it talk like shakespeare.

## setup

needs a specific environment to work with the amd igpu. make a fresh venv.

```bash
python3 -m venv venv
source venv/bin/activate

```

install the rocm pytorch version. standard pip install torch wont work here because it defaults to nvidia stuff.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install tqdm

```

get the training data.

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

```

## training

the 780m is technically unsupported by rocm so you have to lie to it about the architecture version. it also needs the experimental flash attention flag or it complains.

you can add these exports to your zshrc or just run it inline like this.

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python train.py

```

loss should drop to around 1.5 after 5000 steps.