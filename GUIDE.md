# Build Your Own Chatbot LLM — Complete Guide

---

## Project Structure

```
chatbot/
├── config.py          ← all settings in one place
├── model.py           ← neural network (LLaMA-style architecture)
├── tokenizer.py       ← BPE tokenizer, train + save + load
├── dataset.py         ← data loading and formatting
├── train.py           ← training loop (infinite, auto-resume)
├── chat.py            ← talk to your model
├── prepare_data.py    ← download DailyDialog dataset
├── download.py        ← zip and download your model
├── requirements.txt   ← pip dependencies
└── data/
    └── conversations.jsonl   ← training data (created by prepare_data.py)
```

---

## What Makes This Modern & Future-Proof

| Feature | What it is | Why it matters |
|---------|-----------|----------------|
| **RoPE** | Rotary positional encoding | Better than fixed embeddings, used in LLaMA/Mistral |
| **RMSNorm** | Faster normalization | Used in LLaMA — more stable than LayerNorm |
| **SwiGLU** | Modern activation function | Better than GELU, used in LLaMA/PaLM |
| **Flash Attention** | Fast attention via PyTorch 2.0 | Auto-used when available |
| **Weight tying** | Shared embed/output weights | Fewer params, better quality |
| **AMP (bf16/fp16)** | Mixed precision training | 2x faster on T4, 3x on A100 |
| **Cosine LR** | Smooth learning rate decay | Better convergence than fixed LR |
| **HF export** | Saves in HuggingFace format | Upload to HuggingFace Hub later |

---

## Step 1 — Setup Google Colab

Open a new Colab notebook. Make sure GPU is enabled:
**Runtime → Change runtime type → T4 GPU**

Run in a cell:
```python
# Verify GPU
import torch
print(torch.cuda.is_available())       # should print True
print(torch.cuda.get_device_name(0))   # should say Tesla T4
```

---

## Step 2 — Upload Files

Upload all files into a folder called `chatbot`:
```python
import os
os.makedirs("/content/chatbot/data", exist_ok=True)
```

Then upload via the Colab file sidebar (left panel → upload icon).
Upload all `.py` files into `/content/chatbot/`.

Or run in terminal:
```bash
cd /content/chatbot
```

---

## Step 3 — Install Dependencies

Run in a Colab cell:
```python
!pip install torch tokenizers datasets -q
```

---

## Step 4 — Download Training Data

```bash
cd /content/chatbot
python prepare_data.py
```

This downloads **DailyDialog** (~100,000 real conversation pairs) and saves it to `data/conversations.jsonl`.

Expected output:
```
✅  91,234 pairs saved → data/conversations.jsonl
📄  Sample pairs:
[1] IN : okay , so what do you think is the most important thing...
     OUT: i think the most important thing is to be honest...
```

---

## Step 5 — Start Training

```bash
python train.py
```

Expected output:
```
═══════════════════════════════════════════════════════════════════
  🚀  TRAINING  —  Ctrl+C to stop and save anytime
═══════════════════════════════════════════════════════════════════
  Model         : 13,845,504 params
  Vocab         : 8,000 tokens
  Dataset       : 91,234 samples  (713 batches/epoch)
  Batch size    : 32
  Device        : cuda
  Precision     : fp16 AMP
  Architecture  : RoPE + RMSNorm + SwiGLU  (LLaMA-style)
═══════════════════════════════════════════════════════════════════
   Epoch    Loss    Pplx         LR    Time   Status
  ──────  ───────  ───────  ─────────  ───────  ────────────────
       1   5.2341   187.91   2.98e-04   28.3s  ✅ start
       2   4.1823    65.44   2.95e-04   27.9s  🔥 great drop!
       3   3.8201    45.60   2.91e-04   27.8s  🔥 great drop!
```

---

## Step 6 — When to Stop Training

| Loss | Perplexity | Quality | Action |
|------|-----------|---------|--------|
| > 4.0 | > 55 | Gibberish | Keep training |
| 2.5–4.0 | 12–55 | Partial words | Keep training |
| 1.5–2.5 | 4.5–12 | Understandable | Getting good |
| 1.0–1.5 | 2.7–4.5 | Good replies | ✅ Stop here |
| < 1.0 | < 2.7 | Very good / memorizing | Stop |

**Target: Loss < 1.5**
Press **Ctrl+C** when you're happy. It saves automatically.

---

## Step 7 — Estimated Training Times on T4

With DailyDialog (~91k samples, batch=32 → ~713 batches/epoch):

| Target | Epochs | T4 time |
|--------|--------|---------|
| Loss < 3.0 | ~5 | ~2.5 min |
| Loss < 2.0 | ~15–25 | ~10–20 min |
| Loss < 1.5 | ~30–60 | ~25–50 min |
| Loss < 1.2 | ~80–150 | ~60–120 min |

---

## Step 8 — Chat With Your Model

```bash
python chat.py
```

```
You: hello
Bot: hi there! how are you doing today?

You: how are you?
Bot: i'm doing great, thanks for asking! how about you?

You: temp 1.2       ← make replies more creative
You: temp 0.4       ← make replies more focused
You: quit
```

---

## Step 9 — Resume Training Anytime

Just run `python train.py` again. It auto-detects `checkpoint.pt` and continues.

---

## Step 10 — Download Your Model

Run in a **Colab notebook cell** (not terminal):
```python
exec(open('/content/chatbot/download.py').read())
```

---

## Upgrading to A100 / H100

Change one line in `config.py`:
```python
cfg = ModelConfig()
cfg.upgrade_for_gpu("a100")   # or "h100"
cfg.save()
```

Then delete old checkpoint and retrain:
```bash
rm checkpoint.pt chatbot_model.pt tokenizer.json
python train.py
```

| GPU | batch_size | d_model | n_layers | d_ff | Expected time to loss < 1.5 |
|-----|-----------|---------|---------|------|---------------------------|
| T4  | 32 | 256 | 6 | 1024 | ~30–50 min |
| A100 | 128 | 512 | 8 | 2048 | ~8–15 min |
| H100 | 256 | 768 | 12 | 3072 | ~3–6 min |

---

## Export to HuggingFace (Future)

When you stop training, it auto-saves to `hf_export/`:
```
hf_export/
├── pytorch_model.bin   ← weights
├── config.json         ← architecture
└── model_config.json   ← training config
```

To upload to HuggingFace Hub later:
```python
# pip install huggingface_hub
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="hf_export",
    repo_id="your-username/your-chatbot",
    repo_type="model",
)
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `CUDA out of memory` | Lower `batch_size` in config.py |
| `Loss stays at 5+` | Delete tokenizer.json and checkpoint.pt, retrain |
| `Vocab size: 349` | Old tokenizer — delete tokenizer.json, retrain |
| `No model found` | Run train.py first |
| `files.download fails` | Run in notebook cell, not terminal |
| Loss flat after 50+ epochs | LR too low — reduce `warmup_steps` to 100 |
