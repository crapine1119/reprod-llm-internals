# reprod-llm-internals

A collection of experimental scripts/notebooks to **“see” what’s happening inside an LLM**.  
It mainly covers three things:

- **RoPE (Rotary Positional Embedding) visualization**: inspect `theta / cos / sin` matrices as position×dim heatmaps, and compare the `inv_freq` distribution
- **Training/memory design for vocab expansion**: an idea to reduce gradient/optimizer memory by freezing the original vocab and training only newly-added tokens (splitting embedding / LM head)
- **(Toy) multi-GPU strategy estimation/visualization**: simple models to estimate per-rank memory/peak/communication for DDP/FSDP/ZeRO2/ZeRO3 and plot the results

> ⚠️ This repo is not a “faithful paper reproduction.” It’s closer to a set of **exploration/verification notebooks and scripts**.  
> Some parts make simplifying assumptions, and you may need to tweak things depending on your environment (versions/OS).

---

## Repository layout

```
.
├── model/
│   ├── test_rope.py                  # RoPE heatmap + inv_freq visualization (same as the notebook)
│   ├── test_rope.ipynb               # RoPE visualization notebook
│   ├── test_embedding_gradient.ipynb # Split embedding (freeze base + train new tokens) experiments + memory calc
│   └── test_multinode.ipynb          # (Toy) multi-GPU strategy memory/comm estimation + plots
├── tokenizer/
│   ├── split_morpheme.py             # Korean morpheme segmentation via MeCab + <mecab> tag insertion (corpus builder)
│   ├── add_vocabs.ipynb              # Extract Korean tokens from a trained tokenizer → add_tokens() to a base tokenizer
│   ├── assets/
│   │   ├── mecab_ko.txt              # Example corpus with <mecab> tags
│   │   └── aux_tokenizer.json        # Example output of a BPE tokenizer trained in Rust
│   └── src/main.rs                   # Train a BPE tokenizer in Rust (tokenizers crate)
└── pyproject.toml                    # Python dependencies (experimental environment)
```

---

## Installation

### Python environment

Based on `pyproject.toml`:

- Python **>= 3.13**
- Key deps: `torch`, `transformers`, `accelerate`, `matplotlib`, `datasets`, `jupyter`, `mecab-ko`, `python-mecab-ko`

Quick install:

```bash
# from repo root (or recommended to use uv)
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

> Depending on your setup (especially CUDA), you may want to install `torch` / `transformers` separately.

### (Optional) Rust environment (for tokenizer training)

`tokenizer/src/main.rs` trains a BPE tokenizer using the Rust `tokenizers` crate.

After installing a Rust toolchain:

```bash
cd tokenizer
cargo run --release -- assets/mecab_ko.txt assets/aux_tokenizer.json 32000 2
```

---

## 1) RoPE visualization (`model/test_rope.py`)

### What you’ll see

- Heatmaps with **x-axis: position** and **y-axis: hidden dim** for:
  - `cos`, `sin`, and `theta` (phase)
- A side panel for the **inv_freq distribution** for the same dim indices
- Uses `transformers.ROPE_INIT_FUNCTIONS` so you can compare multiple rope types:
  - `default`, `linear`, `dynamic`, `yarn`, `longrope`, `llama3`, etc.

### Run

```bash
python model/test_rope.py
```

Tune parameters near the bottom of the script:

- Model shape: `hidden_size`, `num_attention_heads`, `partial_rotary_factor`
- Sequence length: `L`
- Phase rendering: `theta_mode = raw | phase_0_2pi | phase_negpi_pi`
- RoPE variants: add/remove entries in the `methods` list (`rope_type`)

> Large `L` (e.g., 8192*4) can be slow on CPU. If needed, switch to `device="cuda"` or reduce `L`.

---

## 2) Embedding expansion & gradient design (`model/test_embedding_gradient.ipynb`)

This notebook assumes a scenario where you add a large number of tokens to the tokenizer, and demonstrates:

- **Freeze the original vocab embedding** (as a buffer)
- **Train only the embeddings for newly-added tokens** (as parameters)
- Optionally split the LM head similarly (or share weights with the new embedding)

Key building blocks:

- `SplitEmbedding`:
  - `embed_base` is registered as a `buffer` (frozen)
  - only `embed_new` is an `nn.Embedding` (trainable)
- `SplitLMHead`:
  - base is a frozen matrix
  - new can reuse/share `embed_new`

It also includes simple utilities to estimate **embedding-related memory** (params / grads / Adam moments) under simplifying assumptions.

> This notebook focuses on **idea validation and memory accounting**, not a complete training pipeline.

Run via Jupyter:

```bash
jupyter lab
# open model/test_embedding_gradient.ipynb
```

---

## 3) (Toy) multi-GPU strategy estimation & plots (`model/test_multinode.ipynb`)

Instead of actually running multi-node training, this notebook uses simple models to estimate
**per-rank memory / peak memory / communication volume** for:

- `ddp`
- `fsdp`
- `zero2`
- `zero3`

Main outputs:

- **Steady-state memory** per world size (P/G/O)
- **Peak memory** (approximated via a layer-wise gather fraction)
- **Per-step communication volume** (rough all-gather/reduce-scatter/all-reduce estimates)

Run via Jupyter:

```bash
jupyter lab
# open model/test_multinode.ipynb
```

You can control parameters via environment variables (see the notebook header):

- `DTYPE=fp16|bf16|fp32`
- `WORLD_SIZES=1,2,4,8`
- `GATHER_PEAK_FRAC=0.35` (affects FSDP/ZeRO3 peak estimation)

> The notebook uses `AutoConfig.from_pretrained(model_id)` to load only the model config (no weights).  
> This may require internet access or a local cache.

---

## 4) Korean tokenizer helper pipeline (`tokenizer/`)

Utilities for “Korean token expansion,” which connects to the RoPE/embedding experiments.

### 4.1 Build a `<mecab>`-tagged corpus (morpheme-aware)

`split_morpheme.py` runs MeCab to get morphemes/POS, then inserts `<mecab>` boundaries so BPE training can better
“see” Korean internal boundaries.

```bash
python tokenizer/split_morpheme.py \
  --input_name allganize/IFEval-Ko \
  --output tokenizer/assets/mecab_ko.txt \
  --tag "<mecab>"
```

> Note: the `chunk_by_pos()` logic may look inverted vs. the comment.  
> If you want to change whether excluded POS tags “attach to previous vs. start a new chunk,” check the
> `if not is_excluded_pos(...)` branch.

### 4.2 Train a BPE tokenizer in Rust

`tokenizer/src/main.rs`:

- whitespace split + split by `<mecab>`
- BPE training
- save as `tokenizer.json`

```bash
cd tokenizer
cargo run --release -- assets/mecab_ko.txt assets/aux_tokenizer.json 32000 2
```

### 4.3 Add trained tokens to a base tokenizer

In `add_vocabs.ipynb`:

- extract Korean token candidates from `aux_tokenizer.json`
- remove tokens already present in the base tokenizer
- add the rest via `add_tokens()`, then compare encodings

---

## Experiment tips

- `tokenizer/target/` is the Rust build output and can be large. You can delete it if you don’t need it.
- Experiments based on `transformers` may download configs/tokenizers on first run (cache required).
- Large `L` (RoPE) or large `seqlen/batch` (multi-GPU estimation) can make visualization slow.


---

## License

No license file is included in this repo yet.  
If you plan to distribute or reuse it externally, it’s recommended to add an explicit license.
