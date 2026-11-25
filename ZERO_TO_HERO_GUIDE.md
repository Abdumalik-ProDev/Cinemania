# Zero to Hero: Neural Networks, PyTorch, Transformers & Interactive NN Visualizers

A complete, self-contained educational guide built around the **Cinemania** IMDb sentiment-classification project. Every concept below maps directly to real code and real experimental results in this repository.

---

## Quick Answers First

### 1. Are there Neural Networks in this project?
**Yes — two completely different neural networks:**
1. **`SentimentRNN`** ([`src/cinemania/models/sentiment_rnn.py`](src/cinemania/models/sentiment_rnn.py)): A custom, from-scratch **Recurrent Neural Network** built with raw PyTorch layers (`nn.Embedding`, `nn.RNN`, `nn.Linear`, `nn.Sigmoid`). It reads a review one word at a time from left to right.
2. **`DistilBERT`** ([`src/cinemania/training/train_transformer.py`](src/cinemania/training/train_transformer.py)): A **6-layer Transformer Neural Network** (`distilbert-base-uncased`, ~67M parameters) pretrained on English Wikipedia and BookCorpus, then fine-tuned on our movie reviews using Hugging Face `Trainer`.

### 2. Do Neural Networks have a UI (like those videos where neurons connect and light up)?
**By themselves, no.** Inside PyTorch, a neural network is pure linear algebra: multi-dimensional arrays of floating-point numbers (`torch.Tensor`) multiplied together on a GPU or CPU.

However, **we can build an interactive UI on top of any PyTorch model** by recording its internal numbers (weights, hidden activations, and attention matrices) during a forward pass and rendering them with HTML5 Canvas / SVG.

That is exactly what we built in [`src/cinemania/visualizer/`](src/cinemania/visualizer/):
- **Live Interactive Web App:**
  ```bash
  uv run cinemania-viz
  ```
  Open `http://127.0.0.1:8765`, type any English movie review, and watch real `SentimentRNN` neurons fire word-by-word and `DistilBERT` attention heads connect tokens together!
- **Standalone Offline HTML Snapshot:**
  Open [`reports/nn_visualizer.html`](reports/nn_visualizer.html) directly in any browser (no server needed) to explore precomputed traces of the Cinemania reviews.

---

## Part 1 — Neural Networks from Zero

### 1.1 The Artificial Neuron
A single artificial neuron takes a vector of input numbers $x = [x_1, x_2, \dots, x_n]$, multiplies each by a learned **weight** $w_i$, adds a learned **bias** $b$, and applies a non-linear **activation function** $\sigma$:

$$y = \sigma\left(\sum_{i=1}^{n} w_i x_i + b\right) = \sigma(w^\top x + b)$$

- **Positive weight ($w_i > 0$):** Input $x_i$ pushes the neuron *up* (e.g., `"excellent"` pushing sentiment toward `positive`).
- **Negative weight ($w_i < 0$):** Input $x_i$ pushes the neuron *down* (e.g., `"boring"` pushing sentiment toward `negative`).
- **Why non-linearity matters:** Without $\sigma$, stacking 100 linear layers collapses mathematically into a single linear equation ($W_2(W_1 x) = W_{combined} x$). Non-linearities let networks learn curves, negations (`"not good"`), and context.

### 1.2 Common Activation Functions
| Activation | Formula / Behavior | Where Used |
|---|---|---|
| **Sigmoid** | $\frac{1}{1 + e^{-z}} \in (0, 1)$ | Final binary probability in `SentimentRNN` (`nn.Sigmoid()`). |
| **Softmax** | $\frac{e^{z_i}}{\sum_j e^{z_j}}$, sums to $1$ | Converting DistilBERT's 2 logits `[neg, pos]` into probabilities; also normalizes attention scores. |
| **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}} \in (-1, 1)$ | Default recurrent activation inside `nn.RNN`! Keeps hidden states bounded. |
| **ReLU / GELU** | $\max(0, z)$ / smooth probabilistic gate | Feed-forward layers inside Transformers (DistilBERT uses `GELU`). |

### 1.3 Forward Pass, Loss, and Backpropagation
Training a neural network is a 4-step loop:
1. **Forward pass:** Input flows through layers to produce a prediction $\hat{y}$.
2. **Loss function:** Measures how wrong $\hat{y}$ is compared to true label $y \in \{0, 1\}$.
   - `SentimentRNN` uses **Binary Cross-Entropy (`nn.BCELoss`)**:
     $$\mathcal{L} = -\big[y \log(\hat{y}) + (1 - y)\log(1 - \hat{y})\big]$$
   - `DistilBERT` uses **Cross-Entropy Loss** over 2 logits internally inside `AutoModelForSequenceClassification`.
3. **Backpropagation (Chain Rule):** Computes the gradient $\frac{\partial \mathcal{L}}{\partial w}$ for every weight $w$ in the network — answering *"if I nudge this weight slightly up, does the loss go up or down?"*
4. **Optimizer step (`Adam`):** Updates weights in the opposite direction of the gradient:
   $$w \leftarrow w - \eta \cdot \text{Adam\_update}\!\left(\frac{\partial \mathcal{L}}{\partial w}\right)$$
   where $\eta$ is the learning rate (`0.001` for our RNN, `5e-5` default for DistilBERT).

### 1.4 Overfitting vs. Generalization (Seen in Our Real Results!)
Look at the actual numbers from [`reports/metrics.json`](reports/metrics.json):

| Model | Train Samples | Final Train Loss | Test Loss (500 unseen) | Test Accuracy | Test F1 |
|---|---:|---:|---:|---:|---:|
| `SentimentRNN` (10 epochs) | 1,500 | `0.0317` | `2.0847` | `0.5240` | `0.5296` |
| `distilbert-base-uncased` (3 epochs) | 2,500 | `0.2538` | `0.5397` | `0.8800` | `0.8790` |

**Why did `SentimentRNN` get `0.0317` train loss (almost 0!) yet only `52.4%` test accuracy?**
That is textbook **memorization / overfitting** combined with **vanishing gradients**:
- A vanilla `nn.RNN` trained from scratch on only 1,500 long reviews (up to 256 words) struggles to carry early-sentence information across 200+ steps, so it memorizes specific training-set word patterns instead of learning general English semantics.
- **Key lesson:** Never judge a model by training loss! Always evaluate on a strictly held-out test set that never touched vocabulary building, training, or checkpoint selection.

---

## Part 2 — Mastering PyTorch (Zero to Hero)

### 2.1 Tensors & Shapes
A `torch.Tensor` is like a NumPy array that can live on a GPU (`device="cuda"`) and track gradients (`requires_grad=True`).
Always track shapes in comments! In NLP, you will constantly see:
- `(B,)` — batch of scalar labels, e.g., `(32,)`
- `(B, T)` — batch of token IDs (`batch_size=32`, `seq_len=256`)
- `(B, T, E)` — embedded tokens (`embedding_dim=128` or `768`)
- `(B, H)` — final sequence representation (`hidden_dim=256`)

### 2.2 Autograd in 5 Lines
```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 3 + 4 * x          # dy/dx = 3x^2 + 4 = 16
y.backward()                # computes gradients
print(x.grad)               # tensor([16.])
```
In training loops, you **must** call `optimizer.zero_grad()` before `loss.backward()`, because PyTorch **accumulates** (adds) gradients by default.

### 2.3 Anatomy of `nn.Module` (Our `SentimentRNN`)
In [`src/cinemania/models/sentiment_rnn.py`](src/cinemania/models/sentiment_rnn.py):
```python
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
```
Why `pack_padded_sequence`?
Reviews have different lengths. `collate_reviews` pads shorter reviews in a batch with `<pad>` (`0`) up to the longest review in that batch. Without `pack_padded_sequence`, the RNN would read dozens of `<pad>` tokens at the end and overwrite its memory! Packing tells PyTorch the true length of each review so `hidden[-1]` is the exact state at the **last real word**.

### 2.4 Top 10 PyTorch Pitfalls (and How We Avoided Them)
1. **Forgetting `optimizer.zero_grad()`** -> gradients explode/corrupt updates.
2. **Forgetting `model.eval()` and `with torch.no_grad():`** during evaluation/inference -> wastes VRAM building a computation graph and alters Dropout/BatchNorm behavior.
3. **Device mismatch (`cpu` vs `cuda`)** -> model and input tensors must be on the same device (except `pack_padded_sequence` lengths, which PyTorch requires to be a **1D CPU `int64` tensor**!).
4. **Shape broadcasting bug in `BCELoss`** -> `model(x)` returns `(B, 1)` while `labels` is `(B,)`. Always `labels.unsqueeze(1)` to `(B, 1)`.
5. **Unweighted epoch loss** -> averaging batch losses when the last batch is smaller skews the epoch loss. Always weight by `batch_size / total_samples`.
6. **Vocabulary / scaler leakage** -> never build a vocab, TF-IDF, or `StandardScaler` using test data.
7. **Passing `local_files_only` blindly or ignoring offline caches** -> handle cache fallback cleanly.
8. **Silent `torch.load` security risks** -> use `weights_only=True` when loading `state_dict`s.
9. **Non-deterministic shuffles** -> pass an explicit `torch.Generator().manual_seed(seed)` to `DataLoader`.
10. **In-place operations breaking autograd** -> avoid `x += 1` on tensors requiring grad.

### 2.5 Zero-to-Hero PyTorch Skill Ladder
- **Level 1 (Foundations):** Tensors, broadcasting, `autograd`, `nn.Linear`, `nn.Module`, `Dataset` & `DataLoader`.
- **Level 2 (Fluency):** Custom `collate_fn`, `pack_padded_sequence`, `torch.save`/`load` (`state_dict`), LR schedulers, gradient clipping (`clip_grad_norm_`).
- **Level 3 (Performance):** Mixed precision (`torch.amp.autocast`), `pin_memory=True`, `torch.compile()`, gradient accumulation for large batches on small GPUs.
- **Level 4 (Introspection & Custom Research):** Forward/backward hooks (`register_forward_hook`), custom `torch.autograd.Function`, `torch.profiler`, distributed `DDP`.

---

## Part 3 — Transformers & Building a Transformer Platform

### 3.1 Why Transformers Beat Vanilla RNNs
An RNN processes words sequentially ($t = 1, 2, \dots, T$). Early words must survive hundreds of matrix multiplications to affect the final hidden state.

A **Transformer** removes recurrence completely and uses **Self-Attention**: every token looks directly at every other token in **one parallel step** (path length = 1 between any two words!).

### 3.2 WordPiece Tokenization & Embeddings
DistilBERT does not use a fixed 4,000-word dictionary. It uses a **30,522-token WordPiece vocabulary**:
- Common words stay whole (`"excellent"`).
- Rare words split into meaningful subwords (`"unforgettable"` -> `"un"`, `"##forgettable"`).
- Special tokens are added: `[CLS]` at the start (aggregates sentence meaning for classification) and `[SEP]` at the end.
- Each token gets `Token Embedding + Positional Embedding` (since there is no recurrence, positional embeddings tell the model word order).

### 3.3 Scaled Dot-Product & Multi-Head Self-Attention
Every token vector $x_i$ is projected into three vectors: **Query ($Q$)** (*what am I looking for?*), **Key ($K$)** (*what do I contain?*), and **Value ($V$)** (*what information do I pass forward?*):

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

- **Multi-Head Attention (12 heads in DistilBERT):** Instead of one attention map, DistilBERT runs 12 smaller attention heads in parallel per layer. One head can track negation (`"not"` -> `"like"`), another can track noun-adjective pairs (`"story"` -> `"weak"`), and another can track punctuation.
- **DistilBERT vs. BERT:** DistilBERT has 6 Transformer layers instead of BERT-base's 12 (~40% smaller, ~60% faster) trained via **knowledge distillation** to mimic BERT's outputs while retaining ~97% of its language understanding.

### 3.4 How to Architect a Clean Transformer Platform
Look at how [`src/cinemania/`](src/cinemania/) is structured — this is the exact blueprint used in real ML engineering teams:
1. **`config.py` (Single Source of Truth):** Centralize paths, seeds, label mappings (`LABEL2ID`/`ID2LABEL`), and hyperparameters. Never scatter magic numbers.
2. **`data/` (Ingestion & Strict Splitting):** Separate raw loading, deterministic cleaning (`data/preproccessed/cleaned_data.csv`), and persisted train/test splits (`artifacts/splits/`) so every model trains and evaluates on identical contracts.
3. **`models/` & `training/` (Decoupled Training):** Training scripts read prepared splits and write versioned artifacts (`artifacts/rnn/`, `artifacts/transformer/checkpoint/`).
4. **`evaluation.py` (Unified Benchmark):** Compute identical metrics (`accuracy`, `precision`, `recall`, `f1`, `loss`) on the exact same held-out test set and output both machine-readable JSON (`reports/metrics.json`) and human-readable Markdown (`reports/comparison.md`).
5. **`inference/predict.py` (Stateless Serving API):** Load saved checkpoints with `eval()` + `no_grad()`, apply the **exact** training preprocessing, and expose reusable functions + CLI.
6. **`tests/` (Fast Regression Guardrails):** Unit tests with tiny synthetic fixtures verify shapes, leakage prevention, and inference contracts in seconds without retraining.

### 3.5 Taking It to Production
Once your ML core is clean like Cinemania, scaling it into a production platform means adding:
- **Serving:** Wrap `predict_sentiment_transformer` in a FastAPI endpoint with batched inference and model warm-up on startup.
- **Speed & Cost:** Export to ONNX / TensorRT or apply `INT8` quantization for 2–4x faster CPU/GPU inference.
- **Large Models:** Use **PEFT / LoRA** (Low-Rank Adaptation) to fine-tune 7B+ LLMs by updating <1% of weights.
- **Observability:** Log prediction distributions, latency, and drift metrics; track experiments with MLflow or Weights & Biases.

---

## Part 4 — Building Interactive Neural Network Visualizer UIs

How do those viral videos and tools (TensorFlow Playground, 3Blue1Brown, BertViz) make neurons light up and connect?

### Step 1: Extract Real Internals from PyTorch
There are three main ways to peek inside a PyTorch model:
1. **Return intermediate tensors in `forward()`** (best for custom models like `SentimentRNN`: extract `embedded = self.embedding(x)` and `all_hidden, _ = self.rnn(embedded)`).
2. **Use `register_forward_hook`:** Attach a callback to any `nn.Module` layer to capture its `(input, output)` without modifying model code:
   ```python
   activations = {}
   handle = model.rnn.register_forward_hook(lambda m, inp, out: activations.update({"rnn": out[0].detach()}))
   ```
3. **Hugging Face `output_attentions=True, output_hidden_states=True`:** With `attn_implementation="eager"`, DistilBERT returns all `(1, 12, seq, seq)` attention tensors and `(1, seq, 768)` hidden states directly!

### Step 2: Serve Traces via JSON (or Embed Them)
In [`src/cinemania/visualizer/inspector.py`](src/cinemania/visualizer/inspector.py) and [`server.py`](src/cinemania/visualizer/server.py):
- `POST /api/inspect` receives `{"review": "..."}`, runs both PyTorch models in `eval()` + `no_grad()`, selects the top influential neurons + real weight matrices (`weight_ih_l0`, `weight_hh_l0`, `fc.weight`), and returns compact JSON.
- `export_static_visualizer` precomputes traces for preset reviews and injects them into [`static/index.html`](src/cinemania/visualizer/static/index.html) so the UI works offline too.

### Step 3: Render Connections on HTML5 `<canvas>`
- **Nodes & Palette:** A restrained neutral graphite shell (`#0f0f10`–`#232325`) keeps UI chrome quiet so neural signals carry the color (`#4cc38a` emerald for positive, `#e5484d` crimson for negative, `#e5a93c` amber for attention focus).
- **Synapses & Attention Ribbons:** Draw smooth cubic Bézier curves between layers, mapping `weight * activation` (or attention probability) to `lineWidth` and alpha. Hovering a node isolates its exact incoming/outgoing pathways.
- **Event-Driven Interactivity:** Redraw cleanly on user interaction (scrubber, hover, click, keyboard shortcuts `Space`, `←`, `→`, `1`, `2`) instead of running wasteful infinite animation loops. Track mouse `(x, y)` against node hitboxes to display exact PyTorch values via safe `textContent` tooltips, and scrub through RNN timesteps (`t = 0..T-1`) alongside a clickable `P(positive)` trajectory chart.
- **DistilBERT Layout:** A vertical bipartite graph (`Query token` on the left rail -> `Attended key token` on the right rail) keeps every WordPiece token readable without horizontal overlap, paired with a `[CLS]` block norm table.

### Great Existing Tools to Explore
- **TensorFlow Playground** (`playground.tensorflow.org`) — Best interactive intuition for 2D decision boundaries and hidden layers.
- **3Blue1Brown / Manim** — Python animation engine used to render mathematical neural network videos.
- **BertViz** — Python/Jupyter library for inspecting Transformer multi-head attention.
- **Netron** (`netron.app`) — Visualizes the layer graph of `.pt`, `.onnx`, and `.safetensors` model files.

---

## Self-Study Roadmap (Order of Mastery)
1. **Linear Algebra & Calculus Intuition:** 3Blue1Brown *Essence of Linear Algebra* + *Neural Networks* series.
2. **PyTorch Fundamentals:** Official *Learn the Basics* tutorial + Andrej Karpathy's *Neural Networks: Zero to Hero* (`micrograd` -> `makemore` -> `nanoGPT`).
3. **Transformers Deep Dive:** Jay Alammar's *The Illustrated Transformer* + Hugging Face *NLP Course*.
4. **Hands-On Practice:** Modify this repo! Try swapping `nn.RNN` for `nn.LSTM` or `nn.GRU` in a branch, compare test accuracy against `0.5240`, and watch how the visualizer's hidden-state strip changes.
