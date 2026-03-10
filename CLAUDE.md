# CLAUDE.md — Efficient TTT: Adaptive Test-Time Training for VQA

## Project Summary

Build an adaptive VQA system where a confidence gate learns to route "easy" test samples around expensive TTT adaptation and "hard" samples through it. The central deliverable is a Pareto frontier showing accuracy vs compute cost at different gating thresholds.

**Authors:** Yugesh Sappidi & Aishwarya Reddy Chinthalapudi
**Course:** CS 518 — Deep Learning for Computer Vision (Prof. Sathya N. Ravi, UIC)
**Execution Environment:** Google Colab Pro (T4 16GB or A100 40GB)
**No local GPU.** All GPU work runs on Colab via `!python gpu/script.py`. All non-GPU work (data prep, evaluation, figures) runs locally as Python scripts.

---

## Architecture Overview

```
Input: Image I + Question Q
         │            │
         ▼            ▼
  ┌─────────────┐  ┌─────────────┐
  │ Frozen       │  │ Frozen       │
  │ ViT-B/16     │  │ BERT-base    │
  │ (patch embed)│  │ (tokenizer)  │
  └──────┬──────┘  └──────┬──────┘
         │ v ∈ R^(197×768)│ t ∈ R^(L×768)
         └────────┬───────┘
                  ▼
         ┌──────────────────┐
         │  Cross-Modal      │   θ_f (~4.7M params)
         │  Fusion Attention  │   2-layer transformer
         │  Q=v,K=t,V=t      │   + reverse: Q=t,K=v,V=v
         │  + layer norm      │
         └────────┬─────────┘
                  │ z ∈ R^(768) (pooled fusion output)
                  ▼
         ┌──────────────────┐
         │  Confidence Gate   │   θ_g (~50K params)
         │  MLP: 768→256→1   │   sigmoid output ∈ [0,1]
         │  if g(z) > τ: SKIP │
         │  if g(z) ≤ τ: ADAPT│
         └───┬──────────┬────┘
          SKIP│          │ADAPT
             │          ▼
             │  ┌──────────────────┐
             │  │  TTT Adaptation   │
             │  │  K gradient steps │
             │  │  on θ_f and θ_d   │
             │  │  using L_self_sup │
             │  │  (masked patch or │
             │  │   rotation pred)  │
             │  └────────┬─────────┘
             │           │ z' (adapted)
             ▼           ▼
         ┌──────────────────┐
         │  Prediction MLP   │   θ_d (~2.3M params)
         │  768→1024→3129    │   (3129 = VQA-v2 answer vocab)
         │  ŷ = argmax(MLP)  │
         └──────────────────┘
```

---

## Repository Structure

```
Efficient-TTT/
├── CLAUDE.md                               # THIS FILE
├── config/
│   └── config.yaml                         # All hyperparameters (single source of truth)
│
├── ttt/                                    # Core Python package (importable, testable)
│   ├── __init__.py
│   ├── data.py                             # Dataset classes (VQA-v2, VizWiz)
│   ├── models.py                           # FusionModule, ConfidenceGate, PredictionHead, FullVQAModel
│   ├── ttt_loop.py                         # TTT adaptation logic (masked patch, rotation)
│   ├── gate.py                             # Gating + routing logic
│   ├── metrics.py                          # VQA accuracy, FLOPs counter, Pareto helpers
│   └── utils.py                            # I/O, config loading, checkpointing, logging
│
├── scripts/                                # CLI entry points — run LOCALLY (no GPU)
│   ├── 01_prepare_data.py                  # Download + preprocess VQA-v2 / VizWiz
│   ├── 02_analyze_results.py               # Compute metrics from saved predictions
│   ├── 03_generate_figures.py              # 6 publication figures
│   └── 04_generate_gate_labels.py          # Compare base vs TTT predictions → gate training labels
│
├── gpu/                                    # Run on Colab ONLY — need GPU
│   ├── train_base.py                       # Train fusion + MLP (standard VQA, no TTT)
│   ├── train_gate.py                       # Train confidence gate using gate labels
│   ├── run_ttt_sweep.py                    # Sweep: K steps × TTT objective × threshold τ
│   ├── run_inference.py                    # Final evaluation with adaptive gating
│   └── run_ablation.py                     # TTT vs TTT+consistency vs TTT+mixup vs TTT+both
│
├── notebooks/
│   └── colab_runner.ipynb                  # Mount drive → pip install → !python gpu/whatever.py
│
├── checkpoints/                            # Saved model weights
│   ├── base/                               # Base VQA model (no TTT)
│   └── gate/                               # Trained gate model
│
├── results/                                # Saved predictions and metrics
│   ├── base_predictions.json               # Base model predictions (no TTT)
│   ├── ttt_predictions/                    # One file per (K, objective, τ) config
│   └── ablation/                           # TTT stabilization ablation results
│
├── figures/                                # Generated publication figures
│
├── tests/
│   ├── test_models.py                      # Model forward pass shapes
│   ├── test_ttt.py                         # TTT loop produces valid gradients
│   ├── test_gate.py                        # Gate routing logic
│   └── test_metrics.py                     # VQA accuracy computation
│
└── requirements.txt
```

---

## Configuration: `config/config.yaml`

```yaml
# === Model ===
vision_encoder: "google/vit-base-patch16-224"   # Frozen
text_encoder: "bert-base-uncased"                # Frozen
fusion_layers: 2                                 # Cross-modal transformer layers
fusion_dim: 768
fusion_heads: 12
fusion_dropout: 0.1
prediction_hidden: 1024
num_answers: 3129                                # VQA-v2 answer vocabulary size

# === Gate ===
gate_hidden: 256
gate_threshold_sweep: [0.5, 0.7, 0.8, 0.9, 0.95]

# === TTT ===
ttt_objectives: ["masked_patch", "rotation"]     # Ablate both
ttt_k_steps_sweep: [0, 1, 2, 3, 5]              # 0 = no TTT (baseline)
ttt_lr: 1.0e-4                                   # TTT inner learning rate
ttt_mask_ratio: 0.25                             # For masked patch prediction
ttt_adapt_modules: ["fusion", "prediction_head"] # Which modules get TTT gradients

# === TTT Stabilization ===
consistency_weight: 0.1                          # λ_cons for consistency regularization
consistency_augmentations: ["random_crop", "color_jitter"]
mixup_alpha_range: [0.7, 1.0]                   # Mixup anchoring interpolation range

# === Training (Base VQA) ===
train_batch_size: 64
train_lr: 1.0e-4
train_epochs: 15
train_optimizer: "adamw"
train_weight_decay: 0.01
train_scheduler: "cosine"
train_warmup_ratio: 0.1
gradient_checkpointing: true                     # For T4 memory

# === Training (Gate) ===
gate_batch_size: 128
gate_lr: 5.0e-4
gate_epochs: 5

# === Data ===
dataset: "vqa_v2"                                # or "vizwiz"
image_size: 224
max_question_length: 20                          # BERT tokenizer max length
train_split: "train"
val_split: "val"
data_dir: "data/"

# === Paths ===
checkpoint_dir: "checkpoints/"
results_dir: "results/"
figures_dir: "figures/"
```

---

## File-by-File Implementation Spec

### `ttt/__init__.py`

```python
from ttt.models import FullVQAModel, FusionModule, ConfidenceGate, PredictionHead
from ttt.ttt_loop import TTTAdapter
from ttt.gate import AdaptiveRouter
from ttt.data import VQADataset, VizWizDataset
```

---

### `ttt/data.py`

Handles VQA-v2 and VizWiz dataset loading.

**VQA-v2:**
- Images: MS-COCO (download train2014, val2014 from COCO website)
- Questions + Annotations: from VQA-v2 official downloads
- Answer vocabulary: top 3129 most-frequent answers (standard)
- Each sample: `{"image_path": str, "question": str, "answer_idx": int, "question_type": str}`
- question_type is important — we need it for per-type analysis (yes/no, number, other)

**VizWiz:**
- Download via HuggingFace `datasets` library or official VizWiz download
- Has "unanswerable" class — use this as a natural difficulty signal
- Each sample same format as VQA-v2

**Key class:**

```python
class VQADataset(torch.utils.data.Dataset):
    """
    Loads VQA-v2 or VizWiz.
    
    __getitem__ returns:
        image: torch.Tensor (3, 224, 224) — preprocessed for ViT
        input_ids: torch.Tensor (max_question_length,) — BERT tokenized question
        attention_mask: torch.Tensor (max_question_length,)
        answer_idx: int — index into answer vocabulary
        question_type: str — "yes/no", "number", "other" (for analysis)
        sample_id: str — unique identifier for this sample
    
    Image preprocessing:
        Resize to 224×224
        Normalize with ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    
    Question preprocessing:
        BERT WordPiece tokenizer
        Pad/truncate to max_question_length
    
    Answer preprocessing:
        Map answer string to index in precomputed answer vocabulary
        Answer vocab = top 3129 most frequent answers in training set
        Answers not in vocab get mapped to <UNK> (index 0)
    """
```

**Collate function** should handle batching of variable-length questions (though we pad to fixed length so this is simple).

**Data download utility function:**
```python
def download_vqa_v2(data_dir):
    """
    Download VQA-v2 data files. Files needed:
    - train2014 images (COCO): http://images.cocodataset.org/zips/train2014.zip (~13GB)
    - val2014 images (COCO): http://images.cocodataset.org/zips/val2014.zip (~6GB)
    - Questions: https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
    - Questions: https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
    - Annotations: https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
    - Annotations: https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
    
    NOTE: Images are large. For Colab, consider using a subset or storing on Google Drive.
    For initial development, use val split only (smaller) and split into train/test.
    """
```

**Answer vocabulary builder:**
```python
def build_answer_vocab(annotations_path, top_k=3129):
    """
    Count answer frequencies across all training annotations.
    Return dict mapping answer_string → index (0 to top_k-1).
    Index 0 is reserved for <UNK>.
    Save to data/answer_vocab.json.
    """
```

---

### `ttt/models.py`

All model components. Frozen encoders are loaded from HuggingFace. Trainable components are defined here.

**FusionModule:**
```python
class FusionModule(nn.Module):
    """
    Cross-modal fusion via bidirectional cross-attention.
    
    Architecture:
        Layer 1:
            - CrossAttention(Q=visual, K=text, V=text) + residual + LayerNorm
            - CrossAttention(Q=text, K=visual, V=visual) + residual + LayerNorm
            - FFN + residual + LayerNorm
        Layer 2:
            - Same structure
        Pool:
            - Take [CLS]-equivalent token (index 0 of visual sequence)
            - Or mean pool across visual tokens
            - Output: z ∈ R^(768)
    
    Parameters: θ_f
        - 2 layers × (2 cross-attention × (3 projection matrices × 768×768) + FFN (768×3072 + 3072×768))
        - Approximately 4.7M parameters
    
    Forward signature:
        def forward(self, visual_tokens, text_tokens, text_mask=None):
            # visual_tokens: (B, 197, 768) from ViT
            # text_tokens: (B, L, 768) from BERT
            # text_mask: (B, L) attention mask
            # Returns: z (B, 768) — pooled fused representation
    """
```

**ConfidenceGate:**
```python
class ConfidenceGate(nn.Module):
    """
    Lightweight MLP that predicts whether TTT will help this sample.
    
    Architecture:
        Linear(768, 256) → ReLU → Dropout(0.1) → Linear(256, 1) → Sigmoid
    
    Parameters: θ_g (~50K params)
    
    Forward:
        def forward(self, z):
            # z: (B, 768) — fused representation
            # Returns: confidence (B, 1) — values in [0, 1]
            # High confidence → SKIP TTT (model is already confident)
            # Low confidence → APPLY TTT (model is uncertain)
    
    Routing:
        def route(self, z, threshold):
            # Returns: mask (B,) — True = SKIP TTT, False = APPLY TTT
            confidence = self.forward(z)
            return confidence.squeeze(-1) > threshold
    """
```

**PredictionHead:**
```python
class PredictionHead(nn.Module):
    """
    MLP that maps fused representation to answer logits.
    
    Architecture:
        Linear(768, 1024) → ReLU → Dropout(0.2) → LayerNorm(1024) → Linear(1024, num_answers)
    
    Parameters: θ_d (~2.3M params for num_answers=3129)
    
    Forward:
        def forward(self, z):
            # z: (B, 768)
            # Returns: logits (B, num_answers)
    """
```

**FullVQAModel:**
```python
class FullVQAModel(nn.Module):
    """
    Complete VQA model combining all components.
    
    Components:
        self.vit = frozen ViT-B/16 from HuggingFace (google/vit-base-patch16-224)
        self.bert = frozen BERT-base from HuggingFace (bert-base-uncased)
        self.fusion = FusionModule()           # θ_f — trainable
        self.gate = ConfidenceGate()           # θ_g — trainable
        self.prediction_head = PredictionHead() # θ_d — trainable
    
    IMPORTANT: 
        - ViT and BERT are FROZEN. Set requires_grad=False for all their params.
        - Use torch.no_grad() when running ViT and BERT forward passes.
        - Only fusion, gate, and prediction_head have trainable params.
    
    Forward (training mode):
        def forward(self, images, input_ids, attention_mask):
            with torch.no_grad():
                visual_tokens = self.vit(images).last_hidden_state    # (B, 197, 768)
                text_output = self.bert(input_ids, attention_mask)
                text_tokens = text_output.last_hidden_state            # (B, L, 768)
            
            z = self.fusion(visual_tokens, text_tokens, attention_mask)  # (B, 768)
            confidence = self.gate(z)                                     # (B, 1)
            logits = self.prediction_head(z)                              # (B, num_answers)
            
            return logits, confidence, z
    
    Methods:
        def get_ttt_params(self):
            # Returns parameters that TTT can update: fusion + prediction_head
            # NOT the gate (gate is frozen at test time)
            return list(self.fusion.parameters()) + list(self.prediction_head.parameters())
        
        def encode(self, images, input_ids, attention_mask):
            # Run frozen encoders only, return visual_tokens, text_tokens
            # Used by TTT loop to avoid re-encoding during gradient steps
        
        def fuse_and_predict(self, visual_tokens, text_tokens, text_mask):
            # Run fusion + prediction on pre-encoded tokens
            # Used by TTT loop for efficient adaptation
    """
```

**Loading frozen encoders:**
```python
def load_frozen_vit(model_name="google/vit-base-patch16-224"):
    """
    Load ViT and freeze ALL parameters.
    
    from transformers import ViTModel
    vit = ViTModel.from_pretrained(model_name)
    for param in vit.parameters():
        param.requires_grad = False
    vit.eval()
    return vit
    """

def load_frozen_bert(model_name="bert-base-uncased"):
    """
    Load BERT and freeze ALL parameters.
    
    from transformers import BertModel
    bert = BertModel.from_pretrained(model_name)
    for param in bert.parameters():
        param.requires_grad = False
    bert.eval()
    return bert
    """
```

---

### `ttt/ttt_loop.py`

The core TTT adaptation logic. This is the most important file.

```python
class TTTAdapter:
    """
    Performs test-time training on a single sample or mini-batch.
    
    At test time:
        1. Get pre-encoded visual_tokens and text_tokens (from frozen encoders)
        2. Save a COPY of current θ_f and θ_d (the "anchor" state)
        3. For k in range(K):
            a. Compute self-supervised loss (masked patch OR rotation)
            b. Optionally add consistency loss
            c. Optionally apply mixup anchoring
            d. Backprop through θ_f and θ_d ONLY
            e. Optimizer step
        4. Run forward pass with adapted θ_f and θ_d → get prediction
        5. RESTORE θ_f and θ_d to the anchor state (critical!)
    
    The restore step is essential — TTT adapts per-sample, then resets.
    Without it, parameters drift across samples.
    
    Args:
        model: FullVQAModel
        objective: "masked_patch" or "rotation"
        k_steps: int — number of gradient steps per sample
        lr: float — inner learning rate for TTT
        use_consistency: bool
        consistency_weight: float
        use_mixup: bool
        mixup_alpha_range: tuple
    """
    
    def __init__(self, model, config):
        self.model = model
        self.objective = config["ttt_objectives"][0]  # or passed explicitly
        self.k_steps = config["ttt_k_steps_sweep"][0]
        self.lr = config["ttt_lr"]
        self.use_consistency = False  # Set per ablation
        self.use_mixup = False
        # ...
    
    def adapt_and_predict(self, images, visual_tokens, text_tokens, text_mask):
        """
        Run TTT adaptation on a batch, then predict.
        
        IMPORTANT implementation details:
        
        1. SAVE original parameters before adaptation:
            anchor_state = {name: param.clone() for name, param in model.get_ttt_params_named()}
        
        2. Create a SEPARATE optimizer for TTT (not the training optimizer):
            ttt_optimizer = torch.optim.Adam(model.get_ttt_params(), lr=self.lr)
        
        3. For each TTT step:
            - Compute self-supervised loss
            - ttt_optimizer.zero_grad()
            - loss.backward()
            - ttt_optimizer.step()
        
        4. After K steps, run prediction:
            z_adapted = model.fuse_and_predict(visual_tokens, text_tokens, text_mask)
            logits = model.prediction_head(z_adapted)
        
        5. RESTORE original parameters:
            for name, param in model.get_ttt_params_named():
                param.data.copy_(anchor_state[name])
        
        Returns:
            logits: (B, num_answers) — predictions AFTER TTT
            ttt_loss: float — final self-supervised loss (for logging)
        """
    
    def masked_patch_loss(self, visual_tokens, text_tokens, text_mask):
        """
        Masked Patch Prediction objective.
        
        1. Randomly select 25% of visual tokens (indices 1-196, skip CLS at 0)
        2. Save originals: target = visual_tokens[:, mask_indices, :].clone()
        3. Zero out selected tokens: visual_tokens[:, mask_indices, :] = 0
        4. Run fusion: z = model.fusion(masked_visual, text_tokens, text_mask)
           BUT we need per-token outputs, not pooled. 
           So modify fusion to optionally return full sequence.
        5. Predict masked tokens from fusion output at those positions
           Use a small linear projection: Linear(768, 768) — add to model
        6. Loss = MSE(predicted_tokens, target)
        
        Alternative simpler approach (recommended for v1):
        1. Mask 25% of visual tokens
        2. Run full model forward (fusion → pooled z)
        3. From z, predict the mean of masked tokens via Linear(768, 768)
        4. Loss = MSE(predicted_mean, actual_mean_of_masked_tokens)
        
        This is simpler and still forces the model to understand the image.
        """
    
    def rotation_loss(self, images):
        """
        Rotation Prediction objective.
        
        1. Pick a random rotation: r ∈ {0, 90, 180, 270}
        2. Apply rotation to images (use torchvision.transforms.functional.rotate)
        3. Re-encode rotated image through ViT (frozen, no_grad)
        4. Run fusion on rotated visual tokens + text tokens
        5. From pooled z, predict rotation class via Linear(768, 4)
           Add this projection head to the model (only used during TTT)
        6. Loss = CrossEntropy(predicted_rotation, true_rotation)
        
        NOTE: The ViT re-encoding is a forward pass only (no grad through ViT).
        Gradients flow through fusion and the rotation head.
        """
    
    def consistency_loss(self, images, visual_tokens, text_tokens, text_mask):
        """
        Consistency regularization WITHIN the TTT loop.
        
        1. Generate two augmented views of the image:
           aug1 = RandomResizedCrop(224) + ColorJitter(0.2, 0.2, 0.2, 0.1)
           aug2 = same augmentation pipeline, different random seed
        2. Encode both through frozen ViT → v1, v2
        3. Fuse both: z1 = fusion(v1, text, mask), z2 = fusion(v2, text, mask)
        4. Get prediction distributions: p1 = softmax(head(z1)), p2 = softmax(head(z2))
        5. Loss = KL(p1 || p2) + KL(p2 || p1)  (symmetric KL)
        
        This loss is ADDED to the self-supervised loss with weight λ_cons:
        total_ttt_loss = L_self_sup + λ_cons * L_consistency
        """
    
    def mixup_anchor(self, z_current, z_anchor, alpha_range=(0.7, 1.0)):
        """
        Mixup anchoring to prevent parameter drift.
        
        alpha = random uniform in alpha_range
        z_mixed = alpha * z_current + (1 - alpha) * z_anchor.detach()
        
        Use z_mixed instead of z_current for the prediction.
        This keeps the adapted representation close to the original.
        
        z_anchor is the representation BEFORE any TTT steps (saved at start).
        """
```

---

### `ttt/gate.py`

Gating and routing logic.

```python
class AdaptiveRouter:
    """
    Routes samples through base model or TTT based on gate confidence.
    
    Usage:
        router = AdaptiveRouter(model, ttt_adapter, threshold=0.8)
        predictions = router.predict(images, input_ids, attention_mask)
    
    Logic:
        1. Encode all samples (frozen ViT + BERT)
        2. Fuse all samples → get z for each
        3. Gate: confidence = gate(z) for each sample
        4. Split batch:
           - high_conf = confidence > threshold → predict directly (SKIP TTT)
           - low_conf = confidence ≤ threshold → run TTT, then predict (ADAPT)
        5. Combine predictions back into original batch order
    
    Returns:
        predictions: (B, num_answers) — logits
        routing_info: dict with:
            - "skip_count": int
            - "adapt_count": int  
            - "confidences": (B,) — gate confidence values
            - "skip_mask": (B,) — boolean mask
    
    IMPORTANT: Process SKIP and ADAPT samples separately for correct FLOPs counting.
    SKIP samples: encode + fuse + predict = cheap
    ADAPT samples: encode + fuse + K*(fuse + loss + backward + step) + predict = expensive
    """
    
    def compute_flops(self, routing_info, k_steps):
        """
        Compute average FLOPs per sample for this batch.
        
        FLOPs estimates (approximate, for ViT-B/16 + BERT-base):
            Encoding (frozen, always runs): ~17.6 GFLOPs (ViT) + ~22.5 GFLOPs (BERT)
            Fusion forward: ~0.8 GFLOPs
            Prediction head forward: ~0.006 GFLOPs
            TTT per step (fusion forward + backward + pred head forward + backward): ~3.2 GFLOPs
        
        SKIP path: encode + fuse + predict ≈ 40.9 GFLOPs
        ADAPT path: encode + fuse + K * 3.2 + predict ≈ 40.9 + K * 3.2 GFLOPs
        
        Average = (n_skip * SKIP_flops + n_adapt * ADAPT_flops) / total_samples
        
        These are rough estimates. For precise numbers, use fvcore.nn.FlopCountAnalysis
        or torch.profiler. But rough estimates are fine for the Pareto plot.
        """
```

---

### `ttt/metrics.py`

```python
def vqa_accuracy(predictions, ground_truth):
    """
    Standard VQA accuracy metric.
    
    For VQA-v2: min(#humans_who_gave_that_answer / 3, 1)
    For simplicity (and common practice): just check if predicted answer
    matches the most-frequent ground truth answer. Report top-1 accuracy.
    
    predictions: list of predicted answer indices
    ground_truth: list of ground truth answer indices
    Returns: float accuracy in [0, 1]
    """

def accuracy_by_question_type(predictions, ground_truth, question_types):
    """
    Break down accuracy by question type (yes/no, number, other).
    Returns dict: {"yes/no": float, "number": float, "other": float}
    """

def pareto_frontier(results_list):
    """
    Given a list of dicts with {"accuracy": float, "avg_flops": float, "config": str},
    compute the Pareto-optimal configurations (highest accuracy for given FLOPs).
    Returns filtered list of Pareto-optimal points.
    """

def compute_gate_statistics(routing_infos):
    """
    Aggregate routing info across all batches.
    Returns:
        total_skip: int
        total_adapt: int
        skip_rate: float (what % skipped TTT)
        avg_confidence_skip: float
        avg_confidence_adapt: float
        accuracy_skip_group: float (accuracy of samples that skipped)
        accuracy_adapt_group: float (accuracy of samples that went through TTT)
    """
```

---

### `ttt/utils.py`

```python
def load_config(path="config/config.yaml"):
    """Load YAML config. Returns dict."""

def save_json(data, path):
    """Save dict/list to JSON with indent=2."""

def load_json(path):
    """Load JSON file. Returns dict/list."""

def save_checkpoint(model, optimizer, epoch, path):
    """
    Save model state dict (trainable params only, not frozen encoders).
    torch.save({
        "fusion": model.fusion.state_dict(),
        "gate": model.gate.state_dict(),
        "prediction_head": model.prediction_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }, path)
    """

def load_checkpoint(model, path, load_optimizer=False):
    """Load checkpoint into model. Frozen encoders are not saved/loaded."""

def setup_logging(log_dir):
    """Basic Python logging to file + console."""
```

---

### `gpu/train_base.py`

```python
"""
Train the base VQA model (fusion + gate + prediction head) on VQA-v2 or VizWiz.
Standard supervised training with cross-entropy loss. No TTT during training.

Usage on Colab:
    !python gpu/train_base.py --config config/config.yaml --epochs 15

What this trains:
    - θ_f (FusionModule): cross-modal attention
    - θ_g (ConfidenceGate): initially just trained as auxiliary, refined later
    - θ_d (PredictionHead): answer classifier

Loss: CrossEntropy(logits, answer_idx)

The gate is trained with auxiliary loss during base training:
    Gate sees z, predicts confidence. But we don't have TTT labels yet,
    so during base training, we just train it to predict whether the base
    model gets the answer right (as a proxy for "easy" vs "hard").
    L_gate = BCE(gate(z), 1[base_prediction == ground_truth])

Full loss: L = L_vqa + 0.1 * L_gate

After base training, we generate proper gate labels (Step 04) and refine the gate.

Training loop:
    for epoch in range(epochs):
        for images, input_ids, attention_mask, answers in train_loader:
            logits, confidence, z = model(images, input_ids, attention_mask)
            
            loss_vqa = F.cross_entropy(logits, answers)
            
            with torch.no_grad():
                correct = (logits.argmax(dim=-1) == answers).float()
            loss_gate = F.binary_cross_entropy(confidence.squeeze(-1), correct)
            
            loss = loss_vqa + 0.1 * loss_gate
            loss.backward()
            optimizer.step()
            ...
        
        # Validate
        val_acc = evaluate(model, val_loader)
        save_checkpoint(model, optimizer, epoch, f"checkpoints/base/epoch_{epoch}.pt")

Saves: checkpoints/base/best.pt
Also saves: results/base_predictions.json (predictions on val set for gate label generation)
"""
```

---

### `scripts/04_generate_gate_labels.py`

```python
"""
Generate training labels for the confidence gate.

This runs LOCALLY (no GPU needed — reads saved predictions).

Logic:
    1. Load base model predictions from results/base_predictions.json
    2. Load TTT predictions from results/ttt_predictions/k1_masked_patch.json
       (run gpu/run_ttt_sweep.py first with K=1 on training/val set)
    3. For each sample, compute:
       - base_correct = (base_pred == ground_truth)
       - ttt_correct = (ttt_pred == ground_truth)
       - ttt_helps = (not base_correct) and ttt_correct
       - ttt_hurts = base_correct and (not ttt_correct)
    4. Gate label:
       - 1.0 (high confidence, SKIP TTT) if base is already correct
       - 0.0 (low confidence, APPLY TTT) if TTT helps
       - 1.0 if TTT hurts (don't apply TTT even though base is wrong)
       - 0.5 if neither helps (uncertain — let threshold decide)

Saves: data/gate_labels.json
    List of {"sample_id": str, "gate_label": float, "base_correct": bool, "ttt_helps": bool}
"""
```

---

### `gpu/train_gate.py`

```python
"""
Refine the confidence gate using proper labels from gate_labels.json.

Usage on Colab:
    !python gpu/train_gate.py --base-checkpoint checkpoints/base/best.pt

Loads the base model (frozen fusion + prediction head), only trains θ_g.
Uses gate labels generated by scripts/04_generate_gate_labels.py.

Training:
    for sample, label in gate_train_loader:
        z = model.encode_and_fuse(sample)  # no grad through frozen parts
        confidence = model.gate(z)
        loss = F.binary_cross_entropy(confidence.squeeze(-1), label)
        loss.backward()  # Only gate params get gradients
        ...

Saves: checkpoints/gate/best.pt
"""
```

---

### `gpu/run_ttt_sweep.py`

```python
"""
The main experiment: sweep K steps × TTT objective × threshold τ.

Usage on Colab:
    !python gpu/run_ttt_sweep.py --checkpoint checkpoints/base/best.pt --k 1 --objective masked_patch
    !python gpu/run_ttt_sweep.py --checkpoint checkpoints/base/best.pt --k 3 --objective rotation
    !python gpu/run_ttt_sweep.py --checkpoint checkpoints/base/best.pt --k 0  # baseline (no TTT)

For each configuration:
    1. Load base model
    2. For each val sample:
        a. Encode (frozen ViT + BERT)
        b. If K > 0: run TTT adaptation with specified objective
        c. Predict answer
        d. Record: prediction, ground_truth, question_type, sample_id
    3. Save results to results/ttt_predictions/k{K}_{objective}.json

The sweep generates one results file per (K, objective) pair.
Threshold τ is applied AFTER, during analysis (scripts/02_analyze_results.py).

Estimated time per configuration:
    K=0 (no TTT): ~10 min on A100 (just forward passes)
    K=1: ~25 min (1 backward pass per sample)
    K=3: ~55 min
    K=5: ~90 min
"""
```

---

### `gpu/run_ablation.py`

```python
"""
Ablation study for TTT stabilization techniques.

Fixed: K=1 (or K=2), masked_patch objective.
Vary: 
    (a) TTT alone (no stabilization)
    (b) TTT + consistency regularization
    (c) TTT + mixup anchoring
    (d) TTT + both

Usage:
    !python gpu/run_ablation.py --checkpoint checkpoints/base/best.pt --k 1 --mode ttt_only
    !python gpu/run_ablation.py --checkpoint checkpoints/base/best.pt --k 1 --mode ttt_consistency
    !python gpu/run_ablation.py --checkpoint checkpoints/base/best.pt --k 1 --mode ttt_mixup
    !python gpu/run_ablation.py --checkpoint checkpoints/base/best.pt --k 1 --mode ttt_both

Saves: results/ablation/{mode}_k{K}.json
"""
```

---

### `gpu/run_inference.py`

```python
"""
Final evaluation with the trained adaptive gate.

Usage:
    !python gpu/run_inference.py \
        --base-checkpoint checkpoints/base/best.pt \
        --gate-checkpoint checkpoints/gate/best.pt \
        --threshold 0.8 \
        --k 1 \
        --objective masked_patch

This runs the FULL adaptive pipeline:
    1. Encode sample (frozen ViT + BERT)
    2. Fuse → z
    3. Gate decides: SKIP or ADAPT
    4. If SKIP: predict from z directly
    5. If ADAPT: run K TTT steps, predict from z'
    6. Record: prediction, routing decision, confidence, FLOPs

Saves: results/adaptive_t{threshold}_k{K}_{objective}.json

Run once per threshold to build the Pareto curve:
    for tau in 0.5 0.7 0.8 0.9 0.95; do
        !python gpu/run_inference.py --threshold $tau --k 1 --objective masked_patch ...
    done
"""
```

---

### `scripts/02_analyze_results.py`

```python
"""
Compute all metrics from saved predictions. No GPU needed.

Usage:
    python scripts/02_analyze_results.py --results-dir results/

Computes:
    1. Overall VQA accuracy for each configuration (K, objective, τ)
    2. Accuracy by question type (yes/no, number, other)
    3. Average FLOPs per sample for each configuration
    4. Pareto frontier points
    5. Gate routing statistics (% skip, % adapt, accuracy per group)
    6. McNemar's test: base vs best TTT configuration

Prints summary table and saves: results/analysis_summary.json
"""
```

---

### `scripts/03_generate_figures.py`

```python
"""
Generate 6 publication-quality figures. No GPU needed.

Usage:
    python scripts/03_generate_figures.py --results-dir results/ --output-dir figures/

Figures:
    1. PARETO FRONTIER (THE MAIN FIGURE)
       X-axis: Average GFLOPs per sample
       Y-axis: VQA Accuracy (%)
       Points: each (K, τ) configuration
       Annotate: "K=0 (no TTT)", "K=1, τ=0.9", "K=1, τ=0.8", "K=3 (full TTT)"
       Color: by K value
       Highlight the Pareto-optimal points
       This is the central deliverable of the project.
    
    2. TTT STEPS vs ACCURACY
       X-axis: K (number of TTT steps)
       Y-axis: VQA Accuracy (%)
       One line per TTT objective (masked_patch, rotation)
       Shows diminishing returns: most gain at K=1, marginal at K=5
    
    3. GATE ROUTING ANALYSIS
       Stacked bar chart showing what % of samples are routed to SKIP vs ADAPT
       at each threshold τ
       Overlay accuracy for each group
    
    4. PER-QUESTION-TYPE BREAKDOWN
       Grouped bar chart: question types (yes/no, number, other) × (base, TTT, adaptive)
       Shows WHERE TTT helps most
       Expected: TTT helps "other" (compositional) questions more than "yes/no"
    
    5. TTT STABILIZATION ABLATION
       Bar chart: base, TTT-only, TTT+consistency, TTT+mixup, TTT+both
       Shows whether stabilization techniques help and which combination is best
    
    6. CONFIDENCE DISTRIBUTION
       Histogram of gate confidence values for correct vs incorrect base predictions
       Shows whether the gate can distinguish easy from hard samples
       Expected: bimodal — high confidence for correct, low for incorrect
"""
```

---

### `notebooks/colab_runner.ipynb`

```python
# Cell 1: Setup (run once per session)
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/Efficient-TTT')

!pip install -q torch torchvision transformers datasets pillow pyyaml

!nvidia-smi

# Cell 2: GPU tasks (uncomment the one you need)

# --- Step 1: Train base VQA model ---
# !python /content/drive/MyDrive/Efficient-TTT/gpu/train_base.py --config config/config.yaml

# --- Step 2: Run TTT sweep (one config at a time) ---
# !python /content/drive/MyDrive/Efficient-TTT/gpu/run_ttt_sweep.py --checkpoint checkpoints/base/best.pt --k 0
# !python /content/drive/MyDrive/Efficient-TTT/gpu/run_ttt_sweep.py --checkpoint checkpoints/base/best.pt --k 1 --objective masked_patch
# !python /content/drive/MyDrive/Efficient-TTT/gpu/run_ttt_sweep.py --checkpoint checkpoints/base/best.pt --k 1 --objective rotation
# !python /content/drive/MyDrive/Efficient-TTT/gpu/run_ttt_sweep.py --checkpoint checkpoints/base/best.pt --k 3 --objective masked_patch

# --- Step 3: Train gate (after generating gate labels locally) ---
# !python /content/drive/MyDrive/Efficient-TTT/gpu/train_gate.py --base-checkpoint checkpoints/base/best.pt

# --- Step 4: Run adaptive inference at each threshold ---
# !python /content/drive/MyDrive/Efficient-TTT/gpu/run_inference.py --base-checkpoint checkpoints/base/best.pt --gate-checkpoint checkpoints/gate/best.pt --threshold 0.8 --k 1 --objective masked_patch

# --- Step 5: Run ablation ---
# !python /content/drive/MyDrive/Efficient-TTT/gpu/run_ablation.py --checkpoint checkpoints/base/best.pt --k 1 --mode ttt_only
```

---

## End-to-End Workflow

```
LOCAL:
  python scripts/01_prepare_data.py          # Download + preprocess dataset

COLAB:
  !python gpu/train_base.py                   # Train base VQA model (~4-6 hrs)

COLAB:
  !python gpu/run_ttt_sweep.py --k 0          # Baseline (no TTT)
  !python gpu/run_ttt_sweep.py --k 1 --objective masked_patch
  !python gpu/run_ttt_sweep.py --k 1 --objective rotation
  !python gpu/run_ttt_sweep.py --k 2 --objective masked_patch
  !python gpu/run_ttt_sweep.py --k 3 --objective masked_patch
  !python gpu/run_ttt_sweep.py --k 5 --objective masked_patch

LOCAL:
  python scripts/04_generate_gate_labels.py   # Base vs TTT predictions → gate labels

COLAB:
  !python gpu/train_gate.py                   # Train confidence gate (~30 min)

COLAB:
  !python gpu/run_inference.py --threshold 0.5 --k 1 --objective masked_patch
  !python gpu/run_inference.py --threshold 0.7 --k 1 --objective masked_patch
  !python gpu/run_inference.py --threshold 0.8 --k 1 --objective masked_patch
  !python gpu/run_inference.py --threshold 0.9 --k 1 --objective masked_patch
  !python gpu/run_inference.py --threshold 0.95 --k 1 --objective masked_patch

COLAB:
  !python gpu/run_ablation.py --mode ttt_only
  !python gpu/run_ablation.py --mode ttt_consistency
  !python gpu/run_ablation.py --mode ttt_mixup
  !python gpu/run_ablation.py --mode ttt_both

LOCAL:
  python scripts/02_analyze_results.py        # Compute all metrics
  python scripts/03_generate_figures.py       # Generate 6 figures
```

---

## Tests

```python
# tests/test_models.py
def test_fusion_output_shape():
    """FusionModule(visual(B,197,768), text(B,20,768)) → (B, 768)"""

def test_gate_output_range():
    """ConfidenceGate output is in [0, 1]"""

def test_prediction_head_shape():
    """PredictionHead(B, 768) → (B, 3129)"""

def test_frozen_encoders():
    """ViT and BERT parameters have requires_grad=False"""

def test_full_model_forward():
    """FullVQAModel forward pass produces correct shapes"""

# tests/test_ttt.py
def test_ttt_restores_params():
    """After TTT adaptation, model params return to original values"""

def test_ttt_masked_patch_loss():
    """Masked patch loss is a positive scalar"""

def test_ttt_rotation_loss():
    """Rotation prediction loss is a positive scalar"""

def test_ttt_changes_prediction():
    """TTT with K>0 produces different logits than K=0"""

# tests/test_gate.py
def test_routing_split():
    """Router correctly splits batch into skip and adapt groups"""

def test_routing_recombine():
    """Predictions are correctly recombined after split routing"""

# tests/test_metrics.py
def test_vqa_accuracy():
    """VQA accuracy computation is correct"""

def test_pareto_frontier():
    """Pareto frontier correctly filters dominated points"""
```

---

## requirements.txt

```
torch>=2.0
torchvision>=0.15
transformers>=4.36
datasets
pillow
pyyaml
numpy
matplotlib
scipy
tqdm
```

---

## Memory Budget (Colab T4 16GB)

| Component | VRAM |
|-----------|------|
| ViT-B/16 (frozen, fp32) | ~340 MB |
| BERT-base (frozen, fp32) | ~420 MB |
| Fusion (2 layers, fp32) | ~36 MB |
| Gate | ~0.4 MB |
| Prediction head | ~18 MB |
| Activations (batch=64, training) | ~4 GB |
| Gradients (trainable only) | ~200 MB |
| Optimizer states (AdamW, trainable) | ~400 MB |
| **Training total** | **~5.5 GB** ✅ |
| TTT (K=1, batch=1, extra backward) | +~1.5 GB |
| **TTT inference total** | **~7 GB** ✅ |

Fits comfortably on T4 (16GB). With gradient checkpointing, even batch=128 is possible for training.

---

## Key Implementation Gotchas

1. **TTT parameter restore is CRITICAL.** After every TTT adaptation, restore θ_f and θ_d to their pre-adaptation values. Without this, params drift and accuracy collapses after ~100 samples. Use `param.data.copy_(saved_state)`.

2. **Frozen encoders must stay frozen.** Double-check `requires_grad=False` for all ViT and BERT params. Also wrap encoder forward passes in `torch.no_grad()` for memory savings.

3. **TTT creates a NEW optimizer per sample/batch.** Don't reuse the training optimizer for TTT. Create a fresh `Adam(model.get_ttt_params(), lr=ttt_lr)` for each adaptation episode.

4. **Gate is frozen at test time.** During adaptive inference, only θ_f and θ_d are adapted by TTT. The gate θ_g makes routing decisions but is never updated at test time.

5. **FLOPs counting for Pareto.** Track FLOPs per sample, not per batch. SKIP samples have constant FLOPs. ADAPT samples have FLOPs proportional to K.

6. **VQA-v2 images are large.** Don't try to load all images into memory. Use lazy loading in the Dataset class. Keep images on disk (Google Drive), load per-batch.

7. **Answer vocabulary.** Build it ONCE from the training set annotations, save as `data/answer_vocab.json`, and load it everywhere. Don't rebuild it each time.
