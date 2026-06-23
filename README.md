# cnn-ops-in-c

109 standalone C programs — one per CNN / transformer operator — written as teaching demos.  
Every file compiles to a single binary, runs a smoke test, and prints the operator's output as numeric grids so you can see exactly what the math does.

**Coverage:** convolutions → activations → pooling → normalization → shape/layout ops → linear → recurrent → losses → post-processing → attention  
**Model focus:** ResNet · MobileNet · EfficientNet · YOLOv5/v8 · ViT · MobileViT · LLaMA · RT-2 · OpenVLA · π0 · Octo

---

## Quick start

```bash
# Clone and enter the repo
git clone https://github.com/venkataramanas1/cnn-ops-in-c.git
cd cnn-ops-in-c

# Build every file (outputs go to bin/)
make

# Run the full test suite and see PASS/FAIL counts
make test

# Or use the included shell script directly
bash build_and_test.sh 01_convolutions
bash build_and_test.sh 10_attention
```

Requirements: `gcc` with `-lm` support (any Linux/WSL/macOS). No external dependencies.

---

## Build a single file

```bash
# Generic pattern
gcc -O2 <section>/<file>.c -o <file> -lm
./<file>

# Examples
gcc -O2 01_convolutions/05_gelu.c        # wrong section — see below
gcc -O2 02_activations/05_gelu.c -o gelu -lm && ./gelu
gcc -O2 10_attention/05_rope_embedding.c -o rope -lm && ./rope
gcc -O2 07_recurrent/03_kv_cache.c       -o kv   -lm && ./kv
```

---

## Build one section at a time

```bash
make convolutions    # 01 — 40 files
make activations     # 02 — 12 files
make pooling         # 03 —  9 files
make normalization   # 04 —  7 files
make shape_ops       # 05 — 15 files
make linear          # 06 —  3 files
make recurrent       # 07 —  3 files
make losses          # 08 —  8 files
make postproc        # 09 —  6 files
make attention       # 10 —  6 files
```

Binaries land in `bin/<section>/`.

---

## Run one section and see output

```bash
# Run every binary in a section, show pass/fail
make run_01    # convolutions
make run_05    # shape ops
make run_10    # attention

# Run a single binary and read its output
./bin/10_attention/05_rope_embedding
./bin/07_recurrent/03_kv_cache
./bin/09_postprocessing/01_nms
```

---

## Directory map

```
cnn-ops-in-c/
├── 01_convolutions/     40 files  — 1×1 through bottleneck, dilated, transposed, SE, CBAM, MHSA, FPN, detection heads
├── 02_activations/      12 files  — ReLU, LeakyReLU, PReLU, ELU, GELU, SiLU, Mish, HardSwish, Sigmoid, Tanh, Softmax, HardSigmoid
├── 03_pooling/           9 files  — MaxPool, AvgPool, GlobalAvgPool, GlobalMaxPool, AdaptiveAvgPool, RoI Align, LpPool
├── 04_normalization/     7 files  — BatchNorm, LayerNorm, GroupNorm, InstanceNorm, RMSNorm, L2Norm, LocalResponseNorm
├── 05_shape_ops/        15 files  — Reshape, NCHW↔NHWC, Concat, Split, Pad, Crop, Upsample ×2 (nearest+bilinear),
│                                    PixelShuffle, ChannelShuffle, PatchEmbed, FlattenToSeq, Squeeze/Unsqueeze, GatherToken
├── 06_linear/            3 files  — FC layer [N][T][D], Embedding lookup, SwiGLU FFN
├── 07_recurrent/         3 files  — LSTM cell (unrolled), GRU cell (unrolled), KV-cache autoregressive demo
├── 08_losses/            8 files  — CrossEntropy, BCE, MSE, FocalLoss, Huber, KL-divergence, CosineLoss, CTC greedy decode
├── 09_postprocessing/    6 files  — NMS, Argmax [N][T][C], TopK, AnchorDecode, Softmax+Temperature, GreedyDecode
├── 10_attention/         6 files  — SDPA, SelfAttention, CrossAttention, MHSA+residual, RoPE, GQA
├── build_and_test.sh           — compile + run all .c in a given directory, print PASS/FAIL
├── Makefile                    — build all / sections / clean / test
└── README.md                   — this file
```

---

## Section-by-section file reference

### 01 — Convolutions (40 files)

| File | Operator | Key concept |
|------|----------|-------------|
| 01_conv1x1.c | 1×1 conv | channel projection, no spatial footprint |
| 02_conv3x3.c | 3×3 conv | zero-padding, weight index `((co*Ci+ci)*K+kh)*K+kw` |
| 03_conv5x5.c | 5×5 conv | wider receptive field, PAD=2 |
| 04_conv7x7.c | 7×7 conv | ResNet stem layer |
| 05_conv1x1_then_3x3.c | 1×1 → 3×3 | expand then spatial |
| 06_conv3x3_then_1x1.c | 3×3 → 1×1 | spatial then compress |
| 07_bottleneck_1x1_3x3_1x1.c | bottleneck | ResNet inner block |
| 08_depthwise3x3.c | depthwise 3×3 | per-channel lane, no cross-mixing |
| 09_pointwise1x1.c | pointwise | cross-wire depthwise lanes |
| 10_depthwise_separable.c | DW-sep | MobileNet core: DW + PW |
| 11_depthwise5x5.c | depthwise 5×5 | wider lane |
| 12_depthwise7x7.c | depthwise 7×7 | ConvNeXt signature layer |
| 13_dilated3x3_r2.c | dilated r=2 | gap taps, wider receptive field |
| 14–40 | dilated r4/r6, ASPP, transposed, bilinear upsample, pixel shuffle, grouped conv, channel shuffle, residual, dense, inverted residual, SE block, CBAM, MHSA, BN-coupled, GN, IN, detection head, FPN | see file headers |

### 02 — Activations (12 files)

| File | Formula | Used in |
|------|---------|---------|
| 01_relu.c | max(0,x) | ResNet, MobileNet, YOLO |
| 02_leaky_relu.c | x>0 ? x : 0.1x | YOLOv3/v5 detection heads |
| 03_prelu.c | x>0 ? x : α[i]x | ArcFace, PReLU-Net |
| 04_elu.c | x≥0 ? x : α(eˣ−1) | ELU-Net |
| 05_gelu.c | x·Φ(x) (tanh approx) | ViT, BERT, GPT, RT-2, OpenVLA, π0 FFN |
| 06_silu_swish.c | x·σ(x) | EfficientNet, LLaMA, RT-2 action decoder |
| 07_mish.c | x·tanh(softplus(x)) | YOLOv4 |
| 08_hard_swish.c | x·relu6(x+3)/6 | MobileNetV3, EfficientNet-Lite (edge) |
| 09_sigmoid.c | 1/(1+e⁻ˣ) | SE gates, LSTM, VLA confidence head |
| 10_tanh.c | tanh(x) | LSTM/GRU cell, VLA recurrent decoder |
| 11_softmax.c | exp(x−max)/Σ | Attention scores (ViT/VLA), classifier head |
| 12_hard_sigmoid.c | clamp((x+3)/6,0,1) | MobileNetV3 SE on edge NPUs |

### 03 — Pooling (9 files)

| File | Op | Used in |
|------|----|---------|
| 01_maxpool2x2_s2.c | 2×2 MaxPool s=2 | ResNet stage downsampling |
| 02_avgpool2x2_s2.c | 2×2 AvgPool s=2 | SqueezeNet |
| 03_global_avg_pool.c | GAP → [C][1][1] | MobileNet/EfficientNet/ViT classifier head |
| 04_global_max_pool.c | GMP → [C][1][1] | CBAM channel branch |
| 05_adaptive_avg_pool.c | AdaptiveAvgPool | PyTorch ResNet flexible resolution |
| 06_maxpool3x3_s2_pad1.c | 3×3 MP s=2 p=1 | ResNet first-stage pool |
| 07_avg_pool_3x3_s1.c | 3×3 AvgPool s=1 | local smoothing |
| 08_roi_align.c | RoI Align (bilinear) | Mask R-CNN, VLA detect-then-act |
| 09_lp_pool.c | LpPool p=2 | ONNX LpPool op, audio nets |

### 04 — Normalization (7 files)

| File | Normalizes over | Used in |
|------|-----------------|---------|
| 01_batch_norm.c | spatial H×W per channel | ResNet, MobileNet, EfficientNet |
| 02_layer_norm.c | all C×H×W per token | ViT, BERT, GPT, ALL VLA transformer blocks |
| 03_group_norm.c | group of C/G channels + H×W | Mask R-CNN, π0 UNet backbone |
| 04_instance_norm.c | H×W per channel per sample | Style transfer, CycleGAN, VLA domain adapt |
| 05_rms_norm.c | RMS scale only, no mean | LLaMA, Mistral, RT-2, OpenVLA language decoder |
| 06_l2_norm.c | unit L2 norm | ArcFace embeddings, VLA vision-language alignment |
| 07_local_response_norm.c | cross-channel window | AlexNet (historical), ONNX LRN op |

### 05 — Shape / Layout Ops (15 files)

| File | Op | Key index formula |
|------|----|-------------------|
| 01_reshape_chw_to_vec.c | Reshape → 1D | same flat index, shape-only change |
| 02_nchw_to_nhwc.c | NCHW → NHWC | `n*C*H*W+c*H*W+h*W+w` → `n*H*W*C+h*W*C+w*C+c` |
| 03_nhwc_to_nchw.c | NHWC → NCHW | inverse of above |
| 04_concat_channels.c | Concat along C | output channel offset by C1 |
| 05_split_channels.c | Split along C | two windows into same buffer |
| 06_spatial_pad.c | Zero-pad H,W | `ih = oh - PAD` |
| 07_spatial_crop_slice.c | Crop sub-region | `ih = oh + H1` |
| 08_upsample_nearest2x.c | Nearest ×2 | `ih = oh / 2` (floor) |
| 09_upsample_bilinear2x.c | Bilinear ×2 | 4-corner weighted blend |
| 10_pixel_shuffle_r2.c | PixelShuffle r=2 | `out[c][h*r+rh][w*r+rw] = in[c*r*r+rh*r+rw][h][w]` |
| 11_channel_shuffle_g2.c | ChannelShuffle G=2 | `[G][CG] → transpose → [CG][G]` |
| 12_patch_embed.c | PatchEmbed P×P | `ih = ph*P + py` — ViT tokeniser |
| 13_flatten_to_seq.c | [C][H][W] → [T][C] | `t = h*W + w` — feed to attention |
| 14_squeeze_unsqueeze.c | Squeeze / Unsqueeze | rank change, data unchanged |
| 15_gather_token.c | Gather by index | VLA: extract last A action tokens |

> **Edge / ONNX note:** `02_nchw_to_nhwc.c` is exactly the `Transpose` node ONNX Runtime inserts  
> before an ARM NEON or Mali EP kernel. `12_patch_embed.c` is the first op in every ViT-based  
> VLA vision encoder (RT-2 uses ViT-B/16: 196 tokens of dim 768 from 224×224 input).

### 06 — Linear (3 files)

| File | Op | Tensor shape |
|------|----|-------------|
| 01_linear_fc.c | FC / Linear | [N][T][D_in] → [N][T][D_out] |
| 02_embedding_lookup.c | Embedding | indices[N][T] → [N][T][D] |
| 03_swiglu_ffn.c | SwiGLU FFN | SiLU(gate) ⊙ up → down — LLaMA/RT-2/OpenVLA |

### 07 — Recurrent (3 files)

| File | Op | Notes |
|------|----|-------|
| 01_lstm_cell.c | LSTM cell unrolled over T | 4 gates, [N][T][D] input, T is dynamic |
| 02_gru_cell.c | GRU cell unrolled over T | 2 gates, simpler than LSTM |
| 03_kv_cache.c | KV-cache autoregressive demo | appends K/V each step, T grows at runtime; static T_MAX for edge |

### 08 — Losses (8 files)

| File | Loss | Used in |
|------|------|---------|
| 01_cross_entropy.c | Softmax + CE, [N][T][C] | ViT/VLA classification heads |
| 02_binary_cross_entropy.c | BCE | SE training, binary detection |
| 03_mse_loss.c | MSE over [N][T][D_action] | VLA continuous action regression (RT-2, ACT) |
| 04_focal_loss.c | Focal (α=0.25 γ=2) | YOLOv5/v8, RetinaNet anchor imbalance |
| 05_huber_loss.c | Huber / Smooth-L1 | Faster R-CNN bbox, VLA EE position |
| 06_kl_divergence.c | KL(P‖Q) | Knowledge distillation, VAE, policy distillation |
| 07_cosine_embedding_loss.c | Cosine similarity loss | VLA vision-language alignment |
| 08_ctc_loss_simple.c | CTC greedy decode | Speech (Wav2Vec2), OCR, VLA language output |

### 09 — Post-processing (6 files)

| File | Op | Notes |
|------|----|-------|
| 01_nms.c | NMS (IoU threshold) | YOLOv5/v8, SSD, VLA detect-then-act |
| 02_argmax.c | Argmax [N][T][C] | Classification, VLA greedy action selection |
| 03_topk.c | Top-K selection | Beam search, nucleus sampling |
| 04_anchor_decode.c | Anchor delta decode [N][A][4] | YOLO/SSD/DETR box decode |
| 05_softmax_temperature.c | Softmax with temperature T | VLA inference knob: T<1 deterministic, T>1 exploratory |
| 06_greedy_decode_tokens.c | Autoregressive greedy decode | RT-2/OpenVLA action token generation; dynamic T until EOS |

### 10 — Attention (6 files)

| File | Op | Tensor layout |
|------|----|--------------|
| 01_scaled_dot_product_attn.c | SDPA | Q/K/V: [N][H][T][Dh] |
| 02_self_attention.c | Self-attention + QKV proj | [N][T][D] → reshape → [N][H][T][Dh] |
| 03_cross_attention.c | Cross-attention | Q from action tokens, K/V from vision+language — VLA fusion |
| 04_mhsa_full.c | MHSA + residual + LayerNorm | full ViT/VLA transformer block |
| 05_rope_embedding.c | RoPE position embedding | LLaMA/Mistral/RT-2/OpenVLA; relative position in QK dot |
| 06_gqa_grouped_query_attn.c | GQA (Hq=4 Hkv=2) | LLaMA-3/Mistral/Gemma; 4× smaller KV cache on edge |

---

## Key concepts taught across all files

### Tensor layouts

```
3D image:    [C][H][W]        flat: c*H*W + h*W + w
4D batch:    [N][C][H][W]     flat: n*C*H*W + c*H*W + h*W + w
4D attn:     [N][H][T][Dh]   flat: n*H*T*Dh + h*T*Dh + t*Dh + d
5D video:    [N][T][C][H][W]  flat: n*T*C*H*W + t*C*H*W + ...
NHWC (edge): [N][H][W][C]    flat: n*H*W*C + h*W*C + w*C + c
```

### Static vs dynamic shapes

```c
/* STATIC: dims known at compile time — safe for edge NPUs, ONNX fixed-shape export */
#define T 128
float kv_cache[T][Dh];

/* DYNAMIC: dims are runtime parameters — variable-length sequences, adaptive resolution */
void forward(int T, int D) {
    int stride_t = D;           /* computed at call time */
    float *buf = malloc(T * D * sizeof(float));
    /* access: t*stride_t + d */
}
```

ONNX models exported with dynamic shapes require a `max_seq_len` cap when targeting edge NPUs  
(Qualcomm HTP, Apple ANE) — see `07_recurrent/03_kv_cache.c` for the pattern.

### NCHW → NHWC (the edge transpose)

PyTorch trains in NCHW. ARM NEON, Mali GPU, and TFLite all prefer NHWC — channels  
co-located in memory at each pixel = one SIMD load. ONNX Runtime edge EPs insert a  
`Transpose(0,2,3,1)` node automatically. See `05_shape_ops/02_nchw_to_nhwc.c`.

### VLA model anatomy (where each operator appears)

```
Image (224×224×3)
    │
    ▼  PatchEmbed (05/12)              — split into 16×16 patches → 196 tokens
    ▼  Embedding + RoPE (06/02, 10/05) — add position info
    ▼  MHSA blocks ×N (10/04)          — self-attention, RMSNorm (04/05), SwiGLU (06/03)
    ▼  Cross-attention (10/03)          — attend to language tokens
    ▼  Action head Linear (06/01)       — project to action vocab
    ▼  Greedy/Temperature decode (09/05,06) — generate action tokens
    ▼  Anchor decode / Inverse kinematics (09/04) — back to robot joints
```

---

## Running a specific operator and reading its output

Every binary prints:
1. **INPUT** — the demo tensor as a numeric grid
2. **KERNEL / WEIGHT** (where applicable)  
3. **OUTPUT** — result tensor
4. **check line** — `[XX] check: <value> (expected <value>)` — if values match, the demo is correct

Example output from `./bin/10_attention/05_rope_embedding`:

```
Q BEFORE RoPE  [N=1 H=1 T=4 Dh=4]
     1.0  1.0  1.0  1.0
     1.0  1.0  1.0  1.0
     ...

Q AFTER RoPE   [N=1 H=1 T=4 Dh=4]
     1.0  1.0  1.0  1.0     <- t=0: rotation by 0, unchanged
     0.5  1.5  0.9  1.1     <- t=1: rotated by theta_i
     ...

[05] check: t=0 Q unchanged = 1.0000 (expected 1.0000)
```

---

## License

Copyright (c) 2026 Venkata Ramana S. All rights reserved.  
This repository is for personal learning and reference. No part of this code
may be reproduced or distributed without explicit permission from the author.
