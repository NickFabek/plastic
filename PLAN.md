# PLASTIC — research plan & queue

## Question
Goodfire ("Neural networks are already clean", Cammarata 2026) argues neurons are mostly monosemantic, encoding
*abstractions* that get coarser when neurons are scarce and sharper with scale, rather than superposed buckets.
If so, we should be able to (a) see LLM-nameable abstractions in small models' neurons, with granularity tracking
width, and (b) *write* such neurons directly from data statistics instead of learning them, and skip part of training.

## Pipeline
1. **Abstractions**: gpt-6-luna reads a sample of ClimbMix and proposes a neuron basis: facets of mutually exclusive
   concepts (topic: ~160 fine concepts in ~16 coarse groups; format/genre: ~40).
2. **Binning**: Jev (TypeSafe System One, Choice questions -> calibrated distribution over options) bins ~24k
   passages (<=128 tokens) into each facet. Soft labels p(c | passage). 20k for construction / righting force, 4k probe.
3. **Construction (no gradient descent)**: closed-form statistics from the bins write a working circuit into a GPT:
   - token embedding E gets concept-PMI coordinates  e_v[c] = log p(v|c) - log p(v)       (naive-Bayes evidence)
   - a layer-0 head with uniform causal attention averages that evidence over the prefix   (context detector)
   - one MLP neuron per concept reads its evidence coordinate (+ prior bias)               (concept neuron)
   - its output direction maps through U to  dlogit_c = log p(next|c) - log p(next)        (concept effect)
   - lm_head bias = log unigram; optional bigram-PMI SVD block in E/U                      (statistical control)
4. **Training**: normal AdamW training from the constructed init; compare to matched baselines.
5. **Righting force (plasticity)**: keep the constructed neurons "there" while training:
   - weight anchor  lambda*||theta - theta0||^2 (L2-SP), constant or decaying ("critical period")
   - functional anchor: concept-neuron activations must track Jev labels on the labeled subset
     (identity = what activates it + what it writes, per attribution-graph / microscope view)
   - moving anchor: pull toward an EMA of the weights (pattern that drifts with learning)

## Hypotheses / tests
- H1 (Goodfire, small scale): trained-baseline neurons match Jev concepts better than random directions; narrow
  models match *coarse* groups, wide models match *fine* concepts.
- H2 (bypass): constructed init reaches a target loss in fewer tokens than random init. Controls: statistical-only
  construction, shuffled-label construction, k-means-cluster construction (does LLM semantics matter?).
- H3 (righting force): anchors preserve concept fidelity (AUROC of neuron vs label); measure loss cost/benefit.

## Stack
- CUDA C++ trainer (src/): SIMT kernels + cuBLAS GEMMs (TF32), RoPE, ReLU^2 MLP, param-free RMSNorm, AdamW.
  (CUDA Tile C++ supports sm_86 in this toolkit, but no Tile kernels were written: experiment time took priority.)
- Python (uv) only for glue: parquet conversion, API calls (cached + spend-tracked), analysis/plots.
- Web: static page + WebGPU inference of baseline vs constructed model with a live "concept neuron microscope".

## Status (2026-10-16, end of session)
- [x] data -> data_bin/{train,val}.u16 (GPT-2 BPE, doc-shuffled; 548M / 4M tokens)
- [x] gpt-6-luna taxonomy (160 topics / 16 groups, 40 formats; 64 token classes)
- [x] Jev binning: 24k passages ($2.28) + top-16k tokens ($0.58)
- [x] CUDA trainer + gradient check + NumPy reference + browser engine (CPU/WebGPU) cross-checks
- [x] LR sweep (13 runs) -> lr 4e-3, emb x30, head x3
- [x] H1: neurons vs random directions (passage + token level), growth over checkpoints, width 256/1024/4096
- [x] H2: construction variants + controls (unigram, shuffled, k-means, coarse, random token classes, no-neuron), budget curve, early phase, 3 seeds
- [x] H3: weight anchors (lambda, crit, EMA, random-init control), functional anchors (w, crit, tau, layer, live init, mixing control), ablations
- [x] H4: hand-built copy circuit (+ anchors)
- [x] scale check: 6-layer d=384, 100M tokens: m_base 3.4654, m_ind (copy circuit) 3.4170
- [x] results page + WebGPU microscope, published as a private artifact

## Follow-ups worth doing
- text-through-images ViT (rendered ClimbMix) as a vision testbed for the same questions
- copy circuit + concept-evidence features combined; at larger scale and longer budgets
- functional anchor with a loss that does not compete with prediction (e.g. anchored only on activations' rank / sign, or through a separate read-out)
- better width study (per-width LR tuning; finer concept probes)
