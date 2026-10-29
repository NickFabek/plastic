# PLASTIC

Can you write a language model's neurons instead of learning them?

Goodfire's essay [*Neural networks are already clean*](https://already-clean.goodfire.com) argues that neurons mostly encode single, nameable abstractions. If that's true, you should be able to name the abstractions up front, sort the training data into them, and turn each bin's statistics into a neuron's weights. Then train normally and skip part of the expensive early phase. A second idea: add a "righting force" during training that keeps the written neurons from drifting away.

This repo tests both on a small GPT trained from scratch on one RTX 3080.

## What was done

- gpt-6-luna read samples of [ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix) and proposed abstractions: 160 topics, 40 formats, 64 token classes.
- Jev (TypeSafe) binned 24k passages and the 16k most common tokens into them (about $2.90 of API calls in total).
- Those bins became weights with counts and linear algebra only. Token classes give a class-based bigram in the embeddings. Topics give an evidence block, an averaging attention head, and one MLP neuron per concept. Separately, I hand-wrote a copy (induction) circuit.
- A 4-layer, 29M-parameter GPT then trained from these inits, with and without anchoring forces, against matched random-init baselines. The main comparisons use 3 seeds.

## Results

- **Neurons do line up with LLM-named abstractions**, much more than random directions in the same activation space. Token classes show up in early layers, topics and formats in the last one, and the alignment sharpens during training.
- **Written inits help a little, durably.** They start 4 nats ahead and keep 0.013–0.032 nats at 50M tokens across 3 seeds, or 5–17% fewer tokens for the same loss.
- **The LLM's abstractions aren't special for this.** Bag-of-words clusters work as well, and random token classes match the LLM's.
- **The network ignores the written neurons.** Removing them after training costs almost nothing, and a construction without them does just as well. What helps is the features they would read.
- **Anchoring weights preserves the neurons but doesn't get them used.** A functional anchor, trained to keep each neuron detecting its concept, makes them sharp and used. It costs 0.04–0.12 nats.
- **Writing a mechanism worked best.** The hand-built copy circuit gave the largest gain (−0.031 nats at 4 layers / 50M tokens), and a bigger one on a 6-layer model at 100M tokens (−0.048, single seed).

Full log of every run: [`experiments.md`](experiments.md). The results page with an in-browser neuron microscope is in `web/`.

## Layout

```
src/        CUDA C++ trainer (train.cu, kernels.cuh): GPT, AdamW, anchors, concept loss, probe dumps, grad check
tools/      Python glue: data prep, API binning, closed-form construction, calibration, analysis, web export
queues/     run lists for tools/queue.sh
labels/     the abstractions and Jev's bins (taxonomy, token classes, passage labels)
web/        results page + WebGPU/JS inference engine
figures/    a couple of quick plots
```

## Running it

Needs the CUDA toolkit (built with 13.4, sm_86), MSVC on Windows, [uv](https://github.com/astral-sh/uv) for the Python tools, and Node for the page.

```bash
uv run --with pyarrow --with numpy python tools/prep_climbmix.py   # parquet shards in data/ -> data_bin/
uv run --with numpy --with scipy python tools/stats.py             # unigram / bigram stats
./build.sh && ./build/train.exe mode=gradcheck                      # build + gradient check
./build/train.exe d=256 L=4 B=64 lr=4e-3 lr_emb_mult=30 lr_head_mult=3 tokens=50000000 run_dir=runs/base
```

Constructed inits come from `tools/construct.py` (see `tools/mk_h2.sh` and `tools/mk_h3.sh` for the exact flags used), and are passed to the trainer with `init_from=`. API keys go in `.env` (`OPENAI_API_KEY`, `TYPESAFE_API_KEY`), but the labels they produced are already in `labels/`.

To view the page, export some checkpoints with `tools/export_web.py <ckpt> web/models/<name>` and run `node web/serve.mjs`.

## Caveats

Small models, short runs: 50–100M tokens. The 6-layer result is one seed with an untuned learning rate. No CUDA Tile kernels were written; the trainer is plain CUDA plus cuBLAS.
