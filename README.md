# rys-splice-ik-llama

`rys-splice-ik-llama` is a focused fork of [`ikawrakow/ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp) for running the custom **Qwen3.5-27B Uncensored RYS Splice IQ4_NL** deployment with a narrow, opt-in fastpath.

This fork does **not** try to replace `ik_llama.cpp` with a separate runtime. Instead, it keeps the existing `ik-llama` server/runtime model intact and adds a specialization for the exact RYS splice deployment shape:

- `Qwen3.5-27B-Uncensored-RYS-Splice-IQ4_NL-custom.gguf`
- split mode `graph`
- F32 KV cache
- multi-GPU tensor split serving
- narrow single-token recurrent decode optimization

Public model/release home:

- Hugging Face profile: [`jackasda211233`](https://huggingface.co/jackasda211233)
- Hugging Face model page: [`jackasda211233/Qwen3.5-27B-Uncensored-RYS-Reasoner-GGUF`](https://huggingface.co/jackasda211233/Qwen3.5-27B-Uncensored-RYS-Reasoner-GGUF)

## Why this fork exists

The target deployment already worked in `ik_llama.cpp`, but it was carrying more generic runtime logic than necessary for this one exact model/setup. The goal of this fork is to keep everything good about `ik-llama`:

- GGUF loading
- OpenAI-compatible server behavior
- CUDA backend
- long-context serving
- tensor split
- split mode `graph`

while adding a **safe RYS-specific fastpath** that activates only when explicitly requested and when the loaded model matches the expected signature.

## Results

On the real server and the real model, using the same 8001-style deployment layout on **three RTX 3090s**:

| Mode | Prompt speed | Decode speed |
| --- | ---: | ---: |
| baseline `ik-llama` | 87.6 tok/s | 44.4 tok/s |
| `--rys-splice-fastpath` | 91.6 tok/s | 56.7 tok/s |

That is about a **28% decode throughput gain** in the tested configuration.

## Exact target model

This fork is centered on:

- Hugging Face release page: [`jackasda211233/Qwen3.5-27B-Uncensored-RYS-Reasoner-GGUF`](https://huggingface.co/jackasda211233/Qwen3.5-27B-Uncensored-RYS-Reasoner-GGUF)
- exact GGUF filename referenced by this fork: `Qwen3.5-27B-Uncensored-RYS-Splice-IQ4_NL-custom.gguf`

Important facts established during reverse-engineering:

- it is **standard `qwen35`** at the architecture-metadata level
- it has:
  - 72 blocks
  - 262144 context length
  - 24 attention heads
  - 4 KV heads
  - full attention every 4th block
- the deployment uses **F32 KV cache**
- the custom tensor delta is not a new architecture, but a mixed-quant profile

## What is special about the fork

The custom path is enabled by:

```bash
--rys-splice-fastpath
```

When enabled, the fork:

1. gates itself to the expected RYS/Qwen3.5 model signature
2. preserves `split-mode graph`
3. auto-enables existing load-time tensor repacking
4. forces `F32` K/V cache for the fastpath
5. specializes the narrow recurrent single-token decode path

## Inspirations and borrowed ideas

This fork was influenced by [`Luce-Org/luce-megakernel`](https://github.com/Luce-Org/luce-megakernel), but only at the **idea level**.

### Ideas borrowed from Luce

| Idea | How it was used here |
| --- | --- |
| fixed model contract | the fastpath is explicitly gated to the expected RYS signature |
| decode-first optimization | the specialization focuses on single-token recurrent decode, not generic prefill |
| reduced hot-path branching | the recurrent path gets a narrow decode branch instead of always carrying the generic path |
| persistent/reuse-oriented state thinking | recurrent state and KV assumptions were tightened around the real deployment contract |

### Ideas deliberately **not** copied from Luce

These were **not** transplanted:

- a literal single-dispatch megakernel
- a separate custom runtime outside `ik-llama`
- single-GPU assumptions
- BF16 HuggingFace tensor assumptions

Those do not fit this multi-GPU GGUF + `split-mode graph` deployment.

## What comes from upstream `ik_llama.cpp`

This fork stands on top of `ik_llama.cpp`, not beside it.

The key nearby upstream surfaces this work reuses are:

- CUDA kernels for the relevant quant families
- `split-mode graph`
- Qwen3.5 hybrid recurrent support
- `repack_tensors`
- existing recurrent F32 state expectations
- existing server behavior and API

## What changed

The main code changes live in:

- `common/common.cpp`
- `common/common.h`
- `docs/parameters.md`
- `include/llama.h`
- `src/llama-cparams.h`
- `src/llama-delta-net.cpp`
- `src/llama-delta-net.h`
- `src/llama-model.cpp`
- `src/llama-model.h`
- `src/llama.cpp`

### Summary of changes

#### 1. Added the fastpath flag and plumbing

- added `--rys-splice-fastpath`
- threaded it through public and internal context/model params

#### 2. Added safe model gating

The fastpath is not intended for generic models. It now checks for the actual RYS signature, including:

- `LLM_ARCH_QWEN35`
- expected 72-layer shape
- expected head and SSM dimensions
- full-attention-every-4th-block pattern
- `IQ5_K`/`IQ5_K_R4` on the full-attention `attn_v.weight` tensors

#### 3. Preserved graph split

The optimization is intentionally kept inside the existing `ik-llama` scheduling/runtime contract instead of bypassing it.

#### 4. Forced F32 KV cache for the fastpath

This matches the actual production deployment and the recurrent-state expectations in the fused DeltaNet path.

#### 5. Specialized the recurrent decode path

The narrow decode case:

- one token
- one sequence
- same-sequence decode
- fastpath enabled

is routed through an explicit specialization in `src/llama-delta-net.cpp`.

## Reverse-engineering result: the `140` quant mystery

An important part of this work was proving what the custom GGUF actually was.

The critical result:

- GGUF tensor type `140` in this fork/runtime is **`GGML_TYPE_IQ5_K`**

That means the custom file does **not** require a brand new quant runtime just to load. The relevant conclusion was:

- the runtime already knows the quant
- the real optimization work should go into the **execution path**, not a fictional “custom quant decoder”

## Build

### CPU build

```bash
cmake -B build -DGGML_NATIVE=ON
cmake --build build --config Release -j$(nproc)
```

### CUDA build

```bash
cmake -B build -DGGML_NATIVE=ON -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
```

### Example server-side CUDA build used for the custom deployment

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/opt/cuda \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_CURL=OFF

cmake --build build --target llama-server -j4
```

## Usage

### Minimal fastpath example

```bash
./build/bin/llama-server \
  -m /path/to/Qwen3.5-27B-Uncensored-RYS-Splice-IQ4_NL-custom.gguf \
  --host 0.0.0.0 \
  --port 8001 \
  -ngl 99 \
  -c 262144 \
  --split-mode graph \
  --tensor-split 1,1,1 \
  --cache-type-k f32 \
  --cache-type-v f32 \
  --parallel 1 \
  --threads 12 \
  --threads-batch 24 \
  --batch-size 1024 \
  --ubatch-size 512 \
  --flash-attn on \
  --cache-ram 51200 \
  --timeout 21600 \
  --jinja \
  --reasoning-format deepseek \
  --alias rys-splice-fastpath \
  --temp 0.7 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  --rys-splice-fastpath
```

## Reference deployment used in testing

The tested custom deployment cloned the working offline 8001 layout on **three RTX 3090s**:

- `CUDA_VISIBLE_DEVICES=1,2,3`
- `--tensor-split 1,1,1`
- `--split-mode graph`
- F32 KV cache

In the tested server environment, that mapping resolved to the three physical RTX 3090s actually used by the process.

## Output quality notes

Spot checks showed **no obvious fastpath-specific quality regression** versus normal `ik-llama`.

One important caveat:

- if you run with `--reasoning-format deepseek`
- and you use a low `max_tokens`

the model may spend the whole token budget in `reasoning_content` and never reach final `content`.

That is a serving-mode / token-budget issue, not a fastpath-specific bug.

## Comparison to base `llama.cpp`

A base `llama.cpp` comparison was also run against the likely llama-compatible sibling model:

- `Qwen3.5-27B-Uncensored-RYS-Splice-IQ4_NL-mainline.gguf`

using:

- the same 3-GPU mapping
- `--tensor-split 1,1,1`
- F32 KV cache

Important caveat:

- upstream `llama.cpp` does **not** support `split-mode graph`
- so the closest available upstream comparison had to use:
  - `--split-mode layer`

That means the upstream comparison is useful, but not perfectly apples-to-apples with the `ik-llama` `graph` deployment.

## Relationship to upstream

This repository should be read as:

> `ik_llama.cpp` + a RYS splice fastpath

not as a fresh inference runtime from scratch.

If you want the full general upstream feature surface, start with:

- upstream project: <https://github.com/ikawrakow/ik_llama.cpp>

This fork is intentionally narrower and more deployment-specific.

## Recommended reading

For the most detailed internal implementation history, see the authored build/deployment handoff:

- `rys-splice-fastpath-build-history.md`

## License

This fork inherits the upstream project licensing. See the repository license files for details.
