# C → CUDA workflow (opencode + RTX 3070)

Turn the C reference programs in `benchmark/` into CUDA programs in `cuda/`, then
verify each one against its CPU baseline on this laptop's GPU — driven by
[opencode](https://opencode.ai) as the conversion agent.

> Want to know *how the agent works inside* (prompt layers + the agent loop)?
> See [`AGENT_WORKFLOW.md`](AGENT_WORKFLOW.md). This file is the *usage* guide.

```
benchmark/<tier>/<field>/<name>.c   ──/cudaify──►   cuda/<tier>/<field>/<name>.cu
        (CPU baseline, read-only)      (opencode)         (GPU conversion)
                                                              │
                                              cuda/build_run.sh <tier>/<field>/<name>
                                                              │
                                       compile both ▸ run both ▸ diff `checksum=` ▸ PASS/FAIL
```

## Machine setup (already done on this laptop)

| Piece | Where |
|---|---|
| GPU | NVIDIA RTX 3070 Laptop, compute capability **sm_86**, via WSL2 |
| CUDA toolchain | **CUDA 12.4 + nvcc** inside the `faiss` conda env (`conda activate faiss`) |
| opencode | `~/.opencode/bin/opencode` (on PATH via `~/.bashrc`), v1.17.7 |
| Model | OpenCode **Zen free** model `opencode/deepseek-v4-flash-free` — no API key |

The conda `faiss` env was completed with the CUDA dev headers + static runtime
(`cuda-cudart-dev`, `cuda-cudart-static`) so `nvcc` can both compile and link.

## The pieces

| File | Role |
|---|---|
| `opencode.json` | opencode config: default free model + lets the agent edit files / run the verifier |
| `AGENTS.md` | The conversion **rules** the agent must follow (output contract, error checking, per-tier parallelization, sm_86, complex-number mapping) |
| `.opencode/command/cudaify.md` | The `/cudaify <path>` slash command: convert → verify → fix until PASS |
| `cuda/env.sh` | Activates the `faiss` env and sets `nvcc`/`cc` flags (`-arch=sm_86`) |
| `cuda/build_run.sh` | Compiles the CPU `.c` and GPU `.cu`, runs both, diffs the `checksum=` |
| `cuda/<tier>/<field>/<name>.cu` | The generated conversions (e.g. `cuda/easy/dense-linalg/saxpy.cu`) |

## Usage

### Option A — interactive (TUI)

```sh
cd ~/ME450
opencode                                   # launch the TUI
# then, in the prompt:
/cudaify easy/dense-linalg/saxpy
```

The agent reads the baseline, writes the `.cu`, runs `cuda/build_run.sh`, and
iterates until it prints `RESULT: PASS`. Press `Tab` to toggle plan/build mode.

### Option B — headless (one shot, scriptable)

```sh
cd ~/ME450
opencode run "/cudaify easy/dense-linalg/saxpy"
```

### Verify by hand (no agent)

```sh
cd ~/ME450
./cuda/build_run.sh easy/dense-linalg/saxpy
# knobs:  TOL=1e-3 ./cuda/build_run.sh ...      (looser checksum tolerance)
#         ./cuda/build_run.sh moderate/dense-linalg/gemm 1024   (pass program args)
```

## Worked example — saxpy (verified end-to-end)

```
==> compiling CPU baseline   (cc)
==> compiling CUDA conversion (sm_86)
==> running CPU baseline
    saxpy n=16777216  checksum=140737513521151  time=0.009 s
==> running GPU conversion
    saxpy n=16777216  checksum=140737513521151  time=0.001 s
==> checksum  cpu=140737513521151  gpu=140737513521151  rel_err=0.000e+00  tol=1e-4
RESULT: PASS
```

`cuda/easy/dense-linalg/saxpy.cu` was produced by opencode from the C baseline
and matches the baseline checksum exactly.

## Scaling out

Convert a whole tier (each is independent):

```sh
for f in benchmark/easy/*/*.c; do
  rel="${f#benchmark/}"; rel="${rel%.c}"
  opencode run "/cudaify $rel"
done
```

The `moderate/` and especially `complex/` programs need real GPU design (tiling,
multi-kernel orchestration, complex-number mapping). The free model handles the
`easy/` tier well; for harder ones, use a stronger model.

## Choosing a different / stronger model

```sh
opencode models                                        # list available models
opencode run --model opencode/north-mini-code-free "/cudaify moderate/dense-linalg/gemm 1024"
```

Free Zen models: `opencode/deepseek-v4-flash-free`, `opencode/north-mini-code-free`,
`opencode/mimo-v2.5-free`, `opencode/nemotron-3-ultra-free`, `opencode/big-pickle`.
For best quality, run `opencode auth login`, add an Anthropic key, and set
`"model": "anthropic/claude-..."` in `opencode.json`.

> Note: free Zen models may use your code/prompts to improve the model. Don't use
> them for sensitive code; switch to BYOK (your own API key) for that.

## Troubleshooting

- **`nvcc: command not found`** — `conda activate faiss` first (or just use
  `cuda/build_run.sh`, which sources `cuda/env.sh` and does it for you).
- **`cuda_runtime.h: No such file`** — the env is missing dev headers:
  `conda install -n faiss -c nvidia cuda-cudart-dev cuda-cudart-static`.
- **`RESULT: FAIL` with a tiny rel_err** — GPU sum reordering; raise tolerance
  (`TOL=1e-3 ./cuda/build_run.sh ...`) or reduce the checksum on the host in
  `double` (see `AGENTS.md` rule 4).
- **Wrong GPU arch** — this laptop is `sm_86`; override with
  `CUDA_ARCH=sm_89 ./cuda/build_run.sh ...` on different hardware.
