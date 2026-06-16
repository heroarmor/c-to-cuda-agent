# Agent workflow — how the C→CUDA conversion agent works

This explains the *internals* of the conversion agent: what prompt drives it, how
its execution loop runs, and why a cheap model produces correct CUDA. For
day-to-day *usage* (commands, setup, troubleshooting) see [`WORKFLOW.md`](WORKFLOW.md).

The agent is [opencode](https://opencode.ai) configured for one job: turn a C
reference program in `benchmark/` into a verified CUDA program in `cuda/`.

---

## 1. Big picture

```
   developer                opencode agent                    local GPU
 ───────────       ─────────────────────────────────       ─────────────
 /cudaify <path>  ─►  expand prompt ─► plan ─► tool loop ─►  nvcc + run
                                                  │              │
                                                  ▼              ▼
                                        ./cuda/build_run.sh  ◄── checksum diff
                                                  │
                              FAIL ──── fix & retry ◄──┘
                                                  │
                                         PASS ─► report
```

Two ingredients make this reliable:

1. A **layered prompt** that pins down the output contract and the CUDA rules.
2. A **closed verification loop** — the agent compiles, runs, and diffs a
   deterministic `checksum` itself, then fixes and retries until it passes. The
   ground-truth oracle is what lets a weak model converge to correct code.

---

## 2. Prompt anatomy

When you run `opencode run "/cudaify easy/dense-linalg/saxpy"`, the model receives
a prompt assembled from **three layers**:

```
┌──────────────────────────────────────────────────────────────┐
│ Layer 1  opencode "build" agent system prompt                 │
│          (built in: available tools, edit/verify behavior)    │
├──────────────────────────────────────────────────────────────┤
│ Layer 2  AGENTS.md  — project rules, auto-injected            │
│          output contract • error checks • sm_86 • per-tier     │
│          parallelization • complex-number mapping              │
├──────────────────────────────────────────────────────────────┤
│ Layer 3  /cudaify command body  (.opencode/command/cudaify.md)│
│          the task + numbered steps, with $ARGUMENTS replaced   │
└──────────────────────────────────────────────────────────────┘
```

### Layer 2 — `AGENTS.md` (the rules / domain knowledge)

Auto-loaded for every session (and referenced explicitly via `@AGENTS.md`). The
load-bearing rules:

- **Never edit `benchmark/`** — those are the ground-truth baselines. Only write
  under `cuda/`.
- **Reproduce the baseline's result line verbatim** — same labels, same `printf`
  formats, same values, whatever token it uses (`checksum=`, `total_heat=`,
  `pi=`, `trace=`, …). Only `time=` and `(...)` perf/annotation fields may differ.
- **Faithful host-side `double` reduction** so the checksum is comparable (avoids
  GPU sum-reordering noise).
- **Always check CUDA errors** and `cudaDeviceSynchronize()` before reading back.
- **Target sm_86** (RTX 3070, Ampere), CUDA 12.4.
- **Parallelize by tier**: `easy` = one thread per element; `moderate` =
  shared-memory tiling / multi-kernel host orchestration / time recurrence;
  `complex` = match the checksum first, then optimize.
- **Complex numbers** (`<complex.h>`) → `cuDoubleComplex` / `thrust::complex`.
- **Done = `./cuda/build_run.sh <path>` prints `RESULT: PASS`.**

### Layer 3 — `/cudaify` command (the task template)

`.opencode/command/cudaify.md` is a Markdown file with frontmatter + a body. The
body *is* the prompt; `$ARGUMENTS` is substituted with the path you pass, and
`@AGENTS.md` pulls in Layer 2:

```text
Convert the CPU reference program `benchmark/$ARGUMENTS.c` into an efficient CUDA
program at `cuda/$ARGUMENTS.cu`, following every rule in @AGENTS.md.

Steps:
1. Read `benchmark/$ARGUMENTS.c` and name the parallel pattern.
2. Create `cuda/$ARGUMENTS.cu` (make parent dirs). Reproduce the baseline's exact
   result line, the CLI handling and default size, and a faithful host-side
   `double` reduction.
3. Verify by running:  `./cuda/build_run.sh $ARGUMENTS`
4. If it fails to compile or prints `RESULT: FAIL`, fix and re-run until `PASS`.

Finish by reporting the checksum compare and the GPU-vs-CPU timing.
```

> Command authoring tricks: `$ARGUMENTS` (the args), `@path` (inject a file),
> and `` !`shell cmd` `` (inject command output) are all expanded before the
> model sees the prompt.

---

## 3. The execution loop

What actually happens after the prompt is assembled — this is the ReAct-style
agent loop you can watch scroll by in the TUI or in `opencode run` output:

```
  /cudaify easy/dense-linalg/saxpy
            │
            ▼
  ① Expand prompt  (Layers 1+2+3, $ARGUMENTS → easy/dense-linalg/saxpy)
            │
            ▼
  ② Plan           agent emits a TODO list
                   [ ] read baseline C   [ ] read reference saxpy.cu
                   [ ] write saxpy.cu    [ ] run build_run.sh
            │
            ▼
  ③ Tool loop      think → call a tool → read result → think → …
     ├─ Read   benchmark/easy/dense-linalg/saxpy.c   (understand the algorithm)
     ├─ Read   cuda/easy/dense-linalg/saxpy.cu       (mirror the reference style)
     ├─ Write  cuda/<path>.cu                         (emit CUDA)
     └─ Bash   ./cuda/build_run.sh <path>             (compile + run + diff)
            │
            ▼
  ④ Read the verdict
     ├─ compile error / RESULT: FAIL ─► Edit the .cu ─► back to ③ (retry)
     └─ RESULT: PASS                 ─► ⑤
            │
            ▼
  ⑤ Report   checksum compare + GPU-vs-CPU timing, then stop
```

Step ③→④ is the loop that matters: **write → verify → inspect PASS/FAIL → fix →
re-verify.** Because LSP diagnostics and the harness verdict are fed back as real
tool output, the model corrects against facts, not guesses.

---

## 4. The verification oracle (`cuda/build_run.sh`)

The agent's "definition of done" is delegated to a script so the signal is
objective and identical whether a human or the agent runs it:

1. Compile `benchmark/<path>.c` (CPU baseline) and `cuda/<path>.cu` (GPU).
2. Run both with the same CLI args.
3. Strip the `time=` field and any `(...)` perf-rate / annotation, then compare
   **every remaining number** with a combined absolute+relative tolerance
   (`ATOL`/`RTOL`).
4. Print `RESULT: PASS` (exit 0) or `RESULT: FAIL` (exit 1).

This is deliberately benchmark-agnostic: programs print different result tokens,
so the harness compares the numbers that should match rather than hunting for one
fixed keyword.

---

## 5. Extending it

| Want to… | Do this |
|---|---|
| Add a new task command | Drop `name.md` in `.opencode/command/`; invoke as `/name` |
| Change conversion rules | Edit `AGENTS.md` (applies to every run) |
| Use a stronger model | `opencode run --model anthropic/claude-... "/cudaify …"` or set `model` in `opencode.json` |
| Tighten/loosen verification | `RTOL=1e-5 ./cuda/build_run.sh <path>` (or `ATOL`) |
| Target different GPU | `CUDA_ARCH=sm_89 ./cuda/build_run.sh <path>` |

## 6. File map

| File | Layer / role |
|---|---|
| `opencode.json` | Model + permissions; wires `AGENTS.md` as instructions |
| `AGENTS.md` | Prompt **Layer 2** — rules & domain knowledge |
| `.opencode/command/cudaify.md` | Prompt **Layer 3** — the `/cudaify` task template |
| `cuda/build_run.sh` | The verification oracle (closed loop) |
| `cuda/env.sh` | Toolchain setup (CUDA env, `-arch=sm_86`) |
| `cuda/<tier>/<field>/<name>.cu` | The agent's output (verified conversions) |
