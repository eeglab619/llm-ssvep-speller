# llm-ssvep-speller
This is the official code repository for the ICASSP 2027 submitted paper, "LLM-Guided Training-Free Dynamic Stopping and Text Completion for Efficient SSVEP BCI Spelling".

> **`llm_policy.py` is currently a placeholder.** The prompt and
> COMMIT/CONTINUE decision logic are withheld pending peer review; the
> file keeps the public interface (function signature, config names) so
> the rest of the pipeline stays readable, but importing it prints a
> notice and calling `llm_stopping_policy` raises `NotImplementedError`.
> The full implementation will be published in this repository once the
> accompanying paper is accepted.

## How it works

```
decode.py        SSVEP signal processing: filter-bank extended CCA
                  (FBECCA / eCCA, Nakanishi et al., 2018) against both
                  synthetic sine/cosine references and the subject's own
                  per-class calibration templates -- honest leave-one-
                  block-out cross-validation, one correlation-score
                  vector per class per "look".
fbecca/           The eCCA library (filter bank + template fitting +
                  multi-component CCA fusion) used by decode.py.
llm_policy.py     Everything about the LLM: the system prompt, the
                  OpenAI-compatible chat-completions call, and the
                  COMMIT/CONTINUE decision loop. This is the only file
                  you need to touch to change the prompt or point at a
                  different model/endpoint. Currently a placeholder --
                  see the notice above.
keyboard_map.py   class id <-> character lookup (cosmetic, for readable
                  candidate strings in the prompt).
run.py            Main entry point: decode -> LLM policy -> accuracy /
                  average looks used / information transfer rate (ITR).
```

## Setup

```bash
pip install -r requirements.txt
```

1. **Dataset**: download the public Benchmark SSVEP dataset (Wang et al.,
   2017) from http://bci.med.tsinghua.edu.cn/ and place `S1.mat` ...
   `S35.mat` under `./data/Benchmark/`. Not included here.
2. **Model**: serve any OpenAI-compatible chat-completions endpoint (we
   used a locally hosted Qwen2.5-7B-Instruct via
   [vLLM](https://github.com/vllm-project/vllm)) and point `API_URL` /
   `MODEL_NAME` in `llm_policy.py` at it. No model weights are included in
   this repo.

Note: `decode.py` needs each subject's own **labeled** trials to build the
per-class templates FBECCA scores against (that's what the leave-one-
block-out split is doing internally) — it is not calibration-free. A
calibration-free variant (plain FBCCA against synthetic references only,
no subject templates) is possible but needs a longer look-window to reach
comparable accuracy, since it loses the subject-specific template's extra
discriminative power; it is not included in this trimmed demo.

## Run

```bash
python run.py --subjects 1 8 15 --looks 4 --window 0.3
```
