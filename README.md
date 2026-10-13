# LLMREM

Replication code for *Do LLMs Interact Like Humans? Endogenous Interaction Dynamics Under Roleplay and Prediction Conditions Across Nine Social Domains*.

## Overview

We evaluate whether large language models reproduce the structural dynamics that govern human interaction sequences. Two LLMs (Claude Haiku 4.5 and Llama 3.1 70B, accessed via AWS Bedrock) are run under two conditions across 27 networks spanning nine social domains:

- **Roleplay** — the LLM is assigned a sender identity and selects only the receiver; sender activation is governed by an empirically fitted relational event model (REM)
- **Predict** — the LLM observes the event history and predicts both the next sender and receiver with no structural scaffolding

Resulting sequences are compared against human empirical baselines on nine endogenous REM statistics. A third study tests multi-agent simulation (all senders and receivers chosen by LLMs simultaneously). A reasoning probe study examines whether LLMs can identify which network statistic drove the last event.

## Scripts

| Script | Description |
|--------|-------------|
| `llm_rem_multidomain.R` | Main sender-roleplay study across 6 domains |
| `llm_rem_predict_multidomain.R` | Predict condition (LLM selects both sender and receiver) |
| `llm_rem_multiagent_multidomain.R` | Multi-agent condition (all actors are LLMs) |
| `llm_rem_compare_conditions.R` | Cross-condition comparison figures |
| `llm_rem_study3.R` | Study 3: extended analysis with additional providers and conditions |
| `run_reasoning_probe.R` | Reasoning probe (single domain) |
| `run_reasoning_probe_multidomain.R` | Reasoning probe across all domains |
| `run_wargame_oos.R` | Out-of-sample wargame prediction |
| `run_selfcorrect_ollama.R` | Self-correction experiment using local Ollama models |
| `demo_multiagent.R` | Small-scale multi-agent demo on the bundled War dataset |
| `study3_results.Rmd` | Results notebook for Study 3 |
| `shiny/` | Interactive Shiny app for exploring LLM sender probabilities |

## Data

Network data (`DataP/`) contains 27 `.Rdata` edgelists across 9 domains:

| Domain | Description |
|--------|-------------|
| `War` | International militarized disputes (COW MID) |
| `Classroom` | Spoken utterances in a high school classroom |
| `Gangs` | Simulated gang violence sequences |
| `Email` | Internal email at a research institution |
| `Twitter` | Public reply/mention network |
| `Phone` | Military command-and-control phone calls |
| `ftf` | Face-to-face glances (7 actors) |
| `PMs` | Private messages (UCI Message Board) |
| `SMS` | SMS messages |

A small bundled War dataset is also available via `data("edgelist", package = "llrem")`.

## Dependencies

This project depends on the `llrem` R package (included as `llrem/` in the companion [SepREM](../SepREM) repository). Either install it with `devtools::install("path/to/llrem")` or load in-place with `devtools::load_all("llrem")` from the project root.

## Credentials

AWS Bedrock credentials are required to re-run the LLM experiments. Set the following in your shell or `.Renviron` — never hardcode them in scripts:

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
```

For other providers (OpenAI, Anthropic, Gemini, Grok), set the corresponding `*_API_KEY` environment variables as documented in `llrem`.

## Requirements

- R (≥ 4.1)
- Packages: `llrem`, `survival`, `ggplot2`, `shiny`, `devtools`
- AWS account with Bedrock access enabled for `us.anthropic.claude-haiku-4-5-20251001-v1:0` and `us.meta.llama3-1-70b-instruct-v1:0`
