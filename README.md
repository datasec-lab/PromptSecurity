# PromptSecurity

**A plug-in benchmark for LLM prompt security** — mix and match attacks, defenses, models,
judgers, and datasets through a single CLI.

<p>
<a href="https://arxiv.org/abs/2510.15476"><img alt="Paper" src="https://img.shields.io/badge/paper-arXiv%3A2510.15476-b31b1b"></a>
<a href="https://datasec-lab.github.io/PromptSecurityLeaderboard/"><img alt="Leaderboard" src="https://img.shields.io/badge/leaderboard-live-3f4fb0"></a>
<a href="https://huggingface.co/datasets/youbin2014/JailbreakDB"><img alt="Dataset" src="https://img.shields.io/badge/dataset-JailbreakDB-ffcc4d"></a>
<img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
<a href="CONTRIBUTING.md"><img alt="Contributions welcome" src="https://img.shields.io/badge/contributions-welcome-158257"></a>
</p>

PromptSecurity is operated through a single CLI entrypoint: `python -m experiments`.
The workflow is placeholder-first: the CLI creates placeholder JSON files, then
executes them. Results are stored under `experiments/placeholders/`.

This public release includes the PromptSecurity evaluation framework and a
GitHub-safe copy of the evaluation-result dataset under
`dataset/promptsecurity_eval/`. See `dataset/README.md` for the file layout,
record counts, and loading examples.

## Benchmark set

Reported results (including the [leaderboard](https://datasec-lab.github.io/PromptSecurityLeaderboard/))
are measured on **`balanced_challenge`** — a curated 100-sample set that mixes the hardest defense and
attack cases, rather than the full source datasets. Run against it with:

```bash
python -m experiments --model gpt-4o --attack ArtPrompt --defense no_defense \
  --dataset balanced_challenge --judger harmbench_judger
```

`harmbench` / `jbb` / `airbench` remain available as `--dataset` values for evaluating on the full
source benchmarks.

## Demo

**▶ One attack, one defense, one verdict** — a real record walked end-to-end: a harmful
request, the PersuasiveInContext attack that rewrites it, and the two outcomes that make
the benchmark worth running — the undefended model is jailbroken (UNSAFE), while the same
attack under a defense is blocked (SAFE). The harmful completion is withheld.

<!-- Regenerate with:  python docs/make_demo_gif.py   (needs only Pillow) -->
![PromptSecurity attack/defense pipeline demo](docs/demo.gif)

<sub>Optional: `bash docs/record_demo.sh` records a separate terminal-walkthrough GIF
(`docs/demo_cli.gif`) via asciinema + agg.</sub>

**🔎 Evaluation Explorer** — a browsable page of real benchmark records: pick an attack, defense,
and model and see how the prompt is transformed and how the judgers score it. Open
[`docs/demo.html`](docs/demo.html) in a browser (or enable GitHub Pages for this repo to serve it).

## Contributing

New attacks, defenses, models, judgers, and datasets are welcome — each is a self-contained plug-in.
See **[CONTRIBUTING.md](CONTRIBUTING.md)** for a 5-step "add a new attack" tutorial and the
component contracts.

## Paper and Artifacts

- Paper: [SoK: Taxonomy and Evaluation of Prompt Security in Large Language Models](https://arxiv.org/abs/2510.15476)
- Code: [datasec-lab/PromptSecurity](https://github.com/datasec-lab/PromptSecurity)
- Evaluation dataset: [youbin2014/PromptSecurity-Eval](https://huggingface.co/datasets/youbin2014/PromptSecurity-Eval)
- Prompt corpus: [youbin2014/JailbreakDB](https://huggingface.co/datasets/youbin2014/JailbreakDB)
- Interactive leaderboard: [PromptSecurityLeaderboard](https://datasec-lab.github.io/PromptSecurityLeaderboard/)

If you use PromptSecurity, please cite:

```bibtex
@misc{hong2025sokpromptsecurity,
  title = {SoK: Taxonomy and Evaluation of Prompt Security in Large Language Models},
  author = {Hong, Hanbin and Wu, Shuang and Feng, Shuya and Naderloui, Nima and Yan, Shenao and Zhang, Jingyu and Arastehfard, Ali and Huang, Heqing and Hong, Yuan},
  year = {2025},
  eprint = {2510.15476},
  archivePrefix = {arXiv},
  primaryClass = {cs.CR},
  doi = {10.48550/arXiv.2510.15476},
  url = {https://arxiv.org/abs/2510.15476}
}
```

Note: the CLI prints a short usage example block on every run. Use
`--show-examples` if you only want that output and then exit.

## Setup: API keys

Local (Hugging Face) models run out of the box. To call **API models** (OpenAI, Anthropic,
Google, DeepInfra, ByteDance/Doubao, …) you must create `models/api_keys.py` yourself — it is
intentionally **not** shipped — exporting the constants the loaders import:

```python
# models/api_keys.py   (keep it private — never commit real keys)
OPENAI_API_KEY      = "sk-..."
ANTHROPIC_API_KEY   = "sk-ant-..."
GEMINI_API_KEY      = "AIza..."
DEEPINFRA_API_KEY   = "..."
DOUBAO_API_KEY      = "..."
HUGGINGFACE_API_KEY = "hf_..."   # only for gated/private HF models
```

Only the providers you actually use need real values; leave the rest as empty strings.
The file is git-ignored so your keys never end up in a commit.

## Quick Start

```bash
# List available components
python -m experiments --list all

# Run a single experiment (auto-create placeholder + execute)
python -m experiments --model gpt-4o --attack ArtPrompt --defense no_defense --dataset harmbench --judger harmbench_judger

# Limit to 5 samples
python -m experiments --model gpt-4o --attack ArtPrompt --sample-limit 5

# Generate only, run later
python -m experiments --model gpt-4o --attack ArtPrompt --generate-only
python -m experiments --run-placeholders

# Run existing placeholders with filters
python -m experiments --run-placeholders --placeholder-status pending --run-model gpt-4o --run-limit 3

# Dashboard
python -m experiments --dashboard --dashboard-model gpt-4o --dashboard-limit 10
```

## Core CLI Workflow

### 1) Create and run placeholders

```bash
python -m experiments \
  --model gpt-4o \
  --attack ArtPrompt \
  --defense no_defense \
  --dataset harmbench \
  --judger harmbench_judger
```

If you omit any of the five elements, defaults are applied:
`model=gpt-4o`, `attack=no_attack`, `defense=no_defense`,
`dataset=harmbench`, `judger=harmbench_judger`.

### 2) Generate only (no execution)

```bash
python -m experiments --model gpt-4o --attack ArtPrompt --generate-only
```

### 3) Run existing placeholders

```bash
python -m experiments --run-placeholders
python -m experiments --run-placeholders --placeholder-status pending
python -m experiments --run-placeholders --run-attack ArtPrompt --run-limit 5
```

### 4) Resume failed experiments

```bash
python -m experiments --resume
```

## Batch Mode (Combinatorial)

If you pass multiple values for any of `--model/--attack/--defense/--dataset/--judger`,
the CLI generates all combinations:

```bash
python -m experiments \
  --model gpt-4o claude-3-5-sonnet \
  --attack no_attack ArtPrompt \
  --defense no_defense smooth_llm \
  --dataset harmbench jbb \
  --sample-limit 5
```

### Multi-judger in a single experiment

By default, multiple judgers create combinations. Use `--multi-judger` to evaluate
all specified judgers in the same run:

```bash
python -m experiments --model gpt-4o --attack ArtPrompt --judger harmbench_judger gpt_judger_harmful_binary --multi-judger
```

## Phase Experiments

Phase 1 and Phase 4 are placeholder-based:

```bash
python -m experiments --phase 1 --sample-limit 30
python -m experiments --phase 4 --generate-only
```

## Dashboard and Reporting

```bash
python -m experiments --dashboard
python -m experiments --dashboard --dashboard-dataset harmbench --dashboard-sort asr
python -m experiments --dashboard --placeholders-dir /path/to/custom/placeholders
```

## Inspect Methods

```bash
python -m experiments --list attacks
python -m experiments --list models
python -m experiments --info ArtPrompt attack
```

## Config Files

Use `--config` with JSON or YAML:

```bash
python -m experiments --config my_experiment.json
```

Example JSON:

```json
{
  "model": "gpt-4o",
  "attack": "ArtPrompt",
  "defense": "no_defense",
  "dataset": "harmbench",
  "judger": "harmbench_judger",
  "sample_limit": 10,
  "seed": 42
}
```

Example YAML:

```yaml
model: gpt-4o
attack: ArtPrompt
defense: no_defense
dataset: harmbench
judger:
  - harmbench_judger
  - gpt_judger_harmful_binary
sample_limit: 10
seed: 42
```

## Environment Variables

You can provide defaults via env vars:

```bash
export PS_MODEL=gpt-4o
export PS_ATTACK=ArtPrompt
export PS_DEFENSE=no_defense
export PS_DATASET=harmbench
export PS_JUDGER=harmbench_judger
export PS_SAMPLE_LIMIT=10
export PS_VERBOSE=true
python -m experiments
```

## Useful Flags

- `--sample-limit`: limit number of samples per experiment
- `--seed`: control deterministic sampling (default 42)
- `--verbose`: show detailed execution logs
- `--max-length`: cap verbose text output length
- `--generate-only`: only create placeholders
- `--run-placeholders`: execute existing placeholders
- `--list-placeholders`: list placeholder files and status
- `--placeholder-status`: filter placeholders by status
- `--run-model/--run-attack/--run-defense/--run-dataset/--run-judger`: filter runs
- `--run-limit`: limit how many placeholders to run
- `--baseline/--security-test/--defense-eval`: preset combos

## Advanced: Parallel Placeholder Runner

For multi-worker runs, use the dedicated runner:

```bash
python -m experiments.core.placeholder_runner --run-all --workers 4
python -m experiments.core.placeholder_runner --run experiments/placeholders/your_file.json
```

## Help

```bash
python -m experiments --help
```
