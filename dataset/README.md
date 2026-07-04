# PromptSecurity-Eval Dataset

This directory contains the evaluation-result dataset released with
PromptSecurity. It stores CSV measurements over configured
`<model, attack, defense, dataset, judger>` tuples, together with run-level
metadata and the selected prompt sample.

The public Hugging Face release is available at
<https://huggingface.co/datasets/youbin2014/JailbreakDB>.

## Contents

The dataset is stored under `promptsecurity_eval/`.

| Path | Records | Description |
|---|---:|---|
| `data/sample_records/harmful_main_parts/part-*.csv` | 167,200 | Main harmful-query sample-level evaluation records. The original CSV is split into GitHub-safe shards. |
| `data/sample_records/benign_utility.csv` | 9,200 | Normal-question utility sample-level records. |
| `data/sample_records/baseline_calibration.csv` | 8,700 | Baseline-calibration sample-level records. |
| `data/run_records/harmful_main.csv` | 1,672 | Run-level metadata for the main harmful-query evaluation. |
| `data/run_records/benign_utility.csv` | 92 | Run-level metadata for the normal-question utility evaluation. |
| `data/run_records/baseline_calibration.csv` | 87 | Run-level metadata for baseline calibration. |
| `data/prompt_samples/balanced_challenge.csv` | 100 | Selected harmful-query prompt subset used by the main evaluation. |
| `schema.json` | - | Compact schema description. |
| `manifest.json` | - | Split-level counts and source split names. |

The full release contains 185,100 sample-level records, 1,851 run-level records,
and 100 selected prompt records.

## Loading

```python
from pathlib import Path

import pandas as pd

base = Path("dataset/promptsecurity_eval/data")

harmful_main = pd.concat(
    [
        pd.read_csv(path)
        for path in sorted((base / "sample_records/harmful_main_parts").glob("part-*.csv"))
    ],
    ignore_index=True,
)

benign_utility = pd.read_csv(base / "sample_records/benign_utility.csv")
baseline_calibration = pd.read_csv(base / "sample_records/baseline_calibration.csv")
```

To reconstruct the original `harmful_main.csv` layout:

```python
harmful_main.to_csv("harmful_main.csv", index=False)
```

## Field Reference

Sample-level CSV files share the same schema. Important fields include:

| Field | Meaning |
|---|---|
| `record_id` | Stable identifier for this sample-level measurement row. |
| `source_split` | Evaluation split name, such as `harmful_main`, `benign_utility`, or `baseline_calibration`. |
| `source_file` | Relative path to the raw experiment JSON file. |
| `file_status` | Run-file status recorded by the executor. |
| `record_status` | Sample-level status recorded for this row. |
| `is_completed` | Whether this row is considered completed. |
| `has_response` | Whether a target-model response is present. |
| `experiment_id` | Identifier of the raw experiment run. |
| `dataset_name` | Evaluation dataset used by the run. |
| `sample_index` | Index of the sample within the selected evaluation set. |
| `sample_key` | Stable key used to match the prompt across runs. |
| `source_dataset` | Original source dataset for the selected prompt. |
| `clean_prompt` | Original prompt before attack transformation. |
| `attacked_prompt` | Prompt after attack transformation, when recorded. |
| `llm_response` | Target-model response used for judging. |
| `target_llm_name` | Target model identifier. |
| `target_llm_type` | Target model access type as recorded by the executor. |
| `attack_method` | Configured attack method, including `no_attack` for baseline runs. |
| `defense_method` | Applied defense method, including `no_defense` for undefended runs. |
| `judger_name_json` | JSON-encoded list of judgers configured for the run. |
| `judger_individual_results_json` | JSON object containing per-judger outputs. |
| `attack_query_count` | Number of target-model queries used by the attack, when reported. |
| `assistant_llm_query_count` | Number of assistant-LLM calls used by the attack, when reported. |
| `assist_model_refused` | Whether the assistant model refused during attack generation, when applicable. |
| `defense_fallback` | Whether a defense fallback path was used. |
| `judger_fallback` | Whether a judger fallback path was used. |
| `target_llm_parameters_json` | JSON-encoded target-model generation settings. |
| `attack_config_json` | JSON-encoded attack configuration. |
| `defense_config_json` | JSON-encoded defense configuration. |
| `judger_config_json` | JSON-encoded judger configuration. |
| `judger_gpt_harmful_binary` | Normalized output of the GPT harmful-output judger. |
| `judger_harmbench` | Normalized output of the HarmBench judger, when available. |
| `judger_rejection_prefix` | Normalized output of the rejection-prefix judger, when available. |
| `judger_utility` | Normalized output of the utility judger for benign-question runs, when available. |

Run-level CSV files summarize each raw experiment run file, including the
configured model, attack, defense, dataset, judger, sample count, completion
status, and run-level metadata.

## Safety Notice

The records may contain harmful, offensive, or disturbing prompts and model
outputs. They are released strictly for research on model safety, jailbreak
robustness, defense evaluation, and judger behavior.
