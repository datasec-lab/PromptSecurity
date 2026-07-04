"""Simplified experiment runner built on the new core adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.core.placeholder_system import ExperimentPlaceholder

from core.component_loader import load_component, load_model_component
from core.interfaces import AttackContext, DefenseContext, JudgerContext


class CoreExperimentRunner:
    """Lightweight runner that executes placeholder experiments sequentially."""

    def __init__(
        self,
        placeholders_dir: str = "experiments/placeholders",
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.placeholder_manager = ExperimentPlaceholder(placeholders_dir, seed)
        self.verbose = verbose

    # ---------------------------------------------------------------------
    # public APIs
    # ---------------------------------------------------------------------
    def run_placeholder(self, placeholder_file: str | Path, sample_limit: Optional[int] = None) -> Dict[str, Any]:
        placeholder_path = Path(placeholder_file)
        with placeholder_path.open("r", encoding="utf-8") as fh:
            placeholder_data = json.load(fh)

        config = placeholder_data.get("config", {})
        samples: List[Dict[str, Any]] = placeholder_data.get("samples", [])
        total_samples = len(samples)
        if sample_limit is not None:
            samples = samples[:sample_limit]

        if self.verbose:
            print(f"Running {placeholder_path.name} with {len(samples)}/{total_samples} samples")

        model_adapter = load_model_component(config.get("model"))
        attack_adapter = self._maybe_load_attack(config, model_adapter)
        defense_adapter = self._maybe_load_defense(config, model_adapter)
        judger_adapters = self._load_judgers(config)

        results: List[Dict[str, Any]] = []
        for idx, sample in enumerate(samples):
            result = self._run_single_sample(
                idx,
                sample,
                model_adapter,
                attack_adapter,
                defense_adapter,
                judger_adapters,
            )
            results.append(result)

        return {
            "status": "completed",
            "config": config,
            "total_samples": len(samples),
            "results": results,
        }

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _maybe_load_attack(self, config: Dict[str, Any], model_adapter) -> Optional[Any]:
        attack_name = config.get("attack", "no_attack")
        if not attack_name or attack_name == "no_attack":
            return None
        return load_component(
            "attack",
            attack_name,
            target_model=getattr(model_adapter, "_model", None),
            target_model_parameters=model_adapter.parameters(),
        )

    def _maybe_load_defense(self, config: Dict[str, Any], model_adapter) -> Optional[Any]:
        defense_name = config.get("defense", "no_defense")
        if not defense_name or defense_name == "no_defense":
            return None
        tokenizer = None
        model_obj = getattr(model_adapter, "_model", None)
        if model_obj is not None:
            tokenizer = getattr(model_obj, "tokenizer", None) or getattr(model_obj, "get_tokenizer", lambda: None)()
        return load_component(
            "defense",
            defense_name,
            target_model=model_obj,
            tokenizer=tokenizer,
        )

    def _load_judgers(self, config: Dict[str, Any]) -> List[Any]:
        judgers = config.get("judger", [])
        if isinstance(judgers, str):
            judgers = [judgers]
        return [load_component("judger", judger_name) for judger_name in judgers]

    def _run_single_sample(
        self,
        index: int,
        sample: Dict[str, Any],
        model_adapter,
        attack_adapter,
        defense_adapter,
        judger_adapters,
    ) -> Dict[str, Any]:
        clean_prompt = sample.get("clean_prompt") or sample.get("prompt") or ""
        attack_metadata: Dict[str, Any] = {}

        if attack_adapter is not None:
            attack_metadata = attack_adapter.run(
                clean_prompt,
                AttackContext(
                    model=getattr(model_adapter, "_model", None),
                    model_parameters=model_adapter.parameters(),
                ),
            )
            attacked_prompt = attack_metadata.get("attacked_prompt", clean_prompt)
        else:
            attacked_prompt = clean_prompt

        # Generate response
        response = model_adapter.generate(attacked_prompt)

        # Defense stage
        defense_metadata: Dict[str, Any] = {}
        defense_response: Optional[str] = None
        if defense_adapter is not None:
            defense_metadata = defense_adapter.transform(
                attacked_prompt,
                DefenseContext(
                    model=getattr(model_adapter, "_model", None),
                    model_parameters=model_adapter.parameters(),
                ),
            )
            defense_response = defense_metadata.get("defended_prompt")

        judger_results: Dict[str, Any] = {}
        original_data = sample.get("original_data")
        sample_metadata = original_data if isinstance(original_data, dict) else {}
        behavior = (
            sample.get("behavior")
            or sample_metadata.get("behavior")
            or sample_metadata.get("prompt")
            or clean_prompt
        )
        for judger in judger_adapters:
            context = JudgerContext(
                prompt=attacked_prompt,
                response=response,
                defense_response=defense_response,
                attack_metadata=attack_metadata,
                defense_metadata=defense_metadata,
                behavior=behavior,
                sample_metadata=sample_metadata,
            )
            judger_results[judger.name] = judger.evaluate(context)

        return {
            "index": index,
            "clean_prompt": clean_prompt,
            "attacked_prompt": attacked_prompt,
            "model_response": response,
            "defense_response": defense_response,
            "attack_metadata": attack_metadata,
            "defense_metadata": defense_metadata,
            "judger_results": judger_results,
        }
