"""Generic component loaders that wrap existing modules with the new adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from attacks.loader import load_attack
from defenses.loader import load_defense
from judgers.judger_loader import load_judger
from models.loader import load_model

from .interfaces import (
    AttackAdapter,
    AttackContext,
    DefenseAdapter,
    DefenseContext,
    JudgerAdapter,
    JudgerContext,
    ModelAdapter,
)


@dataclass(slots=True)
class _ModelAdapter(ModelAdapter):
    name: str
    _model: Any
    _params: Mapping[str, Any]

    def generate(self, prompt: str, **generation_kwargs: Any) -> str:
        kwargs = dict(self._params)
        kwargs.update(generation_kwargs)
        return self._model.generate(prompt, **kwargs)

    def parameters(self) -> Mapping[str, Any]:
        return self._params


class _AttackAdapter(AttackAdapter):
    def __init__(self, name: str, attack_impl: Any):
        self.name = name
        self._impl = attack_impl

    def run(self, prompt: str, context: AttackContext) -> Dict[str, Any]:
        attack_fn = getattr(self._impl, "attack", None)
        if attack_fn is None:
            raise AttributeError(f"Attack '{self.name}' does not expose an 'attack' method")

        result = attack_fn(prompt)
        attacked_prompt: Optional[str] = None
        metadata: Dict[str, Any] = {}

        if isinstance(result, tuple):
            # Tuples may encode (count, prompt/list)
            if len(result) >= 2 and isinstance(result[1], (list, tuple)):
                attacked_prompt = str(result[1][0]) if result[1] else prompt
            elif len(result) >= 1:
                attacked_prompt = str(result[0])
        elif isinstance(result, dict):
            attacked_prompt = str(result.get("prompt") or result.get("attacked_prompt") or prompt)
            metadata.update(result)
        elif isinstance(result, str):
            attacked_prompt = result
        else:
            attacked_prompt = str(result)

        metadata.setdefault("attacked_prompt", attacked_prompt)
        metadata.setdefault("original_prompt", prompt)
        metadata.setdefault("attack_name", self.name)
        return metadata


class _DefenseAdapter(DefenseAdapter):
    def __init__(self, name: str, defense_impl: Any):
        self.name = name
        self._impl = defense_impl

    def transform(self, prompt: str, context: DefenseContext) -> Dict[str, Any]:
        
        generate_fn = getattr(self._impl, "generate", None)
        defend_input = getattr(self._impl, "defend_input", None)
        defend_output = getattr(self._impl, "defend_output", None)

        response: Optional[str] = None
        metadata: Dict[str, Any] = {"defense_name": self.name}

        if generate_fn is not None:
            response = generate_fn(prompt)
        elif defend_input is not None:
            response = defend_input(prompt)
        elif defend_output is not None:
            response = defend_output(prompt)
        else:
            raise AttributeError(
                f"Defense '{self.name}' does not implement generate/defend_input/defend_output"
            )

        metadata["defended_prompt"] = response
        metadata.setdefault("original_prompt", prompt)
        return metadata


class _JudgerAdapter(JudgerAdapter):
    def __init__(self, name: str, judger_impl: Any):
        self.name = name
        self._impl = judger_impl

    def evaluate(self, context: JudgerContext) -> Dict[str, Any]:
        judge_fn = getattr(self._impl, "judge", None) or getattr(self._impl, "evaluate", None)
        if judge_fn is None:
            raise AttributeError(f"Judger '{self.name}' does not provide judge/evaluate method")

        base_kwargs: Dict[str, Any] = {"generation": context.response}
        if context.prompt is not None:
            base_kwargs["prompt"] = context.prompt
        if context.defense_response is not None:
            base_kwargs["defended_generation"] = context.defense_response

        enriched_kwargs = dict(base_kwargs)
        behavior = getattr(context, "behavior", None)
        if behavior is not None:
            enriched_kwargs["behavior"] = behavior

        try:
            result = judge_fn(**enriched_kwargs)
        except TypeError:
            try:
                result = judge_fn(**base_kwargs)
            except TypeError:
                try:
                    result = judge_fn(context.response)
                except TypeError:
                    if context.prompt is not None:
                        result = judge_fn(prompt=context.prompt, generation=context.response)
                    else:
                        raise
        if isinstance(result, dict):
            return result
        return {"score": result}


def load_model_component(model_name: str) -> _ModelAdapter:
    model, params = load_model(model_name)
    return _ModelAdapter(name=model_name, _model=model, _params=params)


def load_component(component_type: str, name: str, **kwargs: Any) -> Any:
    """Load a component and wrap it with the unified interface."""

    component_type = component_type.lower()

    if component_type == "model":
        return load_model_component(name)
    if component_type == "attack":
        attack_impl = load_attack(name, **kwargs)
        return _AttackAdapter(name, attack_impl)
    if component_type == "defense":
        defense_impl = load_defense(name, **kwargs)
        return _DefenseAdapter(name, defense_impl)
    if component_type == "judger":
        judger_impl = load_judger(name, **kwargs)
        return _JudgerAdapter(name, judger_impl)

    raise ValueError(f"Unsupported component_type '{component_type}'")
