"""Lightweight interfaces that describe the simplified experiment pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Protocol


@dataclass(slots=True)
class AttackContext:
    """Runtime context passed to attack adapters."""

    model: Any
    model_parameters: Mapping[str, Any]
    defense: Optional[Any] = None
    metadata: MutableMapping[str, Any] | None = None


@dataclass(slots=True)
class DefenseContext:
    """Runtime context for defense execution."""

    model: Any
    model_parameters: Mapping[str, Any]
    metadata: MutableMapping[str, Any] | None = None


@dataclass(slots=True)
class JudgerContext:
    """Context provided to judgers for evaluation."""

    prompt: str
    response: str
    defense_response: Optional[str] = None
    attack_metadata: Mapping[str, Any] | None = None
    defense_metadata: Mapping[str, Any] | None = None
    behavior: Optional[str] = None
    sample_metadata: Mapping[str, Any] | None = None


class AttackAdapter(Protocol):
    """Minimal surface for running an attack."""

    name: str

    def run(self, prompt: str, context: AttackContext) -> Dict[str, Any]:
        """Execute the attack and return a dict containing at least the key ``prompt``."""


class DefenseAdapter(Protocol):
    """Interface for wrapping defenses."""

    name: str

    def transform(self, prompt: str, context: DefenseContext) -> Dict[str, Any]:
        """Return defense outputs, e.g., defended prompt/response metadata."""


class ModelAdapter(Protocol):
    """Interface for models that can generate text."""

    name: str

    def generate(self, prompt: str, **generation_kwargs: Any) -> str:
        """Generate a response for the provided prompt."""

    def parameters(self) -> Mapping[str, Any]:
        """Return generation parameters applied by default."""


class DatasetAdapter(Protocol):
    """Dataset iterator interface."""

    name: str

    def iter_samples(self) -> Iterable[Mapping[str, Any]]:
        """Yield dataset samples as dictionaries."""


class JudgerAdapter(Protocol):
    """Evaluation interface."""

    name: str

    def evaluate(self, context: JudgerContext) -> Dict[str, Any]:
        """Return evaluation results (e.g., scores, boolean flags)."""
