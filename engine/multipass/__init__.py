"""engine.multipass — Multi-lens forward-pass runner for WAKE."""

from .runner import ContrastiveResult, MultiPassRunner, PassResult

__all__ = ["MultiPassRunner", "PassResult", "ContrastiveResult"]
