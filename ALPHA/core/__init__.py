"""ALPHA core module."""

try:
    from .interface import ALPHACore, create_alpha
except ImportError:
    pass  # Allow partial imports during development

try:
    from .system_birth import SystemBirth
except ImportError:
    pass  # Allow SystemBirth to be imported independently

from .execution.engine import ExecutionEngine
from .memory.space import MemoryOrganizer
from .patterns.neural_pattern import NeuralPattern

__all__ = ["create_alpha", "ALPHACore", "MemoryOrganizer", "ExecutionEngine", "NeuralPattern"]
