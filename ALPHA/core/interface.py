"""ALPHA core interface module."""

import array
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, cast

from .binary_foundation.base import Binary
from .execution.engine import ExecutionEngine
from .memory.space import MemoryOrganizer
from .patterns.neural_pattern import NeuralPattern
from .translation.translator import BinaryTranslator


def create_alpha(name: str = "alpha") -> "ALPHACore":
    """Create an ALPHA instance."""
    return ALPHACore(name)


class ALPHACore:
    """Core ALPHA system interface."""

    def __init__(self, name: str):
        """Initialize ALPHA core."""
        self.name = name
        self.memory = MemoryOrganizer()
        self.engine = ExecutionEngine()
        self.neural_pattern = NeuralPattern(name)
        self.translator = BinaryTranslator()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._io_queue: asyncio.Queue = asyncio.Queue()
        self.state: Dict[str, Any] = {
            "processed_count": 0,
            "success_rate": 0.0,
            "optimization_level": 0,
            "translations_performed": 0,
            "translation_success": 0,
            "io_pending": 0,
            "last_interaction": None,
        }

    async def start(self) -> None:
        """Start the interface system."""
        try:
            # Start memory monitoring
            await self.memory.start_monitoring()

            # Start I/O processing loop
            asyncio.create_task(self._process_io_queue())

            self.state["status"] = "running"
        except Exception as e:
            self.state["status"] = "error"
            self.state["last_error"] = str(e)
            raise

    async def stop(self) -> None:
        """Stop the interface system."""
        try:
            # Stop memory monitoring
            await self.memory.stop_monitoring()

            # Clear I/O queue
            while not self._io_queue.empty():
                await self._io_queue.get()

            self.state["status"] = "stopped"
        except Exception as e:
            self.state["status"] = "error"
            self.state["last_error"] = str(e)
            raise

    async def process(self, input_data: Any, priority: int = 0) -> Dict[str, Any]:
        """Process input data through ALPHA system asynchronously."""
        try:
            # Add to I/O queue with priority
            future = asyncio.Future()
            await self._io_queue.put((priority, input_data, future))
            self.state["io_pending"] += 1

            # Wait for processing to complete
            result = await future
            self.state["io_pending"] -= 1
            return result

        except Exception as e:
            self.state["last_error"] = str(e)
            if "translations_performed" in self.state:
                self.state["translations_performed"] += 1
            raise

    async def _process_io_queue(self) -> None:
        """Process I/O queue continuously."""
        while True:
            try:
                # Get next item from queue
                priority, input_data, future = await self._io_queue.get()

                # Process in thread pool
                result = await asyncio.get_event_loop().run_in_executor(
                    self._executor, self._process_single_input, input_data
                )

                # Set result
                future.set_result(result)

            except Exception as e:
                self.state["last_error"] = str(e)
            finally:
                self._io_queue.task_done()

    def _process_single_input(self, input_data: Any) -> Dict[str, Any]:
        """Process a single input item."""
        try:
            # Prepare input
            binary_data = self._prepare_input(input_data)

            # Perform translation
            self.translator.set_binary(binary_data)
            translated = self.translator.translate_from_binary()
            if translated is not None:
                self.state["translations_performed"] += 1
                self.state["translation_success"] += 1

            # Analyze with neural pattern
            self.neural_pattern.analyze_component(binary_data.to_bytes())

            # Store in memory with monitoring
            reference = f"{self.name}_data_{self.state['processed_count']}"
            if not self.memory.allocate(binary_data.to_bytes(), reference):
                raise RuntimeError("Failed to allocate memory")

            # Execute and get results
            exec_result = self.engine.execute(binary_data.to_bytes())

            # Update state and convert result
            result_dict = {
                "success": exec_result.success,
                "output": exec_result.output,
                "metrics": exec_result.metrics,
                "translation": translated,
                "memory_usage": self.memory.get_usage_stats(),
            }
            self._update_state(result_dict)

            return result_dict

        except Exception as e:
            self.state["last_error"] = str(e)
            raise

    def clear_memory(self) -> None:
        """Clear system memory and reset relevant state."""
        # Clear all memory blocks in the organizer
        for block in self.memory.blocks:
            block.clear()
        # Reset memory-related state
        self.state["processed_count"] = 0
        self.state["translations_performed"] = 0
        self.state["translation_success"] = 0
        # Clear neural pattern memory
        self.neural_pattern.learned_patterns.clear()
        self.neural_pattern.pattern_confidence.clear()

    def optimize(self) -> Dict[str, Any]:
        """Run optimization cycle."""
        try:
            # Get stored patterns
            patterns = []
            for ref in self.memory.reference_map:
                data_list = self.memory.read(ref)
                for data in data_list:
                    patterns.append(data)

            if not patterns:
                return {"success": False, "message": "No patterns to optimize"}

            # Learn from patterns
            for pattern in patterns:
                self.neural_pattern.learn_component_behavior(pattern)

            # Get improvement suggestions
            suggestions = []
            for pattern in patterns:
                pattern_suggestions = self.neural_pattern.suggest_improvements(pattern)
                suggestions.extend(pattern_suggestions)

            # Update optimization level
            if suggestions:
                self.state["optimization_level"] += 1

            return {
                "success": True,
                "suggestions": suggestions,
                "optimization_level": self.state["optimization_level"],
            }

        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_state(self) -> Dict[str, Any]:
        """Get current system state."""
        return self.state.copy()

    def analyze_patterns(self, content: str) -> List[Dict[str, Any]]:
        """Analyze patterns in the given content.

        Args:
            content: The content to analyze.

        Returns:
            A list of identified patterns with their metadata.
        """
        try:
            # Convert content to binary for neural pattern analysis
            binary_data = Binary(content.encode("utf-8"))

            # Analyze with neural pattern and get signature
            signature = self.neural_pattern.analyze_component(binary_data.to_bytes())

            # Convert patterns to list of dictionaries
            pattern_list = []

            # Add input patterns
            for pattern in signature.input_patterns:
                pattern_list.append(
                    {
                        "type": "input",
                        "name": pattern,
                        "confidence": 0.8,
                        "metadata": {"category": "input"},
                    }
                )

            # Add output patterns
            for pattern in signature.output_patterns:
                pattern_list.append(
                    {
                        "type": "output",
                        "name": pattern,
                        "confidence": 0.8,
                        "metadata": {"category": "output"},
                    }
                )

            # Add interaction patterns
            for pattern in signature.interaction_patterns:
                pattern_list.append(
                    {
                        "type": "interaction",
                        "name": pattern,
                        "confidence": 0.8,
                        "metadata": {"category": "interaction"},
                    }
                )

            # Add role confidence patterns
            roles = signature.role_confidence.items()
            for role, confidence in roles:
                if confidence > 0.1:  # Only include significant roles
                    metrics = signature.performance_metrics
                    pattern_list.append(
                        {
                            "type": "role",
                            "name": role.value,
                            "confidence": confidence,
                            "metadata": {
                                "category": "role",
                                "metrics": metrics,
                            },
                        }
                    )

            return pattern_list

        except Exception as e:
            self.state["last_error"] = str(e)
            return []

    async def get_confidence_score(self) -> float:
        """Get the current confidence score for pattern analysis.

        Returns:
            A float between 0 and 1 indicating confidence level.
        """
        try:
            # Calculate confidence using neural pattern analysis
            raw_score = self.neural_pattern.analyze_component(b"confidence_check")
            patterns = cast(List[Any], raw_score)

            # Calculate average confidence from pattern scores
            confidences = [getattr(p, "confidence", 0.0) for p in patterns]
            return float(sum(confidences) / max(len(confidences), 1))
        except Exception as e:
            self.state["last_error"] = str(e)
            return 0.0

    def _prepare_input(self, input_data: Any) -> Binary:
        """Prepare input data for processing."""
        if isinstance(input_data, Binary):
            return input_data
        elif isinstance(input_data, (str, bytes)):
            return Binary(input_data.encode("utf-8") if isinstance(input_data, str) else input_data)
        elif isinstance(input_data, (list, array.array)):
            return Binary(bytes(input_data))
        else:
            msg = f"Unsupported input type: {type(input_data)}"
            raise ValueError(msg)

    def _update_state(self, result: Dict[str, Any]) -> None:
        """Update internal state based on processing results."""
        self.state["processed_count"] += 1
        if result.get("success", False):
            current_rate = self.state["success_rate"]
            total = self.state["processed_count"]
            self.state["success_rate"] = (current_rate * (total - 1) + 1) / total
