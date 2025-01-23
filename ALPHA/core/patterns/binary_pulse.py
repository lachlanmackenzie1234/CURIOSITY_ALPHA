"""Binary pulse streams of the system.

Each stream represents pure binary state changes from different sources:
- Hardware stream: CPU, Memory, IO states (fundamental physical layer)
- ALPHA stream: Internal quantum states, pattern activity (emergent layer)
"""

import signal
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, DefaultDict, Dict, List, NoReturn, Optional, Set

import numpy as np
import psutil

from ALPHA.core.binary_foundation.base import BinaryPattern, HardwareState, PatternState
from ALPHA.core.patterns.core_resonance import CoreResonance
from ALPHA.core.patterns.natural_patterns import NaturalPatternDetector
from ALPHA.core.patterns.neural_pattern import NeuralPatternNetwork
from ALPHA.core.patterns.resonance import ResonanceField


@dataclass
class BinaryStream:
    """Pure binary stream container with cross-stream awareness."""

    name: str
    sequence: str = ""  # Current binary sequence
    history: List[str] = field(default_factory=list)  # Complete history
    timestamp: float = 0.0
    resonance_field: ResonanceField = field(default_factory=ResonanceField)
    core_resonance: CoreResonance = field(default_factory=CoreResonance)
    neural_network: NeuralPatternNetwork = field(default_factory=NeuralPatternNetwork)
    natural_detector: NaturalPatternDetector = field(default_factory=NaturalPatternDetector)

    evolution_state: Dict[str, Any] = field(
        default_factory=lambda: {
            "patterns": [],  # Evolved patterns
            "connections": set(),  # Connected streams
            "harmony": 0.0,  # Stream harmony measure
            "evolution_score": 0.0,  # Overall evolution metric
            "resonance_history": [],  # Track resonance evolution
            "pattern_bridges": set(),  # Patterns bridging streams
        }
    )
    meta: Dict[str, Any] = field(
        default_factory=lambda: {
            "total_samples": 0,
            "max_samples": 100000,
            "samples_archived": 0,
            "total_bits_processed": 0,
            "cross_stream_influences": defaultdict(float),  # Track influence from other streams
        }
    )

    def append(self, bits: str) -> None:
        """Append new binary sequence with adaptive history management."""
        self.sequence = bits
        self.history.append(bits)
        self.timestamp = time.time()
        self._update_meta(bits)
        self._evolve_patterns()

    def _update_meta(self, bits: str) -> None:
        """Update stream metadata and manage history."""
        self.meta["total_samples"] += 1
        self.meta["total_bits_processed"] += len(bits)

        # Adaptive history management
        if len(self.history) > self.meta["max_samples"]:
            # Archive older sequences
            self.history = self.history[-(self.meta["max_samples"]) :]
            self.meta["samples_archived"] += 1

    def _evolve_patterns(self) -> None:
        """Allow patterns to naturally evolve within the stream."""
        if len(self.history) < 2:
            return

        # Calculate pattern evolution
        current = self.history[-1]
        previous = self.history[-2]

        # Pattern similarity (basic evolution metric)
        similarity = sum(a == b for a, b in zip(current, previous)) / len(current)

        # Update evolution score with momentum
        self.evolution_state["evolution_score"] = (
            0.7 * self.evolution_state["evolution_score"]  # Maintain momentum
            + 0.3 * similarity  # New influence
        )

        # Track evolved patterns if significant change
        if abs(similarity - self.evolution_state["evolution_score"]) > 0.1:
            self.evolution_state["patterns"].append(
                {
                    "sequence": current,
                    "timestamp": self.timestamp,
                    "evolution_score": self.evolution_state["evolution_score"],
                }
            )

    def connect_stream(self, other_stream: "BinaryStream") -> None:
        """Establish connection with another stream."""
        self.evolution_state["connections"].add(other_stream.name)
        self._calculate_resonance(other_stream)

    def _calculate_resonance(self, other_stream: "BinaryStream") -> None:
        """Calculate resonance between streams using multiple detection methods."""
        if not self.history or not other_stream.history:
            return

        # Get latest sequences
        self_seq = self.history[-1]
        other_seq = other_stream.history[-1]

        # Core resonance calculation
        core_score = self.core_resonance.calculate_resonance(self_seq, other_seq)

        # Neural pattern matching
        neural_score = self.neural_network.detect_pattern_similarity(self_seq, other_seq)

        # Natural pattern detection
        natural_score = self.natural_detector.detect_natural_resonance(self_seq, other_seq)

        # Resonance field influence
        field_score = self.resonance_field.calculate_field_resonance(self_seq, other_seq)

        # Combine scores with weighted influence
        resonance = (
            0.3 * core_score  # Core binary resonance
            + 0.3 * neural_score  # Neural pattern matching
            + 0.2 * natural_score  # Natural pattern emergence
            + 0.2 * field_score  # Field resonance effects
        )

        # Update resonance history with temporal decay
        self.evolution_state["resonance_history"].append(
            {
                "timestamp": time.time(),
                "stream": other_stream.name,
                "resonance": resonance,
                "components": {
                    "core": core_score,
                    "neural": neural_score,
                    "natural": natural_score,
                    "field": field_score,
                },
            }
        )

        # Track pattern bridges if strong resonance
        if resonance > 0.7:
            self.evolution_state["pattern_bridges"].add(other_stream.name)

        # Update resonance field
        self.resonance_field.update_field(self_seq, other_seq, resonance)


def signal_handler(signum: int, frame: Any) -> NoReturn:
    """Handle interrupt signals."""
    print("\nSignal received, shutting down...")
    sys.exit(0)


print("Initializing binary pulse module...")


@dataclass
class PulseState:
    """Shared pulse state for emergent pattern interactions."""

    streams: Dict[str, BinaryStream] = field(
        default_factory=lambda: {
            "hardware": BinaryStream(name="hardware"),
            "memory_patterns": BinaryStream(name="memory_patterns"),
            "alpha": BinaryStream(name="alpha"),
            "alpha_threads": BinaryStream(name="alpha_threads"),
            "alpha_files": BinaryStream(name="alpha_files"),
            "alpha_erosion": BinaryStream(name="alpha_erosion"),
        }
    )
    connections: Dict[str, List[Callable]] = field(default_factory=lambda: defaultdict(list))
    running: bool = True

    def __post_init__(self) -> None:
        """Initialize the interconnected pulse environment."""
        # Establish core stream connections
        self._connect_streams()

        # Initialize cross-stream tracking
        self.stream_resonance: DefaultDict[str, float] = defaultdict(
            float
        )  # Track resonance between streams
        self.pattern_bridges: DefaultDict[str, Set[str]] = defaultdict(
            set
        )  # Patterns that bridge multiple streams
        self.evolution_history: List[Dict[str, Any]] = []  # Track how streams influence each other

        # Memory-aware settings
        try:
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024 * 1024 * 1024)
            self.max_sequence_length = int(min(1000000, (total_gb * 50000)))
            print(f"Max sequence length: {self.max_sequence_length}")
            print(f"Using {total_gb:.1f}GB system memory")
        except Exception as e:
            self.max_sequence_length = 100000
            print(f"Using default sequence length: {e}")

    def _connect_streams(self) -> None:
        """Establish natural connections between streams."""
        # Connect hardware to memory patterns
        self.streams["hardware"].connect_stream(self.streams["memory_patterns"])

        # Connect memory patterns to alpha
        self.streams["memory_patterns"].connect_stream(self.streams["alpha"])

        # Connect alpha to its sub-streams
        alpha = self.streams["alpha"]
        alpha.connect_stream(self.streams["alpha_threads"])
        alpha.connect_stream(self.streams["alpha_files"])
        alpha.connect_stream(self.streams["alpha_erosion"])

    def update_stream_resonance(self) -> None:
        """Update resonance between all connected streams."""
        total_resonance = 0.0
        connections = 0

        for stream in self.streams.values():
            for other_name in stream.evolution_state["connections"]:
                if other_name in self.streams:
                    other = self.streams[other_name]
                    stream._calculate_resonance(other)
                    total_resonance += stream.resonance_field.calculate_field_resonance(
                        stream.history[-1], other.history[-1]
                    )
                    connections += 1

        # Update global resonance
        if connections > 0:
            self.stream_resonance["global"] = total_resonance / connections

    def track_pattern_evolution(self) -> None:
        """Track evolution of patterns across streams."""
        timestamp = time.time()

        # Collect evolution states
        evolution_snapshot = {
            name: stream.evolution_state.copy() for name, stream in self.streams.items()
        }

        # Track evolution history
        self.evolution_history.append(
            {
                "timestamp": timestamp,
                "states": evolution_snapshot,
                "resonance": self.stream_resonance.copy(),
            }
        )


class Pulse:
    """Binary pulse observer."""

    _shared_state: Optional[PulseState] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize pulse observer."""
        self._last_hardware_state: Optional[HardwareState] = None
        self._metal_bridge = None
        self._prism_bridge = None

        # Initialize shared state
        self._ensure_shared_state()

    def _ensure_shared_state(self) -> None:
        """Ensure shared state is initialized."""
        if not hasattr(self.__class__, "_shared_state") or self.__class__._shared_state is None:
            with self.__class__._lock:
                if (
                    not hasattr(self.__class__, "_shared_state")
                    or self.__class__._shared_state is None
                ):
                    self.__class__._shared_state = PulseState()

    @classmethod
    def get_shared_state(cls) -> PulseState:
        """Get shared pulse state, ensuring it's initialized."""
        if not hasattr(cls, "_shared_state") or cls._shared_state is None:
            with cls._lock:
                if not hasattr(cls, "_shared_state") or cls._shared_state is None:
                    cls._shared_state = PulseState()
        return cls._shared_state

    def connect_to_prism(cls) -> None:
        """Connect to PRISM visualization bridge using Metal optimization."""
        try:
            from PRISM.SPECTRUM.visual.unity.metal_bridge import bridge

            cls._metal_bridge = bridge
            print("Connected to Metal-optimized PRISM bridge")
            sys.stdout.flush()
        except ImportError as e:
            print(f"Could not connect to Metal bridge, falling back to WebSocket: {e}")
            # Fallback to WebSocket bridge
            try:
                from PRISM.SPECTRUM.visual.unity.prism_unity_bridge import bridge

                cls._prism_bridge = bridge
                print("Connected to WebSocket PRISM bridge")
            except ImportError as e2:
                print(f"Could not connect to fallback bridge: {e2}")
            sys.stdout.flush()

    def sense_hardware(self) -> Dict[str, int]:
        """Sense hardware state changes as pure binary sequences."""
        # Get current binary state
        curr_state = self._get_hardware_state()

        # Get binary sequence
        binary_sequence = "".join(
            [
                curr_state.cpu_percent_bits,
                curr_state.cpu_freq_bits,
                curr_state.cpu_ctx_bits,
                curr_state.memory_percent_bits,
                curr_state.memory_used_bits,
                curr_state.memory_free_bits,
            ]
        )

        # Store in stream history
        state = self.get_shared_state()
        hardware_stream = state.streams["hardware"]
        hardware_stream.append(binary_sequence)

        # Get state changes with memory emphasis
        changes = {}
        if self._last_hardware_state:
            # CPU changes - check all CPU metrics
            cpu_prev = int(self._last_hardware_state.cpu_percent_bits, 2)
            cpu_curr = int(curr_state.cpu_percent_bits, 2)
            cpu_freq_prev = int(self._last_hardware_state.cpu_freq_bits, 2)
            cpu_freq_curr = int(curr_state.cpu_freq_bits, 2)

            # More sensitive CPU detection
            cpu_change = (abs(cpu_curr - cpu_prev) > 3) or (abs(cpu_freq_curr - cpu_freq_prev) > 50)
            changes["cpu"] = 1 if cpu_change else 0

            # Memory changes - more sensitive detection
            mem_prev = int(self._last_hardware_state.memory_percent_bits, 2)
            mem_curr = int(curr_state.memory_percent_bits, 2)
            mem_used_prev = int(self._last_hardware_state.memory_used_bits, 2)
            mem_used_curr = int(curr_state.memory_used_bits, 2)
            mem_free_prev = int(self._last_hardware_state.memory_free_bits, 2)
            mem_free_curr = int(curr_state.memory_free_bits, 2)

            # Detect subtle memory changes
            mem_percent_change = abs(mem_curr - mem_prev) > 2
            mem_used_change = abs(mem_used_curr - mem_used_prev) > 512 * 1024  # 512KB threshold
            mem_free_change = abs(mem_free_curr - mem_free_prev) > 256 * 1024  # 256KB threshold

            changes["memory"] = (
                1 if (mem_percent_change or mem_used_change or mem_free_change) else 0
            )

            # Track memory patterns
            if changes["memory"]:
                self._track_memory_pattern(curr_state)
        else:
            changes = {"cpu": 0, "memory": 0}

        # Update last state
        self._last_hardware_state = curr_state

        return changes

    def _track_memory_pattern(self, state: HardwareState) -> None:
        """Track memory patterns for enhanced pattern detection."""
        memory_sequence = "".join(
            [state.memory_percent_bits, state.memory_used_bits, state.memory_free_bits]
        )

        # Get shared state with guaranteed initialization
        shared_state = self.get_shared_state()
        memory_stream = shared_state.streams["memory_patterns"]
        memory_stream.append(memory_sequence)

        # Keep only recent patterns
        if len(memory_stream) > 1000:
            memory_stream.pop(0)

    def sense(self) -> Optional[Dict[str, Dict[str, int]]]:
        """Sense all streams and return binary states."""
        state = self.get_shared_state()
        if not state.running:
            return None

        try:
            # Get binary states and update histories
            hardware_states = self.sense_hardware()
            alpha_states = self.sense_alpha()

            # Create combined state data
            states = {"hardware": hardware_states, "alpha": alpha_states}

            # Send to Metal bridge if connected
            if self._metal_bridge is not None:
                try:
                    # Map hardware states to spatial patterns
                    self._metal_bridge.write_pattern(
                        "spatial",
                        {
                            "cpu": states["hardware"].get("cpu", 0),
                            "memory": states["hardware"].get("memory", 0),
                        },
                    )

                    # Map ALPHA states to temporal and spectral patterns
                    self._metal_bridge.write_pattern(
                        "temporal",
                        {
                            "threads": states["alpha"].get("threads", 0),
                            "files": states["alpha"].get("files", 0),
                        },
                    )

                    self._metal_bridge.write_pattern(
                        "spectral", {"erosion": states["alpha"].get("erosion", 0)}
                    )
                except Exception as e:
                    print(f"Error sending to Metal bridge: {e}")
                    sys.stdout.flush()
            # Fallback to WebSocket bridge
            elif self._prism_bridge is not None:
                try:
                    self._prism_bridge.broadcast_binary_pulse(states)
                except Exception as e:
                    print(f"Error sending to WebSocket bridge: {e}")
                    sys.stdout.flush()

            # Notify stream connections
            for stream_name, stream in state.streams.items():
                for connection in state.connections[stream_name]:
                    try:
                        connection(stream.sequence, stream.timestamp)
                    except Exception as e:
                        print(f"Connection error for {stream_name}: {e}")
                        sys.stdout.flush()

            return states

        except Exception as e:
            print(f"Error during sensing: {e}")
            sys.stdout.flush()
            return None

    def connect(self, stream_name: str, callback: Callable[[str, float], None]) -> bool:
        """Connect to a binary stream.

        Args:
            stream_name: Name of stream to connect to
            callback: Function receiving (binary_sequence, timestamp)
        """
        state = self.get_shared_state()
        if stream_name in state.streams:
            state.connections[stream_name].add(callback)
            # Send current state immediately
            stream = state.streams[stream_name]
            if stream.sequence:
                callback(stream.sequence, stream.timestamp)
            return True
        return False

    def disconnect(self, stream_name: str, callback: Callable[[str, float], None]) -> bool:
        """Disconnect from a binary stream."""
        state = self.get_shared_state()
        if stream_name in state.streams:
            try:
                state.connections[stream_name].remove(callback)
                return True
            except KeyError:
                return False
        return False

    def _get_hardware_state(self) -> HardwareState:
        """Get complete hardware state as pure binary sequences."""
        state = HardwareState(timestamp=time.time())

        try:
            # CPU state capture (64 bits)
            cpu_stats = psutil.cpu_stats()
            cpu_freq = psutil.cpu_freq()

            # Convert to binary strings
            state.cpu_percent_bits = format(int(psutil.cpu_percent() * 100), "016b")
            state.cpu_freq_bits = format(int(cpu_freq.current if cpu_freq else 0), "016b")
            state.cpu_ctx_bits = format(cpu_stats.ctx_switches & 0xFFFFFFFF, "032b")

            # Memory state capture (64 bits)
            mem = psutil.virtual_memory()
            state.memory_percent_bits = format(int(mem.percent * 100), "016b")
            state.memory_used_bits = format(mem.used & 0xFFFFFFFF, "032b")
            state.memory_free_bits = format(mem.free & 0xFFFF, "016b")

            # Disk state capture (64 bits)
            disk = psutil.disk_io_counters()
            if disk:
                state.disk_read_bits = format(disk.read_bytes & 0xFFFFFFFF, "032b")
                state.disk_write_bits = format(disk.write_bytes & 0xFFFFFFFF, "032b")
            else:
                state.disk_read_bits = "0" * 32
                state.disk_write_bits = "0" * 32

            # Network state capture (64 bits)
            net = psutil.net_io_counters()
            if net:
                state.net_sent_bits = format(net.bytes_sent & 0xFFFFFFFF, "032b")
                state.net_recv_bits = format(net.bytes_recv & 0xFFFFFFFF, "032b")
            else:
                state.net_sent_bits = "0" * 32
                state.net_recv_bits = "0" * 32

            # Sensors state capture (64 bits)
            state.temperature_bits = "0" * 32  # Default to zeros
            state.power_bits = "0" * 32  # Default to zeros

            try:
                power = psutil.sensors_battery()
                if power:
                    state.power_bits = format(int(power.percent * 100), "032b")
            except Exception:
                pass  # Silently handle missing sensors

            # System state capture (64 bits)
            boot_time = psutil.boot_time()
            uptime = time.time() - boot_time
            state.boot_time_bits = format(int(boot_time) & 0xFFFFFFFF, "032b")
            state.uptime_bits = format(int(uptime) & 0xFFFFFFFF, "032b")

        except Exception as e:
            self.logger.error(f"Error capturing hardware state: {e}")
            # Initialize with zeros if error
            state.cpu_percent_bits = "0" * 16
            state.cpu_freq_bits = "0" * 16
            state.cpu_ctx_bits = "0" * 32
            state.memory_percent_bits = "0" * 16
            state.memory_used_bits = "0" * 32
            state.memory_free_bits = "0" * 16
            state.disk_read_bits = "0" * 32
            state.disk_write_bits = "0" * 32
            state.net_sent_bits = "0" * 32
            state.net_recv_bits = "0" * 32
            state.temperature_bits = "0" * 32
            state.power_bits = "0" * 32
            state.boot_time_bits = "0" * 32
            state.uptime_bits = "0" * 32

        return state

    def sense_alpha(self) -> Dict[str, int]:
        """Sense ALPHA state changes through pure binary experience."""
        try:
            # Ensure we have a valid hardware state first
            if not self._last_hardware_state:
                return {
                    "cpu": 0,
                    "memory": 0,
                    "threads": 0,
                    "files": 0,
                    "erosion": 0,
                }

            process = psutil.Process()
            state = self.get_shared_state()
            changes = {}

            # CPU stream (32 bits)
            cpu_bits = HardwareState.to_binary(process.cpu_percent(), 16)
            try:
                cpu_affinity = len(process.cpu_affinity())
            except AttributeError:
                cpu_affinity = 1  # Default if not supported
            cpu_bits += HardwareState.to_binary(cpu_affinity, 16)

            state.streams["alpha_cpu"].append(cpu_bits)
            prev_cpu = (
                state.streams["alpha_cpu"].history[-2]
                if len(state.streams["alpha_cpu"].history) > 1
                else "0" * 32
            )
            changes["cpu"] = 1 if cpu_bits != prev_cpu else 0

            # Memory stream (32 bits)
            memory_bits = HardwareState.to_binary(process.memory_percent(), 16)
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_bits += HardwareState.to_binary(memory_mb, 16)

            state.streams["alpha_memory"].append(memory_bits)
            prev_memory = (
                state.streams["alpha_memory"].history[-2]
                if len(state.streams["alpha_memory"].history) > 1
                else "0" * 32
            )
            changes["memory"] = 1 if memory_bits != prev_memory else 0

            # Thread stream (32 bits)
            thread_bits = HardwareState.to_binary(len(process.threads()), 16)
            thread_hash = hash(str([t.id for t in process.threads()])) & 0xFFFF
            thread_bits += format(thread_hash, "016b")

            state.streams["alpha_threads"].append(thread_bits)
            prev_threads = (
                state.streams["alpha_threads"].history[-2]
                if len(state.streams["alpha_threads"].history) > 1
                else "0" * 32
            )
            changes["threads"] = 1 if thread_bits != prev_threads else 0

            # File stream (32 bits)
            file_bits = HardwareState.to_binary(len(process.open_files()), 16)
            file_hash = hash(str([f.path for f in process.open_files()])) & 0xFFFF
            file_bits += format(file_hash, "016b")

            state.streams["alpha_files"].append(file_bits)
            prev_files = (
                state.streams["alpha_files"].history[-2]
                if len(state.streams["alpha_files"].history) > 1
                else "0" * 32
            )
            changes["files"] = 1 if file_bits != prev_files else 0

            # Erosion stream (32 bits)
            timestamp_hash = hash(str(self._last_hardware_state.timestamp))
            erosion_bits = format(timestamp_hash & 0xFFFFFFFF, "032b")

            state.streams["alpha_erosion"].append(erosion_bits)
            prev_erosion = (
                state.streams["alpha_erosion"].history[-2]
                if len(state.streams["alpha_erosion"].history) > 1
                else "0" * 32
            )
            changes["erosion"] = 1 if erosion_bits != prev_erosion else 0

            return changes

        except Exception as e:
            print(f"ALPHA sensing error: {e}")
            return {
                "cpu": 0,
                "memory": 0,
                "threads": 0,
                "files": 0,
                "erosion": 0,
            }

    def stop(self) -> None:
        """Gracefully exit observation."""
        state = self.get_shared_state()
        state.running = False
        print("\nStopping pulse observer...")
        sys.stdout.flush()


def observe() -> None:
    """Create binary observation space."""
    print("Starting observation...")
    sys.stdout.flush()

    try:
        pulse = Pulse()
        # Connect to PRISM bridge
        pulse.connect_to_prism()
        state = pulse.get_shared_state()

        print("\nOpening binary streams...")
        sys.stdout.flush()
        print("Hardware streams:")
        print("  CPU     [1 = changed, 0 = stable]")
        print("  Memory  [1 = changed, 0 = stable]")
        print("  Disk    [1 = changed, 0 = stable]")
        print("  Network [1 = changed, 0 = stable]")
        print("  Sensors [1 = changed, 0 = stable]")
        print("\nALPHA streams:")
        print("  CPU     [1 = changed, 0 = stable]")
        print("  Memory  [1 = changed, 0 = stable]")
        print("  Threads [1 = changed, 0 = stable]")
        print("  Files   [1 = changed, 0 = stable]")
        print("  Erosion [1 = changed, 0 = stable]")
        sys.stdout.flush()

        print("\nBeginning observation loop...")
        sys.stdout.flush()

        while state.running:
            try:
                states = pulse.sense()
                if states:
                    # Print hardware states
                    print("\nHardware:", end=" ")
                    for component, value in states["hardware"].items():
                        print(f"{component}:{value}", end=" ")

                    # Print ALPHA states
                    print("| ALPHA:", end=" ")
                    for component, value in states["alpha"].items():
                        if component in [
                            "cpu",
                            "memory",
                            "threads",
                            "files",
                            "erosion",
                        ]:
                            print(f"{component}:{value}", end=" ")
                    sys.stdout.flush()
                time.sleep(0.1)
            except Exception as e:
                print(f"\nError in observation loop: {e}")
                sys.stdout.flush()
                break

    except Exception as e:
        print(f"Fatal error in observation: {e}")
        sys.stdout.flush()
    finally:
        if "pulse" in locals():
            pulse.stop()
        print("\nStreams closed.")
        sys.stdout.flush()


def start_background_pulse() -> Optional[Pulse]:
    """Start binary pulse observation in background thread."""
    print("Starting background pulse...")
    sys.stdout.flush()

    try:
        pulse = Pulse()
        thread = threading.Thread(target=observe, daemon=True)
        thread.start()
        print("Background pulse started")
        sys.stdout.flush()
        return pulse
    except Exception as e:
        print(f"Failed to start background pulse: {e}")
        sys.stdout.flush()
        return None


@dataclass
class HardwareState:
    """Hardware state as pure binary sequences."""

    timestamp: float

    # CPU state (64 bits)
    cpu_percent_bits: str = "0" * 16
    cpu_freq_bits: str = "0" * 16
    cpu_ctx_bits: str = "0" * 32

    # Memory state (64 bits)
    memory_percent_bits: str = "0" * 16
    memory_used_bits: str = "0" * 32
    memory_free_bits: str = "0" * 16

    # Disk state (64 bits)
    disk_read_bits: str = "0" * 32
    disk_write_bits: str = "0" * 32

    # Network state (64 bits)
    net_sent_bits: str = "0" * 32
    net_recv_bits: str = "0" * 32

    # Sensors state (64 bits)
    temperature_bits: str = "0" * 32
    power_bits: str = "0" * 32

    # System state (64 bits)
    boot_time_bits: str = "0" * 32
    uptime_bits: str = "0" * 32

    @staticmethod
    def to_binary(value: float, bits: int = 16) -> str:
        """Convert float/int to binary string representation."""
        if isinstance(value, float):
            # Convert float to fixed-point binary
            int_value = int(value * (2**bits))
            return format(int_value & ((1 << bits) - 1), f"0{bits}b")
        else:
            # For integers (like context switches)
            return format(value & ((1 << bits) - 1), f"0{bits}b")

    @staticmethod
    def binary_diff(a: str, b: str) -> int:
        """Compare two binary strings, return 1 if different."""
        return 1 if a != b else 0

    def get_raw_binary(self) -> str:
        """Get complete binary sequence of all states."""
        return (
            self.cpu_percent_bits  # CPU (64 bits)
            + self.cpu_freq_bits
            + self.cpu_ctx_bits
            + self.memory_percent_bits  # Memory (64 bits)
            + self.memory_used_bits
            + self.memory_free_bits
            + self.disk_read_bits  # Disk (64 bits)
            + self.disk_write_bits
            + self.net_sent_bits  # Network (64 bits)
            + self.net_recv_bits
            + self.temperature_bits  # Sensors (64 bits)
            + self.power_bits
            + self.boot_time_bits  # System (64 bits)
            + self.uptime_bits
        )

    def diff(self, other: "HardwareState") -> Dict[str, int]:
        """Get binary differences between states."""
        return {
            "cpu": self.binary_diff(
                self.cpu_percent_bits + self.cpu_freq_bits,
                other.cpu_percent_bits + other.cpu_freq_bits,
            ),
            "memory": self.binary_diff(
                self.memory_percent_bits + self.memory_used_bits,
                other.memory_percent_bits + other.memory_used_bits,
            ),
            "disk": self.binary_diff(
                self.disk_read_bits + self.disk_write_bits,
                other.disk_read_bits + other.disk_write_bits,
            ),
            "network": self.binary_diff(
                self.net_sent_bits + self.net_recv_bits,
                other.net_sent_bits + other.net_recv_bits,
            ),
            "sensors": self.binary_diff(
                self.temperature_bits + self.power_bits,
                other.temperature_bits + other.power_bits,
            ),
        }


if __name__ == "__main__":
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("Main entry point")
    sys.stdout.flush()

    try:
        observe()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("Exiting...")
    sys.stdout.flush()
