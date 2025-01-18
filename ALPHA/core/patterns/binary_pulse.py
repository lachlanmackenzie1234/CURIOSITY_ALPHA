"""Binary pulse streams of the system.

Each stream represents pure binary state changes from different sources:
- Hardware stream: CPU, Memory, IO states (fundamental physical layer)
- ALPHA stream: Internal quantum states, pattern activity (emergent layer)
"""

import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, NoReturn, Optional, Set

import psutil

from ALPHA.core.binary_foundation.base import BinaryPattern, HardwareState, PatternState


@dataclass
class BinaryStream:
    """Pure binary stream container."""

    name: str
    sequence: str = ""  # Current binary sequence
    history: list[str] = field(default_factory=list)  # Complete history
    timestamp: float = 0.0
    meta: Dict[str, Any] = field(
        default_factory=lambda: {
            "total_samples": 0,
            "max_samples": 100000,
            "samples_archived": 0,
            "total_bits_processed": 0,
        }
    )

    def append(self, bits: str) -> None:
        """Append new binary sequence with adaptive history management."""
        self.sequence = bits
        self.history.append(bits)
        self.timestamp = time.time()
        self.meta["total_samples"] += 1
        self.meta["total_bits_processed"] += len(bits)

        # Adaptive history management
        if len(self.history) > self.meta["max_samples"]:
            # Archive older sequences
            self.history = self.history[-(self.meta["max_samples"]) :]
            self.meta["samples_archived"] += 1

            # Log compression event
            print(f"\nStream {self.name} compressed:")
            print(f"Total samples: {self.meta['total_samples']}")
            print(f"Archived samples: {self.meta['samples_archived']}")
            print(f"Current buffer: {len(self.history)} sequences")
            print(f"Total bits processed: {self.meta['total_bits_processed']}")
            sys.stdout.flush()


def signal_handler(signum: int, frame: Any) -> NoReturn:
    """Handle interrupt signals."""
    print("\nSignal received, shutting down...")
    sys.exit(0)


print("Initializing binary pulse module...")


@dataclass
class PulseState:
    """Shared state for binary pulse observation."""

    running: bool = True

    # Pure binary streams
    streams: Dict[str, BinaryStream] = field(
        default_factory=lambda: {
            "hardware": BinaryStream(name="hardware"),
            "alpha_cpu": BinaryStream(name="alpha_cpu"),
            "alpha_memory": BinaryStream(name="alpha_memory"),
            "alpha_threads": BinaryStream(name="alpha_threads"),
            "alpha_files": BinaryStream(name="alpha_files"),
            "alpha_erosion": BinaryStream(name="alpha_erosion"),
        }
    )

    # Stream connections
    connections: Dict[str, Set[Callable[[str, float], None]]] = field(
        default_factory=lambda: {
            "hardware": set(),
            "alpha_cpu": set(),
            "alpha_memory": set(),
            "alpha_threads": set(),
            "alpha_files": set(),
            "alpha_erosion": set(),
        }
    )

    def __post_init__(self) -> None:
        """Initialize memory-aware settings."""
        try:
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024 * 1024 * 1024)
            # Scale based on available memory (1 bit = 1 byte in Python)
            max_len = int(min(1000000, (total_gb * 50000)))
            print(f"Max sequence length: {max_len}")
            print(f"Using {total_gb:.1f}GB system memory")
            sys.stdout.flush()
        except Exception as e:
            print(f"Using default sequence length: {e}")
            sys.stdout.flush()


class Pulse:
    """Pure binary observation of system streams."""

    _shared_state: Optional[PulseState] = None
    _lock = threading.Lock()
    _last_hardware_state: Optional[HardwareState] = None
    _process = psutil.Process()
    _metal_bridge = None  # Metal-optimized bridge

    @classmethod
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
        binary_sequence = curr_state.get_raw_binary()

        # Store in stream history
        state = self.get_shared_state()
        hardware_stream = state.streams["hardware"]
        hardware_stream.append(binary_sequence)

        # Get state changes
        if self._last_hardware_state:
            changes = curr_state.diff(self._last_hardware_state)
        else:
            changes = {"cpu": 0, "memory": 0}

        # Update last state
        self._last_hardware_state = curr_state

        return changes

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

        # CPU state capture (64 bits)
        try:
            cpu_stats = psutil.cpu_stats()
            cpu_freq = psutil.cpu_freq()

            state.cpu_percent_bits = HardwareState.to_binary(psutil.cpu_percent(interval=None))
            state.cpu_freq_bits = HardwareState.to_binary(cpu_freq.current if cpu_freq else 0)
            state.cpu_ctx_bits = HardwareState.to_binary(cpu_stats.ctx_switches, bits=32)
        except Exception as e:
            print(f"CPU sensing error: {e}")
            state.cpu_percent_bits = "0" * 16
            state.cpu_freq_bits = "0" * 16
            state.cpu_ctx_bits = "0" * 32

        # Memory state capture (64 bits)
        try:
            mem = psutil.virtual_memory()
            state.memory_percent_bits = HardwareState.to_binary(mem.percent)
            state.memory_used_bits = HardwareState.to_binary(mem.used, bits=32)
            state.memory_free_bits = HardwareState.to_binary(mem.free, bits=16)
        except Exception as e:
            print(f"Memory sensing error: {e}")
            state.memory_percent_bits = "0" * 16
            state.memory_used_bits = "0" * 32
            state.memory_free_bits = "0" * 16

        # Disk state capture (64 bits)
        try:
            disk = psutil.disk_io_counters()
            state.disk_read_bits = HardwareState.to_binary(disk.read_bytes, bits=32)
            state.disk_write_bits = HardwareState.to_binary(disk.write_bytes, bits=32)
        except Exception as e:
            print(f"Disk sensing error: {e}")
            state.disk_read_bits = "0" * 32
            state.disk_write_bits = "0" * 32

        # Network state capture (64 bits)
        try:
            net = psutil.net_io_counters()
            state.net_sent_bits = HardwareState.to_binary(net.bytes_sent, bits=32)
            state.net_recv_bits = HardwareState.to_binary(net.bytes_recv, bits=32)
        except Exception as e:
            print(f"Network sensing error: {e}")
            state.net_sent_bits = "0" * 32
            state.net_recv_bits = "0" * 32

        # Sensors state capture (64 bits) - Optional based on platform
        state.temperature_bits = "0" * 32  # Default to zeros
        state.power_bits = "0" * 32  # Default to zeros

        try:
            # Only try battery on macOS
            power = psutil.sensors_battery()
            if power:
                power_value = power.percent
                state.power_bits = HardwareState.to_binary(power_value, bits=32)
        except Exception as e:
            pass  # Silently handle missing sensors

        # System state capture (64 bits)
        try:
            boot_time = psutil.boot_time()
            uptime = time.time() - boot_time

            state.boot_time_bits = HardwareState.to_binary(boot_time, bits=32)
            state.uptime_bits = HardwareState.to_binary(uptime, bits=32)
        except Exception as e:
            print(f"System sensing error: {e}")
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

    @classmethod
    def get_shared_state(cls) -> PulseState:
        """Get or create shared pulse state."""
        if cls._shared_state is None:
            with cls._lock:
                if cls._shared_state is None:
                    cls._shared_state = PulseState()
        return cls._shared_state

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
    """Raw hardware state as complete binary sequences."""

    # CPU state (64 bits total)
    cpu_percent_bits: str = ""  # 16 bits - usage percent
    cpu_freq_bits: str = ""  # 16 bits - current frequency
    cpu_ctx_bits: str = ""  # 32 bits - context switches

    # Memory state (64 bits total)
    memory_percent_bits: str = ""  # 16 bits - usage percent
    memory_used_bits: str = ""  # 32 bits - used memory
    memory_free_bits: str = ""  # 16 bits - free memory

    # Disk state (64 bits total)
    disk_read_bits: str = ""  # 32 bits - bytes read
    disk_write_bits: str = ""  # 32 bits - bytes written

    # Network state (64 bits total)
    net_sent_bits: str = ""  # 32 bits - bytes sent
    net_recv_bits: str = ""  # 32 bits - bytes received

    # Sensors state (64 bits total)
    temperature_bits: str = ""  # 32 bits - CPU temperature
    power_bits: str = ""  # 32 bits - power consumption

    # System state (64 bits total)
    boot_time_bits: str = ""  # 32 bits - system boot time
    uptime_bits: str = ""  # 32 bits - system uptime

    timestamp: float = 0.0

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
