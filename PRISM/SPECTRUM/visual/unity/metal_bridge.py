"""Metal-accelerated bridge for PRISM visualization on Apple Silicon."""

import json
import mmap
import os
import signal
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Gracefully handle Metal import
try:
    import Metal
    import MetalKit

    METAL_AVAILABLE = True
except ImportError:
    print("Metal framework not available - using CPU fallback mode")
    METAL_AVAILABLE = False
    Metal = None
    MetalKit = None


@dataclass
class MetalPattern:
    """Hardware-optimized pattern container."""

    pattern_type: str
    timestamp: float
    data: np.ndarray
    metal_buffer: Optional[Any] = None  # Metal.MTLBuffer when available
    transition_state: float = 0.0


@dataclass
class BirthState:
    """Track birth visualization state."""

    phase: float = 0.0
    last_update: float = 0.0
    pattern_data: List[int] = field(default_factory=list)
    pattern_type: str = ""
    is_transitioning: bool = False
    transition_progress: float = 0.0


class MetalBridge:
    """Direct memory bridge optimized for Apple Silicon."""

    # Pattern type mappings
    PATTERN_TYPES = {
        "ignition": 1,
        "sustain": 2,
        "essence": 3,
        "shutdown": 0,
        "spatial": 4,
        "temporal": 5,
        "spectral": 6,
    }

    def __init__(self, buffer_size: int = 1024 * 1024):  # 1MB default
        """Initialize Metal bridge with optional buffer size."""
        # Metal setup
        self.device = None
        self.command_queue = None
        self.birth_pipeline = None
        self.pattern_pipeline = None

        # State tracking
        self.birth_state = BirthState()
        self.last_pattern: Optional[MetalPattern] = None
        self._state_path = Path("PRISM/SPECTRUM/state/bridge_state.json")
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        # Threading and memory
        self.lock = threading.Lock()
        self.buffer_size = buffer_size
        self.mm_file = mmap.mmap(-1, buffer_size)

        # Initialize Metal if available
        if METAL_AVAILABLE:
            try:
                self.device = Metal.MTLCreateSystemDefaultDevice()
                if self.device:
                    self.command_queue = self.device.newCommandQueue()
                    self._setup_metal_pipelines()
                    self._load_preserved_state()
                    self._register_shutdown_handler()
            except Exception as e:
                print(f"Metal initialization error: {e}")
                self.device = None

    def _register_shutdown_handler(self) -> None:
        """Register graceful shutdown handler."""
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """Handle graceful shutdown."""
        print("\nPreserving visualization state...")
        self._preserve_state()
        if self.last_pattern:
            self.write_pattern("shutdown", self.last_pattern.data)
        print("Bridge state preserved successfully")
        os._exit(0)

    def _preserve_state(self) -> None:
        """Preserve current bridge state."""
        try:
            state_data = {
                "birth_state": {
                    "phase": self.birth_state.phase,
                    "pattern_type": self.birth_state.pattern_type,
                    "transition_progress": self.birth_state.transition_progress,
                },
                "last_pattern": (
                    {
                        "type": self.last_pattern.pattern_type,
                        "timestamp": self.last_pattern.timestamp,
                        "transition": self.last_pattern.transition_state,
                    }
                    if self.last_pattern
                    else None
                ),
            }

            temp_path = self._state_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(state_data, f, indent=2)
            temp_path.replace(self._state_path)
        except Exception as e:
            print(f"Error preserving bridge state: {e}")

    def _load_preserved_state(self) -> None:
        """Load preserved bridge state if available."""
        try:
            if self._state_path.exists():
                with open(self._state_path) as f:
                    state_data = json.load(f)

                if "birth_state" in state_data:
                    birth = state_data["birth_state"]
                    self.birth_state.phase = birth["phase"]
                    self.birth_state.pattern_type = birth["pattern_type"]
                    self.birth_state.transition_progress = birth["transition_progress"]

                print(f"Restored bridge state - Birth phase: {self.birth_state.phase}")
        except Exception as e:
            print(f"Error loading preserved state: {e}")

    def _setup_metal_pipelines(self) -> None:
        """Initialize Metal compute pipelines."""
        if not self.device:
            return

        # Combined kernel source for all pattern processing
        kernel_source = """
        #include <metal_stdlib>
        using namespace metal;

        struct PatternData {
            float4 data;
            int type;
            float timestamp;
            float phase;
            float transition;
        };

        // Birth process visualization functions
        float4 process_birth_ignition(float4 data, float time, float phase, float transition) {
            float3 center = float3(0.5, 0.5, 0.0);
            float spark = pow(phase * (1.0 - transition), 2.0);
            float3 direction = normalize(data.xyz - center);
            float distance = length(data.xyz - center);
            float wave = sin(distance * 10.0 - time * 2.0) * 0.5 + 0.5;
            return float4(direction * wave * spark, data.w * phase);
        }

        float4 process_birth_sustain(float4 data, float time, float phase, float transition) {
            float phi = 1.618033988749895;  // Golden ratio
            float angle = atan2(data.y, data.x);
            float radius = length(data.xyz);
            float spiral = angle + radius * phi + time;
            float flow = (sin(spiral) * 0.5 + 0.5) * phase * (1.0 - transition);
            return float4(data.xyz * flow, data.w * phase);
        }

        float4 process_birth_essence(float4 data, float time, float phase, float transition) {
            float3 center = float3(0.5, 0.5, 0.0);
            float distance = length(data.xyz - center);
            float crystal = pow(1.0 - distance, 2.0) * phase * (1.0 - transition);
            float pulse = (sin(time * 3.0) * 0.5 + 0.5) * phase;
            return float4(lerp(data.xyz, center, crystal), pulse * data.w);
        }

        // Pattern type visualization functions
        float4 process_spatial_pattern(float4 data, float time, float phase) {
            float3 position = data.xyz;
            float frequency = data.w;
            float3 wave = sin(position * frequency + time);
            return float4(wave, frequency);
        }

        float4 process_temporal_pattern(float4 data, float time, float phase) {
            float3 position = data.xyz;
            float timeScale = data.w;
            float3 temporal = cos(position + time * timeScale);
            return float4(temporal, timeScale);
        }

        float4 process_spectral_pattern(float4 data, float time, float phase) {
            float3 spectrum = data.xyz;
            float intensity = data.w;
            float3 spectral = spectrum * sin(time * intensity);
            return float4(spectral, intensity);
        }

        float4 process_shutdown(float4 data, float time, float transition) {
            float fade = 1.0 - transition;
            return data * fade;
        }

        kernel void process_patterns(
            device const PatternData* patterns [[buffer(0)]],
            device float4* output [[buffer(1)]],
            constant float& time [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            PatternData pattern = patterns[id];
            float4 processed;

            switch(pattern.type) {
                case 0:  // Shutdown
                    processed = process_shutdown(pattern.data, time, pattern.transition);
                    break;
                case 1:  // Birth Ignition
                    processed = process_birth_ignition(pattern.data, time, pattern.phase, pattern.transition);
                    break;
                case 2:  // Birth Sustain
                    processed = process_birth_sustain(pattern.data, time, pattern.phase, pattern.transition);
                    break;
                case 3:  // Birth Essence
                    processed = process_birth_essence(pattern.data, time, pattern.phase, pattern.transition);
                    break;
                case 4:  // Spatial
                    processed = process_spatial_pattern(pattern.data, time, pattern.phase);
                    break;
                case 5:  // Temporal
                    processed = process_temporal_pattern(pattern.data, time, pattern.phase);
                    break;
                case 6:  // Spectral
                    processed = process_spectral_pattern(pattern.data, time, pattern.phase);
                    break;
                default:
                    processed = pattern.data;
            }

            output[id] = processed;
        }
        """

        try:
            library = self.device.newLibraryWithSource_options_error_(
                kernel_source, Metal.MTLCompileOptions.alloc().init(), None
            )
            self.pattern_pipeline = library.newFunctionWithName_("process_patterns")
        except Exception as e:
            print(f"Error setting up Metal pipeline: {e}")

    def write_pattern(self, pattern_type: str, data: np.ndarray, phase: float = 1.0) -> None:
        """Write and process pattern data using Metal acceleration."""
        if not data.size:
            return

        with self.lock:
            try:
                if METAL_AVAILABLE and self.device and self.pattern_pipeline:
                    self._process_metal_pattern(pattern_type, data, phase)
                else:
                    self._process_cpu_fallback(pattern_type, data, phase)
            except Exception as e:
                print(f"Error processing pattern: {e}")
                self._process_cpu_fallback(pattern_type, data, phase)

    def write_birth_pattern(self, pattern_data: Dict) -> None:
        """Process and visualize birth pattern."""
        if not pattern_data.get("data"):
            return

        with self.lock:
            # Update birth state
            self.birth_state.phase = pattern_data.get("birth_phase", 0.0)
            self.birth_state.pattern_type = pattern_data.get("type", "")
            self.birth_state.pattern_data = pattern_data["data"]
            self.birth_state.last_update = time.time()

            # Handle transitions
            if pattern_data.get("type") == "shutdown":
                self.birth_state.is_transitioning = True
                self.birth_state.transition_progress = 0.0

            # Process pattern
            self.write_pattern(
                pattern_data["type"],
                np.array(pattern_data["data"], dtype=np.float32),
                self.birth_state.phase,
            )

    def _process_metal_pattern(self, pattern_type: str, data: np.ndarray, phase: float) -> None:
        """Process pattern using Metal acceleration."""
        if not self.device or not self.pattern_pipeline:
            return

        # Prepare pattern data
        pattern_buffer = self.device.newBuffer(
            data=data.tobytes(), length=data.nbytes, options=Metal.MTLResourceStorageModeShared
        )

        # Create output buffer
        output_buffer = self.device.newBuffer(
            length=data.nbytes, options=Metal.MTLResourceStorageModeShared
        )

        # Process with Metal
        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()

        # Set up compute pipeline
        compute_encoder.setComputePipelineState_(
            self.device.newComputePipelineStateWithFunction_error_(self.pattern_pipeline, None)
        )

        # Set buffers and parameters
        compute_encoder.setBuffer_offset_atIndex_(pattern_buffer, 0, 0)
        compute_encoder.setBuffer_offset_atIndex_(output_buffer, 0, 1)

        # Current time for animation
        current_time = np.array([time.time()], dtype=np.float32)
        compute_encoder.setBytes_length_atIndex_(current_time.tobytes(), 4, 2)

        # Dispatch
        threads = (data.size // 4, 1, 1)  # 4 components per float4
        compute_encoder.dispatchThreadgroups_threadsPerThreadgroup_((threads[0], 1, 1), (64, 1, 1))

        compute_encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        # Update transition state if needed
        if self.birth_state.is_transitioning:
            self.birth_state.transition_progress = min(
                1.0, self.birth_state.transition_progress + 0.1
            )

        # Store last pattern
        self.last_pattern = MetalPattern(
            pattern_type=pattern_type,
            timestamp=time.time(),
            data=data,
            transition_state=self.birth_state.transition_progress,
        )

    def _process_cpu_fallback(self, pattern_type: str, data: np.ndarray, phase: float) -> None:
        """Process pattern on CPU when Metal is unavailable."""
        transition = (
            self.birth_state.transition_progress if self.birth_state.is_transitioning else 0.0
        )

        processed = np.zeros_like(data)
        time_val = time.time()

        for i in range(0, len(data), 4):
            chunk = data[i : i + 4]
            if pattern_type == "shutdown":
                processed[i : i + 4] = chunk * (1.0 - transition)
            elif pattern_type == "ignition":
                processed[i : i + 4] = chunk * (phase * 2.0) * (1.0 - transition)
            elif pattern_type == "sustain":
                flow = np.sin(chunk[0] * 0.1 + phase * 6.28318)
                processed[i : i + 4] = chunk * (0.5 + flow * 0.5) * (1.0 - transition)
            elif pattern_type == "essence":
                crystal = np.power(chunk * 0.1, phase * 2.0)
                processed[i : i + 4] = chunk * crystal * (1.0 - transition)
            else:
                processed[i : i + 4] = chunk

        # Store processed data
        self.mm_file.seek(0)
        self.mm_file.write(processed.tobytes())

        # Store last pattern
        self.last_pattern = MetalPattern(
            pattern_type=pattern_type,
            timestamp=time_val,
            data=processed,
            transition_state=transition,
        )


# Global bridge instance
bridge = MetalBridge() if METAL_AVAILABLE else None
