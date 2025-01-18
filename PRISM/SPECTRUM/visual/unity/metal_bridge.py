"""Metal-optimized bridge for PRISM pattern visualization on Apple Silicon."""

import mmap
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import metal
except ImportError:
    print("Metal framework not available, visualization will use fallback mode")
    metal = None


@dataclass
class MetalPattern:
    """Hardware-optimized pattern container."""

    pattern_type: int  # 1=spatial, 2=temporal, 3=spectral
    timestamp: float
    data: np.ndarray
    metal_buffer: Optional[Any] = None  # metal.Buffer when available
    transition_state: float = 0.0  # For smooth state transitions


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

    PATTERN_TYPES = {"spatial": 1, "temporal": 2, "spectral": 3}

    def __init__(self, buffer_size: int = 1024 * 1024):  # 1MB default
        self.device = None
        self.command_queue = None
        self.compute_function = None
        self.birth_pipeline = None

        if metal is not None:
            try:
                self.device = metal.MTLCreateSystemDefaultDevice()
                self.command_queue = self.device.newCommandQueue()
                self._setup_metal_pipeline()
            except Exception as e:
                print(f"Metal initialization error: {e}")
                metal = None

        # Create shared memory buffer
        self.mm_file = mmap.mmap(-1, buffer_size)
        self.buffer_size = buffer_size
        self.lock = threading.Lock()

        # State tracking
        self.birth_state = BirthState()
        self.last_pattern: Optional[MetalPattern] = None

    def _setup_metal_pipeline(self) -> None:
        """Initialize Metal compute pipelines."""
        if not self.device:
            return

        # Birth visualization pipeline
        birth_kernel = """
        #include <metal_stdlib>
        using namespace metal;

        struct PatternData {
            float4 data;
            int type;
            float timestamp;
            float birth_phase;
            float transition_state;
        };

        float4 process_birth_ignition(float4 data, float time, float birth_phase, float transition) {
            float3 center = float3(0.5, 0.5, 0.0);
            float spark = pow(birth_phase * (1.0 - transition), 2.0);
            float3 direction = normalize(data.xyz - center);
            float distance = length(data.xyz - center);
            float wave = sin(distance * 10.0 - time * 2.0) * 0.5 + 0.5;
            return float4(direction * wave * spark, data.w * birth_phase);
        }

        float4 process_birth_sustain(float4 data, float time, float birth_phase, float transition) {
            float phi = 1.618033988749895;
            float angle = atan2(data.y, data.x);
            float radius = length(data.xyz);
            float spiral = angle + radius * phi + time;
            float flow = (sin(spiral) * 0.5 + 0.5) * birth_phase * (1.0 - transition);
            return float4(data.xyz * flow, data.w * birth_phase);
        }

        float4 process_birth_essence(float4 data, float time, float birth_phase, float transition) {
            float3 center = float3(0.5, 0.5, 0.0);
            float distance = length(data.xyz - center);
            float crystal = pow(1.0 - distance, 2.0) * birth_phase * (1.0 - transition);
            float pulse = (sin(time * 3.0) * 0.5 + 0.5) * birth_phase;
            return float4(lerp(data.xyz, center, crystal), pulse * data.w);
        }

        float4 process_shutdown(float4 data, float time, float transition) {
            float fade = 1.0 - transition;
            return data * fade;
        }

        kernel void process_birth_patterns(
            device const PatternData* patterns [[buffer(0)]],
            device float4* output [[buffer(1)]],
            constant float& time [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            PatternData pattern = patterns[id];
            float4 processed;

            if (pattern.type == 0) { // Shutdown
                processed = process_shutdown(pattern.data, time, pattern.transition_state);
            } else {
                switch(pattern.type) {
                    case 1: // Ignition
                        processed = process_birth_ignition(pattern.data, time, pattern.birth_phase, pattern.transition_state);
                        break;
                    case 2: // Sustain
                        processed = process_birth_sustain(pattern.data, time, pattern.birth_phase, pattern.transition_state);
                        break;
                    case 3: // Essence
                        processed = process_birth_essence(pattern.data, time, pattern.birth_phase, pattern.transition_state);
                        break;
                    default:
                        processed = pattern.data;
                }
            }

            output[id] = processed;
        }
        """

        try:
            library = self.device.newLibraryWithSource_options_error_(
                birth_kernel, metal.MTLCompileOptions.alloc().init(), None
            )
            self.birth_pipeline = library.newFunctionWithName_("process_birth_patterns")
        except Exception as e:
            print(f"Error setting up Metal pipeline: {e}")

    def write_birth_pattern(self, pattern_data: Dict) -> None:
        """Process and visualize birth pattern using Metal."""
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

            if metal is not None:
                try:
                    self._process_metal_pattern(pattern_data)
                except Exception as e:
                    print(f"Metal processing error: {e}")
                    self._process_cpu_fallback(pattern_data)
            else:
                self._process_cpu_fallback(pattern_data)

    def _process_metal_pattern(self, pattern_data: Dict) -> None:
        """Process pattern using Metal acceleration."""
        if not self.device or not self.birth_pipeline:
            return

        # Prepare pattern data with transition state
        np_data = np.array(pattern_data["data"], dtype=np.float32)
        pattern_buffer = self.device.newBuffer(
            data=np_data.tobytes(),
            length=np_data.nbytes,
            options=metal.MTLResourceStorageModeShared,
        )

        # Create output buffer
        output_buffer = self.device.newBuffer(
            length=np_data.nbytes, options=metal.MTLResourceStorageModeShared
        )

        # Process with Metal
        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()

        # Set up compute pipeline
        compute_encoder.setComputePipelineState_(
            self.device.newComputePipelineStateWithFunction_error_(self.birth_pipeline, None)
        )

        # Set buffers and parameters
        compute_encoder.setBuffer_offset_atIndex_(pattern_buffer, 0, 0)
        compute_encoder.setBuffer_offset_atIndex_(output_buffer, 0, 1)

        # Current time for animation
        current_time = np.array([time.time()], dtype=np.float32)
        compute_encoder.setBytes_length_atIndex_(current_time.tobytes(), 4, 2)

        # Dispatch
        threads = (len(pattern_data["data"]), 1, 1)
        compute_encoder.dispatchThreadgroups_threadsPerThreadgroup_((threads[0], 1, 1), (64, 1, 1))

        compute_encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        # Update transition state if needed
        if self.birth_state.is_transitioning:
            self.birth_state.transition_progress = min(
                1.0, self.birth_state.transition_progress + 0.1
            )

    def _process_cpu_fallback(self, pattern_data: Dict) -> None:
        """Process pattern on CPU when Metal is unavailable."""
        phase = pattern_data.get("birth_phase", 0.0)
        transition = (
            self.birth_state.transition_progress if self.birth_state.is_transitioning else 0.0
        )

        processed = []
        for value in pattern_data["data"]:
            if pattern_data.get("type") == "shutdown":
                processed.append(value * (1.0 - transition))
            elif phase < 0.4:  # Ignition
                processed.append(value * (phase * 2.0) * (1.0 - transition))
            elif phase < 0.7:  # Sustain
                flow = np.sin(value * 0.1 + phase * 6.28318)
                processed.append(value * (0.5 + flow * 0.5) * (1.0 - transition))
            else:  # Essence
                crystal = np.power(value * 0.1, phase * 2.0)
                processed.append(value * crystal * (1.0 - transition))

        return np.array(processed, dtype=np.float32).tobytes()


# Global bridge instance
bridge = MetalBridge() if metal is not None else None
