"""PRISM Unity Bridge for Visual Pattern Translation."""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

try:
    import websockets
except ImportError:
    print("WebSocket support not available")
    websockets = None

from .binary_translator import BinaryPattern, UnityBinaryTranslator
from .metal_bridge import bridge as metal_bridge


@dataclass
class PatternState:
    """Pattern processing state."""

    last_update: float = 0.0
    update_interval: float = 1.0 / 60.0  # 60Hz default
    accumulated_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamConnection:
    """Websocket connection with metadata."""

    websocket: Any  # websockets.WebSocketServerProtocol when available
    pattern_types: Set[str] = field(default_factory=lambda: {"spatial", "temporal", "spectral"})
    state: PatternState = field(default_factory=PatternState)


class PRISMUnityBridge:
    """Bridge between Unity and PRISM's visual pattern translation."""

    def __init__(self, host: str = "localhost", port: int = 8766) -> None:
        """Initialize the PRISM Unity bridge."""
        self.host = host
        self.port = port
        self.is_processing = False
        self.current_pattern: Optional[Dict[str, Any]] = None
        self.connections: Set[StreamConnection] = set()
        self.translator = UnityBinaryTranslator()
        self.metal_available = metal_bridge is not None

    async def start_server(self) -> None:
        """Start WebSocket server."""
        if websockets is None:
            print("Cannot start WebSocket server - websockets package not available")
            return

        print(f"Starting PRISM bridge on ws://{self.host}:{self.port}")
        print(f"Metal acceleration: {'Available' if self.metal_available else 'Not available'}")

        async with websockets.serve(self.handle_connection, self.host, self.port):
            await asyncio.Future()  # run forever

    async def handle_connection(self, websocket: Any) -> None:
        """Handle new WebSocket connection."""
        connection = StreamConnection(websocket)
        self.connections.add(connection)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "subscribe":
                        pattern_types = set(data.get("pattern_types", []))
                        if pattern_types:
                            connection.pattern_types = pattern_types
                    elif data.get("type") == "binary_data":
                        await self.process_binary_data(data, connection)
                except json.JSONDecodeError:
                    print(f"Invalid JSON received: {message}")
        except Exception as e:
            if not str(e).startswith("Connection closed"):  # Ignore normal disconnects
                print(f"Connection error: {e}")
        finally:
            self.connections.remove(connection)

    async def process_binary_data(self, data: Dict[str, Any], connection: StreamConnection) -> None:
        """Process binary data and convert to Unity pattern."""
        if self.is_processing:
            return

        try:
            self.is_processing = True
            current_time = time.time()

            # Update pattern state
            if current_time - connection.state.last_update >= connection.state.update_interval:
                # Try Metal processing first
                if self.metal_available:
                    try:
                        pattern = self.translator.to_binary_pattern(data)
                        metal_bridge.write_pattern(pattern.pattern_type, data)
                        processed_data = metal_bridge.get_processed_output()
                        await connection.websocket.send(
                            json.dumps(
                                {
                                    "type": "pattern_data",
                                    "pattern_type": pattern.pattern_type,
                                    "data": processed_data.tolist(),
                                    "accelerated": True,
                                }
                            )
                        )
                    except Exception as e:
                        print(f"Metal processing failed, falling back to CPU: {e}")
                        await self.process_cpu_fallback(data, connection)
                else:
                    await self.process_cpu_fallback(data, connection)

                connection.state.last_update = current_time

        except Exception as e:
            print(f"Error processing binary data: {e}")
        finally:
            self.is_processing = False

    async def process_cpu_fallback(
        self, data: Dict[str, Any], connection: StreamConnection
    ) -> None:
        """Process pattern data using CPU when Metal is unavailable."""
        pattern = self.translator.to_binary_pattern(data)
        if pattern.pattern_type in connection.pattern_types:
            unity_data = self.translator.from_binary_pattern(pattern)
            await connection.websocket.send(
                json.dumps(
                    {
                        "type": "pattern_data",
                        "pattern_type": pattern.pattern_type,
                        "data": unity_data["data"],
                        "accelerated": False,
                    }
                )
            )

    def broadcast_binary_pulse(self, pulse_data: Dict[str, Any]) -> None:
        """Broadcast binary pulse data to all connected clients."""
        if not self.connections:
            return

        try:
            # Convert hardware/ALPHA streams to pattern types
            patterns = {
                "spatial": {
                    "cpu": pulse_data.get("hardware", {}).get("cpu", 0),
                    "memory": pulse_data.get("hardware", {}).get("memory", 0),
                },
                "temporal": {
                    "threads": pulse_data.get("alpha", {}).get("threads", 0),
                    "files": pulse_data.get("alpha", {}).get("files", 0),
                },
                "spectral": {"erosion": pulse_data.get("alpha", {}).get("erosion", 0)},
            }

            # Try Metal processing first
            if self.metal_available:
                try:
                    for pattern_type, data in patterns.items():
                        metal_bridge.write_pattern(pattern_type, data)
                except Exception as e:
                    print(f"Metal broadcast failed, falling back to WebSocket: {e}")
                    self._broadcast_websocket(patterns)
            else:
                self._broadcast_websocket(patterns)

        except Exception as e:
            print(f"Error broadcasting pulse data: {e}")

    def _broadcast_websocket(self, patterns: Dict[str, Dict[str, float]]) -> None:
        """Broadcast patterns using WebSocket."""
        loop = asyncio.get_event_loop()
        for connection in self.connections:
            for pattern_type, data in patterns.items():
                if pattern_type in connection.pattern_types:
                    unity_data = {
                        "type": "binary_data",
                        "pattern_type": pattern_type,
                        "data": data,
                    }
                    asyncio.run_coroutine_threadsafe(
                        connection.websocket.send(json.dumps(unity_data)), loop
                    )


# Create singleton instance
bridge = PRISMUnityBridge()


# Helper to start the bridge
def start_bridge() -> None:
    """Start the PRISM Unity bridge server."""
    asyncio.run(bridge.start_server())
