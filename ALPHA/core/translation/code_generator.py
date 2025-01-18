"""Code generation utilities for ALPHA."""

import ast
from typing import Optional, Union

from ..patterns.pattern import Pattern


class CodeGenerator:
    """Advanced code generation from patterns with fallback options."""

    @staticmethod
    def generate_code(pattern: Pattern) -> Optional[str]:
        """Generate code from a pattern using the most appropriate method."""
        try:
            # Try advanced generation first
            code = CodeGenerator._advanced_generation(pattern)
            if code:
                return code

            # Fall back to simple generation if advanced fails
            return CodeGenerator._simple_generation(pattern)
        except Exception as e:
            print(f"Code generation error: {e}")
            return None

    @staticmethod
    def _advanced_generation(pattern: Pattern) -> Optional[str]:
        """Advanced pattern-to-code conversion."""
        # ... existing advanced generation code ...
        return None  # Placeholder for existing implementation

    @staticmethod
    def _simple_generation(pattern: Pattern) -> Optional[str]:
        """Simple fallback pattern-to-code conversion."""
        try:
            # Convert pattern data to binary string
            binary = "".join(format(b, "08b") for b in pattern.data)

            # Convert to bytes
            byte_length = len(binary) // 8
            bytes_list = []

            for i in range(byte_length):
                byte = binary[i * 8 : (i + 1) * 8]
                byte_val = int(byte, 2)
                if byte_val < 32:  # Skip control chars
                    continue
                bytes_list.append(byte_val)

            # Convert to string
            code = bytes(bytes_list).decode("utf-8", errors="replace")

            # Ensure valid syntax
            try:
                ast.parse(code)
                return code
            except SyntaxError:
                return f'''def evolved_pattern_{pattern.id}():
    """Evolved from pattern {pattern.id}."""
    return {repr(code)}
'''
        except Exception as e:
            print(f"Simple code generation error: {e}")
            return None

    @staticmethod
    def validate_code(code: str) -> bool:
        """Validate generated code."""
        try:
            ast.parse(code)
            return True
        except (SyntaxError, ValueError):
            return False
