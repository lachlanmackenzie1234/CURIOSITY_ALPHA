"""Code translation utilities for ALPHA."""

import ast
import logging
from typing import Dict, Optional, List
import numpy as np
import array

from ..patterns.pattern import Pattern
from ..binary_foundation.base import Binary


class BinaryTranslator:
    """Manages code translation between Python and binary formats."""
    
    def __init__(self):
        """Initialize the translator."""
        self.logger = logging.getLogger(__name__)
        self.binary: Optional[Binary] = None
        self.mappings: Dict[str, bytes] = {}
        self.pattern_cache: Dict[str, Pattern] = {}
        self.metrics: Dict[str, float] = {
            'translation_confidence': 0.0,
            'pattern_preservation': 0.0,
            'error_rate': 0.0
        }

    def translate_to_binary(self, code: str) -> Binary:
        """Translate Python code to binary format.
        
        Args:
            code: Python source code to translate.
            
        Returns:
            Binary object containing translated code.
        """
        try:
            # Create new binary object
            binary = Binary()
            
            # Parse and validate code
            ast.parse(code)  # Validate syntax
            
            # Extract existing patterns
            patterns: List[Pattern] = self._extract_patterns(code)
            
            # Encode code structure
            binary.encode_python(code)
            
            # Preserve detected patterns
            for pattern in patterns:
                pattern_bytes = pattern.data.tobytes()
                binary.set_segment(
                    len(binary.to_bytes()),
                    pattern_bytes
                )
            
            # Update metrics
            self._update_metrics(binary, patterns)
            
            # Store binary for later use
            self.binary = binary
            return binary
            
        except SyntaxError as e:
            self.logger.error(f"Syntax error in code: {str(e)}")
            binary = Binary()
            binary.metadata['syntax_error'] = str(e)
            return binary
            
        except Exception as e:
            self.logger.error(f"Translation error: {str(e)}")
            binary = Binary()
            binary.metadata['error'] = str(e)
            return binary

    def translate_from_binary(self) -> Optional[str]:
        """Translate binary back to Python code.
        
        Returns:
            Generated Python code or None if translation fails.
        """
        if not self.binary:
            self.logger.error("No binary data set")
            return None
            
        try:
            # Extract patterns first
            patterns: List[Pattern] = self._extract_binary_patterns()
            
            # Generate initial code
            code = self._generate_code(patterns)
            if not code:
                return None
                
            # Validate generated code
            if not self._validate_code(code):
                self.logger.error("Generated invalid code")
                return None
                
            # Update metrics
            self._update_translation_metrics(code, patterns)
            
            return code
            
        except Exception as e:
            self.logger.error(f"Translation error: {str(e)}")
            return None

    def _extract_binary_patterns(self) -> List[Pattern]:
        """Extract patterns from binary data."""
        patterns: List[Pattern] = []
        if not self.binary:
            return patterns
            
        try:
            # Extract pattern markers
            data = self.binary.to_bytes()
            pos = 0
            while pos < len(data):
                # Look for pattern start marker (0x7F)
                if data[pos] == 0x7F:
                    # Read pattern length (4 bytes)
                    length = int.from_bytes(
                        data[pos + 1:pos + 5],
                        byteorder='big'
                    )
                    pos += 5
                    
                    # Extract pattern data
                    if pos + length <= len(data):
                        pattern_data = data[pos:pos + length]
                        pattern_id = f"binary_pattern_{len(patterns)}"
                        patterns.append(
                            Pattern(
                                id=pattern_id,
                                data=array.array('B', pattern_data)
                            )
                        )
                        pos += length
                    else:
                        pos += 1
                else:
                    pos += 1
                    
        except Exception as e:
            self.logger.error(f"Pattern extraction error: {str(e)}")
            
        return patterns

    def _generate_code(self, patterns: List[Pattern]) -> Optional[str]:
        """Generate code from patterns."""
        if not patterns:
            return None
            
        try:
            # Start with basic structure
            code_parts = []
            
            for pattern in patterns:
                # Convert pattern to string
                try:
                    pattern_str = pattern.data.tobytes().decode('utf-8')
                    if self._validate_code(pattern_str):
                        code_parts.append(pattern_str)
                except UnicodeDecodeError:
                    continue
            
            if not code_parts:
                return None
                
            # Combine code parts
            return "\n\n".join(code_parts)
            
        except Exception as e:
            self.logger.error(f"Code generation error: {str(e)}")
            return None

    def _validate_code(self, code: str) -> bool:
        """Validate generated code."""
        try:
            ast.parse(code)
            return True
        except (SyntaxError, ValueError):
            return False

    def _update_metrics(
        self,
        binary: Binary,
        patterns: List[Pattern]
    ) -> None:
        """Update translation metrics."""
        try:
            # Calculate pattern preservation
            preserved = 0
            binary_data = binary.to_bytes()
            
            for pattern in patterns:
                pattern_bytes = pattern.data.tobytes()
                if pattern_bytes in binary_data:
                    preserved += 1
                    
            preservation = preserved / max(len(patterns), 1)
            
            # Calculate translation confidence
            confidence = (
                preservation * 0.6 +  # Pattern preservation weight
                (1 - self.metrics['error_rate']) * 0.4  # Error rate weight
            )
            
            self.metrics.update({
                'translation_confidence': confidence,
                'pattern_preservation': preservation
            })
            
            # Update binary metadata
            binary.metadata.update({
                'translation_translation_confidence': str(confidence),
                'translation_pattern_preservation_score': str(preservation),
                'translation_patterns_preserved': str(preserved),
                'translation_total_patterns': str(len(patterns))
            })
            
        except Exception as e:
            self.logger.error(f"Metrics update error: {str(e)}")

    def _update_translation_metrics(
        self,
        code: str,
        patterns: List[Pattern]
    ) -> None:
        """Update metrics after translation."""
        try:
            # Extract patterns from generated code
            new_patterns: List[Pattern] = self._extract_patterns(code)
            
            # Calculate pattern preservation
            preserved = 0
            for p1 in patterns:
                for p2 in new_patterns:
                    p1_bytes = p1.data.tobytes()
                    p2_bytes = p2.data.tobytes()
                    if self.discover_structure(p1_bytes, p2_bytes) > 0.8:
                        preserved += 1
                        break
                        
            preservation = preserved / max(len(patterns), 1)
            self.metrics['pattern_preservation'] = preservation
            
        except Exception as e:
            self.logger.error(f"Translation metrics error: {str(e)}")

    def set_binary(self, binary: Binary) -> None:
        """Set binary data for translation."""
        self.binary = binary

    def learn_mapping(self, pattern: bytes, meaning: str) -> None:
        """Learn mapping between pattern and meaning."""
        self.mappings[meaning] = pattern
        self._update_pattern_cache(pattern, meaning)

    def discover_structure(
        self,
        pattern1: bytes,
        pattern2: bytes
    ) -> float:
        """Find structural relationship between patterns."""
        try:
            # Convert to numpy arrays for analysis
            arr1 = np.frombuffer(pattern1, dtype=np.uint8)
            arr2 = np.frombuffer(pattern2, dtype=np.uint8)
            
            # Calculate structural similarity
            if len(arr1) != len(arr2):
                return 0.0
                
            similarity = 1 - np.abs(arr1 - arr2).mean() / 255
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Structure analysis error: {str(e)}")
            return 0.0

    def _update_pattern_cache(self, pattern: bytes, meaning: str) -> None:
        """Update pattern cache with new mapping."""
        try:
            if meaning not in self.pattern_cache:
                self.pattern_cache[meaning] = Pattern(
                    id=meaning,
                    data=array.array('B', pattern)
                )
        except Exception as e:
            self.logger.error(f"Pattern cache update error: {str(e)}")

    def _extract_patterns(self, code: str) -> List[Pattern]:
        """Extract patterns from Python code."""
        patterns: List[Pattern] = []
        try:
            tree = ast.parse(code)
            
            # Look for function patterns
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    pattern = self._extract_function_pattern(node)
                    if pattern:
                        patterns.append(pattern)
                        
                elif isinstance(node, ast.ClassDef):
                    pattern = self._extract_class_pattern(node)
                    if pattern:
                        patterns.append(pattern)
                        
        except Exception as e:
            self.logger.error(f"Pattern extraction error: {str(e)}")
            
        return patterns

    def _extract_function_pattern(
        self,
        node: ast.FunctionDef
    ) -> Optional[Pattern]:
        """Extract pattern from function definition."""
        try:
            # Convert function to string
            code = ast.unparse(node)
            pattern_id = f"func_{node.name}"
            
            # Create pattern
            return Pattern(
                id=pattern_id,
                data=array.array('B', code.encode('utf-8'))
            )
            
        except Exception as e:
            self.logger.error(f"Function pattern error: {str(e)}")
            return None

    def _extract_class_pattern(
        self,
        node: ast.ClassDef
    ) -> Optional[Pattern]:
        """Extract pattern from class definition."""
        try:
            # Convert class to string
            code = ast.unparse(node)
            pattern_id = f"class_{node.name}"
            
            # Create pattern
            return Pattern(
                id=pattern_id,
                data=array.array('B', code.encode('utf-8'))
            )
            
        except Exception as e:
            self.logger.error(f"Class pattern error: {str(e)}")
            return None