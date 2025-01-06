"""Memory space organization module."""

import logging
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
import json
import os
from .monitor import MemoryMonitor


@dataclass
class MemoryMetrics:
    """Metrics for memory block usage and performance."""
    access_count: int = 0
    last_access_time: float = 0.0
    importance_score: float = 0.0
    pattern_connections: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.pattern_connections is None:
            self.pattern_connections = set()


class MemoryBlock:
    """Memory block for storing binary data with enhanced metrics."""
    
    def __init__(self, size: int = 4096):
        """Initialize memory block."""
        self.size = size
        self.data_map: Dict[str, bytes] = {}  # Store data per reference
        self.used = 0
        self.references: Set[str] = set()
        self.metrics: Dict[str, MemoryMetrics] = {}
        self.logger = logging.getLogger('memory_block')
    
    def has_space(self, data_size: int) -> bool:
        """Check if block has space for data."""
        return self.used + data_size <= self.size
    
    def write(
        self,
        data: bytes,
        reference: str,
        importance: float = 0.5
    ) -> bool:
        """Write data to memory block with importance score."""
        try:
            if not self.has_space(len(data)):
                return False
            
            # Get existing metrics if available
            existing_metrics = self.metrics.get(reference)
            if existing_metrics:
                # Keep existing importance score
                importance = existing_metrics.importance_score
            
            # Initialize metrics if needed
            if reference not in self.metrics:
                self.metrics[reference] = MemoryMetrics()
                self.metrics[reference].importance_score = importance
            
            self.logger.debug(
                f"Writing pattern {reference} with importance {importance}"
            )
            
            # Update data and usage
            if reference in self.data_map:
                self.used -= len(self.data_map[reference])
            self.data_map[reference] = data
            self.used += len(data)
            self.references.add(reference)
            
            # Update access metrics
            self.metrics[reference].access_count += 1
            self.metrics[reference].last_access_time = time.time()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Write failed: {str(e)}")
            return False
    
    def read(
        self,
        offset: int,
        length: int,
        reference: Optional[str] = None
    ) -> Optional[bytes]:
        """Read data from memory block and update access metrics."""
        try:
            if not self.references:
                return None
            
            # If reference is provided, return data for that reference
            if reference and reference in self.references:
                data = self.data_map.get(reference)
                if data:
                    # Update access metrics
                    self.metrics[reference].access_count += 1
                    self.metrics[reference].last_access_time = time.time()
                return data
            
            # Return data for the first reference (backward compatibility)
            ref = next(iter(self.references))
            data = self.data_map.get(ref)
            if data:
                self.metrics[ref].access_count += 1
                self.metrics[ref].last_access_time = time.time()
            return data
            
        except Exception as e:
            self.logger.error(f"Read failed: {str(e)}")
            return None
    
    def clear(self) -> None:
        """Clear memory block."""
        self.data_map.clear()
        self.used = 0
        self.references.clear()
        self.metrics.clear()
    
    def get_metrics(self, reference: str) -> Optional[MemoryMetrics]:
        """Get metrics for a reference."""
        return self.metrics.get(reference)
    
    def add_pattern_connection(
        self,
        source_ref: str,
        target_ref: str
    ) -> bool:
        """Add connection between patterns."""
        try:
            if (
                source_ref in self.metrics
                and target_ref in self.references
            ):
                self.metrics[source_ref].pattern_connections.add(target_ref)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to add pattern connection: {str(e)}")
            return False


class MemoryOrganizer:
    """Organizes and manages memory blocks with enhanced features."""
    
    def __init__(
        self,
        initial_block_size: int = 4096,
        persistence_path: Optional[str] = None
    ):
        """Initialize memory organizer."""
        self.blocks: List[MemoryBlock] = []
        self.reference_map: Dict[str, List[int]] = {}
        self.block_size = initial_block_size
        self.persistence_path = persistence_path
        self.logger = logging.getLogger('memory_organizer')
        
        # Memory management parameters
        self.prune_threshold = 0.2  # Lower threshold for pruning
        self.consolidation_threshold = 0.5
        self.last_maintenance_time = time.time()
        self.maintenance_interval = 3600  # 1 hour
        
        # Initialize memory monitor
        self.monitor = MemoryMonitor()
        self.monitor.start_monitoring()
        
        # Load persisted patterns if available
        if persistence_path:
            self._load_persisted_patterns()
            
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'monitor'):
            self.monitor.stop_monitoring()
    
    def _persist_patterns(self):
        """Persist patterns to storage."""
        try:
            if not self.persistence_path:
                return
                
            persistence_data = {
                'patterns': {},
                'connections': {}
            }
            
            # Store pattern data and metrics
            for ref, block_indices in self.reference_map.items():
                if not block_indices:
                    continue
                    
                block = self.blocks[block_indices[0]]
                data = block.data_map.get(ref)
                metrics = block.get_metrics(ref)
                
                if data and metrics:
                    persistence_data['patterns'][ref] = {
                        'data': data.hex(),  # Convert bytes to hex string
                        'importance': metrics.importance_score,
                        'access_count': metrics.access_count,
                        'connections': list(metrics.pattern_connections)
                    }
                    
            # Write to file atomically
            temp_path = f"{self.persistence_path}.tmp"
            with open(temp_path, 'w') as f:
                json.dump(persistence_data, f)
            os.replace(temp_path, self.persistence_path)
                
        except Exception as e:
            self.logger.error(f"Pattern persistence failed: {str(e)}")
    
    def _load_persisted_patterns(self):
        """Load patterns from persistence storage."""
        try:
            if not os.path.exists(self.persistence_path):
                return
                
            with open(self.persistence_path, 'r') as f:
                persistence_data = json.load(f)
            
            # Load patterns and their metrics
            for ref, pattern_info in persistence_data['patterns'].items():
                try:
                    data = bytes.fromhex(pattern_info['data'])
                    importance = pattern_info['importance']
                    
                    # Allocate pattern
                    if self.allocate(data, ref, importance):
                        # Restore access count
                        block_indices = self.reference_map.get(ref, [])
                        if block_indices:
                            block = self.blocks[block_indices[0]]
                            metrics = block.get_metrics(ref)
                            if metrics:
                                metrics.access_count = (
                                    pattern_info['access_count']
                                )
                                
                        # Store connections for later restoration
                        persistence_data['connections'][ref] = (
                            pattern_info['connections']
                        )
                                
                except Exception as e:
                    self.logger.error(
                        f"Failed to load pattern {ref}: {str(e)}"
                    )
                    continue
            
            # Restore pattern connections after all patterns are loaded
            for ref, connections in persistence_data['connections'].items():
                for target_ref in connections:
                    self.connect_patterns(ref, target_ref)
                    
        except Exception as e:
            self.logger.error(f"Pattern loading failed: {str(e)}")
    
    def _perform_maintenance(self):
        """Perform memory maintenance operations."""
        try:
            current_time = time.time()
            pruned_refs = set()
            
            # Check each block for patterns to prune
            for block in self.blocks:
                # Create copy to modify during iteration
                for ref in list(block.references):
                    metrics = block.get_metrics(ref)
                    if not metrics:
                        continue
                        
                    # Calculate effective importance based on access history
                    decay_factor = (
                        (current_time - metrics.last_access_time) / 3600
                    )
                    time_factor = max(
                        0.1,
                        min(1.0, 1.0 / (1.0 + decay_factor))
                    )
                    effective_importance = (
                        metrics.importance_score * time_factor
                    )
                    
                    if effective_importance < self.prune_threshold:
                        self._deallocate_pattern(ref)
                        pruned_refs.add(ref)
            
            # Update reference map
            for ref in pruned_refs:
                if ref in self.reference_map:
                    del self.reference_map[ref]
            
            # Trigger defragmentation if needed
            if self._should_defragment():
                self.defragment()
                
            self.last_maintenance_time = current_time
            
        except Exception as e:
            self.logger.error(f"Maintenance operation failed: {str(e)}")
    
    def _should_defragment(self) -> bool:
        """Check if defragmentation is needed."""
        if not self.blocks:
            return False
            
        total_space = sum(block.size for block in self.blocks)
        used_space = sum(block.used for block in self.blocks)
        
        return (used_space / total_space) < self.consolidation_threshold
    
    def defragment(self):
        """Defragment memory blocks."""
        try:
            if not self.blocks:
                return
                
            # Log initial state
            self.logger.debug(
                f"Starting defragmentation with {len(self.blocks)} blocks"
            )
            for i, block in enumerate(self.blocks):
                self.logger.debug(
                    f"Block {i}: {len(block.references)} patterns, "
                    f"used: {block.used}/{block.size}"
                )
                for ref in block.references:
                    self.logger.debug(f"  Pattern in block {i}: {ref}")
            
            # Collect all valid patterns and their complete state
            patterns = []
            pattern_map = {}  # Track patterns by reference
            
            self.logger.debug("Starting pattern collection")
            for i, block in enumerate(self.blocks):
                self.logger.debug(f"Scanning block {i}")
                for ref in block.references:
                    # Skip if already collected
                    if ref in pattern_map:
                        self.logger.debug(f"  Skipping {ref} (already collected)")
                        continue
                        
                    data = block.data_map.get(ref)
                    metrics = block.get_metrics(ref)
                    if data and metrics:
                        pattern_info = {
                            'ref': ref,
                            'data': data,
                            'importance': metrics.importance_score,
                            'access_count': metrics.access_count,
                            'last_access_time': metrics.last_access_time,
                            'connections': metrics.pattern_connections.copy()
                        }
                        patterns.append(pattern_info)
                        pattern_map[ref] = pattern_info
                        self.logger.debug(
                            f"  Collected {ref} "
                            f"(size: {len(data)}, importance: "
                            f"{metrics.importance_score})"
                        )
                    else:
                        self.logger.debug(
                            f"  Failed to collect {ref} "
                            f"(data: {data is not None}, "
                            f"metrics: {metrics is not None})"
                        )
            
            self.logger.debug(
                f"Total patterns collected: {len(patterns)}, "
                f"References: {sorted(pattern_map.keys())}"
            )
            
            # Store old state
            old_blocks = self.blocks
            old_ref_map = self.reference_map.copy()
            self.logger.debug(
                f"Old reference map: {sorted(old_ref_map.keys())}"
            )
            
            # Clear current state
            self.blocks = []
            self.reference_map.clear()
            
            # Sort by size (largest first) then importance
            patterns.sort(
                key=lambda x: (len(x['data']), x['importance']),
                reverse=True
            )
            
            # Track successful reallocations
            reallocated = set()
            
            # Reallocate all patterns
            for pattern in patterns:
                data_size = len(pattern['data'])
                allocated = False
                
                self.logger.debug(
                    f"Attempting to reallocate pattern {pattern['ref']} "
                    f"(size: {data_size})"
                )
                
                # Try to find a block with enough space
                for block_index, block in enumerate(self.blocks):
                    if block.has_space(data_size):
                        if block.write(
                            pattern['data'],
                            pattern['ref'],
                            pattern['importance']
                        ):
                            self._update_reference_map(
                                pattern['ref'],
                                block_index
                            )
                            reallocated.add(pattern['ref'])
                            
                            # Restore metrics
                            metrics = block.get_metrics(pattern['ref'])
                            if metrics:
                                metrics.access_count = pattern['access_count']
                                metrics.last_access_time = (
                                    pattern['last_access_time']
                                )
                                metrics.pattern_connections = (
                                    pattern['connections']
                                )
                            
                            # Restore connections
                            for connected_ref in pattern['connections']:
                                self.connect_patterns(
                                    pattern['ref'],
                                    connected_ref
                                )
                            
                            allocated = True
                            self.logger.debug(
                                f"Reallocated pattern {pattern['ref']} "
                                f"to block {block_index}"
                            )
                            break
                
                # Create new block if needed
                if not allocated:
                    new_size = self._calculate_block_size(data_size)
                    new_block = MemoryBlock(new_size)
                    
                    self.logger.debug(
                        f"Creating new block for pattern {pattern['ref']} "
                        f"(size: {new_size})"
                    )
                    
                    if new_block.write(
                        pattern['data'],
                        pattern['ref'],
                        pattern['importance']
                    ):
                        block_index = len(self.blocks)
                        self.blocks.append(new_block)
                        self._update_reference_map(
                            pattern['ref'],
                            block_index
                        )
                        reallocated.add(pattern['ref'])
                        
                        # Restore metrics
                        metrics = new_block.get_metrics(pattern['ref'])
                        if metrics:
                            metrics.access_count = pattern['access_count']
                            metrics.last_access_time = pattern['last_access_time']
                            metrics.pattern_connections = pattern['connections']
                        
                        # Restore connections
                        for connected_ref in pattern['connections']:
                            self.connect_patterns(
                                pattern['ref'],
                                connected_ref
                            )
                        
                        self.logger.debug(
                            f"Created new block {block_index} for pattern "
                            f"{pattern['ref']}"
                        )
                    else:
                        self.logger.error(
                            f"Failed to write pattern {pattern['ref']} "
                            "during defragmentation"
                        )
            
            # Verify all patterns were preserved
            missing_patterns = set(old_ref_map.keys()) - reallocated
            if missing_patterns:
                self.logger.error(
                    f"Patterns lost during defragmentation: {missing_patterns}"
                )
                # Restore old state
                self.blocks = old_blocks
                self.reference_map = old_ref_map
                return
            
            self.logger.debug(
                f"Defragmentation complete. Reallocated patterns: "
                f"{sorted(reallocated)}"
            )
            
            # Clean up old blocks
            for block in old_blocks:
                block.clear()
            
        except Exception as e:
            self.logger.error(f"Defragmentation failed: {str(e)}")
            # Restore old state if defragmentation fails
            self.blocks = old_blocks
            self.reference_map = old_ref_map
    
    def _deallocate_pattern(self, reference: str) -> bool:
        """Deallocate a pattern from memory."""
        try:
            if reference not in self.reference_map:
                return False
                
            # Get pattern size before deallocation
            pattern_size = 0
            for block_index in self.reference_map[reference]:
                if block_index < len(self.blocks):
                    block = self.blocks[block_index]
                    if reference in block.data_map:
                        pattern_size = len(block.data_map[reference])
                        break
                        
            # Remove from blocks
            for block_index in self.reference_map[reference]:
                if block_index < len(self.blocks):
                    block = self.blocks[block_index]
                    if reference in block.data_map:
                        block.used -= len(block.data_map[reference])
                    block.data_map.pop(reference, None)
                    block.references.discard(reference)
                    block.metrics.pop(reference, None)
            
            # Track deallocation
            if pattern_size > 0:
                self.monitor.track_deallocation(reference, reference)
                
            # Remove from reference map
            del self.reference_map[reference]
            return True
            
        except Exception as e:
            self.logger.error(f"Deallocation failed: {str(e)}")
            return False
    
    def _update_reference_map(self, reference: str, block_index: int):
        """Update reference map with block index."""
        if reference not in self.reference_map:
            self.reference_map[reference] = []
        if block_index not in self.reference_map[reference]:
            self.reference_map[reference].append(block_index)
    
    def _check_maintenance(self):
        """Check if maintenance is needed and perform if necessary."""
        current_time = time.time()
        if (current_time - self.last_maintenance_time >=
                self.maintenance_interval):
            self._perform_maintenance()
    
    def _calculate_block_size(self, data_size: int) -> int:
        """Calculate appropriate block size for data."""
        return max(self.block_size, data_size * 2)  # Allow for growth
    
    def allocate(
        self,
        data: bytes,
        reference: str,
        importance: float = 0.5
    ) -> bool:
        """Allocate memory for data with importance score."""
        try:
            # Check if maintenance is needed
            self._check_maintenance()
            
            # Check if pattern already exists
            if reference in self.reference_map:
                # Get existing metrics
                block_indices = self.reference_map[reference]
                if block_indices:
                    block = self.blocks[block_indices[0]]
                    metrics = block.get_metrics(reference)
                    if metrics:
                        # Keep original importance score
                        imp_score = metrics.importance_score
                        msg = (
                            f"Pattern {reference} exists "
                            f"with importance {imp_score}"
                        )
                        self.logger.debug(msg)
                        # Update access metrics
                        metrics.access_count += 1
                        metrics.last_access_time = time.time()
                        # Track cache hit
                        self.monitor.track_cache_access(reference, True)
                return True
                
            msg = (
                f"Allocating pattern {reference} "
                f"with importance {importance}"
            )
            self.logger.debug(msg)
            
            data_size = len(data)
            
            # Try existing blocks first
            for i, block in enumerate(self.blocks):
                if block.has_space(data_size):
                    if block.write(data, reference, importance):
                        self._update_reference_map(reference, i)
                        # Track allocation
                        self.monitor.track_allocation(
                            reference,
                            data_size,
                            reference,
                            importance
                        )
                        return True
                        
            # Create new block with appropriate size
            new_size = self._calculate_block_size(data_size)
            msg = (
                f"New block for {reference} "
                f"(size: {new_size}, data: {data_size})"
            )
            self.logger.debug(msg)
            
            new_block = MemoryBlock(new_size)
            if new_block.write(data, reference, importance):
                self.blocks.append(new_block)
                self._update_reference_map(reference, len(self.blocks) - 1)
                # Track allocation
                self.monitor.track_allocation(
                    reference,
                    data_size,
                    reference,
                    importance
                )
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error during allocation: {e}")
            return False
    
    def connect_patterns(
        self,
        source_ref: str,
        target_ref: str
    ) -> bool:
        """Establish connection between patterns."""
        try:
            if source_ref not in self.reference_map:
                return False
            
            success = False
            for block_index in self.reference_map[source_ref]:
                block = self.blocks[block_index]
                if block.add_pattern_connection(source_ref, target_ref):
                    success = True
            
            return success
            
        except Exception as e:
            self.logger.error(f"Pattern connection failed: {str(e)}")
            return False
    
    def get_connected_patterns(
        self,
        reference: str
    ) -> Set[str]:
        """Get all patterns connected to the reference."""
        try:
            connected = set()
            if reference in self.reference_map:
                for block_index in self.reference_map[reference]:
                    block = self.blocks[block_index]
                    metrics = block.get_metrics(reference)
                    if metrics:
                        connected.update(metrics.pattern_connections)
            return connected
            
        except Exception as e:
            self.logger.error(
                f"Failed to get connected patterns: {str(e)}"
            )
            return set() 
    
    def get_memory_metrics(self) -> Dict:
        """Get current memory metrics."""
        metrics = self.monitor.get_metrics()
        return {
            'total_allocated': metrics.total_allocated,
            'total_freed': metrics.total_freed,
            'current_usage': metrics.current_usage,
            'pattern_count': metrics.pattern_count,
            'cache_hit_rate': metrics.cache_hit_rate,
            'fragmentation_ratio': metrics.fragmentation_ratio,
            'alerts': [str(alert) for alert in metrics.alerts]
        } 