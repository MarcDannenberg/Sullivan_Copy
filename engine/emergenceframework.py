# sully_engine/kernel_modules/emergence_framework.py
# 🧠 Sully's Emergence Framework - Creating conditions for genuine emergent properties

from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
import random
import json
import os
from datetime import datetime, timedelta
import time
import uuid
import re
import math
import numpy as np
from collections import defaultdict, Counter

class EmergencePattern:
    """
    Represents a detected emergent pattern in the system.
    """
    
    def __init__(self, pattern_id: Optional[str] = None, pattern_type: str = "unknown"):
        """
        Initialize an emergence pattern.
        
        Args:
            pattern_id: Optional pattern identifier
            pattern_type: Type of pattern
        """
        self.pattern_id = pattern_id or str(uuid.uuid4())
        self.pattern_type = pattern_type
        self.creation_time = datetime.now()
        self.last_observed = self.creation_time
        self.observation_count = 1
        self.stability = 0.1  # 0.0 - 1.0
        self.contributing_modules = set()
        self.activation_history = []
        self.description = ""
        self.properties = {}
        
    def register_observation(self, modules: List[str], properties: Dict[str, Any] = None) -> None:
        """
        Register a new observation of this pattern.
        
        Args:
            modules: Modules involved in this observation
            properties: Optional properties observed
        """
        self.observation_count += 1
        self.last_observed = datetime.now()
        self.contributing_modules.update(modules)
        
        # Record activation
        self.activation_history.append({
            "timestamp": self.last_observed.isoformat(),
            "modules": list(modules),
            "properties": properties or {}
        })
        
        # Update stability based on observation frequency
        time_diff = (self.last_observed - self.creation_time).total_seconds()
        if time_diff > 0:
            # Higher stability for patterns observed frequently over time
            observation_rate = self.observation_count / time_diff
            self.stability = min(1.0, max(0.1, self.stability + 0.1 * observation_rate))
        
        # Update properties
        if properties:
            for key, value in properties.items():
                self.properties[key] = value
                
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "creation_time": self.creation_time.isoformat(),
            "last_observed": self.last_observed.isoformat(),
            "observation_count": self.observation_count,
            "stability": self.stability,
            "contributing_modules": list(self.contributing_modules),
            "activation_history": self.activation_history,
            "description": self.description,
            "properties": self.properties
        }
        
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'EmergencePattern':
        """
        Create from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Created pattern
        """
        pattern = EmergencePattern(
            pattern_id=data.get("pattern_id"),
            pattern_type=data.get("pattern_type", "unknown")
        )
        
        # Parse timestamps
        try:
            pattern.creation_time = datetime.fromisoformat(data["creation_time"])
        except (KeyError, ValueError, TypeError):
            pattern.creation_time = datetime.now()
            
        try:
            pattern.last_observed = datetime.fromisoformat(data["last_observed"])
        except (KeyError, ValueError, TypeError):
            pattern.last_observed = datetime.now()
            
        # Set properties
        pattern.observation_count = data.get("observation_count", 1)
        pattern.stability = data.get("stability", 0.1)
        pattern.contributing_modules = set(data.get("contributing_modules", []))
        pattern.activation_history = data.get("activation_history", [])
        pattern.description = data.get("description", "")
        pattern.properties = data.get("properties", {})
        
        return pattern


class EmergenceMonitor:
    """
    Monitors system activity to detect potential emergent patterns.
    """
    
    def __init__(self):
        """Initialize the emergence monitor."""
        # Detection parameters
        self.detection_threshold = 0.7  # Minimum confidence for pattern detection
        self.correlation_threshold = 0.6  # Minimum correlation for related activities
        
        # Monitoring state
        self.module_activations = {}  # module -> [timestamps]
        self.cross_activations = {}  # (module1, module2) -> [timestamps]
        self.activation_sequences = []  # Sequences of module activations
        self.feedback_loops = {}  # module -> [modules that activated it]
        
        # Pattern storage
        self.detected_patterns = {}  # pattern_id -> EmergencePattern
        self.pattern_registry = {}  # pattern_type -> [pattern_ids]
        
    def track_module_activation(self, module_name: str, context: Dict[str, Any] = None) -> None:
        """
        Track individual module activation.
        
        Args:
            module_name: Name of activated module
            context: Optional activation context
        """
        timestamp = datetime.now()
        
        # Record activation time
        if module_name not in self.module_activations:
            self.module_activations[module_name] = []
        self.module_activations[module_name].append(timestamp)
        
        # Keep only recent activations (last week)
        week_ago = timestamp - timedelta(days=7)
        self.module_activations[module_name] = [
            t for t in self.module_activations[module_name] if t >= week_ago
        ]
        
        # Update activation sequences
        if self.activation_sequences:
            # If last sequence is recent (within 5 seconds), append to it
            last_seq = self.activation_sequences[-1]
            last_time = last_seq[-1]["timestamp"]
            
            if (timestamp - last_time).total_seconds() < 5:
                self.activation_sequences[-1].append({
                    "module": module_name,
                    "timestamp": timestamp,
                    "context": context
                })
            else:
                # Start new sequence
                self.activation_sequences.append([{
                    "module": module_name,
                    "timestamp": timestamp,
                    "context": context
                }])
        else:
            # First activation
            self.activation_sequences.append([{
                "module": module_name,
                "timestamp": timestamp,
                "context": context
            }])
            
        # Limit sequence history
        if len(self.activation_sequences) > 1000:
            self.activation_sequences = self.activation_sequences[-1000:]
            
    def track_cross_module_activation(self, source_module: str, target_module: str,
                                    context: Dict[str, Any] = None) -> None:
        """
        Track activation of one module by another.
        
        Args:
            source_module: Source module that triggered activation
            target_module: Target module that was activated
            context: Optional activation context
        """
        timestamp = datetime.now()
        
        # Record cross-activation
        cross_key = (source_module, target_module)
        if cross_key not in self.cross_activations:
            self.cross_activations[cross_key] = []
        
        self.cross_activations[cross_key].append({
            "timestamp": timestamp,
            "context": context
        })
        
        # Keep only recent cross-activations (last week)
        week_ago = timestamp - timedelta(days=7)
        self.cross_activations[cross_key] = [
            act for act in self.cross_activations[cross_key] 
            if isinstance(act["timestamp"], datetime) and act["timestamp"] >= week_ago
            or isinstance(act["timestamp"], str) and datetime.fromisoformat(act["timestamp"]) >= week_ago
        ]
        
        # Update feedback loops
        if target_module not in self.feedback_loops:
            self.feedback_loops[target_module] = {}
            
        if source_module not in self.feedback_loops[target_module]:
            self.feedback_loops[target_module][source_module] = 0
            
        self.feedback_loops[target_module][source_module] += 1
        
        # Track individual module activations as well
        self.track_module_activation(target_module, context)
        
    def detect_patterns(self) -> List[EmergencePattern]:
        """
        Analyze tracked activations to detect potential emergent patterns.
        
        Returns:
            List of newly detected patterns
        """
        new_patterns = []
        
        # Check for recurring sequences
        recurring_sequences = self._identify_recurring_sequences()
        for seq_pattern, count in recurring_sequences.items():
            if count >= 3:  # Seen at least 3 times
                # Check if already detected
                already_detected = False
                for pattern in self.detected_patterns.values():
                    if pattern.pattern_type == "sequence" and str(seq_pattern) in pattern.description:
                        already_detected = True
                        break
                        
                if not already_detected:
                    # Create new pattern
                    pattern = EmergencePattern(pattern_type="sequence")
                    pattern.description = f"Recurring activation sequence: {seq_pattern}"
                    
                    # Extract modules
                    modules = [module for module in seq_pattern]
                    pattern.contributing_modules.update(modules)
                    
                    # Add properties
                    pattern.properties["sequence"] = seq_pattern
                    pattern.properties["observation_count"] = count
                    
                    # Register in system
                    self.detected_patterns[pattern.pattern_id] = pattern
                    
                    if "sequence" not in self.pattern_registry:
                        self.pattern_registry["sequence"] = []
                    self.pattern_registry["sequence"].append(pattern.pattern_id)
                    
                    new_patterns.append(pattern)
                    
        # Check for feedback loops
        feedback_patterns = self._identify_feedback_loops()
        for loop, strength in feedback_patterns.items():
            if strength >= self.detection_threshold:
                # Check if already detected
                already_detected = False
                for pattern in self.detected_patterns.values():
                    if pattern.pattern_type == "feedback_loop" and str(loop) in pattern.description:
                        already_detected = True
                        # Update existing pattern
                        pattern.register_observation(loop, {"strength": strength})
                        break
                        
                if not already_detected:
                    # Create new pattern
                    pattern = EmergencePattern(pattern_type="feedback_loop")
                    pattern.description = f"Feedback loop between modules: {' -> '.join(loop)}"
                    
                    # Set properties
                    pattern.contributing_modules.update(loop)
                    pattern.properties["loop"] = loop
                    pattern.properties["strength"] = strength
                    
                    # Register in system
                    self.detected_patterns[pattern.pattern_id] = pattern
                    
                    if "feedback_loop" not in self.pattern_registry:
                        self.pattern_registry["feedback_loop"] = []
                    self.pattern_registry["feedback_loop"].append(pattern.pattern_id)
                    
                    new_patterns.append(pattern)
                    
        # Check for module clusters
        clusters = self._identify_module_clusters()
        for cluster, cohesion in clusters.items():
            if cohesion >= self.detection_threshold and len(cluster) >= 3:
                # Check if already detected
                already_detected = False
                for pattern in self.detected_patterns.values():
                    if pattern.pattern_type == "module_cluster":
                        existing_cluster = set(pattern.properties.get("cluster", []))
                        if existing_cluster == set(cluster):
                            already_detected = True
                            # Update existing pattern
                            pattern.register_observation(cluster, {"cohesion": cohesion})
                            break
                            
                if not already_detected:
                    # Create new pattern
                    pattern = EmergencePattern(pattern_type="module_cluster")
                    pattern.description = f"Module cluster with high interaction: {', '.join(cluster)}"
                    
                    # Set properties
                    pattern.contributing_modules.update(cluster)
                    pattern.properties["cluster"] = cluster
                    pattern.properties["cohesion"] = cohesion
                    
                    # Register in system
                    self.detected_patterns[pattern.pattern_id] = pattern
                    
                    if "module_cluster" not in self.pattern_registry:
                        self.pattern_registry["module_cluster"] = []
                    self.pattern_registry["module_cluster"].append(pattern.pattern_id)
                    
                    new_patterns.append(pattern)
                    
        return new_patterns
        
    def _identify_recurring_sequences(self) -> Dict[Tuple[str, ...], int]:
        """
        Identify recurring sequences of module activations.
        
        Returns:
            Dictionary mapping sequences to observation counts
        """
        # Extract sequences of length 3-5
        sequences = {}
        
        for activation_sequence in self.activation_sequences:
            if len(activation_sequence) < 3:
                continue
                
            # Extract module names from sequence
            modules = [act["module"] for act in activation_sequence]
            
            # Generate subsequences of length 3-5
            for length in range(3, min(6, len(modules) + 1)):
                for i in range(len(modules) - length + 1):
                    subseq = tuple(modules[i:i+length])
                    if subseq not in sequences:
                        sequences[subseq] = 0
                    sequences[subseq] += 1
                    
        # Filter to sequences seen multiple times
        return {seq: count for seq, count in sequences.items() if count >= 3}
        
    def _identify_feedback_loops(self) -> Dict[Tuple[str, ...], float]:
        """
        Identify feedback loops between modules.
        
        Returns:
            Dictionary mapping feedback loops to strength
        """
        loops = {}
        
        # Look for cycles in the feedback graph
        for module, sources in self.feedback_loops.items():
            for source, count in sources.items():
                # Check for direct loop (A -> B -> A)
                if source in self.feedback_loops and module in self.feedback_loops[source]:
                    loop = (source, module)
                    strength = math.sqrt(count * self.feedback_loops[source][module]) / 10
                    loops[loop] = min(1.0, strength)
                    
                # Try to find longer loops (limited to 3-node loops for simplicity)
                if source in self.feedback_loops:
                    for indirect, indirect_count in self.feedback_loops[source].items():
                        if indirect != module and indirect in self.feedback_loops and module in self.feedback_loops[indirect]:
                            # Found 3-node loop: module -> source -> indirect -> module
                            loop = (module, source, indirect)
                            # Calculate strength as geometric mean of connection counts
                            counts = [count, indirect_count, self.feedback_loops[indirect][module]]
                            strength = math.pow(counts[0] * counts[1] * counts[2], 1/3) / 15
                            loops[loop] = min(1.0, strength)
                            
        return loops
        
    def _identify_module_clusters(self) -> Dict[Tuple[str, ...], float]:
        """
        Identify clusters of modules with high cross-activation.
        
        Returns:
            Dictionary mapping module clusters to cohesion
        """
        # Build connection strength matrix
        modules = list(self.module_activations.keys())
        n_modules = len(modules)
        
        if n_modules < 3:
            return {}  # Need at least 3 modules for meaningful clustering
            
        # Create connection strength matrix
        conn_matrix = np.zeros((n_modules, n_modules))
        
        for i, module1 in enumerate(modules):
            for j, module2 in enumerate(modules):
                if i != j:
                    key1 = (module1, module2)
                    key2 = (module2, module1)
                    
                    # Count cross-activations in both directions
                    conn_strength = 0.0
                    if key1 in self.cross_activations:
                        conn_strength += len(self.cross_activations[key1])
                    if key2 in self.cross_activations:
                        conn_strength += len(self.cross_activations[key2])
                        
                    # Normalize
                    total_activations = len(self.module_activations.get(module1, [])) + len(self.module_activations.get(module2, []))
                    if total_activations > 0:
                        conn_matrix[i, j] = conn_strength / total_activations
                        
        # Simple clustering - find modules with high mutual connectivity
        clusters = {}
        
        # Start with pairs that have high connection strength
        for i in range(n_modules):
            for j in range(i + 1, n_modules):
                if conn_matrix[i, j] >= self.correlation_threshold:
                    # Found potential cluster seed
                    cluster = [modules[i], modules[j]]
                    
                    # Try to extend cluster
                    for k in range(n_modules):
                        if k != i and k != j:
                            # Check if strongly connected to both existing members
                            if (conn_matrix[i, k] >= self.correlation_threshold and 
                                conn_matrix[j, k] >= self.correlation_threshold):
                                cluster.append(modules[k])
                                
                    if len(cluster) >= 3:
                        # Calculate cluster cohesion (average connection strength)
                        strength_sum = 0
                        pair_count = 0
                        
                        for x in range(len(cluster)):
                            for y in range(x + 1, len(cluster)):
                                ix = modules.index(cluster[x])
                                iy = modules.index(cluster[y])
                                strength_sum += conn_matrix[ix, iy]
                                pair_count += 1
                                
                        cohesion = strength_sum / pair_count if pair_count > 0 else 0
                        
                        # Store if cohesion is high enough
                        if cohesion >= self.correlation_threshold:
                            clusters[tuple(sorted(cluster))] = cohesion
                            
        return clusters
        
    def get_pattern(self, pattern_id: str) -> Optional[EmergencePattern]:
        """
        Get a pattern by ID.
        
        Args:
            pattern_id: Pattern identifier
            
        Returns:
            Pattern or None if not found
        """
        return self.detected_patterns.get(pattern_id)
        
    def get_patterns_by_type(self, pattern_type: str) -> List[EmergencePattern]:
        """
        Get patterns by type.
        
        Args:
            pattern_type: Pattern type
            
        Returns:
            List of matching patterns
        """
        if pattern_type not in self.pattern_registry:
            return []
            
        return [self.detected_patterns[pid] for pid in self.pattern_registry[pattern_type]
               if pid in self.detected_patterns]
               
    def get_module_activation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about module activations.
        
        Returns:
            Activation statistics
        """
        stats = {
            "total_modules": len(self.module_activations),
            "total_activations": sum(len(acts) for acts in self.module_activations.values()),
            "module_counts": {module: len(acts) for module, acts in self.module_activations.items()},
            "cross_activation_count": sum(len(acts) for acts in self.cross_activations.values()),
            "sequences_tracked": len(self.activation_sequences),
            "patterns_detected": len(self.detected_patterns)
        }
        
        # Most active modules
        if stats["total_modules"] > 0:
            sorted_modules = sorted([(m, len(a)) for m, a in self.module_activations.items()],
                                  key=lambda x: x[1], reverse=True)
            stats["most_active_modules"] = sorted_modules[:5]
            
        # Most common cross-activations
        if self.cross_activations:
            sorted_cross = sorted([(k, len(v)) for k, v in self.cross_activations.items()],
                                key=lambda x: x[1], reverse=True)
            stats["most_common_cross_activations"] = sorted_cross[:5]
            
        return stats


class ModuleInteractionMap:
    """
    Maps interactions between modules with configurable connection strengths.
    """
    
    def __init__(self):
        """Initialize the module interaction map."""
        self.connections = {}  # (source, target) -> connection_strength (0.0-1.0)
        self.connection_metadata = {}  # (source, target) -> metadata dict
        self.default_strength = 0.5
        
    def set_connection(self, source: str, target: str, strength: float = None,
                      metadata: Dict[str, Any] = None) -> None:
        """
        Set connection between modules.
        
        Args:
            source: Source module
            target: Target module
            strength: Connection strength (0.0-1.0)
            metadata: Optional connection metadata
        """
        connection_key = (source, target)
        
        # Set connection strength
        self.connections[connection_key] = min(1.0, max(0.0, strength if strength is not None else self.default_strength))
        
        # Set metadata
        if metadata:
            if connection_key not in self.connection_metadata:
                self.connection_metadata[connection_key] = {}
            self.connection_metadata[connection_key].update(metadata)
            
    def get_connection_strength(self, source: str, target: str) -> float:
        """
        Get connection strength between modules.
        
        Args:
            source: Source module
            target: Target module
            
        Returns:
            Connection strength (0.0-1.0)
        """
        return self.connections.get((source, target), 0.0)
        
    def get_targets(self, source: str, min_strength: float = 0.0) -> Dict[str, float]:
        """
        Get all potential target modules for a source.
        
        Args:
            source: Source module
            min_strength: Minimum connection strength
            
        Returns:
            Dictionary mapping target modules to connection strengths
        """
        targets = {}
        
        for (src, tgt), strength in self.connections.items():
            if src == source and strength >= min_strength:
                targets[tgt] = strength
                
        return targets
        
    def get_sources(self, target: str, min_strength: float = 0.0) -> Dict[str, float]:
        """
        Get all potential source modules for a target.
        
        Args:
            target: Target module
            min_strength: Minimum connection strength
            
        Returns:
            Dictionary mapping source modules to connection strengths
        """
        sources = {}
        
        for (src, tgt), strength in self.connections.items():
            if tgt == target and strength >= min_strength:
                sources[src] = strength
                
        return sources
        
    def adjust_connection(self, source: str, target: str, adjustment: float) -> float:
        """
        Adjust connection strength by a delta.
        
        Args:
            source: Source module
            target: Target module
            adjustment: Strength adjustment
            
        Returns:
            New connection strength
        """
        connection_key = (source, target)
        
        # Get current strength
        current = self.connections.get(connection_key, 0.0)
        
        # Calculate new strength
        new_strength = min(1.0, max(0.0, current + adjustment))
        
        # Update connection
        self.connections[connection_key] = new_strength
        
        return new_strength
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        # Convert connection keys to strings for JSON serialization
        connections_dict = {}
        for (source, target), strength in self.connections.items():
            connections_dict[f"{source}|{target}"] = strength
            
        metadata_dict = {}
        for (source, target), metadata in self.connection_metadata.items():
            metadata_dict[f"{source}|{target}"] = metadata
            
        return {
            "connections": connections_dict,
            "metadata": metadata_dict,
            "default_strength": self.default_strength
        }
        
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ModuleInteractionMap':
        """
        Create from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Created module interaction map
        """
        interaction_map = ModuleInteractionMap()
        
        # Set default strength
        if "default_strength" in data:
            interaction_map.default_strength = data["default_strength"]
            
        # Parse connections
        if "connections" in data:
            for key_str, strength in data["connections"].items():
                try:
                    source, target = key_str.split("|")
                    interaction_map.connections[(source, target)] = strength
                except:
                    pass
                    
        # Parse metadata
        if "metadata" in data:
            for key_str, metadata in data["metadata"].items():
                try:
                    source, target = key_str.split("|")
                    interaction_map.connection_metadata[(source, target)] = metadata
                except:
                    pass
                    
        return interaction_map


class CognitiveParameters:
    """
    Dynamic system parameters that can self-adjust based on emergent behavior.
    """
    
    def __init__(self):
        """Initialize the cognitive parameters."""
        # Core parameters with default values and allowed ranges
        self.parameters = {
            "creativity": {
                "value": 0.5,  # Current value
                "min": 0.1,  # Minimum allowed value
                "max": 1.0,  # Maximum allowed value
                "step": 0.05,  # Adjustment step size
                "description": "Tendency to generate novel connections and ideas"
            },
            "focus": {
                "value": 0.7,
                "min": 0.3,
                "max": 1.0,
                "step": 0.05,
                "description": "Concentration on specific cognitive pathways"
            },
            "plasticity": {
                "value": 0.6,
                "min": 0.2,
                "max": 1.0,
                "step": 0.05,
                "description": "Ability to form new connections and patterns"
            },
            "persistence": {
                "value": 0.5,
                "min": 0.1,
                "max": 0.9,
                "step": 0.05,
                "description": "Tendency to maintain existing patterns"
            },
            "feedback_sensitivity": {
                "value": 0.6,
                "min": 0.2,
                "max": 1.0,
                "step": 0.05,
                "description": "Sensitivity to feedback loops"
            },
            "abstraction": {
                "value": 0.5,
                "min": 0.1,
                "max": 1.0,
                "step": 0.05,
                "description": "Tendency to form high-level patterns from lower-level ones"
            },
            "energy": {
                "value": 0.8,
                "min": 0.3,
                "max": 1.0,
                "step": 0.05,
                "description": "Overall activation energy in the system"
            }
        }
        
        # Parameter adjustment history
        self.adjustment_history = []
        
    def get(self, parameter_name: str) -> float:
        """
        Get a parameter value.
        
        Args:
            parameter_name: Parameter name
            
        Returns:
            Parameter value
        """
        if parameter_name in self.parameters:
            return self.parameters[parameter_name]["value"]
        return 0.5  # Default value for unknown parameters
        
    def adjust(self, parameter_name: str, adjustment: float, reason: str) -> float:
        """
        Adjust a parameter value.
        
        Args:
            parameter_name: Parameter name
            adjustment: Adjustment amount
            reason: Reason for adjustment
            
        Returns:
            New parameter value
        """
        if parameter_name not in self.parameters:
            return 0.5  # Unknown parameter
            
        # Get parameter info
        param = self.parameters[parameter_name]
        
        # Apply adjustment
        old_value = param["value"]
        new_value = min(param["max"], max(param["min"], old_value + adjustment))
        param["value"] = new_value
        
        # Record adjustment
        self.adjustment_history.append({
            "timestamp": datetime.now().isoformat(),
            "parameter": parameter_name,
            "old_value": old_value,
            "new_value": new_value,
            "adjustment": adjustment,
            "reason": reason
        })
        
        return new_value
        
    def adjust_for_pattern(self, pattern_type: str) -> Dict[str, float]:
        """
        Adjust parameters to nurture a specific pattern type.
        
        Args:
            pattern_type: Type of pattern to nurture
            
        Returns:
            Dictionary of adjusted parameters
        """
        adjustments = {}
        
        if pattern_type == "sequence":
            # For sequences, increase focus and persistence
            adjustments["focus"] = self.adjust("focus", 0.05, "Nurturing sequence pattern")
            adjustments["persistence"] = self.adjust("persistence", 0.05, "Nurturing sequence pattern")
            
        elif pattern_type == "feedback_loop":
            # For feedback loops, increase feedback sensitivity and energy
            adjustments["feedback_sensitivity"] = self.adjust("feedback_sensitivity", 0.05, "Nurturing feedback loop")
            adjustments["energy"] = self.adjust("energy", 0.05, "Nurturing feedback loop")
            
        elif pattern_type == "module_cluster":
            # For module clusters, increase plasticity and abstraction
            adjustments["plasticity"] = self.adjust("plasticity", 0.05, "Nurturing module cluster")
            adjustments["abstraction"] = self.adjust("abstraction", 0.05, "Nurturing module cluster")
            
        elif pattern_type == "creative":
            # For creative patterns, increase creativity and plasticity
            adjustments["creativity"] = self.adjust("creativity", 0.05, "Nurturing creative pattern")
            adjustments["plasticity"] = self.adjust("plasticity", 0.05, "Nurturing creative pattern")
            
        return adjustments
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "parameters": self.parameters,
            "adjustment_history": self.adjustment_history
        }
        
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'CognitiveParameters':
        """
        Create from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Created cognitive parameters
        """
        params = CognitiveParameters()
        
        # Set parameters
        if "parameters" in data:
            params.parameters = data["parameters"]
            
        # Set adjustment history
        if "adjustment_history" in data:
            params.adjustment_history = data["adjustment_history"]
            
        return params


class EmergenceFramework:
    """
    System for creating conditions favorable to genuine emergent properties
    by monitoring, nurturing, and analyzing emergent patterns in Sully.
    """

    def __init__(self, all_cognitive_modules=None):
        """
        Initialize the emergence framework.
        
        Args:
            all_cognitive_modules: Dictionary of all cognitive modules
        """
        self.modules = all_cognitive_modules or {}
        
        # Core emergence components
        self.monitor = EmergenceMonitor()
        self.interaction_map = ModuleInteractionMap()
        self.parameters = CognitiveParameters()
        
        # Tracking state
        self.emergence_candidates = []
        self.detected_emergent_properties = []
        self.emergence_logs = []
        
        # Initialize interaction map with module connections
        if self.modules:
            self._initialize_interaction_map()
            
    def _initialize_interaction_map(self) -> None:
        """Initialize interaction map with default connections between modules."""
        module_names = list(self.modules.keys())
        
        # Set up default connections
        for i, source in enumerate(module_names):
            for j, target in enumerate(module_names):
                if i != j:
                    # Default connection strength
                    strength = 0.3  # Conservative default
                    self.interaction_map.set_connection(source, target, strength)
                    
    def register_module_activation(self, module_name: str, 
                                 context: Dict[str, Any] = None) -> None:
        """
        Register a module activation.
        
        Args:
            module_name: Name of the activated module
            context: Optional activation context
        """
        # Track in monitor
        self.monitor.track_module_activation(module_name, context)
        
        # Check for potential cross-module activation
        targets = self.interaction_map.get_targets(module_name, min_strength=0.4)
        for target, strength in targets.items():
            # Probabilistic activation based on connection strength
            if random.random() < strength:
                # Record cross-activation
                self.monitor.track_cross_module_activation(module_name, target, context)
                
                # Log message
                self.emergence_logs.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "cross_activation",
                    "source": module_name,
                    "target": target,
                    "strength": strength,
                    "context": context
                })
                
    def register_direct_activation(self, source_module: str, target_module: str,
                                 context: Dict[str, Any] = None) -> None:
        """
        Register a direct activation of one module by another.
        
        Args:
            source_module: Source module that triggered activation
            target_module: Target module that was activated
            context: Optional activation context
        """
        # Track cross-module activation
        self.monitor.track_cross_module_activation(source_module, target_module, context)
        
        # Strengthen connection slightly
        current = self.interaction_map.get_connection_strength(source_module, target_module)
        new_strength = self.interaction_map.adjust_connection(source_module, target_module, 0.01)
        
        # Log message
        self.emergence_logs.append({
            "timestamp": datetime.now().isoformat(),
            "type": "direct_activation",
            "source": source_module,
            "target": target_module,
            "old_strength": current,
            "new_strength": new_strength,
            "context": context
        })
        
    def detect_emergence(self) -> List[Dict[str, Any]]:
        """
        Detect new emergent patterns.
        
        Returns:
            List of newly detected emergent patterns
        """
        # Run pattern detection
        new_patterns = self.monitor.detect_patterns()
        
        # Process each new pattern
        results = []
        for pattern in new_patterns:
            # Analyze pattern
            analysis = self._analyze_pattern(pattern)
            
            # Check if it's a candidate for emergent property
            is_candidate = analysis.get("emergence_potential", 0) >= 0.7
            
            # Record result
            result = {
                "pattern_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type,
                "description": pattern.description,
                "analysis": analysis,
                "is_candidate": is_candidate
            }
            results.append(result)
            
            # Add to candidates if applicable
            if is_candidate:
                self.emergence_candidates.append({
                    "pattern_id": pattern.pattern_id,
                    "potential": analysis["emergence_potential"],
                    "detected": datetime.now().isoformat()
                })
                
                # Nurture the pattern
                self._nurture_pattern(pattern)
                
            # Log detection
            self.emergence_logs.append({
                "timestamp": datetime.now().isoformat(),
                "type": "pattern_detected",
                "pattern_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type,
                "description": pattern.description,
                "is_candidate": is_candidate
            })
            
        return results
        
    def _analyze_pattern(self, pattern: EmergencePattern) -> Dict[str, Any]:
        """
        Analyze a detected pattern for emergence potential.
        
        Args:
            pattern: Pattern to analyze
            
        Returns:
            Analysis results
        """
        # Factors contributing to emergence potential
        factors = {}
        
        # Number of contributing modules (more is better)
        module_count = len(pattern.contributing_modules)
        factors["module_diversity"] = min(1.0, module_count / 5)
        
        # Pattern stability (higher is better)
        factors["stability"] = pattern.stability
        
        # Pattern complexity
        if pattern.pattern_type == "sequence":
            # Length of sequence
            seq_length = len(pattern.properties.get("sequence", []))
            factors["complexity"] = min(1.0, seq_length / 5)
        elif pattern.pattern_type == "feedback_loop":
            # Length of loop
            loop_length = len(pattern.properties.get("loop", []))
            factors["complexity"] = min(1.0, loop_length / 3)
        elif pattern.pattern_type == "module_cluster":
            # Size and cohesion of cluster
            cluster_size = len(pattern.properties.get("cluster", []))
            cohesion = pattern.properties.get("cohesion", 0.0)
            factors["complexity"] = min(1.0, (cluster_size / 5) * cohesion)
        else:
            factors["complexity"] = 0.5  # Default
            
        # Calculate overall emergence potential
        weights = {
            "module_diversity": 0.3,
            "stability": 0.4,
            "complexity": 0.3
        }
        
        emergence_potential = sum(factor * weights[name] for name, factor in factors.items())
        
        return {
            "factors": factors,
            "emergence_potential": emergence_potential
        }
        
    def _nurture_pattern(self, pattern: EmergencePattern) -> None:
        """
        Adjust system parameters to nurture an emergent pattern.
        
        Args:
            pattern: Pattern to nurture
        """
        # Adjust cognitive parameters
        self.parameters.adjust_for_pattern(pattern.pattern_type)
        
        # Strengthen connections between contributing modules
        modules = list(pattern.contributing_modules)
        for i in range(len(modules)):
            for j in range(i + 1, len(modules)):
                # Strengthen in both directions
                self.interaction_map.adjust_connection(modules[i], modules[j], 0.05)
                self.interaction_map.adjust_connection(modules[j], modules[i], 0.05)
                
        # Log nurturing
        self.emergence_logs.append({
            "timestamp": datetime.now().isoformat(),
            "type": "pattern_nurtured",
            "pattern_id": pattern.pattern_id,
            "pattern_type": pattern.pattern_type,
            "modules": modules
        })
        
    def identify_emergent_properties(self) -> List[Dict[str, Any]]:
        """
        Identify properties that have genuinely emerged from patterns.
        
        Returns:
            List of newly identified emergent properties
        """
        # Criteria for genuine emergence:
        # 1. Pattern has been stable for some time
        # 2. Pattern involves multiple diverse modules
        # 3. Pattern exhibits novel behavior not explicitly programmed
        
        new_properties = []
        candidates_to_remove = []
        
        # Check each candidate
        for candidate in self.emergence_candidates:
            pattern_id = candidate["pattern_id"]
            pattern = self.monitor.get_pattern(pattern_id)
            
            if not pattern:
                # Pattern no longer exists
                candidates_to_remove.append(candidate)
                continue
                
            # Check stability - pattern has been observed consistently
            if pattern.observation_count >= 10 and pattern.stability >= 0.7:
                # Check if already identified as emergent
                already_identified = any(prop["pattern_id"] == pattern_id 
                                       for prop in self.detected_emergent_properties)
                
                if not already_identified:
                    # Analyze for novelty and functional impact
                    impact = self._analyze_functional_impact(pattern)
                    
                    if impact["novelty"] >= 0.6 and impact["functional_impact"] >= 0.5:
                        # This is a genuine emergent property
                        property_data = {
                            "id": str(uuid.uuid4()),
                            "pattern_id": pattern_id,
                            "pattern_type": pattern.pattern_type,
                            "description": pattern.description,
                            "modules": list(pattern.contributing_modules),
                            "detection_time": datetime.now().isoformat(),
                            "novelty": impact["novelty"],
                            "functional_impact": impact["functional_impact"],
                            "characteristics": impact["characteristics"]
                        }
                        
                        self.detected_emergent_properties.append(property_data)
                        new_properties.append(property_data)
                        
                        # Remove from candidates
                        candidates_to_remove.append(candidate)
                        
                        # Log detection
                        self.emergence_logs.append({
                            "timestamp": datetime.now().isoformat(),
                            "type": "emergent_property_detected",
                            "property_id": property_data["id"],
                            "pattern_id": pattern_id,
                            "description": property_data["description"]
                        })
            elif (datetime.now() - datetime.fromisoformat(candidate["detected"])).days > 7:
                # Candidate has been around for a week without meeting criteria
                candidates_to_remove.append(candidate)
                
        # Remove processed candidates
        for candidate in candidates_to_remove:
            if candidate in self.emergence_candidates:
                self.emergence_candidates.remove(candidate)
            
        return new_properties
        
    def _analyze_functional_impact(self, pattern: EmergencePattern) -> Dict[str, Any]:
        """
        Analyze functional impact and novelty of an emergent pattern.
        
        Args:
            pattern: Pattern to analyze
            
        Returns:
            Impact analysis
        """
        # Determine novelty - how unexpected is this pattern?
        # Higher novelty for patterns involving disparate modules or unusual sequences
        
        # Get module categories (simplified for demonstration)
        module_categories = self._get_module_categories()
        
        # Check if pattern crosses module categories
        categories_involved = set()
        for module in pattern.contributing_modules:
            category = module_categories.get(module, "unknown")
            categories_involved.add(category)
            
        # More categories = higher novelty
        category_diversity = len(categories_involved) / max(1, len(pattern.contributing_modules))
        novelty = 0.4 + (category_diversity * 0.6)  # Base novelty + diversity bonus
        
        # Determine functional impact
        # What new capabilities does this pattern enable?
        characteristics = []
        
        if pattern.pattern_type == "feedback_loop":
            # Feedback loops can enable self-regulation
            characteristics.append("self_regulation")
            
            # Check if it's a reinforcing or balancing loop
            loop_modules = pattern.properties.get("loop", [])
            if loop_modules and self._is_reinforcing_loop(loop_modules):
                characteristics.append("amplification")
            else:
                characteristics.append("equilibrium")
                
        elif pattern.pattern_type == "module_cluster":
            # Module clusters can enable specialized processing
            characteristics.append("specialized_processing")
            
            # Check if it's a perception, reasoning, or memory cluster
            if self._is_perception_cluster(pattern.contributing_modules):
                characteristics.append("enhanced_perception")
            elif self._is_reasoning_cluster(pattern.contributing_modules):
                characteristics.append("enhanced_reasoning")
            elif self._is_memory_cluster(pattern.contributing_modules):
                characteristics.append("enhanced_memory")
                
        elif pattern.pattern_type == "sequence":
            # Sequences can enable procedural capabilities
            characteristics.append("procedural_capability")
            
            # Check if it's a creative, analytical, or learning sequence
            seq = pattern.properties.get("sequence", [])
            if seq and self._is_creative_sequence(seq):
                characteristics.append("creative_process")
            elif seq and self._is_analytical_sequence(seq):
                characteristics.append("analytical_process")
            elif seq and self._is_learning_sequence(seq):
                characteristics.append("learning_process")
                
        # Calculate functional impact based on characteristics
        functional_impact = min(1.0, len(characteristics) * 0.25)
        
        return {
            "novelty": novelty,
            "functional_impact": functional_impact,
            "characteristics": characteristics
        }
        
    def _get_module_categories(self) -> Dict[str, str]:
        """
        Get categories for all modules.
        
        Returns:
            Dictionary mapping module names to categories
        """
        # In a real implementation, this would extract from module metadata
        # For demonstration, using a simplified mapping
        categories = {
            "reasoning_node": "reasoning",
            "judgment": "reasoning",
            "intuition": "reasoning",
            "dream": "creative",
            "fusion": "creative",
            "translator": "integration",
            "paradox": "integration",
            "codex": "knowledge",
            "memory": "knowledge",
            "identity": "self",
            "visual_system": "perception",
            "learning_system": "learning",
            "goal_system": "motivation",
            "neural_modifier": "adaptation"
        }
        
        return categories
        
    def _is_reinforcing_loop(self, modules: List[str]) -> bool:
        """
        Check if a feedback loop is reinforcing.
        
        Args:
            modules: Modules in the loop
            
        Returns:
            Whether the loop is reinforcing
        """
        # Simplified implementation - in a real system, would analyze connection types
        # For demonstration, use a probabilistic approach
        return random.random() < 0.5
        
    def _is_perception_cluster(self, modules: Set[str]) -> bool:
        """
        Check if a module cluster is perception-focused.
        
        Args:
            modules: Modules in the cluster
            
        Returns:
            Whether the cluster is perception-focused
        """
        perception_modules = {"visual_system", "translator"}
        return len(perception_modules.intersection(modules)) >= 1
        
    def _is_reasoning_cluster(self, modules: Set[str]) -> bool:
        """
        Check if a module cluster is reasoning-focused.
        
        Args:
            modules: Modules in the cluster
            
        Returns:
            Whether the cluster is reasoning-focused
        """
        reasoning_modules = {"reasoning_node", "judgment", "intuition"}
        return len(reasoning_modules.intersection(modules)) >= 1
        
    def _is_memory_cluster(self, modules: Set[str]) -> bool:
        """
        Check if a module cluster is memory-focused.
        
        Args:
            modules: Modules in the cluster
            
        Returns:
            Whether the cluster is memory-focused
        """
        memory_modules = {"memory", "codex", "learning_system"}
        return len(memory_modules.intersection(modules)) >= 1
        
    def _is_creative_sequence(self, sequence: List[str]) -> bool:
        """
        Check if a module sequence is creativity-focused.
        
        Args:
            sequence: Module sequence
            
        Returns:
            Whether the sequence is creativity-focused
        """
        creative_modules = {"dream", "fusion", "intuition"}
        return any(module in creative_modules for module in sequence)
        
    def _is_analytical_sequence(self, sequence: List[str]) -> bool:
        """
        Check if a module sequence is analytically-focused.
        
        Args:
            sequence: Module sequence
            
        Returns:
            Whether the sequence is analytically-focused
        """
        analytical_modules = {"reasoning_node", "judgment", "translator"}
        return any(module in analytical_modules for module in sequence)
        
    def _is_learning_sequence(self, sequence: List[str]) -> bool:
        """
        Check if a module sequence is learning-focused.
        
        Args:
            sequence: Module sequence
            
        Returns:
            Whether the sequence is learning-focused
        """
        learning_modules = {"learning_system", "neural_modifier", "memory"}
        return any(module in learning_modules for module in sequence)
        
    def get_activation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about module activations.
        
        Returns:
            Activation statistics
        """
        return self.monitor.get_module_activation_stats()
        
    def get_emergent_properties(self) -> List[Dict[str, Any]]:
        """
        Get list of detected emergent properties.
        
        Returns:
            List of emergent properties
        """
        return self.detected_emergent_properties
        
    def get_interaction_map(self) -> Dict[str, Any]:
        """
        Get the module interaction map.
        
        Returns:
            Interaction map data
        """
        return self.interaction_map.to_dict()
        
    def get_cognitive_parameters(self) -> Dict[str, Any]:
        """
        Get current cognitive parameters.
        
        Returns:
            Cognitive parameter values
        """
        return {name: param["value"] for name, param in self.parameters.parameters.items()}
        
    def save_state(self, filepath: str) -> Dict[str, Any]:
        """
        Save emergence framework state to file.
        
        Args:
            filepath: Path to save the state
            
        Returns:
            Save results
        """
        try:
            # Prepare state data
            state = {
                "interaction_map": self.interaction_map.to_dict(),
                "parameters": self.parameters.to_dict(),
                "detected_patterns": {pid: pattern.to_dict() for pid, pattern in self.monitor.detected_patterns.items()},
                "pattern_registry": self.monitor.pattern_registry,
                "emergence_candidates": self.emergence_candidates,
                "detected_emergent_properties": self.detected_emergent_properties,
                "recent_logs": self.emergence_logs[-100:] if self.emergence_logs else [],
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
                
            return {
                "success": True,
                "filepath": filepath,
                "timestamp": state["timestamp"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "filepath": filepath
            }
            
    def load_state(self, filepath: str) -> Dict[str, Any]:
        """
        Load emergence framework state from file.
        
        Args:
            filepath: Path to load the state from
            
        Returns:
            Load results
        """
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                return {
                    "success": False,
                    "error": f"File not found: {filepath}",
                    "filepath": filepath
                }
                
            # Load from file
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
                
            # Load interaction map
            if "interaction_map" in state:
                self.interaction_map = ModuleInteractionMap.from_dict(state["interaction_map"])
                
            # Load parameters
            if "parameters" in state:
                self.parameters = CognitiveParameters.from_dict(state["parameters"])
                
            # Load detected patterns
            if "detected_patterns" in state:
                self.monitor.detected_patterns = {}
                for pid, pattern_data in state["detected_patterns"].items():
                    self.monitor.detected_patterns[pid] = EmergencePattern.from_dict(pattern_data)
                    
            # Load pattern registry
            if "pattern_registry" in state:
                self.monitor.pattern_registry = state["pattern_registry"]
                
            # Load emergence candidates
            if "emergence_candidates" in state:
                self.emergence_candidates = state["emergence_candidates"]
                
            # Load detected emergent properties
            if "detected_emergent_properties" in state:
                self.detected_emergent_properties = state["detected_emergent_properties"]
                
            # Load recent logs
            if "recent_logs" in state:
                self.emergence_logs = state["recent_logs"]
                
            return {
                "success": True,
                "filepath": filepath,
                "timestamp": state.get("timestamp", "unknown")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "filepath": filepath
            }