# sully_engine/kernel_integration.py
# 🧠 SuperPowered Kernel Integration System

from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import json
import os
from datetime import datetime
import random
import re

# Import PDF reader
from sully_engine.pdf_reader import PDFReader, extract_kernel_from_text

# Import the enhanced codex
from sully_engine.kernel_modules.enhanced_codex import EnhancedCodex

# Import kernels
from sully_engine.kernel_modules.dream import DreamCore
from sully_engine.kernel_modules.fusion import SymbolFusionEngine
from sully_engine.kernel_modules.paradox import ParadoxLibrary
from sully_engine.kernel_modules.math_translator import SymbolicMathTranslator
from sully_engine.conversation_engine import ConversationEngine
from sully_engine.memory_integration import MemoryIntegration, integrate_with_sully as integrate_memory_with_sully

class KernelIntegrationSystem:
    """
    Central integration system for Sully's cognitive kernels.
    
    Connects and enhances Dream, Fusion, Paradox, Math Translation, Conversation,
    Memory Integration, and other kernels through the enhanced codex, enabling 
    cross-module reasoning and emergent capabilities.
    """
    
    def __init__(self, 
                codex: Optional[EnhancedCodex] = None,
                dream_core: Optional[DreamCore] = None,
                fusion_engine: Optional[SymbolFusionEngine] = None,
                paradox_library: Optional[ParadoxLibrary] = None,
                math_translator: Optional[SymbolicMathTranslator] = None,
                conversation_engine: Optional[ConversationEngine] = None,
                memory_integration: Optional[MemoryIntegration] = None,
                sully_instance = None):
        """
        Initialize the kernel integration system.
        
        Args:
            codex: Optional existing EnhancedCodex instance
            dream_core: Optional existing DreamCore instance
            fusion_engine: Optional existing SymbolFusionEngine instance
            paradox_library: Optional existing ParadoxLibrary instance
            math_translator: Optional existing SymbolicMathTranslator instance
            conversation_engine: Optional existing ConversationEngine instance
            memory_integration: Optional existing MemoryIntegration instance
            sully_instance: Optional reference to main Sully instance
        """
        # Initialize or use provided components
        self.codex = codex if codex else EnhancedCodex()
        self.dream_core = dream_core if dream_core else DreamCore()
        self.fusion_engine = fusion_engine if fusion_engine else SymbolFusionEngine()
        self.paradox_library = paradox_library if paradox_library else ParadoxLibrary()
        self.math_translator = math_translator if math_translator else SymbolicMathTranslator()
        self.conversation_engine = conversation_engine
        self.memory_integration = memory_integration
        self.sully = sully_instance
        
        # Initialize PDF reader
        self.pdf_reader = PDFReader(ocr_enabled=True, dpi=300)
        
        # Integration mappings between kernels
        self.kernel_mappings = {
            "dream_fusion": {},      # How dream symbols map to fusion concepts
            "dream_paradox": {},     # How dream patterns relate to paradoxes
            "fusion_paradox": {},    # How fused concepts generate paradoxes
            "math_dream": {},        # How mathematical concepts map to dream symbols
            "math_fusion": {},       # How mathematical operations relate to fusion styles
            "math_paradox": {},      # How mathematical structures relate to paradox types
            "conversation_dream": {}, # How conversation topics map to dream styles
            "conversation_fusion": {}, # How conversation tones map to fusion styles
            "memory_context": {}      # How memory traces affect other kernel operations
        }
        
        # Cross-kernel transformation functions
        self.transformations = {
            "dream_to_fusion": self._transform_dream_to_fusion,
            "dream_to_paradox": self._transform_dream_to_paradox,
            "fusion_to_dream": self._transform_fusion_to_dream,
            "fusion_to_paradox": self._transform_fusion_to_paradox,
            "paradox_to_dream": self._transform_paradox_to_dream,
            "paradox_to_fusion": self._transform_paradox_to_fusion,
            "math_to_dream": self._transform_math_to_dream,
            "math_to_fusion": self._transform_math_to_fusion,
            "math_to_paradox": self._transform_math_to_paradox,
            "conversation_to_dream": self._transform_conversation_to_dream,
            "conversation_to_fusion": self._transform_conversation_to_fusion,
            "memory_to_dream": self._transform_memory_to_dream,
            "memory_to_fusion": self._transform_memory_to_fusion
        }
        
        # Enhance kernels with integration capabilities
        self._enhance_dream_core()
        self._enhance_fusion_engine()
        self._enhance_paradox_library()
        self._enhance_math_translator()
        
        # Enhance communication and memory if provided
        if conversation_engine:
            self._enhance_conversation_engine()
        
        if memory_integration:
            self._enhance_memory_integration()
        
        # Create initial integration mappings
        self._initialize_kernel_mappings()
        
        # Initialize cross-kernel concept space
        self.concept_space = {}
        
        # Track kernel integration history
        self.integration_history = []

    # Note: methods to enhance kernels would go here
    # Including _enhance_dream_core, _enhance_fusion_engine, etc.
    # And transformation methods like _transform_dream_to_fusion, etc.
    
    def _enhance_dream_core(self):
        """Enhances the DreamCore with integration capabilities."""
        # Implementation would go here
        pass
        
    def _enhance_fusion_engine(self):
        """Enhances the SymbolFusionEngine with integration capabilities."""
        # Implementation would go here
        pass
        
    def _enhance_paradox_library(self):
        """Enhances the ParadoxLibrary with integration capabilities."""
        # Implementation would go here
        pass
        
    def _enhance_math_translator(self):
        """Enhances the SymbolicMathTranslator with integration capabilities."""
        # Implementation would go here
        pass
        
    def _enhance_conversation_engine(self):
        """Enhances the ConversationEngine with integration capabilities."""
        # Implementation would go here
        pass
        
    def _enhance_memory_integration(self):
        """Enhances the MemoryIntegration with integration capabilities."""
        # Implementation would go here
        pass
    
    def _initialize_kernel_mappings(self):
        """Creates initial mappings between kernel concepts."""
        # Implementation would go here
        pass
        
    def _transform_dream_to_fusion(self, dream_result):
        """Transforms a dream result into a fusion operation."""
        # Implementation would go here
        pass
        
    def _transform_dream_to_paradox(self, dream_result):
        """Transforms a dream result into a paradox query."""
        # Implementation would go here
        pass
    
    def _transform_fusion_to_dream(self, fusion_result):
        """Transforms a fusion result into a dream generation."""
        # Implementation would go here
        pass
    
    def _transform_fusion_to_paradox(self, fusion_result):
        """Transforms a fusion result into a paradox generation."""
        # Implementation would go here
        pass
    
    def _transform_paradox_to_dream(self, paradox_result):
        """Transforms a paradox result into a dream generation."""
        # Implementation would go here
        pass
    
    def _transform_paradox_to_fusion(self, paradox_result):
        """Transforms a paradox result into a fusion operation."""
        # Implementation would go here
        pass
    
    def _transform_math_to_dream(self, math_result):
        """Transforms a math translation result into a dream generation."""
        # Implementation would go here
        pass
    
    def _transform_math_to_fusion(self, math_result):
        """Transforms a math translation result into a fusion operation."""
        # Implementation would go here
        pass
    
    def _transform_math_to_paradox(self, math_result):
        """Transforms a math translation result into a paradox query."""
        # Implementation would go here
        pass
    
    def _transform_conversation_to_dream(self, conversation_data):
        """Transforms conversation data into dream parameters."""
        # Implementation would go here
        pass
    
    def _transform_conversation_to_fusion(self, conversation_data):
        """Transforms conversation data into fusion parameters."""
        # Implementation would go here
        pass
    
    def _transform_memory_to_dream(self, memory_data):
        """Transforms memory data into dream parameters."""
        # Implementation would go here
        pass
    
    def _transform_memory_to_fusion(self, memory_data):
        """Transforms memory data into fusion parameters."""
        # Implementation would go here
        pass

    def _add_integration_record(self, integration_type, data):
        """
        Adds a record of kernel integration to the history.
        
        Args:
            integration_type: Type of integration performed
            data: Integration data
        """
        record = {
            "type": integration_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self.integration_history.append(record)

    def cross_kernel_operation(self, source_kernel: str, target_kernel: str, 
                            input_data: Any) -> Dict[str, Any]:
        """
        Performs a cross-kernel operation, transforming the output of one kernel
        into the input for another kernel.
        
        Args:
            source_kernel: Source kernel ("dream", "fusion", "paradox", "math", "conversation", "memory")
            target_kernel: Target kernel ("dream", "fusion", "paradox", "math", "conversation", "memory")
            input_data: Input data for the source kernel
            
        Returns:
            Dictionary with results from both kernels
        """
        # Validate kernels
        valid_kernels = ["dream", "fusion", "paradox", "math", "conversation", "memory"]
        if source_kernel not in valid_kernels or target_kernel not in valid_kernels:
            return {
                "error": f"Invalid kernel specified. Valid kernels are: {', '.join(valid_kernels)}"
            }
            
        # Process with source kernel
        source_result = None
        
        # Implementation for source kernel processing would go here
        
        # Transform to target kernel
        transformation_key = f"{source_kernel}_to_{target_kernel}"
        transformation_function = self.transformations.get(transformation_key)
        
        if not transformation_function:
            return {
                "error": f"No transformation available from {source_kernel} to {target_kernel}",
                "source_result": source_result
            }
            
        # Transform the result
        target_params = transformation_function(source_result)
        
        # Process with target kernel
        target_result = None
        
        # Implementation for target kernel processing would go here
                
        # Record the cross-kernel operation
        self._add_integration_record("cross_kernel", {
            "source_kernel": source_kernel,
            "target_kernel": target_kernel,
            "transformation": transformation_key,
            "source_input": input_data,
            "target_params": target_params
        })
                
        # Return combined results
        return {
            "source_kernel": source_kernel,
            "source_result": source_result,
            "target_kernel": target_kernel,
            "target_result": target_result,
            "transformation": transformation_key
        }

    def create_concept_network(self, seed_concept: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Creates a rich concept network by applying multiple kernels and 
        integrating the results through the codex.
        
        Args:
            seed_concept: Initial concept to start from
            max_depth: Maximum depth of concept exploration
            
        Returns:
            Dictionary with integrated concept network
        """
        # Implementation would go here
        pass

    def recursive_concept_exploration(self, seed_concept: str, max_depth: int = 3, 
                                  exploration_breadth: int = 2) -> Dict[str, Any]:
        """
        Performs a deep recursive exploration of a concept, alternating between
        different cognitive kernels at each depth.
        
        Args:
            seed_concept: Initial concept to explore
            max_depth: Maximum depth of recursion
            exploration_breadth: Number of branches to explore at each level
            
        Returns:
            Dictionary with recursive exploration results
        """
        # Implementation would go here
        pass

    def _generate_exploration_insights(self, exploration: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyzes a concept exploration to generate insights.
        
        Args:
            exploration: Recursive exploration results
            
        Returns:
            List of insights
        """
        # Implementation would go here
        pass

    def generate_cross_kernel_narrative(self, concept: str, include_kernels: List[str] = None) -> Dict[str, Any]:
        """
        Generates a cohesive narrative that integrates insights from multiple kernels
        around a central concept.
        
        Args:
            concept: Central concept for the narrative
            include_kernels: Optional list of specific kernels to include
            
        Returns:
            Dictionary with the integrated narrative
        """
        # Use all available kernels if none specified
        if not include_kernels:
            include_kernels = ["dream", "fusion", "paradox", "math"]
            
            # Add conversation and memory if available
            if self.conversation_engine:
                include_kernels.append("conversation")
            if self.memory_integration:
                include_kernels.append("memory")
            
        # Initialize narrative structure
        narrative = {
            "concept": concept,
            "title": f"Integrated Exploration of {concept.title()}",
            "sections": [],
            "kernel_outputs": {},
            "integrations": [],
            "conclusion": ""
        }
        
        # Generate content from each kernel
        kernel_outputs = {}
        
        # Dream kernel
        if "dream" in include_kernels:
            dream_result = self.dream_core.generate(concept, depth="deep")
            kernel_outputs["dream"] = dream_result
            
            # Extract concepts for integration
            dream_concepts = []
            if isinstance(dream_result, str):
                dream_concepts = re.findall(r'\'([^\']+)\'', dream_result)
                
            # Add narrative section
            narrative["sections"].append({
                "title": f"Dream Exploration of {concept}",
                "content": dream_result if isinstance(dream_result, str) else str(dream_result),
                "type": "dream",
                "concepts": dream_concepts[:3]  # Take first 3 concepts
            })
            
        # Fusion kernel
        if "fusion" in include_kernels:
            # Find a concept to fuse with
            fusion_with = None
            
            # Use a concept from dream if available
            if "dream" in kernel_outputs and dream_concepts:
                fusion_with = dream_concepts[0]
            else:
                # Otherwise get a related concept from codex
                related = self.codex.get_related_concepts(concept, max_depth=1)
                if related:
                    fusion_with = next(iter(related.keys()))
                else:
                    # Default second concept
                    fusion_with = "understanding"
                    
            # Generate fusion
            fusion_result = self.fusion_engine.fuse_with_options(
                concept, 
                fusion_with, 
                output_format="dict"
            )
            kernel_outputs["fusion"] = fusion_result
            
            # Add narrative section
            narrative["sections"].append({
                "title": f"Fusion of {concept} with {fusion_with}",
                "content": fusion_result.get("formatted_result", str(fusion_result)),
                "type": "fusion",
                "concepts": [concept, fusion_with]
            })
            
            # Record integration between dream and fusion
            if "dream" in kernel_outputs:
                narrative["integrations"].append({
                    "type": "dream_to_fusion",
                    "description": f"The dream exploration revealed {fusion_with}, which became the fusion partner for {concept}."
                })
            
        # Paradox kernel
        if "paradox" in include_kernels:
            paradox_result = self.paradox_library.get(concept)
            kernel_outputs["paradox"] = paradox_result
            
            # Add narrative section
            paradox_content = paradox_result.get("description", "")
            if "reframed" in paradox_result:
                paradox_content += f"\n\nReframed understanding: {paradox_result['reframed']}"
                
            narrative["sections"].append({
                "title": f"The Paradox of {concept}",
                "content": paradox_content,
                "type": "paradox",
                "concepts": paradox_result.get("related_concepts", [])
            })
            
            # Record integration with previous kernels
            if "dream" in kernel_outputs:
                narrative["integrations"].append({
                    "type": "dream_to_paradox",
                    "description": f"The dream's symbolic representation of {concept} manifests the paradoxical tension explored here."
                })
                
            if "fusion" in kernel_outputs:
                narrative["integrations"].append({
                    "type": "fusion_to_paradox",
                    "description": f"The fusion process reveals complementary aspects that help navigate the paradox of {concept}."
                })
            
        # Math kernel
        if "math" in include_kernels:
            math_result = self.math_translator.translate(concept)
            kernel_outputs["math"] = math_result
            
            # Prepare math content
            math_content = f"Mathematical representation of {concept}:\n"
            if "matches" in math_result:
                for term, symbol in math_result["matches"].items():
                    math_content += f"• {term}: {symbol}\n"
                    
            if "explanation" in math_result:
                math_content += f"\n{math_result['explanation']}"
                
            # Add narrative section
            narrative["sections"].append({
                "title": f"Mathematical Perspective on {concept}",
                "content": math_content,
                "type": "math",
                "concepts": list(math_result.get("matches", {}).keys())
            })
            
            # Record integration with previous kernels
            if "dream" in kernel_outputs:
                narrative["integrations"].append({
                    "type": "dream_to_math",
                    "description": f"The dream's symbolic imagery can be mapped to mathematical notation, providing structure to intuition."
                })
                
            if "paradox" in kernel_outputs:
                narrative["integrations"].append({
                    "type": "paradox_to_math",
                    "description": f"Mathematical formalism offers a framework for containing and exploring the paradox of {concept}."
                })
        
        # Conversation kernel
        if "conversation" in include_kernels and self.conversation_engine:
            # Generate conversation about the concept
            conversation_result = self.conversation_engine.process_message(
                f"Let's explore the concept of {concept} in depth. What are the key aspects, implications, and related ideas?",
                "analytical",
                True
            )
            kernel_outputs["conversation"] = conversation_result
            
            # Extract topics from conversation
            conversation_topics = self.conversation_engine._extract_topics(conversation_result)
            
            # Add narrative section
            narrative["sections"].append({
                "title": f"Conversational Exploration of {concept}",
                "content": conversation_result,
                "type": "conversation",
                "concepts": conversation_topics[:3]  # Take first 3 topics
            })
            
            # Record integration with previous kernels
            if "dream" in kernel_outputs:
                narrative["integrations"].append({
                    "type": "dream_to_conversation",
                    "description": f"The dream imagery provides symbolic depth to the conversational exploration of {concept}."
                })
                
            if "fusion" in kernel_outputs:
                narrative["integrations"].append({
                    "type": "fusion_to_conversation",
                    "description": f"The fusion of concepts enriches the conversational dialogue with unexpected connections."
                })
                
            if "paradox" in kernel_outputs:
                narrative["integrations"].append({
                    "type": "paradox_to_conversation",
                    "description": f"The paradoxical tensions in {concept} drive the dialectical movement of the conversation."
                })
        
        # Memory kernel
        if "memory" in include_kernels and self.memory_integration:
            # Search memory for the concept
            memory_results = self.memory_integration.recall(
                concept,
                limit=3,
                module="kernel_integration"
            )
            kernel_outputs["memory"] = memory_results
            
            # Prepare memory content
            memory_content = f"Memory traces related to {concept}:\n\n"
            
            memory_concepts = []
            for memory in memory_results:
                # Format memory
                if isinstance(memory.get("content"), dict):
                    # From structured memory
                    query = memory["content"].get("query", "")
                    result = memory["content"].get("result", "")
                    if query:
                        memory_content += f"• Relevant query: {query}\n"
                        memory_content += f"  Response: {str(result)[:200]}...\n\n"
                        
                        # Extract concepts
                        words = query.split()
                        significant_words = [w for w in words if len(w) > 4 and w.lower() not in 
                                           ["about", "would", "could", "should", "there"]]
                        if significant_words:
                            memory_concepts.append(significant_words[0])
                            
                elif isinstance(memory.get("content"), str):
                    # From string memory
                    memory_content += f"• Memory: {memory['content'][:200]}...\n\n"
                    
                    # Extract concepts
                    words = memory["content"].split()
                    significant_words = [w for w in words if len(w) > 4 and w.lower() not in 
                                       ["about", "would", "could", "should", "there"]]
                    if significant_words:
                        memory_concepts.append(significant_words[0])
            
            # If no memories found
            if not memory_results:
                memory_content += "No specific memories found for this concept."
            
            # Add narrative section
            narrative["sections"].append({
                "title": f"Memory Traces of {concept}",
                "content": memory_content,
                "type": "memory",
                "concepts": memory_concepts[:3]  # Take first 3 concepts
            })
            
            # Record integration with previous kernels
            if "dream" in kernel_outputs:
                narrative["integrations"].append({
                    "type": "dream_to_memory",
                    "description": f"The dream's imagery resonates with memory traces, creating echoes of past encounters with {concept}."
                })
                
            if "conversation" in kernel_outputs:
                narrative["integrations"].append({
                    "type": "conversation_to_memory",
                    "description": f"The conversational exploration draws upon and enriches the memory context for {concept}."
                })
        
        # Store all kernel outputs
        narrative["kernel_outputs"] = kernel_outputs
        
        # Generate conclusion that integrates all perspectives
        conclusion_components = []
        
        # Analyze which domains were touched
        domains = set()
        for section in narrative["sections"]:
            if "concepts" in section:
                for concept_term in section["concepts"]:
                    concept_data = self.codex.get(concept_term)
                    if "domain" in concept_data:
                        domains.add(concept_data["domain"])
                        
        # Add domain insight
        if domains:
            domains_text = ", ".join(list(domains)[:3])
            conclusion_components.append(f"This exploration of {concept} spans the domains of {domains_text}.")
            
        # Add integration insight
        if len(include_kernels) > 1:
            modes = []
            if "dream" in include_kernels:
                modes.append("symbolic")
            if "fusion" in include_kernels:
                modes.append("synthetic")
            if "paradox" in include_kernels:
                modes.append("dialectical")
            if "math" in include_kernels:
                modes.append("formal")
            if "conversation" in include_kernels:
                modes.append("dialogical")
            if "memory" in include_kernels:
                modes.append("experiential")
                
            modes_text = ", ".join(modes)
            conclusion_components.append(f"By viewing {concept} through {modes_text} modes of cognition, a more complete understanding emerges.")
            
        # Add specifics about what was learned
        if "paradox" in kernel_outputs:
            paradox_type = kernel_outputs["paradox"].get("type", "")
            if paradox_type:
                conclusion_components.append(f"The {paradox_type.replace('_', ' ')} nature of {concept} reveals tensions that drive its conceptual evolution.")
                
        if "math" in kernel_outputs and "matches" in kernel_outputs["math"]:
            conclusion_components.append(f"Mathematical formalization provides precision to our understanding of {concept}, grounding intuition in structure.")
            
        if "conversation" in kernel_outputs:
            conclusion_components.append(f"The conversational exploration reveals how {concept} functions in dialectical exchange, highlighting its communicative dimensions.")
            
        if "memory" in kernel_outputs:
            memory_results = kernel_outputs["memory"]
            if memory_results and len(memory_results) > 0:
                conclusion_components.append(f"Memory traces connect {concept} to experiential contexts, embedding abstract understanding in lived instances.")
            
        # Combine conclusion components
        narrative["conclusion"] = " ".join(conclusion_components)
        
        # Record the integration
        self._add_integration_record("cross_kernel_narrative", {
            "concept": concept,
            "included_kernels": include_kernels,
            "sections": len(narrative["sections"]),
            "integrations": len(narrative["integrations"])
        })
        
        return narrative

    def save_integration_state(self, filepath: str) -> str:
        """
        Saves the current integration system state to a file.
        
        Args:
            filepath: Path to save the state
            
        Returns:
            Confirmation message
        """
        state = {
            "kernel_mappings": self.kernel_mappings,
            "concept_space": self.concept_space,
            "integration_history": self.integration_history,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            return f"Integration state saved to {filepath}"
        except Exception as e:
            return f"Error saving integration state: {e}"

    def load_integration_state(self, filepath: str) -> str:
        """
        Loads integration system state from a file.
        
        Args:
            filepath: Path to load the state from
            
        Returns:
            Confirmation message
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
                
            if "kernel_mappings" in state:
                self.kernel_mappings = state["kernel_mappings"]
            if "concept_space" in state:
                self.concept_space = state["concept_space"]
            if "integration_history" in state:
                self.integration_history = state["integration_history"]
                
            return f"Integration state loaded from {filepath}"
        except Exception as e:
            return f"Error loading integration state: {e}"

    def get_integration_statistics(self) -> Dict[str, Any]:
        """
        Returns statistics about the integration system usage.
        
        Returns:
            Dictionary with integration statistics
        """
        # Count integration operations by type
        operation_counts = {}
        for record in self.integration_history:
            op_type = record["type"]
            if op_type in operation_counts:
                operation_counts[op_type] += 1
            else:
                operation_counts[op_type] = 1
                
        # Count concepts by domain
        domain_counts = {}
        for concept in self.concept_space:
            domain = self.concept_space[concept].get("domain", "unknown")
            if domain in domain_counts:
                domain_counts[domain] += 1
            else:
                domain_counts[domain] = 1
                
        # Analyze cross-kernel operations
        cross_kernel_ops = [record for record in self.integration_history if record["type"] == "cross_kernel"]
        cross_kernel_paths = {}
        
        for op in cross_kernel_ops:
            if "data" in op and "source_kernel" in op["data"] and "target_kernel" in op["data"]:
                path = f"{op['data']['source_kernel']}_to_{op['data']['target_kernel']}"
                if path in cross_kernel_paths:
                    cross_kernel_paths[path] += 1
                else:
                    cross_kernel_paths[path] = 1
        
        # Calculate kernel usage statistics
        kernel_usage = {
            "dream": 0,
            "fusion": 0,
            "paradox": 0,
            "math": 0,
            "conversation": 0,
            "memory": 0
        }
        
        for record in self.integration_history:
            # Count direct kernel operations
            if record["type"] in kernel_usage:
                kernel_usage[record["type"]] += 1
            
            # Count cross-kernel operations
            if record["type"] == "cross_kernel" and "data" in record:
                data = record["data"]
                if "source_kernel" in data and data["source_kernel"] in kernel_usage:
                    kernel_usage[data["source_kernel"]] += 1
                if "target_kernel" in data and data["target_kernel"] in kernel_usage:
                    kernel_usage[data["target_kernel"]] += 1
        
        # Calculate most productive integration pathways
        integration_productivity = {}
        for record in self.integration_history:
            if record["type"] == "cross_kernel_narrative" and "data" in record:
                kernels = record["data"].get("included_kernels", [])
                for i in range(len(kernels)):
                    for j in range(i+1, len(kernels)):
                        pair = f"{kernels[i]}_{kernels[j]}"
                        if pair in integration_productivity:
                            integration_productivity[pair] += 1
                        else:
                            integration_productivity[pair] = 1
        
        # Find top integration pairs
        top_integration_pairs = sorted(integration_productivity.items(), key=lambda x: x[1], reverse=True)[:5]
                
        return {
            "total_integrations": len(self.integration_history),
            "integration_types": operation_counts,
            "concept_domains": domain_counts,
            "cross_kernel_paths": cross_kernel_paths,
            "kernel_usage": kernel_usage,
            "top_integration_pairs": dict(top_integration_pairs),
            "last_integration": self.integration_history[-1]["timestamp"] if self.integration_history else None
        }
    
    def integrate_with_sully(self, sully_instance) -> None:
        """
        Integrates the kernel integration system with the main Sully instance.
        
        Args:
            sully_instance: The main Sully instance to integrate with
        """
        if not sully_instance:
            return
        
        # Store reference to Sully
        self.sully = sully_instance
        
        # Add method to Sully for accessing integrated kernels
        def access_integrated_kernels(self, concept: str, kernels: List[str] = None) -> Dict[str, Any]:
            """
            Access multiple cognitive kernels integrated around a concept.
            
            Args:
                concept: The concept to explore
                kernels: List of kernels to include (defaults to all)
                
            Returns:
                Integrated results from all specified kernels
            """
            # Use kernel integration system to generate a cross-kernel narrative
            return self.kernel_integration.generate_cross_kernel_narrative(concept, kernels)
        
        # Add the method to Sully
        sully_instance.access_integrated_kernels = access_integrated_kernels.__get__(sully_instance)
        
        # Add method for recursive concept exploration
        def explore_concept_network(self, concept: str, depth: int = 2) -> Dict[str, Any]:
            """
            Explore a concept network using all cognitive kernels.
            
            Args:
                concept: The central concept to explore
                depth: Depth of exploration
                
            Returns:
                Concept network with integrated insights
            """
            return self.kernel_integration.create_concept_network(concept, depth)
        
        # Add the method to Sully
        sully_instance.explore_concept_network = explore_concept_network.__get__(sully_instance)
        
        # Add method for deep recursive concept exploration
        def deep_explore_concept(self, concept: str, depth: int = 3, breadth: int = 2) -> Dict[str, Any]:
            """
            Perform deep recursive exploration of a concept.
            
            Args:
                concept: The seed concept to explore
                depth: Maximum depth of recursion
                breadth: Number of branches to explore at each level
                
            Returns:
                Recursive exploration results with insights
            """
            return self.kernel_integration.recursive_concept_exploration(concept, depth, breadth)
        
        # Add the method to Sully
        sully_instance.deep_explore_concept = deep_explore_concept.__get__(sully_instance)
        
        # Add method for cross-kernel operations
        def cross_kernel_process(self, source_kernel: str, target_kernel: str, input_data: Any) -> Dict[str, Any]:
            """
            Perform a cross-kernel operation.
            
            Args:
                source_kernel: Source kernel
                target_kernel: Target kernel
                input_data: Input data for the source kernel
                
            Returns:
                Results from both kernels
            """
            return self.kernel_integration.cross_kernel_operation(source_kernel, target_kernel, input_data)
        
        # Add the method to Sully
        sully_instance.cross_kernel_process = cross_kernel_process.__get__(sully_instance)
        
        # Add the kernel integration instance to Sully
        sully_instance.kernel_integration = self
        
        # Register integration with Sully
        self._add_integration_record("sully_integration", {
            "timestamp": datetime.now().isoformat(),
            "methods_added": [
                "access_integrated_kernels",
                "explore_concept_network",
                "deep_explore_concept",
                "cross_kernel_process"
            ]
        })

    # PDF Integration Methods
    
    def process_pdf(self, pdf_path: str, extract_structure: bool = True, 
                  use_ocr_fallback: bool = True) -> Dict[str, Any]:
        """
        Process a PDF through the kernel integration system.
        
        Args:
            pdf_path: Path to the PDF file
            extract_structure: Whether to attempt extracting document structure
            use_ocr_fallback: Whether to use OCR as a fallback if native extraction fails
            
        Returns:
            Dictionary with extraction results and kernel insights
        """
        # Extract text from the PDF
        extraction_result = self.pdf_reader.extract_text(
            pdf_path, 
            verbose=True,
            use_ocr_fallback=use_ocr_fallback,
            extract_structure=extract_structure
        )
        
        if not extraction_result["success"]:
            return {
                "error": "PDF extraction failed",
                "details": extraction_result.get("error", "Unknown error")
            }
        
        # Extract key concepts from the text
        full_text = extraction_result["text"]
        key_concepts = self._extract_key_concepts(full_text)
        
        # Generate insights from different kernels
        kernel_insights = {
            "dream": None,
            "fusion": None,
            "paradox": None,
            "math": None
        }
        
        # Process the first page for a quick dream if available
        if extraction_result["pages"] and extraction_result["pages"][0]["text"]:
            first_page = extraction_result["pages"][0]["text"]
            # Generate a dream based on the first page
            try:
                dream_seed = key_concepts[0] if key_concepts else "document"
                kernel_insights["dream"] = self.dream_core.generate(dream_seed, "standard")
            except Exception as e:
                kernel_insights["dream"] = f"Dream generation error: {str(e)}"
        
        # Generate fusion of top concepts if multiple concepts found
        if len(key_concepts) >= 2:
            try:
                kernel_insights["fusion"] = self.fusion_engine.fuse_with_options(
                    key_concepts[0], 
                    key_concepts[1],
                    output_format="dict"
                )
            except Exception as e:
                kernel_insights["fusion"] = f"Fusion error: {str(e)}"
        
        # Look for paradoxes in the text
        try:
            if key_concepts:
                kernel_insights["paradox"] = self.paradox_library.get(key_concepts[0])
        except Exception as e:
            kernel_insights["paradox"] = f"Paradox error: {str(e)}"
        
        # Generate mathematical translations for key concepts
        try:
            if key_concepts:
                kernel_insights["math"] = self.math_translator.translate(key_concepts[0])
        except Exception as e:
            kernel_insights["math"] = f"Math translation error: {str(e)}"
        
        # Process with conversation engine if available
        conversation_insight = None
        if self.conversation_engine and full_text:
            try:
                # Take the first 2000 characters for conversation to avoid overload
                summary_prompt = f"Summarize the key insights from this document excerpt: {full_text[:2000]}..."
                conversation_insight = self.conversation_engine.process_message(summary_prompt, "analytical", False)
            except Exception as e:
                conversation_insight = f"Conversation error: {str(e)}"
        
        # Store in memory if integration available
        if self.memory_integration:
            try:
                self.memory_integration.store_experience(
                    content=f"Processed PDF document: {os.path.basename(pdf_path)}",
                    source="pdf_processing",
                    importance=0.8,
                    concepts=key_concepts,
                    emotional_tags={"curiosity": 0.7, "analytical": 0.8}
                )
            except Exception as e:
                print(f"Memory storage error: {str(e)}")
        
        # Assemble the result
        result = {
            "extraction": extraction_result,
            "key_concepts": key_concepts,
            "kernel_insights": kernel_insights,
            "conversation_insight": conversation_insight
        }
        
        # Record the integration
        self._add_integration_record("pdf_processing", {
            "pdf_path": pdf_path,
            "concepts_found": len(key_concepts),
            "page_count": extraction_result.get("page_count", 0)
        })
        
        return result

    def extract_document_kernel(self, pdf_path: str, domain: str = "general") -> Dict[str, Any]:
        """
        Extract a symbolic kernel from a PDF document for cross-kernel operations.
        
        Args:
            pdf_path: Path to the PDF file
            domain: Target domain for the symbolic kernel
            
        Returns:
            Symbolic kernel with domain elements
        """
        # Extract text from the PDF
        extraction_result = self.pdf_reader.extract_text(pdf_path, verbose=True)
        
        if not extraction_result["success"]:
            return {
                "error": "PDF extraction failed",
                "details": extraction_result.get("error", "Unknown error")
            }
        
        # Extract kernel from the text
        kernel = extract_kernel_from_text(extraction_result["text"], domain)
        
        # Enhance the kernel with additional insights
        enhanced_kernel = self._enhance_document_kernel(kernel)
        
        # Store in memory if integration available
        if self.memory_integration:
            try:
                self.memory_integration.store_experience(
                    content=f"Extracted symbolic kernel from: {os.path.basename(pdf_path)}",
                    source="document_kernel",
                    importance=0.8,
                    concepts=[domain] + kernel["symbols"][:3],
                    emotional_tags={"analytical": 0.9}
                )
            except Exception as e:
                print(f"Memory storage error: {str(e)}")
        
        # Record the integration
        self._add_integration_record("document_kernel", {
            "pdf_path": pdf_path,
            "domain": domain,
            "symbols_found": len(kernel["symbols"]),
            "paradoxes_found": len(kernel["paradoxes"]),
            "frames_found": len(kernel["frames"])
        })
        
        return enhanced_kernel

    def _enhance_document_kernel(self, kernel: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance a document kernel with cross-kernel insights.
        
        Args:
            kernel: Basic document kernel
            
        Returns:
            Enhanced kernel with cross-kernel insights
        """
        enhanced_kernel = kernel.copy()
        
        # Add insights from different cognitive kernels
        enhanced_kernel["insights"] = {}
        
        # Process symbols with the math translator
        if kernel["symbols"]:
            math_insights = []
            for symbol in kernel["symbols"][:3]:  # Limit to first 3 symbols
                try:
                    translation = self.math_translator.translate(symbol)
                    if isinstance(translation, dict) and "matches" in translation:
                        math_insights.append({
                            "symbol": symbol,
                            "translation": translation["matches"]
                        })
                except Exception:
                    pass
            enhanced_kernel["insights"]["mathematical"] = math_insights
        
        # Process paradoxes with the paradox library
        if kernel["paradoxes"]:
            paradox_insights = []
            for paradox_text in kernel["paradoxes"][:3]:  # Limit to first 3 paradoxes
                try:
                    # Extract the main concept from the paradox text
                    words = paradox_text.split()
                    significant_words = [w for w in words if len(w) > 4 and w.lower() not in 
                                       ["about", "would", "could", "should", "there"]]
                    if significant_words:
                        paradox_concept = significant_words[0]
                        paradox_result = self.paradox_library.get(paradox_concept)
                        paradox_insights.append({
                            "text": paradox_text,
                            "concept": paradox_concept,
                            "type": paradox_result.get("type", "unknown"),
                            "description": paradox_result.get("description", "")
                        })
                except Exception:
                    pass
            enhanced_kernel["insights"]["paradoxical"] = paradox_insights
        
        # Process frames with the fusion engine
        if kernel["frames"] and len(kernel["frames"]) >= 2:
            fusion_insights = []
            try:
                # Take first two frames for fusion
                frame1 = kernel["frames"][0]
                frame2 = kernel["frames"][1]
                
                # Extract key terms
                words1 = frame1.split()
                words2 = frame2.split()
                
                significant_words1 = [w for w in words1 if len(w) > 4 and w.lower() not in 
                                   ["about", "would", "could", "should", "there"]]
                significant_words2 = [w for w in words2 if len(w) > 4 and w.lower() not in 
                                   ["about", "would", "could", "should", "there"]]
                
                if significant_words1 and significant_words2:
                    concept1 = significant_words1[0]
                    concept2 = significant_words2[0]
                    
                    fusion_result = self.fusion_engine.fuse_with_options(
                        concept1, 
                        concept2,
                        output_format="dict"
                    )
                    
                    fusion_insights.append({
                        "concepts": [concept1, concept2],
                        "frames": [frame1, frame2],
                        "result": fusion_result.get("result", ""),
                        "formatted_result": fusion_result.get("formatted_result", "")
                    })
            except Exception:
                pass
            
            enhanced_kernel["insights"]["fusion"] = fusion_insights
        
        # Process domain with dream core
        dream_insight = None
        try:
            dream_result = self.dream_core.generate(kernel["domain"], "standard")
            dream_insight = {
                "domain": kernel["domain"],
                "dream": dream_result
            }
        except Exception:
            pass
        
        enhanced_kernel["insights"]["dream"] = dream_insight
        
        return enhanced_kernel

    def pdf_to_cross_kernel_narrative(self, pdf_path: str, focus_concept: str = None) -> Dict[str, Any]:
        """
        Process a PDF and generate a cross-kernel narrative about its content.
        
        Args:
            pdf_path: Path to the PDF file
            focus_concept: Optional concept to focus the narrative on
            
        Returns:
            Cross-kernel narrative about the document
        """
        # First extract text from the PDF
        extraction_result = self.pdf_reader.extract_text(pdf_path, verbose=True)
        
        if not extraction_result["success"]:
            return {
                "error": "PDF extraction failed",
                "details": extraction_result.get("error", "Unknown error")
            }
        
        # Extract key concepts from the text
        full_text = extraction_result["text"]
        key_concepts = self._extract_key_concepts(full_text)
        
        # Use focus concept if provided, otherwise use first key concept
        if focus_concept:
            central_concept = focus_concept
        elif key_concepts:
            central_concept = key_concepts[0]
        else:
            central_concept = os.path.basename(pdf_path).split('.')[0]  # Use filename
        
        # Generate the cross-kernel narrative
        narrative = self.generate_cross_kernel_narrative(central_concept)
        
        # Add PDF context to the narrative
        narrative["source"] = {
            "type": "pdf",
            "path": pdf_path,
            "filename": os.path.basename(pdf_path),
            "page_count": extraction_result.get("page_count", 0),
            "extracted_concepts": key_concepts
        }
        
        # Store in memory if integration available
        if self.memory_integration:
            try:
                self.memory_integration.store_experience(
                    content=f"Generated cross-kernel narrative for PDF: {os.path.basename(pdf_path)}",
                    source="pdf_narrative",
                    importance=0.8,
                    concepts=[central_concept] + key_concepts[:2],
                    emotional_tags={"creativity": 0.8, "analytical": 0.7}
                )
            except Exception as e:
                print(f"Memory storage error: {str(e)}")
        
        # Record the integration
        self._add_integration_record("pdf_narrative", {
            "pdf_path": pdf_path,
            "central_concept": central_concept,
            "concepts_found": len(key_concepts),
            "sections_generated": len(narrative.get("sections", []))
        })
        
        return narrative

    def _extract_key_concepts(self, text: str, max_concepts: int = 10) -> List[str]:
        """
        Extract key concepts from text content.
        
        Args:
            text: Text to extract concepts from
            max_concepts: Maximum number of concepts to extract
            
        Returns:
            List of key concepts
        """
        # Use codex to extract concepts if available
        key_concepts = []
        
        try:
            # Limit text to a reasonable size for processing
            truncated_text = text[:10000]  # First 10K characters
            
            # Simple extraction based on word frequency and significance
            import re
            from collections import Counter
            
            # Tokenize text
            tokens = re.findall(r'\b[A-Za-z][A-Za-z\-]{3,}\b', truncated_text)
            
            # Filter out common words and short words
            common_words = {
                "the", "and", "but", "for", "nor", "or", "so", "yet", "a", "an", "to", 
                "in", "on", "with", "by", "at", "from", "this", "that", "these", "those",
                "there", "their", "they", "them", "when", "where", "which", "who", "whom",
                "whose", "what", "whatever", "how", "however", "about", "would", "could", 
                "should"
            }
            
            tokens = [token.lower() for token in tokens if token.lower() not in common_words and len(token) > 3]
            
            # Count token frequencies
            token_counts = Counter(tokens)
            
            # Get most common tokens
            most_common = token_counts.most_common(max_concepts)
            key_concepts = [token for token, count in most_common if count >= 2]
        except Exception as e:
            print(f"Concept extraction error: {str(e)}")
            
        return key_concepts

    def pdf_deep_exploration(self, pdf_path: str, max_depth: int = 2, exploration_breadth: int = 2) -> Dict[str, Any]:
        """
        Perform a deep recursive exploration of PDF content through multiple kernels.
        
        Args:
            pdf_path: Path to the PDF file
            max_depth: Maximum depth of recursion
            exploration_breadth: Number of branches to explore at each level
            
        Returns:
            Dictionary with recursive exploration results
        """
        # First extract text from the PDF
        extraction_result = self.pdf_reader.extract_text(pdf_path, verbose=True)
        
        if not extraction_result["success"]:
            return {
                "error": "PDF extraction failed",
                "details": extraction_result.get("error", "Unknown error")
            }
        
        # Extract key concepts from the text
        full_text = extraction_result["text"]
        key_concepts = self._extract_key_concepts(full_text)
        
        if not key_concepts:
            return {
                "error": "No key concepts found in the document",
                "extraction": extraction_result
            }
        
        # Use the first key concept as the seed for exploration
        seed_concept = key_concepts[0]
        
        # Perform recursive exploration
        exploration = self.recursive_concept_exploration(seed_concept, max_depth, exploration_breadth)
        
        # Add PDF context to the exploration
        exploration["source"] = {
            "type": "pdf",
            "path": pdf_path,
            "filename": os.path.basename(pdf_path),
            "page_count": extraction_result.get("page_count", 0),
            "extracted_concepts": key_concepts
        }
        
        # Store in memory if integration available
        if self.memory_integration:
            try:
                self.memory_integration.store_experience(
                    content=f"Performed deep exploration of PDF: {os.path.basename(pdf_path)}",
                    source="pdf_exploration",
                    importance=0.8,
                    concepts=key_concepts[:3],
                    emotional_tags={"curiosity": 0.9, "analytical": 0.7}
                )
            except Exception as e:
                print(f"Memory storage error: {str(e)}")
        
        # Record the integration
        self._add_integration_record("pdf_exploration", {
            "pdf_path": pdf_path,
            "seed_concept": seed_concept,
            "max_depth": max_depth,
            "exploration_breadth": exploration_breadth,
            "nodes_generated": len(exploration.get("nodes", [])),
            "insights_generated": len(exploration.get("insights", []))
        })
        
        return exploration


# Initialize integration function
def initialize_kernel_integration(
    codex=None, 
    dream_core=None, 
    fusion_engine=None, 
    paradox_library=None, 
    math_translator=None,
    conversation_engine=None,
    memory_integration=None,
    sully_instance=None
) -> KernelIntegrationSystem:
    """
    Initialize the kernel integration system with optional components.
    
    Args:
        codex: Optional EnhancedCodex instance
        dream_core: Optional DreamCore instance
        fusion_engine: Optional SymbolFusionEngine instance
        paradox_library: Optional ParadoxLibrary instance
        math_translator: Optional SymbolicMathTranslator instance
        conversation_engine: Optional ConversationEngine instance
        memory_integration: Optional MemoryIntegration instance
        sully_instance: Optional reference to main Sully instance
        
    Returns:
        Initialized KernelIntegrationSystem
    """
    # Create kernel integration system
    integration_system = KernelIntegrationSystem(
        codex=codex,
        dream_core=dream_core,
        fusion_engine=fusion_engine,
        paradox_library=paradox_library,
        math_translator=math_translator,
        conversation_engine=conversation_engine,
        memory_integration=memory_integration,
        sully_instance=sully_instance
    )
    
    # If Sully instance is provided, integrate with it
    if sully_instance:
        integration_system.integrate_with_sully(sully_instance)
    
    return integration_system

# Example usage
if __name__ == "__main__":
    # Create integration system
    integration = KernelIntegrationSystem()
    
    # Test cross-kernel operation
    result = integration.cross_kernel_operation(
        source_kernel="dream",
        target_kernel="fusion",
        input_data="consciousness"
    )
    
    print("=== Cross-Kernel Operation ===")
    print(f"Source: Dream of 'consciousness'")
    print(f"Target: Fusion based on dream")
    if isinstance(result["target_result"], dict) and "formatted_result" in result["target_result"]:
        print(f"Result: {result['target_result']['formatted_result']}")
    else:
        print(f"Result: {result['target_result']}")
        
    # Test concept network
    network = integration.create_concept_network("truth", max_depth=1)
    
    print("\n=== Concept Network ===")
    print(f"Central concept: {network['central_concept']}")
    print(f"Nodes: {len(network['nodes'])}")
    print(f"Edges: {len(network['edges'])}")
    
    # Test recursive exploration
    exploration = integration.recursive_concept_exploration("infinity", max_depth=2, exploration_breadth=1)
    
    print("\n=== Recursive Exploration ===")
    print(f"Seed concept: {exploration['seed_concept']}")
    print(f"Nodes: {len(exploration['nodes'])}")
    
    # Print insights
    print("\nInsights:")
    for insight in exploration["insights"]:
        print(f"- {insight['description']}")
        
    # Test cross-kernel narrative
    narrative = integration.generate_cross_kernel_narrative("paradox")
    
    print("\n=== Cross-Kernel Narrative ===")
    print(f"Title: {narrative['title']}")
    print(f"Sections: {len(narrative['sections'])}")
    print(f"Conclusion: {narrative['conclusio
        }