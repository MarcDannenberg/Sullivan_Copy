# sully_engine/kernel_modules/__init__.py
# 🧠 Sully Core Cognitive Kernel Modules

"""
Sully Kernel Modules: Specialized Cognitive Components

This package contains specialized cognitive modules that form the kernels
of the Sully system, each providing unique capabilities and perspectives.
"""

__version__ = "1.0.0"

# Core modules
from .judgment import JudgmentProtocol
from .dream import DreamCore
from .math_translator import SymbolicMathTranslator
from .fusion import SymbolFusionEngine
from .paradox import ParadoxLibrary
from .identity import SullyIdentity
from .codex import SullyCodex

# Autonomous brain modules
from .neural_modification import NeuralModification
from .continuous_learning import ContinuousLearningSystem
from .autonomous_goals import AutonomousGoalSystem
from .visual_cognition import VisualCognitionSystem
from .emergence_framework import EmergenceFramework

# Advanced integrations
from .persona import PersonaManager
from .intuition import SomaticIntuition, SocialIntuition, Intuition
from .virtue import VirtueEngine
from .logic_kernel import LogicKernel

# Game modules
from .games import SullyGames

# Dictionary of kernel module classes for programmatic access
KERNEL_MODULES = {
    "codex": SullyCodex,
    "identity": SullyIdentity,
    "dream": DreamCore,
    "fusion": SymbolFusionEngine,
    "paradox": ParadoxLibrary,
    "math_translator": SymbolicMathTranslator,
    "judgment": JudgmentProtocol,
    "virtue": VirtueEngine,
    "intuition": Intuition,
    "neural_modification": NeuralModification,
    "continuous_learning": ContinuousLearningSystem,
    "autonomous_goals": AutonomousGoalSystem,
    "visual_cognition": VisualCognitionSystem,
    "emergence_framework": EmergenceFramework,
    "persona": PersonaManager,
    "somatic_intuition": SomaticIntuition,
    "social_intuition": SocialIntuition,
    "logic_kernel": LogicKernel,
    "games": SullyGames
}

# Define module capabilities for discovery
MODULE_CAPABILITIES = {
    "codex": ["store", "retrieve", "search", "associate"],
    "identity": ["persona", "adaptation", "evolution", "expression"],
    "dream": ["generate", "interpret", "symbolize", "associate"],
    "fusion": ["fuse", "synthesize", "blend", "recombine"],
    "paradox": ["identify", "analyze", "explore", "resolve"],
    "math_translator": ["translate", "formalize", "symbolize", "interpret"],
    "judgment": ["evaluate", "analyze", "critique", "validate"],
    "virtue": ["ethical_analysis", "virtue_evaluation", "action_assessment", "reflection"],
    "intuition": ["leap", "connect", "insight", "emergence"],
    "neural_modification": ["analyze", "optimize", "adapt", "enhance"],
    "continuous_learning": ["process", "consolidate", "transfer", "integrate"],
    "autonomous_goals": ["establish", "prioritize", "pursue", "evaluate"],
    "visual_cognition": ["interpret", "recognize", "understand", "contextualize"],
    "emergence_framework": ["detect", "nurture", "integrate", "synthesize"],
    "persona": ["transform", "blend", "generate", "adapt"],
    "somatic_intuition": ["physical_awareness", "gut_feeling", "embodied_cognition"],
    "social_intuition": ["emotional_intelligence", "social_dynamics", "interpersonal_insight"],
    "logic_kernel": ["infer", "prove", "verify", "formalize"],
    "games": ["interact", "simulate", "engage", "play"]
}

def get_module_class(module_name: str):
    """
    Get a module class by name.
    
    Args:
        module_name: Name of the module
        
    Returns:
        Module class or None if not found
    """
    return KERNEL_MODULES.get(module_name)

def get_module_capabilities(module_name: str):
    """
    Get the capabilities of a specific module.
    
    Args:
        module_name: Name of the module
        
    Returns:
        List of capabilities or empty list if module not found
    """
    return MODULE_CAPABILITIES.get(module_name, [])

def list_modules():
    """
    List all available kernel modules.
    
    Returns:
        Dictionary with module names and their capabilities
    """
    return {name: get_module_capabilities(name) for name in KERNEL_MODULES.keys()}

def find_module_by_capability(capability: str):
    """
    Find modules that provide a specific capability.
    
    Args:
        capability: Capability to search for
        
    Returns:
        List of module names that provide the capability
    """
    return [
        module_name for module_name, capabilities in MODULE_CAPABILITIES.items()
        if capability in capabilities
    ]

__all__ = [
    # Core modules
    "JudgmentProtocol",
    "DreamCore",
    "SymbolicMathTranslator",
    "SymbolFusionEngine",
    "ParadoxLibrary",
    "SullyIdentity",
    "SullyCodex",
    # Autonomous modules
    "NeuralModification",
    "ContinuousLearningSystem",
    "AutonomousGoalSystem",
    "VisualCognitionSystem",
    "EmergenceFramework",
    # Advanced integrations
    "PersonaManager",
    "SomaticIntuition",
    "SocialIntuition",
    "Intuition",
    "VirtueEngine",
    "LogicKernel",
    # Games
    "SullyGames",
    # Helper functions
    "get_module_class",
    "get_module_capabilities",
    "list_modules",
    "find_module_by_capability",
    # Constants
    "KERNEL_MODULES",
    "MODULE_CAPABILITIES"
]