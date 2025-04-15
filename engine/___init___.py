"""
Sully Engine: Core Cognitive Framework Components

The engine module contains the core components that power the Sully
cognitive system, including reasoning, memory, and specialized kernels.
"""

__version__ = "1.0.0"
__author__ = "Marc Dannenberg"
__email__ = "contact@example.com"

# Import core components for easier access
from .reasoning import SymbolicReasoningNode
from .memory import SullySearchMemory
from .kernel_integration import KernelIntegrationSystem, initialize_kernel_integration
from .memory_integration import MemoryIntegration, integrate_with_sully
from .conversation_engine import ConversationEngine
from .logic_kernel import LogicKernel

# Import from kernel modules
from .kernel_modules.codex import SullyCodex
from .kernel_modules.identity import SullyIdentity
from .kernel_modules.dream import DreamCore
from .kernel_modules.fusion import SymbolFusionEngine
from .kernel_modules.paradox import ParadoxLibrary
from .kernel_modules.math_translator import SymbolicMathTranslator
from .kernel_modules.judgment import JudgmentProtocol
from .kernel_modules.virtue import VirtueEngine
from .kernel_modules.intuition import Intuition
from .kernel_modules.neural_modification import NeuralModification
from .kernel_modules.continuous_learning import ContinuousLearningSystem
from .kernel_modules.autonomous_goals import AutonomousGoalSystem
from .kernel_modules.visual_cognition import VisualCognitionSystem
from .kernel_modules.emergence_framework import EmergenceFramework

# Import PDF handling
from .pdf_reader import PDFReader, extract_text_from_pdf

# Constants
COGNITIVE_MODES = [
    "emergent", "analytical", "creative", "critical", "ethereal",
    "humorous", "professional", "casual", "musical", "visual",
    "scientific", "philosophical", "poetic", "instructional"
]

DREAM_DEPTHS = ["shallow", "standard", "deep", "dreamscape"]
DREAM_STYLES = ["recursive", "associative", "symbolic", "narrative"]
MATH_STYLES = ["formal", "intuitive", "applied", "creative"]
LOGICAL_FRAMEWORKS = ["PROPOSITIONAL", "FIRST_ORDER", "MODAL", "TEMPORAL", "FUZZY"]
CORE_KERNELS = ["dream", "fusion", "paradox", "math", "reasoning", "conversation", "memory"]
EVALUATION_FRAMEWORKS = ["balanced", "logical", "ethical", "practical", "scientific", "creative", "combined"]

# Helper functions
def get_available_cognitive_modules():
    """
    Get a list of all available cognitive modules in the Sully engine.
    
    Returns:
        List of available module names
    """
    return [
        "reasoning_node",
        "memory",
        "codex",
        "identity",
        "dream",
        "fusion", 
        "paradox",
        "translator",
        "judgment",
        "intuition",
        "virtue",
        "neural_modification",
        "continuous_learning",
        "autonomous_goals",
        "visual_cognition",
        "emergence"
    ]

def initialize_memory_system(memory_path=None):
    """
    Initialize the memory system with optional persistent storage.
    
    Args:
        memory_path: Path to memory storage file
        
    Returns:
        Initialized memory system
    """
    from .memory import SullySearchMemory
    memory = SullySearchMemory()
    
    # Set up memory persistence if path provided
    if memory_path:
        memory.set_persistence_path(memory_path)
        memory.load_from_disk()
    
    return memory

def initialize_core_modules():
    """
    Initialize the core cognitive modules needed for Sully.
    
    Returns:
        Dictionary of initialized modules
    """
    memory = SullySearchMemory()
    codex = SullyCodex()
    translator = SymbolicMathTranslator()
    
    # Create reasoning node (initially without translator)
    reasoning = SymbolicReasoningNode(
        codex=codex,
        memory=memory,
        translator=None
    )
    
    # Now set translator
    translator.reasoning = reasoning
    reasoning.translator = translator
    
    return {
        "memory": memory,
        "codex": codex,
        "translator": translator,
        "reasoning": reasoning
    }