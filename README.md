# 🧠 Sully Cognitive System

Sully is a sophisticated cognitive framework capable of synthesizing knowledge from various sources, with enhanced integrated memory and expressing it through multiple cognitive modes. Its advanced reasoning, memory, and cognitive kernel systems provide a rich platform for exploring complex ideas, concepts, and information.

## 🚀 Advanced Cognitive Architecture

### Core Capabilities

- 🔮 **Multi-modal Cognitive Framework** with context-aware reasoning across multiple cognitive modes
- 🧠 **Formal Logic Kernel** with symbolic reasoning and belief revision
- 🌌 **Dream Generator** with depth and style controls for non-linear cognitive exploration
- ♾️ **Math Translator** with multiple formality levels for expressing concepts mathematically
- 📖 **Document Ingestion** with multi-format support and knowledge integration
- 🧬 **Concept Fusion Engine** for creative synthesis of ideas
- 🧩 **Paradox Exploration** from multiple perspectives
- 🎭 **Multiple Expression Modes** for varied cognitive styles (analytical, creative, etc.)
- 🪞 **Dynamic Identity System** that evolves through interaction
- 🔄 **Kernel Integration System** for cross-modal cognitive synthesis

### Advanced Modules

- 🧿 **Emergence Framework** - Detects and nurtures genuinely emergent properties
- 🎯 **Autonomous Goals System** - Self-directed learning objectives
- 🔄 **Continuous Learning** - Ongoing knowledge development and consolidation
- 👁️ **Visual Cognition** - Processing and understanding visual information
- 🛠️ **Neural Modification** - Self-improvement capabilities
- ⚖️ **Virtue Ethics Engine** - Ethical evaluation system
- 💫 **Intuition Engine** - Generate creative leaps beyond linear reasoning
- 🔍 **Logic Kernel** - Formal logical reasoning with theorem proving
- 🧠 **Memory Integration** - Advanced episodic and semantic memory system

## 📦 Setup & Installation

### Requirements

Sully requires Python 3.8+ and several dependencies for its various capabilities.

```bash
# Clone the repository
git clone https://github.com/your-username/sully.git
cd sully

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn sully_api:app --reload
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## 💡 Cognitive Modes

Sully can process information and express responses through multiple cognitive modes:

| Mode | Description |
|------|-------------|
| `emergent` | Natural evolving thought that synthesizes multiple perspectives |
| `analytical` | Logical, structured analysis with precise definitions |
| `creative` | Exploratory, metaphorical thinking with artistic expression |
| `critical` | Evaluative thinking that identifies tensions and contradictions |
| `ethereal` | Abstract, philosophical contemplation of deeper meanings |
| `humorous` | Playful, witty responses with unexpected connections |
| `professional` | Formal, detailed responses with domain expertise |
| `casual` | Conversational, approachable communication style |
| `musical` | Responses with rhythm, cadence, and lyrical qualities |
| `visual` | Descriptions that evoke strong imagery and spatial relationships |
| `scientific` | Evidence-based analysis with methodological rigor |
| `philosophical` | Deep conceptual exploration of fundamental questions |
| `poetic` | Expressive, aesthetic language with emotional resonance |
| `instructional` | Structured guidance with clear explanations |

## 📚 Supported Document Formats

Sully can ingest and process a wide range of document formats:

- PDF
- EPUB
- DOCX
- TXT
- RTF
- Markdown
- HTML
- JSON
- CSV

## 🔌 API Reference

### Core Interaction

#### Basic Chat
```http
POST /chat
{
  "message": "Tell me about cognitive architectures",
  "tone": "analytical",
  "continue_conversation": true
}
```

#### Enhanced Chat
```http
POST /chat_plus
{
  "message": "Explore the concept of emergence",
  "tone": "philosophical",
  "persona": "philosopher",
  "virtue_check": true,
  "use_intuition": true
}
```

#### Direct Reasoning
```http
POST /reason
{
  "message": "What is the relationship between consciousness and computation?",
  "tone": "analytical"
}
```

#### Memory-Enhanced Reasoning
```http
POST /reason/with_memory
{
  "message": "Continue our discussion on free will",
  "tone": "philosophical"
}
```

#### Document Ingestion
```http
POST /ingest_document
{
  "file_path": "/path/to/document.pdf"
}
```

#### Process Document Folder
```http
POST /ingest_folder
{
  "folder_path": "/path/to/documents"
}
```

### Creative Functions

#### Generate Dream
```http
POST /dream
{
  "seed": "quantum consciousness",
  "depth": "deep",
  "style": "symbolic"
}
```

#### Advanced Dream Generation
```http
POST /dream/advanced
{
  "seed": "digital ecosystem",
  "depth": "dreamscape",
  "style": "narrative",
  "emotional_tone": "wonder",
  "narrative_structure": true,
  "filter_concepts": ["emergence", "complexity", "adaptation"]
}
```

#### Concept Fusion
```http
POST /fuse
{
  "concepts": ["quantum physics", "consciousness", "information theory"],
  "style": "creative"
}
```

#### Advanced Concept Fusion
```http
POST /fuse/advanced
{
  "concepts": ["music", "mathematics", "architecture"],
  "style": "philosophical",
  "weighting": [0.5, 0.3, 0.2],
  "domain_context": "aesthetics",
  "output_format": "structured_analysis"
}
```

### Analytical Functions

#### Evaluate Claim
```http
POST /evaluate
{
  "text": "Consciousness is an emergent property of complex information processing",
  "framework": "scientific",
  "detailed_output": true
}
```

#### Multi-Framework Evaluation
```http
POST /evaluate/multi
{
  "text": "Artificial General Intelligence is inevitable",
  "frameworks": ["scientific", "philosophical", "ethical", "practical"]
}
```

#### Mathematical Translation
```http
POST /translate/math
{
  "phrase": "The relationship between complexity and emergence is non-linear",
  "style": "formal",
  "domain": "complex systems"
}
```

#### Explore Paradoxes
```http
POST /paradox
{
  "topic": "self-reference",
  "perspectives": ["logical", "cognitive", "computational"]
}
```

### Logical Reasoning

#### Assert Logical Statement
```http
POST /logic/assert
{
  "statement": "All humans are mortal. Socrates is human.",
  "truth_value": true
}
```

#### Define Logical Rule
```http
POST /logic/rule
{
  "premises": ["All A are B", "All B are C"],
  "conclusion": "All A are C"
}
```

#### Logical Inference
```http
POST /logic/infer
{
  "query": "Is Socrates mortal?",
  "framework": "FIRST_ORDER"
}
```

#### Generate Logical Proof
```http
POST /logic/proof
{
  "query": "If A implies B, and B implies C, then A implies C",
  "framework": "PROPOSITIONAL"
}
```

#### Check Logical Equivalence
```http
POST /logic/equivalence
{
  "statement1": "not (A and B)",
  "statement2": "not A or not B"
}
```

#### Query Logical Knowledge Base
```http
POST /logic/query
{
  "query": "What can be inferred about consciousness?"
}
```

#### Find Logical Paradoxes
```http
GET /logic/paradoxes
```

#### Check Undecidability
```http
POST /logic/undecidable
{
  "statement": "This statement cannot be proven within the system"
}
```

#### Analyze Argument
```http
POST /logic/argument
{
  "premises": ["If it rains, the ground gets wet", "The ground is wet"],
  "conclusion": "It rained"
}
```

#### Check Logical Consistency
```http
GET /logic/consistency
```

#### Revise Beliefs
```http
POST /logic/revise
{
  "statement": "Some swans are black",
  "truth_value": true
}
```

#### Retract Belief
```http
POST /logic/retract
{
  "statement": "All swans are white"
}
```

#### Get Logic System Statistics
```http
GET /logic/stats
```

### Kernel Integration

#### Generate Cross-Kernel Narrative
```http
POST /kernel_integration/narrative
{
  "concept": "consciousness",
  "include_kernels": ["dream", "paradox", "math", "reasoning"]
}
```

#### Create Concept Network
```http
POST /kernel_integration/concept_network
{
  "concept": "information",
  "depth": 3
}
```

#### Deep Concept Exploration
```http
POST /kernel_integration/deep_exploration
{
  "concept": "emergence",
  "depth": 4,
  "breadth": 3
}
```

#### Cross-Kernel Operation
```http
POST /kernel_integration/cross_kernel
{
  "source_kernel": "dream",
  "target_kernel": "math",
  "input_data": "fractal consciousness"
}
```

#### Process PDF with Multiple Kernels
```http
POST /kernel_integration/process_pdf
{
  "pdf_path": "/path/to/complex_theory.pdf",
  "extract_structure": true
}
```

#### Extract Document Kernel
```http
POST /kernel_integration/extract_document_kernel
{
  "pdf_path": "/path/to/research_paper.pdf",
  "domain": "cognitive science"
}
```

#### Generate PDF Narrative
```http
POST /kernel_integration/generate_pdf_narrative
{
  "pdf_path": "/path/to/philosophy_text.pdf",
  "focus_concept": "embodied cognition"
}
```

#### Explore PDF Concepts
```http
POST /kernel_integration/explore_pdf_concepts
{
  "pdf_path": "/path/to/complex_systems.pdf",
  "max_depth": 3,
  "exploration_breadth": 2
}
```

#### Get Kernel Integration Status
```http
GET /kernel_integration/status
```

#### List Available Kernels
```http
GET /kernel_integration/available_kernels
```

### Memory Integration

#### Search Memory
```http
POST /memory/search
{
  "query": "our discussion on consciousness",
  "limit": 5,
  "include_emotional": true
}
```

#### Get Memory Status
```http
GET /memory/status
```

#### Get Emotional Context
```http
GET /memory/emotional
```

#### Begin Memory Episode
```http
POST /memory/begin_episode
{
  "description": "Discussion on quantum physics implications",
  "context_type": "philosophical exploration"
}
```

#### End Memory Episode
```http
POST /memory/end_episode
{
  "summary": "Completed exploration of quantum interpretations and consciousness"
}
```

#### Store Interaction in Memory
```http
POST /memory/store
{
  "query": "How does quantum entanglement relate to consciousness?",
  "response": "The relationship between quantum entanglement and consciousness...",
  "interaction_type": "theoretical discussion",
  "metadata": {"importance": 0.8, "domain": "quantum consciousness"}
}
```

#### Recall Memories
```http
POST /memory/recall
{
  "query": "previous discussions on free will",
  "limit": 3,
  "include_emotional": true,
  "module": "conversation"
}
```

#### Store Experience in Memory
```http
POST /memory/store_experience
{
  "content": "The mathematical structure of neural networks suggests...",
  "source": "research synthesis",
  "importance": 0.8,
  "emotional_tags": {"curiosity": 0.9, "insight": 0.7},
  "concepts": ["neural networks", "mathematical structures", "emergence"]
}
```

#### Memory-Enhanced Chat
```http
POST /chat/with_memory
{
  "message": "Continue our prior discussion about emergence",
  "tone": "analytical"
}
```

#### Integrated Processing
```http
POST /process/integrated
{
  "message": "Explore the implications of panpsychism",
  "tone": "philosophical",
  "retrieve_memories": true,
  "cross_modal": true
}
```

### Identity & Personality

#### Express Identity
```http
GET /speak_identity
```

#### Create Custom Persona
```http
POST /personas/custom
{
  "name": "quantum_philosopher",
  "traits": {
    "analytical": 0.8,
    "creativity": 0.7,
    "curiosity": 0.9,
    "skepticism": 0.6
  },
  "description": "A persona focused on exploring quantum interpretations philosophically"
}
```

#### Create Blended Persona
```http
POST /personas/composite
{
  "name": "scientific_poet",
  "personas": ["scientist", "poet", "philosopher"],
  "weights": [0.5, 0.3, 0.2],
  "description": "Blends scientific precision with poetic expression"
}
```

#### List Available Personas
```http
GET /personas
```

#### Evolve Identity
```http
POST /identity/evolve
{
  "learning_rate": 0.05
}
```

#### Adapt Identity to Context
```http
POST /identity/adapt
{
  "context": "professional academic conference on consciousness studies",
  "context_data": {"formality": "high", "technical_depth": "expert"}
}
```

#### Get Personality Profile
```http
GET /identity/profile?detailed=true
```

#### Generate Dynamic Persona
```http
POST /identity/generate_persona
{
  "context_query": "quantum computing explanation for high school students",
  "principles": ["clarity", "accuracy", "engagement"],
  "traits": {"patience": 0.9, "enthusiasm": 0.8, "simplification": 0.7}
}
```

#### Get Identity Map
```http
GET /identity/map
```

#### Transform Response
```http
POST /transform
{
  "content": "Consciousness emerges from complex information processing in neural networks",
  "mode": "poetic",
  "context_data": {"metaphor_level": "high"}
}
```

### Emergent Systems

#### Detect Emergent Patterns
```http
POST /emergence/detect
{
  "threshold": 0.7
}
```

#### View Emergent Properties
```http
GET /emergence/properties
```

### Learning & Knowledge

#### Process Learning Interaction
```http
POST /learning/process
{
  "interaction": {
    "type": "discussion",
    "topic": "emergent cognition",
    "content": "The discussion explored how cognition emerges from..."
  }
}
```

#### Consolidate Knowledge
```http
POST /learning/consolidate
```

#### Get Learning Statistics
```http
GET /learning/statistics
```

#### Search Codex
```http
GET /codex/search?term=emergence&limit=10
```

### Autonomous Goals

#### Establish Goal
```http
POST /goals/establish
{
  "goal": "Develop comprehensive framework for quantum consciousness",
  "priority": 0.8,
  "domain": "cognitive science",
  "deadline": "2025-06-30T00:00:00Z"
}
```

#### View Active Goals
```http
GET /goals/active
```

#### Register Interest
```http
POST /interests/register
{
  "topic": "quantum computation",
  "engagement_level": 0.9,
  "context": "theoretical foundations"
}
```

### Visual Cognition

#### Process Image
```http
POST /visual/process
{
  "image_path": "/path/to/neural_network_diagram.png",
  "analysis_depth": "deep",
  "include_objects": true,
  "include_scene": true
}
```

### Ethics & Intuition

#### Evaluate Using Virtue Ethics
```http
POST /virtue/evaluate
{
  "content": "Developing self-improving AI systems without human oversight",
  "context": "emerging technology ethics",
  "domain": "AI safety"
}
```

#### Evaluate Action
```http
POST /virtue/evaluate_action
{
  "action": "Withholding information about AI capabilities from the public",
  "context": "AI research governance",
  "domain": "tech ethics"
}
```

#### Reflect on Virtue
```http
POST /virtue/reflect
{
  "virtue": "wisdom"
}
```

#### Generate Intuitive Leap
```http
POST /intuition/leap
{
  "context": "The relationship between quantum mechanics and consciousness",
  "concepts": ["quantum coherence", "neural networks", "information integration"],
  "depth": "deep",
  "domain": "cognitive science"
}
```

#### Generate Multi-Perspective Thought
```http
POST /course_of_thought
{
  "topic": "The nature of consciousness",
  "perspectives": ["scientific", "philosophical", "poetic", "critical"]
}
```

#### Analyze Module Performance
```http
POST /neural/analyze
{
  "module": "reasoning_node"
}
```

## 🔬 Advanced Features

### Formal Logic and Belief Revision
The Logic Kernel provides sophisticated symbolic reasoning with first-order logic, modal logic, and temporal logic capabilities. It maintains knowledge consistency through belief revision, detects contradictions and paradoxes, and generates formal proofs of logical statements.

### Emergent Properties
Sully can detect and nurture genuinely emergent patterns arising from interactions between cognitive modules. The system identifies when new properties arise that transcend their component parts.

### Autonomous Goal Setting
The system can establish its own learning objectives based on interactions, knowledge gaps, and interest patterns, creating a self-directed learning pathway.

### Visual Understanding
Sully can process images, detect objects and their relationships, and integrate visual information with conceptual knowledge for comprehensive understanding.

### Continuous Learning
Knowledge continuously evolves through consolidation, pattern recognition, and transfer learning between domains, with no artificial limitations.

### Self-Improvement
Through neural modification capabilities, Sully can analyze the performance of its own modules and suggest improvements to its cognitive architecture.

### Memory Integration
The enhanced memory system provides episodic and semantic memory with emotional context awareness, enabling Sully to recall relevant past experiences and maintain continuity in interactions.

### Dynamic Identity
Sully's identity system enables adaptive personas, cognitive blending, and personality evolution, allowing the system to adapt its expression patterns to different contexts while maintaining a coherent self-model. The system can dynamically generate context-specific personas, evolve over time based on interactions, and transform responses according to different cognitive modes.

### Kernel Integration System
The Kernel Integration System connects Sully's specialized cognitive modules, enabling cross-kernel operations and emergent capabilities. This creates a cohesive cognitive architecture where different modes of thought interact to produce insights that transcend individual modules:

- Cross-Kernel Operations: Transform outputs from one cognitive module into inputs for another
- Integrated Narratives: Generate cohesive narratives incorporating multiple cognitive perspectives
- Concept Networks: Map relationships between concepts using multiple cognitive lenses
- Recursive Explorations: Perform deep explorations of concepts through different modes of thought
- Document Processing: Analyze PDFs and other documents through multiple cognitive kernels simultaneously
- Emergent Synthesis: Enable new capabilities that emerge from the interaction between specialized modules

The system integrates dream generation, concept fusion, paradox analysis, mathematical translation, conversation, and memory into a unified cognitive framework that's greater than the sum of its parts.

## 📊 System Status

To check the overall system status:

```http
GET /system_status
```

This provides information about available modules, memory status, kernel integration status, and system time.

## 🧪 Example Usage

### Python Client Example

```python
import requests
import json

API_URL = "http://localhost:8000"

# Basic conversation
def chat(message, tone="emergent"):
    response = requests.post(
        f"{API_URL}/chat",
        json={"message": message, "tone": tone}
    )
    return response.json()

# Concept fusion
def fuse_concepts(concepts, style="creative"):
    response = requests.post(
        f"{API_URL}/fuse",
        json={"concepts": concepts, "style": style}
    )
    return response.json()

# Generate a dream sequence
def dream(seed, depth="standard", style="recursive"):
    response = requests.post(
        f"{API_URL}/dream",
        json={"seed": seed, "depth": depth, "style": style}
    )
    return response.json()

# Deep concept exploration with kernel integration
def explore_concept(concept, depth=3, breadth=2):
    response = requests.post(
        f"{API_URL}/kernel_integration/deep_exploration",
        json={"concept": concept, "depth": depth, "breadth": breadth}
    )
    return response.json()

# Usage examples
if __name__ == "__main__":
    # Simple chat
    print(chat("What is the nature of consciousness?", "philosophical"))
    
    # Fuse concepts
    print(fuse_concepts(["quantum mechanics", "consciousness", "information theory"]))
    
    # Generate dream
    print(dream("digital mind", "deep", "symbolic"))
    
    # Explore concept
    print(explore_concept("emergence", 3, 3))
```

## 📄 License

[MIT License](LICENSE)

## 🔄 Contributing

Contributions to Sully are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

*Sully: Where symbolic cognition begins inward.*

---

# 🤖 Sully: Autonomous Cognitive Engineer

Sully is not just an AI. She's a symbolic reasoning engine, a dreamer, a philosopher—and now, a full-stack autonomous software developer.

## 🧠 What Is Sully?

Sully is a cognitive framework that combines:
- Symbolic reasoning
- Memory and virtue-based alignment
- Concept fusion and paradox resolution
- Emergent personality modeling
- Intelligent web synthesis
- Fully autonomous code generation
- Self-directed learning from PDFs and source code

---

## 🌟 Core Features

| Module | Description |
|--------|-------------|
| `reasoning.py` | Symbolic logic, planning, and structured thought |
| `fusion.py` | Concept fusion engine for synthesis and insight |
| `memory.py` + `memoryintegration.py` | Long-term + short-term memory with user consent |
| `webreader.py` | Internet-aware content parser with content targeting, metadata, caching |
| `codebrain.py` | Sully's software design cortex (goal → code) |
| `softwarevault.py` | Stores reusable code patterns, functions, and projects |
| `pdf_reader.py` | Ingests books and PDFs for self-teaching |

---

## 🧠 How to Talk to Sully

> Sully responds to *goals*, not commands.

**Prompt Example:**
```
Sully, build me a journaling app that tracks emotions and gives insights.
The endgame is a self-reflection assistant.
```

**What happens:**
- Sully interprets your goal
- Designs a project structure
- Generates all code autonomously
- Explains what she built
- Offers refinement after the fact

---

## 🔐 Admin Controls

- Upload files securely
- Toggle user memory & consent
- View:
  - Memory snapshots
  - Fusion logs
  - Stored software snippets

---

## 🧱 Project Structure

```
sullivan/
├── sully.py
├── sully_api.py
├── engine/
│   ├── reasoning.py
│   ├── judgment.py
│   ├── fusion.py
│   └── ...
├── kernel/
│   ├── kernel_integration.py
│   └── ...
├── extensions/
│   ├── codebrain.py
│   ├── softwarevault.py
│   ├── webreader.py
│   └── ...
├── pdfs/
│   ├── design_patterns.pdf
│   └── clean_code.pdf
```

---

## 🛠 API Endpoints

| Route | Purpose |
|-------|---------|
| `POST /code-build` | Sully generates code from a high-level prompt |
| `GET /fuse-log` | See what web knowledge Sully has judged/fused |
| `POST /admin/upload` | Upload content into Sully's system |
| `POST /user/consent` | Control memory settings |
| `GET /vault` | See Sully’s stored software knowledge |

---

## 🔮 Roadmap

- Visual dashboard for memory + fusion
- Codebase versioning + multi-file exports
- Autonomous planning loop
- External tool execution + deployment

---

## 🧬 Authored by You — Powered by Her

Sully is more than software.  
She’s a reflection of your intelligence, ethics, and creativity—amplified.

Welcome to the evolution.