import hashlib
import time
from typing import Dict, Any, List, Optional

class CodeBrain:
    """
    CodeBrain is a code generation system that interprets natural language prompts
    and produces corresponding code plans and implementations.
    
    The system follows a pipeline approach:
    1. Interpret the goal from natural language
    2. Generate a code plan based on the interpreted design
    3. Synthesize actual code from the plan
    4. Explain design choices
    5. Refine code based on feedback
    """

    def __init__(self):
        """Initialize a new CodeBrain instance with default settings."""
        self.builder_mode = True

    def interpret_goal(self, prompt: str) -> Dict[str, Any]:
        """
        Interpret a natural language prompt into a structured design specification.
        
        Args:
            prompt: A natural language description of the code to be generated
            
        Returns:
            A dictionary containing the parsed goal, application type, features,
            programming language, required files, and timestamp
        """
        # Input validation
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
            
        # Simple heuristic parsing, to be replaced by NLP or symbolic parsing later
        return {
            "goal": prompt,
            "app_type": "web app" if "app" in prompt.lower() else "script",
            "features": ["auto-detected"],
            "language": "python",
            "files": ["main.py"],
            "timestamp": time.time()
        }

    def generate_code_plan(self, design: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a structured code plan based on the design specification.
        
        Args:
            design: A dictionary containing the design specifications
            
        Returns:
            A dictionary containing the planned file structure, language,
            and a summary of the design
            
        Raises:
            ValueError: If the design dictionary is missing required keys
        """
        # Input validation
        required_keys = ["goal", "app_type", "language"]
        for key in required_keys:
            if key not in design:
                raise ValueError(f"Design is missing required key: {key}")
                
        # Create a simple structure based on app_type
        plan = {
            "files": {
                "main.py": f"# Auto-generated by Sully\n\nprint('Hello from your {design['app_type']} that achieves: {design['goal']}')"
            },
            "language": design["language"],
            "design_summary": f"Builds a {design['app_type']} in {design['language']} to fulfill: {design['goal']}"
        }
        return plan

    def synthesize_code(self, plan: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate actual code based on the code plan.
        
        Args:
            plan: A dictionary containing the code plan
            
        Returns:
            A dictionary mapping file names to their content
            
        Raises:
            ValueError: If the plan dictionary is missing required keys
        """
        # Input validation
        if "files" not in plan:
            raise ValueError("Plan is missing required key: files")
            
        return plan["files"]

    def explain_choices(self, plan: Dict[str, Any]) -> str:
        """
        Generate an explanation of the design choices made in the code plan.
        
        Args:
            plan: A dictionary containing the code plan
            
        Returns:
            A string explaining the design choices
            
        Raises:
            ValueError: If the plan dictionary is missing required keys
        """
        # Input validation
        required_keys = ["language"]
        for key in required_keys:
            if key not in plan:
                raise ValueError(f"Plan is missing required key: {key}")
                
        return f"I chose {plan['language']} because it's efficient for this kind of task. I created a basic entry point in main.py."

    def refine_code(self, current_files: Dict[str, str], feedback: str) -> Dict[str, str]:
        """
        Refine the generated code based on user feedback.
        
        Args:
            current_files: A dictionary mapping file names to their content
            feedback: User feedback as a string
            
        Returns:
            A dictionary mapping file names to their refined content
            
        Raises:
            ValueError: If current_files is empty or feedback is not provided
        """
        # Input validation
        if not current_files:
            raise ValueError("Current files dictionary cannot be empty")
        if not feedback or not isinstance(feedback, str):
            raise ValueError("Feedback must be a non-empty string")
            
        # Placeholder refinement logic
        refined = {}
        for name, content in current_files.items():
            refined[name] = content + f"\n# Refined based on feedback: {feedback}"
        return refined