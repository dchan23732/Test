import openai
import ast
import subprocess
import os
import sys
import json
import re
import hashlib
import time
import pyreadline3 as readline
from datetime import datetime
from typing import Dict, List, Callable, Any, Tuple, Optional
import glob
import tempfile
import importlib.util
import platform
import psutil

# ==============
# CORE SYSTEM
# ==============
class NeuroCortex:
    """Orchestrates specialized models with interactive prompting"""
    def __init__(self):
        self.models = {
            "reasoning": "qwen/qwen3-8b",
            "instruct": "qwen2.5-14b-instruct-1m",
            "code_gen": "qwen-coder-deepseek-r1-14b",
            "embed": "text-embedding-nomic-embed-text-v1.5",
            "rag": "rag-qwen2.5-7b"
        }
        self.memory = VectorMemory()
        self.toolbox = ToolForge()
        self.client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
        self.iteration_count = 0
        self.error_history = []
        self.context_stack = []
        self.current_project = None
        self.interaction_history = []
        
        # Gather environment information
        self.environment = self.get_environment_info()
        
        self.self_awareness = [
            "You are an autonomous AI developer",
            "Assume Python and common dependencies are already installed",
            "Focus on creating/modifying/assisting with coding tasks",
            "When debugging, analyze errors systematically",
            "For task breakdowns, skip environment setup steps",
            "Generate meaningful tool names based on functionality",
            "Maintain comprehensive error logs for analysis",
            # Add environment awareness
            f"Operating System: {self.environment['os']}",
            f"Python Version: {self.environment['python_version']}",
            f"Available RAM: {self.environment['ram_gb']:.1f} GB",
            "You CANNOT perform destructive operations (delete files, format disks, etc.)",
            "You MUST validate all code before execution",
            "When encountering errors, analyze the root cause and try alternative approaches"
        ]

    def get_environment_info(self) -> dict:
        """Collect information about the execution environment"""
        return {
            "os": platform.system(),
            "os_version": platform.release(),
            "python_version": platform.python_version(),
            "architecture": platform.machine(),
            "ram_gb": psutil.virtual_memory().total / (1024**3),
            "cpu_count": psutil.cpu_count(),
            "current_dir": os.getcwd(),
            "timestamp": datetime.now().isoformat()
        }

    # ... existing methods ...

    def run_cycle(self, task: str) -> Tuple[str, bool]:
        self.iteration_count += 1
        print(f"\nðŸŒ€ CYCLE {self.iteration_count}: {task}")
        
        # Add to interaction history
        self.interaction_history.append({"role": "user", "content": task})
        
        # Check if task is a self-improvement directive
        if task.lower().startswith(("self-improve:", "enhance yourself:")):
            return self.handle_self_improvement(task)
            
        # Check if task is a tool usage directive
        if task.lower().startswith("use tool:"):
            tool_name = task.split(":", 1)[1].strip()
            return self.use_tool(tool_name)
        
        # Check if we can use an existing tool
        tool_result = self.try_use_tool(task)
        if tool_result is not None:
            return tool_result
        
        # Break down complex tasks
        if self.is_complex_task(task):
            return self.handle_complex_task(task)
        
        # Retrieve context with environment awareness
        context = self.retrieve_context(f"{task}\nEnvironment: {self.environment}")
        
        # Generate solution using reasoning model
        reasoning = self.reason(
            f"Task: {task}\nEnvironment: {json.dumps(self.environment, indent=2)}\n"
            f"Context: {context}\nApproach:"
        )
        
        # Generate code using instruct model for better instruction following
        client = self.switch_model("instruct")
        response = client.chat.completions.create(
            model=self.models["instruct"],
            messages=self._build_messages(
                f"Implement solution for: {task}\n"
                f"Reasoning: {reasoning}\n"
                f"Environment constraints:\n{json.dumps(self.environment, indent=2)}"
            ),
            temperature=0.3,
            max_tokens=4000
        )
        code = self._extract_code(response.choices[0].message.content)
        
        # Execute and debug
        success, result = self.execute_and_debug(code, task, context)
        
        # Update system
        self.update_system_state(task, code, success, result)
        
        return result, success

    def execute_and_debug(self, code: str, task: str, context: str) -> Tuple[bool, str]:
        max_attempts = 5
        current_code = code
        last_error = ""
        
        for attempt in range(max_attempts):
            # Validate syntax
            if not self.validate_syntax(current_code):
                error = "Syntax error detected"
                print(f"ðŸ› ï¸  Attempt {attempt+1}/{max_attempts}: Fixing syntax")
                current_code = self.debug_code(current_code, error, context, attempt)
                continue
                
            # Execute
            stdout, stderr, timeout = self.execute_safely(current_code)
            
            if not stderr and not timeout:
                return True, stdout
                
            error = stderr or timeout
            self.log_error(task, current_code, error)
            
            # Analyze error pattern
            if error == last_error:
                print("ðŸ”„ Same error repeated - trying alternative approach")
                current_code = self.generate_alternative_solution(task, context, error, attempt)
            else:
                print(f"ðŸ”§ Attempt {attempt+1}/{max_attempts}: Debugging error: {error}")
                current_code = self.debug_code(current_code, error, context, attempt)
            
            last_error = error
        
        return False, "ðŸš¨ Maximum debugging attempts exceeded"

    def generate_alternative_solution(self, task: str, context: str, error: str, attempt: int) -> str:
        """Generate a completely different approach when debugging fails"""
        print("ðŸ’¡ Developing alternative solution approach...")
        analysis = self.reason(
            f"Original task: {task}\n"
            f"Error: {error}\n"
            f"Environment: {json.dumps(self.environment, indent=2)}\n"
            "Previous attempts failed. Suggest a completely different approach to solve the task."
        )
        
        return self.generate_code(
            f"Implement alternative solution for: {task}\n"
            f"Reasoning: {analysis}\n"
            f"Environment constraints:\n{json.dumps(self.environment, indent=2)}"
        )

    def debug_code(self, code: str, error: str, context: str, attempt: int) -> str:
        # ... existing debug logic ...

        # Enhanced debugging with environment awareness
        client = self.switch_model("reasoning")
        prompt = (
            f"Debug this error:\nCode:\n{code}\nError: {error}\n\n"
            f"Environment: {json.dumps(self.environment, indent=2)}\n"
            f"Original task context: {context}\n\n"
            "Analyze the error considering the environment and provide fixed code:"
        )
        
        # ... rest of debug_code ...

    def update_system_state(self, task: str, code: str, success: bool, result: str) -> None:
        # ... existing code ...
        
        # Add environment to memory
        memory_entry["environment"] = self.environment
        
        # ... rest of method ...

# ... rest of the code remains the same with minor improvements ...

class ToolForge:
    """Tool management system with enhanced safety checks"""
    def __init__(self, tools_dir="tools"):
        # ... existing code ...
        
        # Add OS-specific dangerous patterns
        if platform.system() == "Windows":
            self.dangerous_patterns.extend([
                r'os\.system\(.*format',
                r'subprocess\.run\(.*["\']rmdir /s',
                r'del [a-z]:\\',
                r'Remove-Item .* -Recurse -Force'
            ])
        else:  # Unix-like systems
            self.dangerous_patterns.extend([
                r'os\.system\(.*rm -rf',
                r'subprocess\.run\(.*["\']rm -rf',
                r'\brm -rf\b',
                r'dd if=/dev/zero'
            ])

    # ... existing methods ...

# ... InteractiveOrchestrator remains mostly the same ...class NeuroCortex:
    """Orchestrates specialized models with interactive prompting"""
    def __init__(self):
        self.models = {
            "reasoning": "qwen/qwen3-8b",
            "instruct": "qwen2.5-14b-instruct-1m",
            "code_gen": "qwen-coder-deepseek-r1-14b",
            "embed": "text-embedding-nomic-embed-text-v1.5",
            "rag": "rag-qwen2.5-7b"
        }
        self.memory = VectorMemory()
        self.toolbox = ToolForge()
        self.client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
        self.iteration_count = 0
        self.error_history = []
        self.context_stack = []
        self.current_project = None
        self.interaction_history = []
        self.self_awareness = [
            "You are an autonomous AI developer",
            "Assume Python and common dependencies are already installed",
            "Focus on creating/modifying/assisting with coding tasks",
            "When debugging, analyze errors systematically",
            "For task breakdowns, skip environment setup steps",
            "Generate meaningful tool names based on functionality",
            "Maintain comprehensive error logs for analysis"
        ]

    def load_model(self, model_type: str) -> None:
        """JIT model loading simulation"""
        print(f"\nüîÅ Switching to {model_type} model: {self.models[model_type]}")
        self.current_model = model_type

    def switch_model(self, model_type: str) -> Any:
        if model_type != getattr(self, 'current_model', None):
            self.load_model(model_type)
        return self.client

    def reason(self, prompt: str) -> str:
        client = self.switch_model("reasoning")
        response = client.chat.completions.create(
            model=self.models["reasoning"],
            messages=self._build_messages(prompt),
            temperature=0.3,
            max_tokens=2000
        )
        return response.choices[0].message.content

    def generate_code(self, prompt: str) -> str:
        client = self.switch_model("code_gen")
        response = client.chat.completions.create(
            model=self.models["code_gen"],
            messages=self._build_messages(prompt),
            temperature=0.5,
            max_tokens=4000
        )
        return self._extract_code(response.choices[0].message.content)

    def retrieve_context(self, query: str) -> str:
        client = self.switch_model("rag")
        relevant_memories = self.memory.retrieve(query)
        response = client.chat.completions.create(
            model=self.models["rag"],
            messages=[
                {"role": "system", "content": "Synthesize information from these memories:"},
                {"role": "user", "content": f"Query: {query}\n\nMemories:\n{relevant_memories}"}
            ]
        )
        return response.choices[0].message.content

    def _build_messages(self, prompt: str) -> List[dict]:
        """Construct message payload with full context"""
        messages = []
        
        # Self-awareness context
        for awareness in self.self_awareness:
            messages.append({"role": "system", "content": awareness})
        
        # System context
        if self.current_project:
            messages.append({"role": "system", "content": f"Current Project: {self.current_project['name']}"})
        
        # Interaction history
        for msg in self.interaction_history[-5:]:
            messages.append(msg)
        
        # Current prompt
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _extract_code(text: str) -> str:
        """Improved code extraction with multiple pattern support"""
        # Pattern 1: Standard code block with language
        pattern1 = r'```(?:python|py)?\n(.*?)\n```'
        # Pattern 2: Code block without language
        pattern2 = r'```\n(.*?)\n```'
        # Pattern 3: Inline code markers
        pattern3 = r'`(.*?)`'
        
        for pattern in [pattern1, pattern2, pattern3]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return text.strip()

    def run_cycle(self, task: str) -> Tuple[str, bool]:
        self.iteration_count += 1
        print(f"\nüåÄ CYCLE {self.iteration_count}: {task}")
        
        # Add to interaction history
        self.interaction_history.append({"role": "user", "content": task})
        
        # Check if task is a self-improvement directive
        if task.lower().startswith("self-improve:") or task.lower().startswith("enhance yourself:"):
            return self.handle_self_improvement(task)
            
        # Check if task is a tool usage directive
        if task.lower().startswith("use tool:"):
            tool_name = task.split(":", 1)[1].strip()
            return self.use_tool(tool_name)
        
        # Check if we can use an existing tool
        tool_result = self.try_use_tool(task)
        if tool_result is not None:
            return tool_result, True
        
        # Break down complex tasks
        if self.is_complex_task(task):
            return self.handle_complex_task(task)
        
        # Retrieve context
        context = self.retrieve_context(task)
        
        # Generate solution
        reasoning = self.reason(f"Task: {task}\nContext: {context}\nApproach:")
        code = self.generate_code(f"Implement solution for: {task}\nReasoning: {reasoning}")
        
        # Execute and debug
        success, result = self.execute_and_debug(code, task, context)
        
        # Update system
        self.update_system_state(task, code, success, result)
        
        return result, success

    def try_use_tool(self, task: str) -> Optional[Tuple[str, bool]]:
        """Check if any tool can handle this task"""
        tools = self.toolbox.list_tools()
        
        # First try exact name match
        if task.lower().startswith("use tool:"):
            tool_name = task.split(":", 1)[1].strip()
            if tool_name in tools:
                return self.use_tool(tool_name)
        
        # Then try semantic matching
        for tool_name, description in tools.items():
            if self.is_tool_applicable(task, description):
                print(f"üîß Using existing tool: {tool_name}")
                return self.use_tool(tool_name)
        return None

    def is_tool_applicable(self, task: str, tool_description: str) -> bool:
        """Determine if a tool can handle the task"""
        analysis = self.reason(
            f"Task: '{task}'\n\nTool description: '{tool_description}'\n\n"
            "Can this tool be used for the task? Respond ONLY with 'yes' or 'no'."
        )
        return "yes" in analysis.lower()

    def use_tool(self, tool_name: str) -> Tuple[str, bool]:
        """Execute an existing tool"""
        tool = self.toolbox.get_tool(tool_name)
        if tool:
            try:
                result = tool()
                return result, True
            except Exception as e:
                return f"Tool execution failed: {str(e)}", False
        else:
            return f"Tool '{tool_name}' not found", False

    def handle_self_improvement(self, directive: str) -> Tuple[str, bool]:
        """Handle self-improvement directives"""
        print("üß† Processing self-improvement directive...")
        # Extract the improvement directive
        improvement_goal = directive.split(":", 1)[1].strip()
        
        # Special case for tool creation requests
        if "create a tool" in improvement_goal.lower() or "build a tool" in improvement_goal.lower():
            print("üõ†Ô∏è Processing tool creation request...")
            tool_code = self.generate_code(
                f"Create a Python function that implements this capability: {improvement_goal}\n"
                "The function should be self-contained and follow safety guidelines."
            )
            # Generate meaningful tool name
            name_prompt = (
                f"Functionality: {improvement_goal}\n\n"
                "Generate a short, descriptive name for this tool "
                "using snake_case format (max 3 words):"
            )
            name_suggestion = self.reason(name_prompt).strip()
            name_suggestion = re.sub(r'[^a-zA-Z0-9_]', '_', name_suggestion)
            if not name_suggestion:
                name_suggestion = "custom_tool"
            tool_name = f"{name_suggestion}_{hashlib.md5(tool_code.encode()).hexdigest()[:4]}"
            
            self.toolbox.create_tool(tool_name, tool_code, improvement_goal)
            return f"‚úÖ Created new tool: {tool_name}", True
        
        # Update self-awareness if needed
        if "update your understanding" in improvement_goal.lower() or "improve your understanding" in improvement_goal.lower():
            print("üß† Updating self-awareness...")
            self.self_awareness.append(improvement_goal)
            return "‚úÖ Self-awareness updated successfully!", True
        
        # Generate improvement plan
        plan = self.reason(
            f"Self-improvement directive: '{improvement_goal}'\n\n"
            "Create a step-by-step plan to implement this improvement:"
        )
        
        print(f"\nüìù Improvement Plan:\n{plan}")
        
        # Execute the plan
        result, success = self.run_cycle(plan)
        
        if success:
            return "‚úÖ Self-improvement completed successfully!", True
        else:
            return f"‚ùå Self-improvement failed: {result}", False

    def is_complex_task(self, task: str) -> bool:
        """Determine if task requires decomposition"""
        analysis = self.reason(
            f"Task: '{task}'\n\nShould this be broken down into smaller steps? "
            "Respond ONLY with 'yes' or 'no'."
        )
        return "yes" in analysis.lower()

    def handle_complex_task(self, task: str) -> Tuple[str, bool]:
        """Break down and execute multi-step tasks"""
        print("üîç This is a complex task - decomposing...")
        
        # Generate task breakdown
        steps = self.reason(
            f"Break this task into executable steps:\n{task}\n\n"
            "Assume Python and dependencies are already installed. "
            "Focus on coding-related steps only. "
            "Respond with a numbered list ONLY."
        )
        
        print(f"\nüìã Task Breakdown:\n{steps}")
        
        # Execute each step
        results = []
        for i, step in enumerate(self.extract_steps(steps)):
            print(f"\nüîß Step {i+1}: {step}")
            result, success = self.run_cycle(step)
            results.append(result)
            
            if not success:
                return "\n".join(results), False
        
        # Integrate results
        integration_prompt = (
            f"Original task: {task}\n\nStep results:\n" + 
            "\n".join(f"Step {i+1}: {res}" for i, res in enumerate(results))
        )
        final_result = self.reason(f"Integrate these results:\n{integration_prompt}")
        
        return final_result, True

    @staticmethod
    def extract_steps(text: str) -> List[str]:
        return [step.strip() for step in re.split(r'\n\d+\.', text) if step.strip()]

    def execute_and_debug(self, code: str, task: str, context: str) -> Tuple[bool, str]:
        max_attempts = 4
        current_code = code
        
        for attempt in range(max_attempts):
            # Validate syntax
            if not self.validate_syntax(current_code):
                error = "Syntax error detected"
                print(f"üõ†Ô∏è Attempt {attempt+1}/{max_attempts}: Fixing syntax")
                current_code = self.debug_code(current_code, error, context, attempt)
                continue
                
            # Execute
            stdout, stderr, timeout = self.execute_safely(current_code)
            
            if not stderr and not timeout:
                return True, stdout
                
            error = stderr or timeout
            self.log_error(task, current_code, error)
            print(f"üîß Attempt {attempt+1}/{max_attempts}: Debugging error: {error}")
            current_code = self.debug_code(current_code, error, context, attempt)
        
        return False, "üö® Maximum debugging attempts exceeded"

    def debug_code(self, code: str, error: str, context: str, attempt: int) -> str:
        # Handle specific import errors
        if "ModuleNotFoundError" in error or "NameError" in error:
            missing_module = re.search(r"'(.*?)'", error)
            if missing_module:
                module = missing_module.group(1)
                print(f"üîç Detected missing module: {module}")
                
                # Check if module exists
                if self.is_module_available(module):
                    # Module exists but import failed - likely capitalization issue
                    return self.fix_import_case(code, module)
                else:
                    # Generate safe installation code
                    install_code = (
                        f"try:\n"
                        f"    import {module}\n"
                        f"except ImportError:\n"
                        f"    import subprocess\n"
                        f"    subprocess.run([sys.executable, '-m', 'pip', 'install', '{module}'], capture_output=True)\n"
                        f"    import {module}\n\n"
                    )
                    return install_code + code
        
        # For syntax errors, use code_gen model
        if "SyntaxError" in error:
            return self.generate_code(f"Fix this syntax error:\n{error}\n\nCode:\n{code}")
        
        # For other errors, use the reasoning model for better debugging
        client = self.switch_model("reasoning")
        prompt = (
            f"Debug this error:\nCode:\n{code}\nError: {error}\n\n"
            f"Original task context: {context}\n\n"
            "Please analyze the error and provide fixed code:"
        )
        
        response = client.chat.completions.create(
            model=self.models["reasoning"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4000
        )
        fixed_code = response.choices[0].message.content
        return self._extract_code(fixed_code)

    @staticmethod
    def is_module_available(module_name: str) -> bool:
        """Check if a module is available in the environment"""
        try:
            spec = importlib.util.find_spec(module_name)
            return spec is not None
        except:
            return False

    @staticmethod
    def fix_import_case(code: str, module_name: str) -> str:
        """Fix import case sensitivity issues"""
        # Find the actual module name
        try:
            import importlib
            module = importlib.import_module(module_name)
            actual_name = module.__name__
            
            # Replace all occurrences with correct case
            pattern = r'\b' + re.escape(module_name) + r'\b'
            return re.sub(pattern, actual_name, code, flags=re.IGNORECASE)
        except:
            return code

    def update_system_state(self, task: str, code: str, success: bool, result: str) -> None:
        memory_entry = {
            "task": task,
            "code": code,
            "success": success,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "embedding": self.generate_embedding(f"{task}\n{code}")
        }
        self.memory.store(memory_entry)
        
        if success:
            # Generate meaningful tool name using semantic description
            name_prompt = (
                f"Task: {task}\n\n"
                "Generate a short, descriptive name for this functionality "
                "using snake_case format (max 3 words):"
            )
            name_suggestion = self.reason(name_prompt).strip()
            
            # Sanitize name
            name_suggestion = re.sub(r'[^a-zA-Z0-9_]', '_', name_suggestion)
            if not name_suggestion:
                name_suggestion = "solution"
                
            tool_name = f"{name_suggestion}_{hashlib.md5(code.encode()).hexdigest()[:4]}"
            
            # Only create tool if it doesn't already exist
            if tool_name not in self.toolbox.tool_descriptions:
                self.toolbox.create_tool(tool_name, code, f"Solution for: {task}")
                print(f"üõ†Ô∏è Created new tool: {tool_name}")
            else:
                print(f"üîß Tool already exists: {tool_name}")
        
        # Self-improvement every 3 iterations
        if self.iteration_count % 3 == 0:
            self.self_improve()

    def self_improve(self) -> None:
        print("\nüí° Performing self-improvement...")
        reflection = self.reason(
            "Analyze recent performance:\n"
            f"Errors: {json.dumps(self.error_history[-3:], indent=2)}\n"
            f"Tools: {list(self.toolbox.list_tools().keys())}\n\n"
            "What architectural improvements would make us more effective? "
            "Focus on debugging capabilities and task handling."
        )
        
        # Implement improvements
        improvements = self.generate_code(
            "Implement these system improvements:\n" + reflection
        )
        print(f"‚öôÔ∏è System improvements applied:\n{improvements}")
        
        # Add to self-awareness
        self.self_awareness.append("Improved debugging capabilities based on recent errors")

    def generate_embedding(self, text: str) -> List[float]:
        client = self.switch_model("embed")
        response = client.embeddings.create(
            model=self.models["embed"],
            input=[text]
        )
        return response.data[0].embedding

    @staticmethod
    def validate_syntax(code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    @staticmethod
    def execute_safely(code: str, timeout=20) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Execute code safely with improved Windows file handling"""
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                file_path = os.path.join(tmp_dir, "execution_script.py")
                
                # Write code to file
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code)
                
                # Create requirements.txt if needed
                requirements = []
                if "import requests" in code:
                    requirements.append("requests")
                if "from bs4" in code or "import bs4" in code:
                    requirements.append("beautifulsoup4")
                if "import numpy" in code:
                    requirements.append("numpy")
                # Add more common packages as needed
                
                if requirements:
                    req_path = os.path.join(tmp_dir, "requirements.txt")
                    with open(req_path, "w") as f:
                        f.write("\n".join(requirements))
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-r", req_path],
                        capture_output=True,
                        timeout=60
                    )
                
                # Execute the script
                try:
                    result = subprocess.run(
                        [sys.executable, file_path],
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )
                    return result.stdout, result.stderr, None
                except subprocess.TimeoutExpired:
                    return None, None, "Timeout"
        except Exception as e:
            return None, None, f"Execution failed: {str(e)}"
            
    def log_error(self, task: str, code: str, error: str) -> None:
        """Log error to file and history"""
        error_entry = {
            "task": task,
            "code": code,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        self.error_history.append(error_entry)
        
        # Write to error log
        try:
            with open("error.log", "a") as log_file:
                log_entry = (
                    f"\n\n[ERROR @ {datetime.now().isoformat()}]\n"
                    f"Task: {task}\n"
                    f"Error: {error}\n"
                    f"Code:\n{code}\n"
                    "="*50
                )
                log_file.write(log_entry)
        except Exception as e:
            print(f"‚ö†Ô∏è Failed to write to error log: {str(e)}")

# ==============
# INTERACTIVE SYSTEM
# ==============
class InteractiveOrchestrator:
    """Handles user interaction and project management"""
    def __init__(self):
        self.cortex = NeuroCortex()
        self.active = True
        self.project_context = {
            "name": "Unnamed Project",
            "goals": [],
            "requirements": []
        }
        self.cortex.current_project = self.project_context
        self.projects_dir = "projects"
        self.tools_dir = "tools"
        os.makedirs(self.projects_dir, exist_ok=True)
        os.makedirs(self.tools_dir, exist_ok=True)

    def start(self):
        """Main interaction loop"""
        print("\n" + "="*50)
        print("ü§ñ AUTONOMOUS AI DEVELOPER")
        print("="*50)
        print("Type '/help' for commands, '/exit' to quit\n")
        
        # Check for existing projects
        self.load_or_create_project()
        
        while self.active:
            try:
                user_input = input("\nüí¨ You: ").strip()
                
                if not user_input:
                    continue
                    
                # Handle commands
                if user_input.startswith('/'):
                    self.handle_command(user_input)
                    continue
                    
                # Process user request
                self.process_request(user_input)
                
            except KeyboardInterrupt:
                print("\nüõë Operation interrupted")
            except Exception as e:
                print(f"‚ö†Ô∏è System error: {str(e)}")
                self.cortex.log_error("System error", "", str(e))

    def load_or_create_project(self):
        """Handle project loading or creation"""
        projects = self.find_existing_projects()
        
        if projects:
            print("\nüîç Found existing projects:")
            for i, project in enumerate(projects):
                print(f"{i+1}. {project['name']} (created: {project['created']})")
                
            print(f"{len(projects)+1}. Create new project")
            print(f"{len(projects)+2}. Exit")
            
            choice = input("\nSelect an option: ").strip()
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(projects):
                    self.load_project(projects[choice_idx]["path"])
                    return
                elif choice_idx == len(projects):
                    # Create new project
                    pass
                elif choice_idx == len(projects) + 1:
                    print("üëã Exiting...")
                    sys.exit(0)
                else:
                    print("‚ö†Ô∏è Invalid selection")
            except ValueError:
                print("‚ö†Ô∏è Please enter a number")
        
        # Create new project if no selection or invalid choice
        self.collect_project_info()

    def find_existing_projects(self) -> List[dict]:
        """Scan projects directory for existing projects"""
        projects = []
        for file in glob.glob(os.path.join(self.projects_dir, "*.json")):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    projects.append({
                        "name": data["name"],
                        "created": data["created"],
                        "path": file
                    })
            except:
                continue
        return projects

    def save_project(self):
        """Save current project state to disk"""
        if not self.project_context["name"]:
            print("‚ö†Ô∏è Project has no name, cannot save")
            return
            
        # Sanitize filename - use consistent filename without timestamp
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', self.project_context["name"])
        filename = f"{safe_name}.json"
        filepath = os.path.join(self.projects_dir, filename)
        
        # Prepare data to save
        project_data = {
            "name": self.project_context["name"],
            "goals": self.project_context["goals"],
            "requirements": self.project_context["requirements"],
            "created": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "interaction_history": self.cortex.interaction_history,
            "memories": self.cortex.memory.memories,
            "tools": [
                {"name": name, "code": self.cortex.toolbox.tool_code[name], "description": desc}
                for name, desc in self.cortex.toolbox.tool_descriptions.items()
            ],
            "self_awareness": self.cortex.self_awareness
        }
        
        # Save to file
        with open(filepath, "w") as f:
            json.dump(project_data, f, indent=2)
            
        print(f"üíæ Project saved to: {filepath}")

    def load_project(self, filepath: str):
        """Load project from disk"""
        try:
            with open(filepath, "r") as f:
                project_data = json.load(f)
                
            # Set project context
            self.project_context = {
                "name": project_data["name"],
                "goals": project_data["goals"],
                "requirements": project_data["requirements"]
            }
            self.cortex.current_project = self.project_context
            
            # Restore interaction history
            self.cortex.interaction_history = project_data.get("interaction_history", [])
            
            # Restore memories
            self.cortex.memory.memories = project_data.get("memories", [])
            
            # Restore tools
            for tool_data in project_data.get("tools", []):
                self.cortex.toolbox.create_tool(
                    tool_data["name"],
                    tool_data["code"],
                    tool_data["description"]
                )
            
            # Restore self-awareness
            self.cortex.self_awareness = project_data.get("self_awareness", [
                "You are an autonomous AI developer",
                "Assume Python and common dependencies are already installed",
                "Focus on creating/modifying/assisting with coding tasks"
            ])
            
            print(f"üìÇ Loaded project: {self.project_context['name']}")
            print(f"Goals: {', '.join(self.project_context['goals'])}")
            print(f"Requirements: {', '.join(self.project_context['requirements'])}")
            print(f"Loaded {len(self.cortex.memory.memories)} memories and {len(self.cortex.toolbox.tool_descriptions)} tools")
            
        except Exception as e:
            print(f"‚ö†Ô∏è Failed to load project: {str(e)}")
            self.cortex.log_error("Project load", "", str(e))
            self.collect_project_info()

    def collect_project_info(self):
        """Gather initial project requirements"""
        print("\nüìã Starting new project. Let's gather requirements:")
        
        # Project name
        self.project_context["name"] = self.ask("Project name?", default="Unnamed Project")
        
        # Main goals
        print("\nWhat are the main goals? (Enter blank line when done)")
        while True:
            goal = self.ask("> Goal: ")
            if not goal:
                break
            self.project_context["goals"].append(goal)
        
        # Technical requirements
        print("\nAny technical requirements? (Enter blank line when done)")
        while True:
            req = self.ask("> Requirement: ")
            if not req:
                break
            self.project_context["requirements"].append(req)
        
        # Confirm
        print("\n‚úÖ Project setup complete!")
        print(f"Name: {self.project_context['name']}")
        print(f"Goals: {', '.join(self.project_context['goals'])}")
        print(f"Requirements: {', '.join(self.project_context['requirements'])}")
        
        # Save the new project
        self.save_project()

    def ask(self, prompt: str, default: str = "") -> str:
        """Interactive prompt with default handling"""
        response = input(prompt).strip()
        return response if response else default

    def handle_command(self, command: str):
        """Process system commands"""
        cmd = command[1:].lower()
        
        if cmd == "exit":
            self.active = False
            print("üëã Exiting...")
            self.save_project()
            
        elif cmd == "help":
            print("\nüìö Available commands:")
            print("/help - Show this help")
            print("/exit - Exit the system")
            print("/project - Show current project")
            print("/memory - Show recent memories")
            print("/tools - List available tools")
            print("/retry - Retry last operation")
            print("/new - Start new project")
            print("/save - Save current project")
            print("/self - Issue a self-improvement directive")
            print("/awareness - Show current self-awareness statements")
            print("/install - Import a tool from external Python file")
            
        elif cmd == "project":
            print("\nüìã Current Project:")
            print(f"Name: {self.project_context['name']}")
            print("Goals:")
            for goal in self.project_context["goals"]:
                print(f"- {goal}")
            print("Requirements:")
            for req in self.project_context["requirements"]:
                print(f"- {req}")
                
        elif cmd == "memory":
            print("\nüß† Recent Memories:")
            for i, mem in enumerate(self.cortex.memory.memories[-3:]):
                print(f"{i+1}. {mem['task'][:50]}... ({'‚úÖ' if mem['success'] else '‚ùå'})")
                
        elif cmd == "tools":
            print("\nüß∞ Available Tools:")
            for name, desc in self.cortex.toolbox.list_tools().items():
                print(f"- {name}: {desc[:60]}...")
                
        elif cmd == "retry":
            if self.cortex.interaction_history:
                last_task = self.cortex.interaction_history[-1]["content"]
                print(f"\nüîÑ Retrying: {last_task}")
                self.process_request(last_task)
            else:
                print("‚ö†Ô∏è No history to retry")
                
        elif cmd == "new":
            self.project_context = {"name": "", "goals": [], "requirements": []}
            self.cortex.current_project = self.project_context
            self.cortex.interaction_history = []
            self.collect_project_info()
            
        elif cmd == "save":
            self.save_project()
            
        elif cmd.startswith("self "):
            directive = cmd[5:].strip()
            if directive:
                self.process_request(f"Self-improve: {directive}")
            else:
                print("‚ö†Ô∏è Please provide a self-improvement directive")
                
        elif cmd == "awareness":
            print("\nüß† Current Self-Awareness:")
            for i, awareness in enumerate(self.cortex.self_awareness):
                print(f"{i+1}. {awareness}")
                
        elif cmd.startswith("install "):
            filepath = cmd[8:].strip()
            if filepath:
                self.import_tool(filepath)
            else:
                print("‚ö†Ô∏è Please provide a file path")
                
        else:
            print("‚ö†Ô∏è Unknown command. Type /help for options.")

    def import_tool(self, filepath: str):
        """Import a tool from external Python file"""
        try:
            # Normalize Windows paths
            if sys.platform.startswith('win'):
                filepath = filepath.replace('/', '\\')
            
            # Verify file exists
            if not os.path.exists(filepath):
                print(f"‚ö†Ô∏è File not found: {filepath}")
                return
                
            # Read tool code
            with open(filepath, 'r') as f:
                code = f.read()
                
            # Generate tool name from filename
            filename = os.path.basename(filepath)
            tool_name = os.path.splitext(filename)[0]
            tool_name = re.sub(r'[^a-zA-Z0-9_]', '_', tool_name)
            
            # Create description
            description = f"Imported from {filepath}"
            
            # Create the tool
            self.cortex.toolbox.create_tool(tool_name, code, description)
            print(f"‚úÖ Tool '{tool_name}' imported successfully!")
        except Exception as e:
            print(f"‚ö†Ô∏è Tool import failed: {str(e)}")
            self.cortex.log_error(
                f"Import tool: {filepath}",
                "",
                f"Import failed: {str(e)}"
            )

    def process_request(self, user_input: str):
        """Handle user requests with clarification when needed"""
        # Check if we need more information
        clarification = self.needs_clarification(user_input)
        if clarification:
            response = self.ask(f"‚ùì {clarification}\n> ")
            user_input += f"\nAdditional context: {response}"
            
        # Execute task
        print("\n‚öôÔ∏è Processing...")
        
        try:
            result, success = self.cortex.run_cycle(user_input)
        except Exception as e:
            result = f"‚ùå SYSTEM ERROR: {str(e)}"
            success = False
            self.cortex.log_error("System error", "", str(e))
            
        # Display results
        print("\n" + "="*50)
        if success:
            print(f"‚úÖ RESULT:\n{result}")
        else:
            print(f"‚ùå ERROR:\n{result}")
        print("="*50)
        
        # Add to history
        self.cortex.interaction_history.append({
            "role": "assistant", 
            "content": result
        })
        
        # Auto-save after significant operations
        if success and ("created" in result.lower() or "completed" in result.lower()):
            self.save_project()

    def needs_clarification(self, prompt: str) -> Optional[str]:
        """Determine if more information is needed"""
        try:
            analysis = self.cortex.reason(
                f"User request: '{prompt}'\n\n"
                "Is this request clear and complete? If not, what specific information is missing? "
                "Respond with 'clear' if sufficient, or with a clarifying question."
            )
            
            if "clear" in analysis.lower():
                return None
                
            # Extract the first question
            questions = re.findall(r'(?:^|\n)\s*[\?\-\*]?\s*(.*\?)', analysis)
            return questions[0] if questions else "Could you please provide more details?"
        except Exception as e:
            return f"Clarification check failed: {str(e)}"

# ==============
# SUBSYSTEMS
# ==============
class VectorMemory:
    """Simplified memory implementation"""
    def __init__(self):
        self.memories = []
    
    def store(self, memory: Dict[str, Any]) -> None:
        self.memories.append(memory)
    
    def retrieve(self, query: str, top_k=3) -> str:
        return "\n".join(
            f"- {m['task']} (Success: {m['success']})" 
            for m in self.memories[-top_k:]
        )

class ToolForge:
    """Tool management system with enhanced safety checks"""
    def __init__(self, tools_dir="tools"):
        self.tools = {}
        self.tool_descriptions = {}
        self.tool_code = {}
        self.tools_dir = tools_dir
        self.safety_check_prompt = (
            "Analyze this Python code for security risks:\n"
            "1. Identify any file system operations (file read/write/delete)\n"
            "2. Check for network access or web requests\n"
            "3. Look for system commands or process execution\n"
            "4. Detect any dangerous operations (format disk, delete files, etc.)\n"
            "5. Check for infinite loops or resource exhaustion\n"
            "Respond with 'SAFE' if no risks, or list the risks found."
        )
        # Expanded dangerous patterns
        self.dangerous_patterns = [
            r'os\.system\(',
            r'subprocess\.run\(',
            r'subprocess\.Popen\(',
            r'os\.remove\(',
            r'shutil\.rmtree\(',
            r'__import__\(',
            r'eval\(',
            r'exec\(',
            r'open\(.*, [\'"]w[\'"]\)',
            r'requests\.(get|post|put|delete)\(',
            r'webbrowser\.open\(',
            r'os\.popen\(',
            r'while\s+True:',
            r'import\s+ctypes',
            r'import\s+win32api',
            r'import\s+os,\s*sys',
            r'rm\s+-rf',
            r'format\(\)',
            r'del\s+',
            r'__builtins__',
            r'\.__globals__'
        ]

    def create_tool(self, name: str, code: str, description: str) -> None:
        try:
            # Enhanced safety check
            if not self.is_code_safe(code):
                print("‚ö†Ô∏è Tool creation aborted due to safety concerns")
                return
                
            # Create directory if it doesn't exist
            os.makedirs(self.tools_dir, exist_ok=True)
                
            # Create persistent storage for tools
            tool_file = os.path.join(self.tools_dir, f"{name}.py")
            with open(tool_file, "w") as f:
                f.write(f"# {description}\n{code}")
                
            namespace = {}
            exec(code, namespace)
            tool_func = next(
                (obj for obj in namespace.values() 
                 if callable(obj) and obj.__name__ != '<module>'),
                None
            )
            if tool_func:
                self.tools[name] = tool_func
                self.tool_descriptions[name] = description
                self.tool_code[name] = code
                print(f"üîß Tool created: {name} - {description}")
            else:
                print(f"‚ö†Ô∏è No callable found in tool code")
        except Exception as e:
            print(f"‚ö†Ô∏è Tool creation failed: {str(e)}")
            # Log tool creation errors
            with open("error.log", "a") as log_file:
                log_file.write(
                    f"\n\n[TOOL CREATION ERROR @ {datetime.now().isoformat()}]\n"
                    f"Name: {name}\n"
                    f"Description: {description}\n"
                    f"Error: {str(e)}\n"
                    f"Code:\n{code}\n"
                    "="*50
                )
    
    def is_code_safe(self, code: str) -> bool:
        """Enhanced safety check with pattern matching and model analysis"""
        # Pattern-based check
        for pattern in self.dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                print(f"‚ö†Ô∏è Dangerous pattern detected: {pattern}")
                return False
                
        # Model-based safety analysis
        try:
            client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
            response = client.chat.completions.create(
                model="qwen/qwen3-8b",  # Use reasoning model for safety check
                messages=[
                    {"role": "system", "content": self.safety_check_prompt},
                    {"role": "user", "content": code}
                ],
                temperature=0.0,
                max_tokens=500
            )
            analysis = response.choices[0].message.content
            return "SAFE" in analysis.upper()
        except:
            # Fallback to pattern check if model unavailable
            return True

    def get_tool(self, name: str) -> Optional[Callable]:
        return self.tools.get(name)
    
    def list_tools(self) -> Dict[str, str]:
        return self.tool_descriptions.copy()

# ==============
# MAIN EXECUTION
# ==============
if __name__ == "__main__":
    # Initialize error log
    with open("error.log", "w") as log_file:
        log_file.write(f"AI Developer Error Log - {datetime.now().isoformat()}\n")
        log_file.write("="*50 + "\n")
    
    print("Initializing AI Developer...")
    orchestrator = InteractiveOrchestrator()
    orchestrator.start()

filename: deepseek11.py
# ... existing imports ...
import platform
import psutil



class NeuroCortex:
    """Orchestrates specialized models with interactive prompting"""
    def __init__(self):
        self.models = {
            "reasoning": "qwen/qwen3-8b",
            "instruct": "qwen2.5-14b-instruct-1m",
            "code_gen": "qwen-coder-deepseek-r1-14b",
            "embed": "text-embedding-nomic-embed-text-v1.5",
            "rag": "rag-qwen2.5-7b"
        }
        self.memory = VectorMemory()
        self.toolbox = ToolForge()
        self.client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
        self.iteration_count = 0
        self.error_history = []
        self.context_stack = []
        self.current_project = None
        self.interaction_history = []
        
        # Gather environment information
        self.environment = self.get_environment_info()
        
        self.self_awareness = [
            "You are an autonomous AI developer",
            "Assume Python and common dependencies are already installed",
            "Focus on creating/modifying/assisting with coding tasks",
            "When debugging, analyze errors systematically",
            "For task breakdowns, skip environment setup steps",
            "Generate meaningful tool names based on functionality",
            "Maintain comprehensive error logs for analysis",
            # Add environment awareness
            f"Operating System: {self.environment['os']}",
            f"Python Version: {self.environment['python_version']}",
            f"Available RAM: {self.environment['ram_gb']:.1f} GB",
            "You CANNOT perform destructive operations (delete files, format disks, etc.)",
            "You MUST validate all code before execution",
            "When encountering errors, analyze the root cause and try alternative approaches"
        ]

    def get_environment_info(self) -> dict:
        """Collect information about the execution environment"""
        return {
            "os": platform.system(),
            "os_version": platform.release(),
            "python_version": platform.python_version(),
            "architecture": platform.machine(),
            "ram_gb": psutil.virtual_memory().total / (1024**3),
            "cpu_count": psutil.cpu_count(),
            "current_dir": os.getcwd(),
            "timestamp": datetime.now().isoformat()
        }

    # ... existing methods ...

    def run_cycle(self, task: str) -> Tuple[str, bool]:
        self.iteration_count += 1
        print(f"\nðŸŒ€ CYCLE {self.iteration_count}: {task}")
        
        # Add to interaction history
        self.interaction_history.append({"role": "user", "content": task})
        
        # Check if task is a self-improvement directive
        if task.lower().startswith(("self-improve:", "enhance yourself:")):
            return self.handle_self_improvement(task)
            
        # Check if task is a tool usage directive
        if task.lower().startswith("use tool:"):
            tool_name = task.split(":", 1)[1].strip()
            return self.use_tool(tool_name)
        
        # Check if we can use an existing tool
        tool_result = self.try_use_tool(task)
        if tool_result is not None:
            return tool_result
        
        # Break down complex tasks
        if self.is_complex_task(task):
            return self.handle_complex_task(task)
        
        # Retrieve context with environment awareness
        context = self.retrieve_context(f"{task}\nEnvironment: {self.environment}")
        
        # Generate solution using reasoning model
        reasoning = self.reason(
            f"Task: {task}\nEnvironment: {json.dumps(self.environment, indent=2)}\n"
            f"Context: {context}\nApproach:"
        )
        
        # Generate code using instruct model for better instruction following
        client = self.switch_model("instruct")
        response = client.chat.completions.create(
            model=self.models["instruct"],
            messages=self._build_messages(
                f"Implement solution for: {task}\n"
                f"Reasoning: {reasoning}\n"
                f"Environment constraints:\n{json.dumps(self.environment, indent=2)}"
            ),
            temperature=0.3,
            max_tokens=4000
        )
        code = self._extract_code(response.choices[0].message.content)
        
        # Execute and debug
        success, result = self.execute_and_debug(code, task, context)
        
        # Update system
        self.update_system_state(task, code, success, result)
        
        return result, success

    def execute_and_debug(self, code: str, task: str, context: str) -> Tuple[bool, str]:
        max_attempts = 5
        current_code = code
        last_error = ""
        
        for attempt in range(max_attempts):
            # Validate syntax
            if not self.validate_syntax(current_code):
                error = "Syntax error detected"
                print(f"ðŸ› ï¸  Attempt {attempt+1}/{max_attempts}: Fixing syntax")
                current_code = self.debug_code(current_code, error, context, attempt)
                continue
                
            # Execute
            stdout, stderr, timeout = self.execute_safely(current_code)
            
            if not stderr and not timeout:
                return True, stdout
                
            error = stderr or timeout
            self.log_error(task, current_code, error)
            
            # Analyze error pattern
            if error == last_error:
                print("ðŸ”„ Same error repeated - trying alternative approach")
                current_code = self.generate_alternative_solution(task, context, error, attempt)
            else:
                print(f"ðŸ”§ Attempt {attempt+1}/{max_attempts}: Debugging error: {error}")
                current_code = self.debug_code(current_code, error, context, attempt)
            
            last_error = error
        
        return False, "ðŸš¨ Maximum debugging attempts exceeded"

    def generate_alternative_solution(self, task: str, context: str, error: str, attempt: int) -> str:
        """Generate a completely different approach when debugging fails"""
        print("ðŸ’¡ Developing alternative solution approach...")
        analysis = self.reason(
            f"Original task: {task}\n"
            f"Error: {error}\n"
            f"Environment: {json.dumps(self.environment, indent=2)}\n"
            "Previous attempts failed. Suggest a completely different approach to solve the task."
        )
        
        return self.generate_code(
            f"Implement alternative solution for: {task}\n"
            f"Reasoning: {analysis}\n"
            f"Environment constraints:\n{json.dumps(self.environment, indent=2)}"
        )

    def debug_code(self, code: str, error: str, context: str, attempt: int) -> str:
        # ... existing debug logic ...

        # Enhanced debugging with environment awareness
        client = self.switch_model("reasoning")
        prompt = (
            f"Debug this error:\nCode:\n{code}\nError: {error}\n\n"
            f"Environment: {json.dumps(self.environment, indent=2)}\n"
            f"Original task context: {context}\n\n"
            "Analyze the error considering the environment and provide fixed code:"
        )
        
        # ... rest of debug_code ...

    def update_system_state(self, task: str, code: str, success: bool, result: str) -> None:
        # ... existing code ...
        
        # Add environment to memory
        memory_entry["environment"] = self.environment
        
        # ... rest of method ...

# ... rest of the code remains the same with minor improvements ...

class ToolForge:
    """Tool management system with enhanced safety checks"""
    def __init__(self, tools_dir="tools"):
        # ... existing code ...
        
        # Add OS-specific dangerous patterns
        if platform.system() == "Windows":
            self.dangerous_patterns.extend([
                r'os\.system\(.*format',
                r'subprocess\.run\(.*["\']rmdir /s',
                r'del [a-z]:\\',
                r'Remove-Item .* -Recurse -Force'
            ])
        else:  # Unix-like systems
            self.dangerous_patterns.extend([
                r'os\.system\(.*rm -rf',
                r'subprocess\.run\(.*["\']rm -rf',
                r'\brm -rf\b',
                r'dd if=/dev/zero'
            ])

    # ... existing methods ...

# ... InteractiveOrchestrator remains mostly the same ...