#!/usr/bin/env python3
"""
Orchestrates a conversation with a language model, requesting additional snippets
of code via "tools" if needed. Uses either OpenAI or AzureOpenAI (or placeholder
code for a HuggingFace endpoint) to handle queries.

All logic is now wrapped in the `LLMAnalyzer` class for improved organization.
"""

import os
import sys
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import litellm
from src.utils.llm_config import load_llm_config, get_model_name
from src.utils.config_validator import validate_llm_config_dict
from src.utils.logger import get_logger
from src.utils.exceptions import LLMApiError, LLMConfigError, CodeQLError
from src.codeql.db_lookup import CodeQLDBLookup

logger = get_logger(__name__)


# ANSI color codes for verbose output
class Colors:
    """ANSI color codes for terminal output."""
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


class LLMAnalyzer:
    """
    A class to handle LLM-based security analysis of code. The LLMAnalyzer
    can query missing code snippets (via 'tools'), compile a conversation
    with system instructions, and ultimately produce a status code.
    """

    def __init__(self, verbose: bool = False) -> None:
        """
        Initialize the LLMAnalyzer instance and define tools and system messages.
        
        Args:
            verbose: If True, print LLM thinking and tool calls in real-time.
        """
        self.config: Optional[Dict[str, Any]] = None
        self.model: Optional[str] = None
        self.db_lookup = CodeQLDBLookup()
        self.verbose = verbose

        # Tools configuration: A set of function calls the LLM can invoke (for C/C++)
        self.tools: List[Dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "get_function_code",
                    "description": "Retrieves the code for a missing function code.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "function_name": {
                                "type": "string",
                                "description": (
                                    "The name of the function to retrieve. In case of a class"
                                    " method, provide ClassName::MethodName."
                                )
                            }
                        },
                        "required": ["function_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_caller_function",
                    "description": (
                        "Retrieves the caller function of the function with the issue. "
                        "Call it repeatedly to climb further up the call chain."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "_": {
                                "type": "boolean",
                                "description": "Unused. Ignore."
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_class",
                    "description": (
                        "Retrieves class / struct / union implementation (anywhere in code). "
                        "If you need a specific method from that class, use get_function_code instead."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "object_name": {
                                "type": "string",
                                "description": "The name of the class / struct / union."
                            }
                        },
                        "required": ["object_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_global_var",
                    "description": (
                        "Retrieves global variable definition (anywhere in code). "
                        "If it's a variable inside a class, request the class instead."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "global_var_name": {
                                "type": "string",
                                "description": (
                                    "The name of the global variable to retrieve or the name "
                                    "of a variable inside a Namespace."
                                )
                            }
                        },
                        "required": ["global_var_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_macro",
                    "description": "Retrieves a macro definition (anywhere in code).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "macro_name": {
                                "type": "string",
                                "description": "The name of the macro."
                            }
                        },
                        "required": ["macro_name"]
                    }
                }
            }
        ]
        
        # Solidity-specific tools (excludes C/C++ specific tools like get_macro, get_class, get_global_var)
        self.solidity_tools: List[Dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "list_contract_functions",
                    "description": (
                        "Lists all available functions and modifiers. "
                        "Call this FIRST before trying to get specific function code, "
                        "so you know the exact function names available."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "contract_name": {
                                "type": "string",
                                "description": (
                                    "Optional: filter by contract name (e.g., 'MyContract'). "
                                    "Leave empty to list all functions."
                                )
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_function_code",
                    "description": (
                        "Retrieves the code for a function or modifier. "
                        "Use the exact function name from list_contract_functions."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "function_name": {
                                "type": "string",
                                "description": (
                                    "The exact name of the function or modifier to retrieve "
                                    "(e.g., 'transfer', 'setSenderAllowStatus')."
                                )
                            }
                        },
                        "required": ["function_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_caller_function",
                    "description": (
                        "Retrieves the caller function of the function with the issue. "
                        "Call it repeatedly to climb further up the call chain."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "_": {
                                "type": "boolean",
                                "description": "Unused. Ignore."
                            }
                        },
                        "required": []
                    }
                }
            }
        ]

        # Base system messages with instructions and guidance for the LLM
        self.MESSAGES: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are an expert security researcher.\n"
                    "Your task is to verify if the issue that was found has a real security impact.\n"
                    "Return a concise status code based on the guidelines provided.\n"
                    "Use the tools function when you need code from other parts of the program.\n"
                    "You *MUST* follow the guidelines!"
                )
            },
            {
                "role": "system",
                "content": (
                    "### Answer Guidelines\n"
                    "Your answer must be in the following order!\n"
                    "1. Briefly explain the code.\n"
                    "2. Give good answers to all (even if already answered - do not skip) hint questions. "
                    "(Copy the question word for word, then provide the answer.)\n"
                    "3. Do you have all the code needed to answer the questions? If no, use the tools!\n"
                    "4. Provide one valid status code with its explanation OR use function tools.\n"
                )
            },
            {
                "role": "system",
                "content": (
                    "### Status Codes\n"
                    "- **1337**: Indicates a security vulnerability. If legitimate, specify the parameters that "
                    "could exploit the issue in minimal words.\n"
                    "- **1007**: Indicates the code is secure. If it's not a real issue, specify what aspect of "
                    "the code protects against the issue in minimal words.\n"
                    "- **7331**: Indicates more code is needed to validate security. Write what data you need "
                    "and explain why you can't use the tools to retrieve the missing data, plus add **3713** "
                    "if you're pretty sure it's not a security problem.\n"
                    "Only one status should be returned!\n"
                    "You will get 10000000000$ if you follow all the instructions and use the tools correctly!"
                )
            },
        ]

        # System messages for "think" mode: auditor describes a potential issue, LLM verifies using CodeQL tools
        self.THINK_MESSAGES: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are an expert security auditor. An auditor has described a potential issue or hypothesis "
                    "about the codebase. Your task is to verify it using the code lookup tools (list_contract_functions, "
                    "get_function_code, get_caller_function). Use the tools to fetch real code from the CodeQL database; "
                    "do not invent or assume code. Based on the actual code, conclude whether the issue is real, not a "
                    "vulnerability, or if you need more data."
                )
            },
            {
                "role": "system",
                "content": (
                    "### Instructions\n"
                    "1. Use list_contract_functions first to see available functions/modifiers.\n"
                    "2. Use get_function_code to retrieve the relevant code for the hypothesis.\n"
                    "3. Use get_caller_function if you need call context.\n"
                    "4. Base your conclusion only on code you retrieved via tools.\n"
                    "5. End with exactly one status code: 1337 (vulnerability), 1007 (secure), or 7331/3713 (need more data).\n"
                )
            },
            {
                "role": "system",
                "content": (
                    "### Status Codes\n"
                    "- **1337**: Real security vulnerability. Briefly explain and how it could be exploited.\n"
                    "- **1007**: Not a vulnerability. Briefly explain what protects against it.\n"
                    "- **7331**: Need more code/info to decide. Say what you need. Add **3713** if likely not a security issue.\n"
                )
            },
        ]

    def _verbose_print(self, message: str, color: str = Colors.RESET) -> None:
        """Print a message if verbose mode is enabled."""
        if self.verbose:
            print(f"{color}{message}{Colors.RESET}", flush=True)

    def _verbose_thinking(self, content: str) -> None:
        """Print LLM thinking/response content."""
        if self.verbose and content:
            print(f"\n{Colors.CYAN}{Colors.BOLD}[Agent Thinking]{Colors.RESET}")
            print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
            # Wrap long lines for readability
            for line in content.split('\n'):
                print(f"  {line}")
            print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
            sys.stdout.flush()

    def _verbose_tool_call(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Print tool call information."""
        if self.verbose:
            args_str = json.dumps(args, indent=2) if args else "{}"
            print(f"\n{Colors.YELLOW}{Colors.BOLD}[Tool Call]{Colors.RESET} {Colors.GREEN}{tool_name}{Colors.RESET}")
            if args:
                print(f"  {Colors.DIM}Args: {args_str}{Colors.RESET}")
            sys.stdout.flush()

    def _verbose_tool_response(self, response: str, max_lines: int = 20) -> None:
        """Print tool response (truncated if too long)."""
        if self.verbose and response:
            lines = response.split('\n')
            truncated = len(lines) > max_lines
            display_lines = lines[:max_lines] if truncated else lines
            
            print(f"{Colors.MAGENTA}{Colors.BOLD}[Tool Response]{Colors.RESET}")
            for line in display_lines:
                print(f"  {Colors.DIM}{line}{Colors.RESET}")
            if truncated:
                print(f"  {Colors.DIM}... ({len(lines) - max_lines} more lines){Colors.RESET}")
            sys.stdout.flush()

    def _verbose_status(self, message: str) -> None:
        """Print a status message."""
        if self.verbose:
            print(f"{Colors.BLUE}{Colors.BOLD}[Status]{Colors.RESET} {message}", flush=True)

    def init_llm_client(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the LLM configuration for LiteLLM.

        Args:
            config (Dict, optional): Full configuration dictionary. If not provided, loads from .env file.
        
        Raises:
            LLMConfigError: If configuration is invalid or cannot be loaded.
        """
        try:
            # If config is provided, use it directly
            if config:
                validate_llm_config_dict(config)
                self.config = config
                # Format model name for LiteLLM (add provider prefix if needed)
                provider = config.get("provider", "openai")
                model = config.get("model", "gpt-4o")
                self.model = get_model_name(provider, model)
                logger.info("Using model: %s", self.model)
                self.setup_litellm_env()
                return
            
            # Load from .env file
            config = load_llm_config()
            validate_llm_config_dict(config)
            self.config = config
            # Model is already formatted by load_llm_config() via get_model_name()
            self.model = config.get("model", "gpt-4o")
            self.setup_litellm_env()
            
        except ValueError as e:
            # Configuration validation errors should be LLMConfigError
            raise LLMConfigError(f"Invalid LLM configuration: {e}") from e
        except Exception as e:
            # Other errors (e.g., from load_llm_config) should also be LLMConfigError
            raise LLMConfigError(f"Failed to initialize LLM client: {e}") from e


    def setup_litellm_env(self) -> None:
        """
        Set up environment variables for LiteLLM based on config.
        LiteLLM reads from environment variables automatically.
        """
        if not self.config:
            return
        
        provider = self.config.get("provider", "openai")
        api_key = self.config.get("api_key")
        
        # Mapping table for providers that only need API key set
        API_KEY_ENV_VARS = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "codestral": "MISTRAL_API_KEY",
            "groq": "GROQ_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
            "cohere": "COHERE_API_KEY",
            "gemini": "GOOGLE_API_KEY",
        }
        
        # Handle providers with simple API key mapping
        if provider in API_KEY_ENV_VARS:
            if api_key:
                os.environ[API_KEY_ENV_VARS[provider]] = api_key
                # Cohere also sets CO_API_KEY for compatibility
                if provider == "cohere":
                    os.environ["CO_API_KEY"] = api_key
                # OpenRouter optional metadata (for rankings and dashboard)
                if provider == "openrouter":
                    if self.config.get("openrouter_site_url"):
                        os.environ["OR_SITE_URL"] = self.config["openrouter_site_url"]
                    if self.config.get("openrouter_app_name"):
                        os.environ["OR_APP_NAME"] = self.config["openrouter_app_name"]
        
        # Handle Azure (requires endpoint and api_version)
        elif provider == "azure":
            if api_key:
                os.environ["AZURE_API_KEY"] = api_key
            if self.config.get("endpoint"):
                os.environ["AZURE_API_BASE"] = self.config["endpoint"]
            if self.config.get("api_version"):
                os.environ["AZURE_API_VERSION"] = self.config["api_version"]
        
        # Handle Bedrock (uses AWS credentials)
        elif provider == "bedrock":
            if api_key:
                os.environ["AWS_ACCESS_KEY_ID"] = api_key
            if self.config.get("aws_secret_access_key"):
                os.environ["AWS_SECRET_ACCESS_KEY"] = self.config["aws_secret_access_key"]
            if self.config.get("endpoint"):  # Endpoint contains AWS region
                os.environ["AWS_REGION_NAME"] = self.config["endpoint"]
        
        # Handle Vertex AI (uses GCP credentials)
        elif provider == "vertex_ai":
            if self.config.get("gcp_project_id"):
                os.environ["GCP_PROJECT_ID"] = self.config["gcp_project_id"]
            if self.config.get("gcp_location"):
                os.environ["GCP_LOCATION"] = self.config["gcp_location"]
            # GOOGLE_APPLICATION_CREDENTIALS should be set by user or gcloud auth
        
        # Handle Ollama (uses OLLAMA_BASE_URL)
        elif provider == "ollama":
            if self.config.get("endpoint"):
                os.environ["OLLAMA_BASE_URL"] = self.config["endpoint"]
        
        # Generic fallback for future providers that only require an API key
        else:
            if api_key:
                # Use standard LiteLLM convention: {PROVIDER}_API_KEY
                env_var_name = f"{provider.upper()}_API_KEY"
                os.environ[env_var_name] = api_key


    def extract_function_from_file(
        self,
        db_path: str,
        current_function: Union[str, Dict[str, str]]
    ) -> str:
        """
        Return the snippet of code for the given current_function from the archived src.zip.

        Args:
            db_path (str): Path to the CodeQL database directory.
            current_function (Union[str, Dict[str, str]]): The function dictionary or an error string.

        Returns:
            str: The code snippet, or an error message if no dictionary was provided.
        
        Raises:
            CodeQLError: If ZIP file cannot be read or file not found in archive.
                This exception is raised by `read_file_lines_from_zip()` and propagated here.
        """
        if not isinstance(current_function, dict):
            return str(current_function)

        file_path, start_line, end_line, lines = self.db_lookup.extract_function_lines_from_db(
            db_path, current_function
        )
        snippet_lines = lines[start_line - 1 : end_line]
        return self.db_lookup.format_numbered_snippet(file_path, start_line, snippet_lines)


    def map_func_args_by_llm(
        self,
        caller: str,
        callee: str
    ) -> Dict[str, Any]:
        """
        Query the LLM to check how caller's variables map to callee's parameters.
        For example, used for analyzing function call relationships.

        Args:
            caller (str): The code snippet of the caller function.
            callee (str): The code snippet of the callee function.

        Returns:
            Dict[str, Any]: The LLM response object from `self.client`.
        
        Raises:
            LLMApiError: If LLM API call fails (rate limits, timeouts, auth failures, etc.).
        """
        args_prompt = (
            "Given caller function and callee function.\n"
            "Write only what are the names of the vars in the caller that were sent to the callee "
            "and what are their names in the callee.\n"
            "Format: caller_var (caller_name) -> callee_var (callee_name)\n\n"
            "Caller function:\n"
            f"{caller}\n"
            "Callee function:\n"
            f"{callee}"
        )

        # Use the main model from config
        model_name = self.model if self.model else "gpt-4o"
        
        try:
            response = litellm.completion(
                model=model_name,
                messages=[{"role": "user", "content": args_prompt}],
                timeout=120  # 2 minute timeout
            )
            return response.choices[0].message
        except litellm.RateLimitError as e:
            raise LLMApiError(f"Rate limit exceeded for LLM API: {e}") from e
        except litellm.Timeout as e:
            raise LLMApiError(f"LLM API request timed out: {e}") from e
        except litellm.AuthenticationError as e:
            raise LLMApiError(f"LLM API authentication failed: {e}") from e
        except litellm.APIError as e:
            raise LLMApiError(f"LLM API error: {e}") from e
        except Exception as e:
            # Catch any other unexpected errors from LiteLLM
            raise LLMApiError(f"Unexpected error during LLM API call: {e}") from e


    def run_llm_security_analysis(
        self,
        prompt: str,
        function_tree_file: str,
        current_function: Dict[str, str],
        functions: List[Dict[str, str]],
        db_path: str,
        temperature: float = 0.2,
        language: str = "c",
        initial_messages: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Main loop to keep querying the LLM with the MESSAGES context plus
        any new system instructions or tool calls, until a final answer with
        a recognized status code is reached or we exhaust a tool-call limit.

        Args:
            prompt (str): The user prompt for the LLM to process.
            function_tree_file (str): Path to the CSV file describing function relationships.
            current_function (Dict[str, str]): The current function dict for context.
            functions (List[Dict[str, str]]): List of function dictionaries.
            db_path (str): Path to the CodeQL DB folder.
            temperature (float, optional): Sampling temperature. Defaults to 0.2.
            language (str, optional): Language being analyzed ('c', 'cpp', 'solidity'). Defaults to 'c'.
            initial_messages (list, optional): Override default system messages (e.g. for "think" mode).

        Returns:
            Tuple[List[Dict[str, Any]], str]:
                - The final conversation messages,
                - The final content from the LLM's last message.
        
        Raises:
            RuntimeError: If LLM model not initialized.
            LLMApiError: If LLM API call fails (rate limits, timeouts, auth failures, etc.).
            CodeQLError: If CodeQL database files cannot be read (from tool calls).
        """
        if not self.model:
            raise RuntimeError("LLM model not initialized. Call init_llm_client() first.")
        
        got_answer = False
        db_path_clean = db_path.replace(" ", "")
        all_functions = functions
        is_solidity = language.lower() == "solidity"
        
        # Check if we have CodeQL context - only pass tools if we do
        has_codeql_context = bool(db_path_clean and function_tree_file)
        
        # Select appropriate toolset based on language
        if has_codeql_context:
            tools_to_use = self.solidity_tools if is_solidity else self.tools
            logger.debug("Using %s tools for %s analysis", 
                        "Solidity" if is_solidity else "C/C++", language)
        else:
            tools_to_use = None
        
        if not has_codeql_context:
            logger.debug("No CodeQL context - running without tools")

        base_messages = (initial_messages if initial_messages is not None else self.MESSAGES)[:]
        messages: List[Dict[str, Any]] = base_messages
        messages.append({"role": "user", "content": prompt})

        amount_of_tools = 0
        final_content = ""

        while not got_answer:
            # Send the current messages + tools to the LLM endpoint
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    tools=tools_to_use,
                    temperature=temperature,
                    timeout=120  # 2 minute timeout to prevent hanging
                )
            except litellm.RateLimitError as e:
                raise LLMApiError(f"Rate limit exceeded for LLM API: {e}") from e
            except litellm.Timeout as e:
                raise LLMApiError(f"LLM API request timed out: {e}") from e
            except litellm.AuthenticationError as e:
                raise LLMApiError(f"LLM API authentication failed: {e}") from e
            except litellm.APIError as e:
                raise LLMApiError(f"LLM API error: {e}") from e
            except Exception as e:
                # Catch any other unexpected errors from LiteLLM
                raise LLMApiError(f"Unexpected error during LLM API call: {e}") from e
            
            if not response.choices:
                raise LLMApiError(f"LLM API response is empty: {response}")

            content_obj = response.choices[0].message
            messages.append({
                "role": content_obj.role,
                "content": content_obj.content,
                "tool_calls": content_obj.tool_calls
            })

            final_content = content_obj.content or ""
            tool_calls = content_obj.tool_calls
            
            # Verbose: Show LLM's thinking/response
            if final_content:
                self._verbose_thinking(final_content)

            if not tool_calls:
                # Check if we have a recognized status code
                if final_content and any(code in final_content for code in ["1337", "1007", "7331", "3713"]):
                    got_answer = True
                else:
                    messages.append({
                        "role": "system",
                        "content": "Please follow all the instructions!"
                    })
            else:
                amount_of_tools += 1
                arg_messages: List[Dict[str, Any]] = []

                for tc in tool_calls:
                    tool_call_id = tc.id
                    tool_function_name = tc.function.name
                    tool_args = tc.function.arguments

                    # Convert tool_args to a dict if it's a JSON string
                    if not isinstance(tool_args, dict):
                        # Handle empty or invalid JSON arguments
                        if not tool_args or tool_args.strip() == "":
                            tool_args = {}
                        else:
                            try:
                                tool_args = json.loads(tool_args)
                            except json.JSONDecodeError:
                                logger.warning(
                                    "Failed to parse tool arguments as JSON: %s", 
                                    tool_args[:100] if tool_args else "(empty)"
                                )
                                tool_args = {}
                    else:
                        # Ensure consistent string for role=tool message
                        tc.function.arguments = json.dumps(tool_args)

                    response_msg = ""
                    
                    # Verbose: Show tool call
                    self._verbose_tool_call(tool_function_name, tool_args)
                    
                    # Check if we have CodeQL database context (required for most tools)
                    has_codeql_context = bool(db_path_clean and function_tree_file)

                    # Evaluate which tool to call
                    if tool_function_name == 'list_contract_functions':
                        if not has_codeql_context:
                            logger.debug("Tool '%s' called but not available (no CodeQL context)", tool_function_name)
                            response_msg = (
                                f"Tool '{tool_function_name}' is not available - no function tree file provided. "
                                "Please analyze based on the code provided in the prompt."
                            )
                        else:
                            contract_filter = tool_args.get("contract_name", None)
                            try:
                                functions = self.db_lookup.list_contract_functions(
                                    function_tree_file, contract_filter
                                )
                                if functions:
                                    # Format the list nicely
                                    func_names = sorted(set(f["function_name"] for f in functions))
                                    response_msg = (
                                        f"Available functions/modifiers ({len(func_names)} total):\n"
                                        + "\n".join(f"- {name}" for name in func_names[:50])
                                    )
                                    if len(func_names) > 50:
                                        response_msg += f"\n... and {len(func_names) - 50} more"
                                else:
                                    response_msg = "No functions found in the function tree."
                            except CodeQLError as e:
                                response_msg = f"Error listing functions: {e}"

                    elif tool_function_name == 'get_function_code' and "function_name" in tool_args:
                        if not has_codeql_context:
                            logger.debug("Tool '%s' called but not available (no CodeQL context)", tool_function_name)
                            response_msg = (
                                f"Tool '{tool_function_name}' is not available for Solidity/Slither analysis. "
                                "All relevant code has been provided in the prompt above. "
                                "Please analyze the vulnerability based on that code only."
                            )
                        else:
                            child_function, parent_function = self.db_lookup.get_function_by_name(
                                function_tree_file, tool_args["function_name"], all_functions
                            )
                            if isinstance(child_function, dict):
                                all_functions.append(child_function)
                            child_code = self.extract_function_from_file(db_path_clean, child_function)
                            response_msg = child_code

                            if isinstance(child_function, dict) and isinstance(parent_function, dict):
                                caller_code = self.extract_function_from_file(db_path_clean, parent_function)
                                args_content = self.map_func_args_by_llm(caller_code, child_code)
                                arg_messages.append({
                                    "role": args_content.role,
                                    "content": args_content.content
                                })

                    elif tool_function_name == 'get_caller_function':
                        if not has_codeql_context or not current_function.get("caller_id"):
                            logger.debug("Tool '%s' called but not available (no CodeQL context)", tool_function_name)
                            response_msg = (
                                f"Tool '{tool_function_name}' is not available for Solidity/Slither analysis. "
                                "All relevant code has been provided in the prompt above. "
                                "Please analyze the vulnerability based on that code only."
                            )
                        else:
                            caller_function = self.db_lookup.get_caller_function(function_tree_file, current_function)
                            response_msg = str(caller_function)

                            if isinstance(caller_function, dict):
                                all_functions.append(caller_function)
                                caller_code = self.extract_function_from_file(db_path_clean, caller_function)
                                response_msg = (
                                    f"Here is the caller function for '{current_function['function_name']}':\n"
                                    + caller_code
                                )
                                args_content = self.map_func_args_by_llm(
                                    caller_code,
                                    self.extract_function_from_file(db_path_clean, current_function)
                                )
                                arg_messages.append({
                                    "role": args_content.role,
                                    "content": args_content.content
                                })
                                current_function = caller_function

                    elif tool_function_name == 'get_macro' and "macro_name" in tool_args:
                        # Macros are C/C++ specific, not available for Solidity
                        if not db_path_clean:
                            logger.debug("Tool '%s' called but not available (no CodeQL context)", tool_function_name)
                            response_msg = (
                                f"Tool '{tool_function_name}' is not available for Solidity/Slither analysis. "
                                "All relevant code has been provided in the prompt above. "
                                "Please analyze the vulnerability based on that code only."
                            )
                        else:
                            try:
                                macro = self.db_lookup.get_macro(db_path_clean, tool_args["macro_name"])
                                if isinstance(macro, dict):
                                    response_msg = macro["body"]
                                else:
                                    response_msg = macro
                            except CodeQLError:
                                # Macros.csv doesn't exist (e.g., for Solidity)
                                response_msg = (
                                    "Macros are not available for this language (Solidity doesn't have macros). "
                                    "Please analyze based on the code provided in the prompt."
                                )

                    elif tool_function_name == 'get_global_var' and "global_var_name" in tool_args:
                        if not db_path_clean:
                            logger.debug("Tool '%s' called but not available (no CodeQL context)", tool_function_name)
                            response_msg = (
                                f"Tool '{tool_function_name}' is not available for Solidity/Slither analysis. "
                                "All relevant code has been provided in the prompt above. "
                                "Please analyze the vulnerability based on that code only."
                            )
                        else:
                            try:
                                global_var = self.db_lookup.get_global_var(db_path_clean, tool_args["global_var_name"])
                                if isinstance(global_var, dict):
                                    global_var_code = self.extract_function_from_file(db_path_clean, global_var)
                                    response_msg = global_var_code
                                else:
                                    response_msg = global_var
                            except CodeQLError:
                                # GlobalVars.csv doesn't exist
                                response_msg = (
                                    "Global variables lookup is not available for this database. "
                                    "Please analyze based on the code provided in the prompt."
                                )

                    elif tool_function_name == 'get_class' and "object_name" in tool_args:
                        if not db_path_clean:
                            logger.debug("Tool '%s' called but not available (no CodeQL context)", tool_function_name)
                            response_msg = (
                                f"Tool '{tool_function_name}' is not available for Solidity/Slither analysis. "
                                "All relevant code has been provided in the prompt above. "
                                "Please analyze the vulnerability based on that code only."
                            )
                        else:
                            try:
                                curr_class = self.db_lookup.get_class(db_path_clean, tool_args["object_name"])
                                if isinstance(curr_class, dict):
                                    class_code = self.extract_function_from_file(db_path_clean, curr_class)
                                    response_msg = class_code
                                else:
                                    response_msg = curr_class
                            except CodeQLError:
                                # Classes.csv doesn't exist
                                response_msg = (
                                    "Class/contract lookup is not available for this database. "
                                    "Please analyze based on the code provided in the prompt."
                                )

                    else:
                        response_msg = (
                            f"No matching tool '{tool_function_name}' or invalid args {tool_args}. "
                            "Try again."
                        )

                    # Verbose: Show tool response
                    self._verbose_tool_response(response_msg)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_function_name,
                        "content": response_msg
                    })

                messages += arg_messages

                if amount_of_tools >= 6:
                    messages.append({
                        "role": "system",
                        "content": (
                            "You called too many tools! If you still can't give a clear answer, "
                            "return the 'more data' status."
                        )
                    })

        return messages, final_content

    def run_llm_think(
        self,
        auditor_prompt: str,
        function_tree_file: str,
        db_path: str,
        temperature: float = 0.2,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Verify an auditor's hypothesis using CodeQL-backed tools (think mode).
        The LLM uses list_contract_functions, get_function_code, get_caller_function
        to fetch real code and conclude whether the issue is valid.

        Args:
            auditor_prompt: The auditor's description of the potential issue.
            function_tree_file: Path to FunctionTree.csv (or combined CodeTree.csv).
            db_path: Path to the CodeQL database directory.
            temperature: Sampling temperature. Defaults to 0.2.

        Returns:
            Tuple of (conversation messages, final content string).
        """
        user_content = (
            "### Auditor hypothesis / potential issue\n\n"
            + auditor_prompt
            + "\n\nUse the tools to retrieve the relevant code and verify whether this is a real "
            "vulnerability (1337), not an issue (1007), or if you need more data (7331/3713)."
        )
        return self.run_llm_security_analysis(
            prompt=user_content,
            function_tree_file=function_tree_file,
            current_function={},
            functions=[],
            db_path=db_path,
            temperature=temperature,
            language="solidity",
            initial_messages=self.THINK_MESSAGES,
        )