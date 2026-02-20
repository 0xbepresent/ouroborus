"""
CodeQL database lookup utilities.

This module provides functions to query CodeQL CSV files (FunctionTree.csv,
Macros.csv, GlobalVars.csv, Classes.csv) and extract code snippets from
the source archive.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.utils.exceptions import CodeQLError
from src.utils.common_functions import read_file_lines_from_zip
from src.utils.csv_parser import parse_csv_row


def tokenize_name(name: str) -> Set[str]:
    """
    Split a camelCase or PascalCase name into lowercase tokens.
    
    Examples:
        'setSenderAllowStatus' -> {'set', 'sender', 'allow', 'status'}
        'HyperliquidForwarder' -> {'hyperliquid', 'forwarder'}
        'get_function_code' -> {'get', 'function', 'code'}
    
    Args:
        name: The function/method name to tokenize.
    
    Returns:
        Set of lowercase tokens.
    """
    # Split on underscores first
    parts = name.replace("_", " ").split()
    tokens = set()
    for part in parts:
        # Split camelCase/PascalCase: insert space before uppercase letters
        split_camel = re.sub(r'([a-z])([A-Z])', r'\1 \2', part)
        for token in split_camel.split():
            if token:
                tokens.add(token.lower())
    return tokens


def token_similarity(name1: str, name2: str) -> float:
    """
    Calculate similarity between two names based on shared tokens.
    
    Returns a score between 0.0 and 1.0 (Jaccard similarity).
    
    Args:
        name1: First name to compare.
        name2: Second name to compare.
    
    Returns:
        Similarity score (0.0 = no overlap, 1.0 = identical tokens).
    """
    tokens1 = tokenize_name(name1)
    tokens2 = tokenize_name(name2)
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)


class CodeQLDBLookup:
    """
    Encapsulates CodeQL database lookup operations for functions, macros,
    global variables, classes, and caller relationships.
    """

    def _iter_csv_lines(
        self,
        file_path: Union[str, Path],
        file_type_name: str
    ):
        """
        Generator that yields lines from a CSV file, handling file I/O errors.

        This helper centralizes CSV file opening, line iteration, and error handling.
        Each method can iterate over the yielded lines and apply method-specific logic.

        Args:
            file_path: Path to the CSV file to read.
            file_type_name: Descriptive name for the file type (e.g., "Function tree file",
                           "Macros CSV", "GlobalVars CSV") for error messages.

        Yields:
            str: Each line from the CSV file (including newline characters).

        Raises:
            CodeQLError: If file cannot be read (not found, permission denied, etc.).
        """
        try:
            with Path(file_path).open("r", encoding="utf-8") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    yield line
        except (FileNotFoundError, PermissionError, OSError) as e:
            raise self._convert_csv_file_error(e, file_path, file_type_name) from e


    @staticmethod
    def _convert_csv_file_error(
        error: Exception,
        file_path: Union[str, Path],
        file_type_name: str
    ) -> CodeQLError:
        """
        Convert file I/O exceptions to CodeQLError with consistent messaging.

        Args:
            error: The original exception (FileNotFoundError, PermissionError, or OSError).
            file_path: Path to the CSV file that caused the error.
            file_type_name: Descriptive name for the file type (e.g., "Function tree file",
                           "Macros CSV", "GlobalVars CSV") for error messages.

        Returns:
            CodeQLError: Converted exception with appropriate message.
        """
        file_path_str = str(file_path)
        if isinstance(error, FileNotFoundError):
            return CodeQLError(f"{file_type_name} not found: {file_path_str}")
        elif isinstance(error, PermissionError):
            return CodeQLError(f"Permission denied reading {file_type_name}: {file_path_str}")
        elif isinstance(error, OSError):
            return CodeQLError(f"OS error while reading {file_type_name}: {file_path_str}")
        else:
            # Fallback for unexpected exception types
            return CodeQLError(f"Error reading {file_type_name}: {file_path_str}")

    def list_contract_functions(
        self,
        function_tree_file: str,
        contract_name: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        List all functions/modifiers in the function tree, optionally filtered by contract.
        
        Args:
            function_tree_file: Path to FunctionTree.csv.
            contract_name: Optional contract name to filter by. If None, returns all functions.
        
        Returns:
            List of function dictionaries with keys: function_name, file, start_line, end_line.
        
        Raises:
            CodeQLError: If function tree file cannot be read.
        """
        keys = ["function_name", "file", "start_line", "function_id", "end_line", "caller_id"]
        functions = []
        
        for row in self._iter_csv_lines(function_tree_file, "Function tree file"):
            if not row.strip():
                continue
            row_dict = parse_csv_row(row, keys)
            if not row_dict:
                continue
            
            func_name = row_dict["function_name"].replace("\"", "").strip()
            file_path = row_dict["file"].replace("\"", "").strip()
            
            # Filter by contract name if provided (check if contract name is in file path)
            if contract_name:
                if contract_name.lower() not in file_path.lower():
                    continue
            
            functions.append({
                "function_name": func_name,
                "file": file_path,
                "start_line": row_dict.get("start_line", ""),
                "end_line": row_dict.get("end_line", "")
            })
        
        return functions

    def find_similar_functions(
        self,
        function_tree_file: str,
        target_name: str,
        max_suggestions: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find functions with similar names using token-based matching.
        
        Args:
            function_tree_file: Path to FunctionTree.csv.
            target_name: The function name to find similar matches for.
            max_suggestions: Maximum number of suggestions to return.
        
        Returns:
            List of (function_name, similarity_score) tuples, sorted by similarity descending.
        
        Raises:
            CodeQLError: If function tree file cannot be read.
        """
        keys = ["function_name", "file", "start_line", "function_id", "end_line", "caller_id"]
        similarities = []
        seen_names = set()
        
        # Extract just the function name (remove contract prefix)
        target_name_normalized = target_name.replace(".", "::").split("::")[-1]
        
        for row in self._iter_csv_lines(function_tree_file, "Function tree file"):
            if not row.strip():
                continue
            row_dict = parse_csv_row(row, keys)
            if not row_dict:
                continue
            
            func_name = row_dict["function_name"].replace("\"", "").strip()
            if func_name in seen_names:
                continue
            seen_names.add(func_name)
            
            similarity = token_similarity(target_name_normalized, func_name)
            if similarity > 0.2:  # Only include if there's some overlap
                similarities.append((func_name, similarity))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_suggestions]


    def get_function_by_line(
        self,
        function_tree_file: str,
        file: str,
        line: int
    ) -> Optional[Dict[str, str]]:
        """
        Retrieve the function dictionary from a CSV (FunctionTree.csv) that matches
        the specified file and line coverage.

        Args:
            function_tree_file (str): Path to the FunctionTree.csv file.
            file (str): Name of the file as it appears in the CSV row.
            line (int): A line number within the function's start_line and end_line range.

        Returns:
            Optional[Dict[str, str]]: The matching function row as a dict, or None if not found.
        
        Raises:
            CodeQLError: If function tree file cannot be read (not found, permission denied, etc.).
        """
        keys = ["function_name", "file", "start_line", "function_id", "end_line", "caller_id"]
        for function in self._iter_csv_lines(function_tree_file, "Function tree file"):
            if file in function:
                row_dict = parse_csv_row(function, keys)
                if row_dict and row_dict["start_line"] and row_dict["end_line"]:
                    start = int(row_dict["start_line"])
                    end = int(row_dict["end_line"])
                    if start <= line <= end:
                        return row_dict
        return None


    def get_function_by_name(
            self,
            function_tree_file: str,
            function_name: str,
            all_function: List[Dict[str, Any]],
            less_strict: bool = False
        ) -> Tuple[Union[str, Dict[str, str]], Optional[Dict[str, str]]]:
            """
            Retrieve a function by searching function_name in FunctionTree.csv.
            If not found, tries partial match if less_strict is True.

            Args:
                function_tree_file (str): Path to FunctionTree.csv.
                function_name (str): Desired function name (e.g., 'MyClass::MyFunc' or 'Contract.func').
                all_function (List[Dict[str, Any]]): A list of known function dictionaries.
                less_strict (bool, optional): If True, use partial matching. Defaults to False.

            Returns:
                Tuple[Union[str, Dict[str, str]], Optional[Dict[str, str]]]:
                    - The found function (dict) or an error message (str).
                    - The "parent function" that references it, if relevant.
            
            Raises:
                CodeQLError: If function tree file cannot be read (not found, permission denied, etc.).
            """
            keys = ["function_name", "file", "start_line", "function_id", "end_line", "caller_id"]
            
            # Handle both C++ style (::) and Solidity/Java style (.) separators
            function_name_normalized = function_name.replace(".", "::")
            function_name_only = function_name_normalized.split("::")[-1]

            # First try: search within known functions (for C/C++ with call chain)
            for current_function in all_function:
                try:
                    with Path(function_tree_file).open("r", encoding="utf-8") as f:
                        while True:
                            row = f.readline()
                            if not row:
                                break
                            if current_function.get("function_id", "") in row:
                                row_dict = parse_csv_row(row, keys)
                                if not row_dict:
                                    continue

                                candidate_name = row_dict["function_name"].replace("\"", "")
                                if (candidate_name == function_name_only
                                        or (less_strict and function_name_only in candidate_name)):
                                    return row_dict, current_function
                except (FileNotFoundError, PermissionError, OSError) as e:
                    raise self._convert_csv_file_error(e, function_tree_file, "Function tree file") from e

            # Second try: direct search in the entire CSV file (for Solidity or when all_function is empty)
            best_token_match = None
            best_token_score = 0.0
            
            try:
                with Path(function_tree_file).open("r", encoding="utf-8") as f:
                    for row in f:
                        if not row.strip():
                            continue
                        row_dict = parse_csv_row(row, keys)
                        if not row_dict:
                            continue
                        
                        candidate_name = row_dict["function_name"].replace("\"", "").strip()
                        
                        # Exact match
                        if candidate_name == function_name_only:
                            return row_dict, None
                        
                        # Case-insensitive exact match
                        if candidate_name.lower() == function_name_only.lower():
                            return row_dict, None
                        
                        # Partial match if less_strict (substring)
                        if less_strict and function_name_only.lower() in candidate_name.lower():
                            return row_dict, None
                        
                        # Token-based similarity matching
                        if less_strict:
                            score = token_similarity(function_name_only, candidate_name)
                            if score > best_token_score and score >= 0.5:
                                best_token_score = score
                                best_token_match = row_dict
            except (FileNotFoundError, PermissionError, OSError) as e:
                raise self._convert_csv_file_error(e, function_tree_file, "Function tree file") from e

            # Return best token match if found
            if best_token_match:
                return best_token_match, None

            # Try partial matching if less_strict is False
            if not less_strict:
                return self.get_function_by_name(function_tree_file, function_name, all_function, True)
            else:
                # Get suggestions for "did you mean?"
                try:
                    suggestions = self.find_similar_functions(function_tree_file, function_name, max_suggestions=3)
                    if suggestions:
                        suggestion_names = [f"'{name}' ({score:.0%})" for name, score in suggestions]
                        err = (
                            f"Function '{function_name}' not found. "
                            f"Did you mean: {', '.join(suggestion_names)}? "
                            "Use list_contract_functions to see all available functions."
                        )
                    else:
                        err = (
                            f"Function '{function_name}' not found. "
                            "Use list_contract_functions to see all available functions."
                        )
                except CodeQLError:
                    err = (
                        f"Function '{function_name}' not found. Make sure you're using "
                        "the correct tool and args."
                    )
                return err, None


    def get_macro(
        self,
        curr_db: str,
        macro_name: str,
        less_strict: bool = False
    ) -> Union[str, Dict[str, str]]:
        """
        Return macro info from Macros.csv for the given macro_name.
        If not found, tries partial match if less_strict is True.

        Args:
            curr_db (str): Path to the current CodeQL database folder.
            macro_name (str): Macro name to search for.
            less_strict (bool, optional): If True, use partial matching.

        Returns:
            Union[str, Dict[str, str]]:
                - A dict with 'macro_name' and 'body' if found,
                - or an error message string if not found.
        
        Raises:
            CodeQLError: If Macros CSV file cannot be read (not found, permission denied, etc.).
        """
        macro_file = Path(curr_db) / "Macros.csv"
        keys = ["macro_name", "body"]

        for macro in self._iter_csv_lines(macro_file, "Macros CSV"):
            if macro_name in macro:
                row_dict = parse_csv_row(macro, keys)
                if not row_dict:
                    continue

                actual_name = row_dict["macro_name"].replace("\"", "")
                if (actual_name == macro_name
                        or (less_strict and macro_name in actual_name)):
                    return row_dict

        if not less_strict:
            return self.get_macro(curr_db, macro_name, True)
        else:
            return (
                f"Macro '{macro_name}' not found. Make sure you're using the correct tool "
                "with correct args."
            )


    def get_global_var(
        self,
        curr_db: str,
        global_var_name: str,
        less_strict: bool = False
    ) -> Union[str, Dict[str, str]]:
        """
        Return a global variable from GlobalVars.csv matching global_var_name.
        If not found, tries partial match if less_strict is True.

        Args:
            curr_db (str): Path to current CodeQL database folder.
            global_var_name (str): The name of the global variable to find.
            less_strict (bool, optional): If True, use partial matching.

        Returns:
            Union[str, Dict[str, str]]:
                - A dict with ['global_var_name','file','start_line','end_line'] if found,
                - or an error message string if not found.
        
        Raises:
            CodeQLError: If GlobalVars CSV file cannot be read (not found, permission denied, etc.).
        """
        global_var_file = Path(curr_db) / "GlobalVars.csv"
        keys = ["global_var_name", "file", "start_line", "end_line"]
        var_name_only = global_var_name.split("::")[-1]

        for line in self._iter_csv_lines(global_var_file, "GlobalVars CSV"):
            if var_name_only in line:
                data_dict = parse_csv_row(line, keys)
                if not data_dict:
                    continue

                actual_name = data_dict["global_var_name"].replace("\"", "")
                if (actual_name == var_name_only
                        or (less_strict and var_name_only in actual_name)):
                    return data_dict

        if not less_strict:
            return self.get_global_var(curr_db, global_var_name, True)
        else:
            return (
                f"Global var '{global_var_name}' not found. "
                "Could it be a macro or should you use another tool?"
            )


    def get_class(
        self,
        curr_db: str,
        class_name: str,
        less_strict: bool = False
    ) -> Union[str, Dict[str, str]]:
        """
        Return class info (type, class_name, file, start_line, end_line, simple_name)
        from Classes.csv for class_name. If not found, tries partial match if less_strict is True.

        Args:
            curr_db (str): Path to current CodeQL database folder.
            class_name (str): The name of the class/struct/union to find.
            less_strict (bool, optional): If True, use partial matching.

        Returns:
            Union[str, Dict[str, str]]:
                - A dict with keys ['type','class_name','file','start_line','end_line','simple_name']
                - or an error message string if not found.
        
        Raises:
            CodeQLError: If Classes CSV file cannot be read (not found, permission denied, etc.).
        """
        classes_file = Path(curr_db) / "Classes.csv"
        keys = ["type", "class_name", "file", "start_line", "end_line", "simple_name"]
        class_name_only = class_name.split("::")[-1]

        for row in self._iter_csv_lines(classes_file, "Classes CSV"):
            if class_name_only in row:
                row_dict = parse_csv_row(row, keys)
                if not row_dict:
                    continue

                actual_class = row_dict["class_name"].replace("\"", "")
                simple_class = row_dict["simple_name"].replace("\"", "")
                if (
                    actual_class == class_name_only
                    or simple_class == class_name_only
                    or (less_strict and class_name_only in actual_class)
                    or (less_strict and class_name_only in simple_class)
                ):
                    return row_dict

        if not less_strict:
            return self.get_class(curr_db, class_name, True)
        else:
            return f"Class '{class_name}' not found. Could it be a Namespace?"


    def get_caller_function(
        self,
        function_tree_file: str,
        current_function: Dict[str, str]
    ) -> Union[str, Dict[str, str]]:
        """
        Return the caller function from function_tree_file that calls current_function.

        Args:
            function_tree_file (str): Path to FunctionTree.csv.
            current_function (Dict[str, str]): The function dictionary whose caller we want.

        Returns:
            Union[str, Dict[str, str]]:
                - Dict describing the caller if found
                - or an error string if the caller wasn't found.
        
        Raises:
            CodeQLError: If function tree file cannot be read (not found, permission denied, etc.).
        """
        keys = ["function_name", "file", "start_line", "function_id", "end_line", "caller_id"]
        caller_id = current_function["caller_id"].replace("\"", "").strip()

        for line in self._iter_csv_lines(function_tree_file, "Function tree file"):
            if caller_id in line:
                data_dict = parse_csv_row(line, keys)
                if not data_dict:
                    continue
                if data_dict["function_id"].replace("\"", "").strip() == caller_id:
                    return data_dict

        # Fallback if 'caller_id' is in format file:line
        maybe_line = caller_id.split(":")
        if len(maybe_line) == 2:
            file_part, line_part = maybe_line
            function = self.get_function_by_line(function_tree_file, file_part[1:], int(line_part))
            if function:
                return function

        return (
            "Caller function was not found. "
            "Make sure you are using the correct tool with the correct args."
        )


    def extract_function_lines_from_db(
        self,
        db_path: str,
        current_function: Dict[str, str],
    ) -> Tuple[str, int, int, List[str]]:
        """
        Extract function lines from the CodeQL database source archive.

        Args:
            db_path (str): Path to the CodeQL database directory.
            current_function (Dict[str, str]): The function dictionary.

        Returns:
            Tuple[str, int, int, List[str]]:
                - file_path (str): The file path (after .replace and [1:])
                - start_line (int): Starting line number
                - end_line (int): Ending line number
                - all_lines (List[str]): Full file splitlines
        """
        src_zip = Path(db_path) / "src.zip"
        file_path = current_function["file"].replace("\"", "")[1:]
        code_file = read_file_lines_from_zip(str(src_zip), file_path)
        lines = code_file.split("\n")

        start_line = int(current_function["start_line"])
        end_line = int(current_function["end_line"])
        return file_path, start_line, end_line, lines


    @staticmethod
    def format_numbered_snippet(file_path: str, start_line: int, snippet_lines: List[str]) -> str:
        """
        Format a code snippet with line numbers.

        Args:
            file_path (str): Path to the source file.
            start_line (int): Starting line number (1-indexed).
            snippet_lines (List[str]): The code lines to format.

        Returns:
            str: Formatted snippet with line numbers.
        """
        snippet = "\n".join(
            f"{start_line + i}: {text}" for i, text in enumerate(snippet_lines)
        )
        return f"file: {file_path}\n{snippet}"
