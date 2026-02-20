#!/usr/bin/env python3
"""
Solidity analysis engine for Ouroborus using Slither.

This module coordinates the analysis of Slither findings with LLM classification.
It processes Slither JSON output, extracts code context, and uses the LLM
to determine if findings are true positives, false positives, or need more data.

Analysis Pipeline Algorithm:
    1. Load Slither JSON results, parse into SlitherFinding objects
    2. Group findings by detector type (reentrancy, unchecked-transfer, etc.)
    3. For each finding: extract code context from source files
    4. Build prompt using Solidity-specific templates
    5. Run LLM analysis; classify by status codes (1337/1007/7331)
    6. Save results to output/results/solidity/<detector_type>/
"""

from pathlib import Path
import json
from typing import Any, Dict, List, Optional

from src.slither.slither_parser import (
    SlitherResultParser,
    SlitherFinding,
    get_detector_description
)
from src.llm.llm_analyzer import LLMAnalyzer
from src.utils.common_functions import read_file, write_file_ascii
from src.utils.config_validator import validate_and_exit_on_error
from src.utils.logger import get_logger
from src.utils.exceptions import OuroborusError, LLMApiError

logger = get_logger(__name__)


class SolidityIssueAnalyzer:
    """
    Analyzes Slither findings for Solidity contracts, extracts code context,
    and uses LLM to classify security issues.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the SolidityIssueAnalyzer.
        
        Args:
            config: Optional LLM configuration dictionary.
        """
        self.lang = "solidity"
        self.config = config
        self.repo_path: Optional[str] = None
        self._project_name: Optional[str] = None

    @property
    def project_name(self) -> str:
        """Get the project name from repo_path."""
        if self._project_name:
            return self._project_name
        if self.repo_path:
            # Extract project name from path (e.g., 'output/repos/solidity/0xbepresent_NC' -> '0xbepresent_NC')
            return Path(self.repo_path).name
        return "unknown_project"

    def _get_results_folder(self, check: str) -> Path:
        """
        Build the results folder path for a detector type.
        
        Format: output/results/solidity/<project_name>/slither-<detector>/
        
        Args:
            check: The detector check name.
        
        Returns:
            Path to the results folder.
        """
        detector_folder = f"slither-{check.replace(' ', '_')}"
        return Path("output/results/solidity") / self.project_name / detector_folder

    def build_prompt_by_template(
        self,
        finding: SlitherFinding,
        code: str
    ) -> str:
        """
        Build the prompt for LLM analysis using Solidity templates.
        
        Args:
            finding: The SlitherFinding to analyze.
            code: The extracted code context.
        
        Returns:
            str: The formatted prompt string.
        """
        templates_base = Path("data/templates/solidity")
        
        # Try detector-specific template first
        hints_path = templates_base / f"{finding.check}.template"
        if not hints_path.exists():
            hints_path = templates_base / "general.template"
        
        hints = read_file(str(hints_path))
        
        # Read the main template
        template_path = templates_base / "template.template"
        template = read_file(str(template_path))
        
        # Get location info
        location = finding.get_primary_location()
        if location:
            filename = location.get("filename_relative", "unknown")
            lines = location.get("lines", [])
            line_info = f"{lines[0]}-{lines[-1]}" if lines else "unknown"
            location_str = f"{filename}:{line_info}"
        else:
            location_str = "unknown location"
        
        # Build the prompt
        prompt = template.format(
            name=finding.check,
            description=get_detector_description(finding.check),
            impact=finding.impact,
            confidence=finding.confidence,
            message=finding.description,
            location=location_str,
            hints=hints,
            code=code
        )
        
        return prompt

    def ensure_directories_exist(self, dirs: List[str]) -> None:
        """Create directories if they don't exist."""
        for d in dirs:
            dir_path = Path(d)
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                except (PermissionError, OSError) as e:
                    raise OuroborusError(f"Failed to create directory {d}: {e}") from e

    def determine_issue_status(self, llm_content: str) -> str:
        """
        Determine issue status from LLM response.
        
        Args:
            llm_content: The LLM response text.
        
        Returns:
            str: "true", "false", or "more"
        """
        if "1337" in llm_content:
            return "true"
        elif "1007" in llm_content:
            return "false"
        else:
            return "more"

    def format_llm_messages(self, messages: List[str]) -> str:
        """Format LLM messages for saving to file."""
        gpt_result = "[\n    " + ",\n    ".join(
            f"'''{item}'''" if "\n" in item else repr(item) for item in messages
        ).replace("\\n", "\n    ").replace("\\t", " ") + "\n]"
        return gpt_result

    def save_raw_input_data(
        self,
        prompt: str,
        finding: SlitherFinding,
        results_folder: str,
        issue_id: int
    ) -> None:
        """Save raw input data before LLM analysis."""
        raw_data = json.dumps({
            "check": finding.check,
            "impact": finding.impact,
            "confidence": finding.confidence,
            "description": finding.description,
            "affected_functions": finding.get_affected_functions(),
            "affected_contracts": finding.get_affected_contracts(),
            "prompt": prompt
        }, ensure_ascii=False, indent=2)
        
        raw_output_file = Path(results_folder) / f"{issue_id}_raw.json"
        write_file_ascii(str(raw_output_file), raw_data)

    def get_next_issue_id(self, issue_type: str) -> int:
        """Get the next available issue ID for a detector type."""
        results_folder = self._get_results_folder(issue_type)
        
        if not results_folder.exists() or len(list(results_folder.glob("*.json"))) == 0:
            return 1
        
        max_issue_id = 1
        for file in results_folder.glob("*.json"):
            try:
                issue_id = int(file.stem.split("_")[0])
                max_issue_id = max(issue_id, max_issue_id)
            except ValueError:
                continue
        
        return max_issue_id + 1

    def process_finding_group(
        self,
        check: str,
        findings: List[SlitherFinding],
        parser: SlitherResultParser,
        llm_analyzer: LLMAnalyzer
    ) -> None:
        """
        Process all findings of a single detector type.
        
        Args:
            check: The detector check name.
            findings: List of findings for this detector.
            parser: The SlitherResultParser instance.
            llm_analyzer: The LLMAnalyzer instance.
        """
        results_folder = self._get_results_folder(check)
        self.ensure_directories_exist([str(results_folder)])
        
        issue_id = self.get_next_issue_id(check)
        real_issues = []
        false_issues = []
        more_data = []
        skipped_issues = []
        
        logger.info("Found %d issues of type %s", len(findings), check)
        logger.info("")
        
        for finding in findings:
            # Extract code context
            code_parts = []
            
            # Get the primary code snippet
            snippet, filename, start_line, end_line = parser.extract_code_snippet(finding)
            if snippet:
                code_parts.append(f"// File: {filename}\n{snippet}")
            
            # Get full function code for affected functions
            func_codes = parser.extract_function_code(finding)
            for func_name, func_file, func_code in func_codes:
                if func_code and func_code not in snippet:
                    code_parts.append(f"\n// Function: {func_name} ({func_file})\n{func_code}")
            
            code = "\n\n".join(code_parts) if code_parts else finding.markdown
            
            # Build prompt
            prompt = self.build_prompt_by_template(finding, code)
            
            # Save raw input
            self.save_raw_input_data(prompt, finding, str(results_folder), issue_id)
            
            # Run LLM analysis
            # If codeql_db and function_tree are provided, enable code lookup tools
            db_path = getattr(self, 'codeql_db', None) or ""
            function_tree_file = getattr(self, 'function_tree', None) or ""
            
            try:
                messages, content = llm_analyzer.run_llm_security_analysis(
                    prompt,
                    function_tree_file,
                    {},  # No current function dict
                    [],  # No function list
                    db_path,
                    language="solidity"
                )
            except LLMApiError as e:
                logger.warning("Issue ID: %s SKIPPED - LLM error: %s", issue_id, e)
                skipped_issues.append(issue_id)
                issue_id += 1
                continue
            
            # Save results
            gpt_result = self.format_llm_messages([str(m.get("content", "")) for m in messages if m.get("content")])
            final_file = Path(results_folder) / f"{issue_id}_final.json"
            write_file_ascii(str(final_file), gpt_result)
            
            # Determine status
            status = self.determine_issue_status(content)
            if status == "true":
                real_issues.append(issue_id)
                status_str = "True Positive"
            elif status == "false":
                false_issues.append(issue_id)
                status_str = "False Positive"
            else:
                more_data.append(issue_id)
                status_str = "LLM needs More Data"
            
            logger.info("Issue ID: %s, LLM decision: → %s", issue_id, status_str)
            issue_id += 1
        
        # Log summary
        logger.info("")
        logger.info("Detector: %s", check)
        logger.info("Total issues: %d", len(findings))
        logger.info("True Positive: %d", len(real_issues))
        logger.info("False Positive: %d", len(false_issues))
        logger.info("LLM needs More Data: %d", len(more_data))
        if skipped_issues:
            logger.warning("Skipped (LLM errors): %d (IDs: %s)", len(skipped_issues), skipped_issues)
        logger.info("")

    def run(
        self, 
        slither_results: Dict[str, Any], 
        repo_path: str,
        codeql_db: Optional[str] = None,
        function_tree: Optional[str] = None,
        verbose: bool = False
    ) -> None:
        """
        Main analysis routine for Solidity contracts.
        
        Args:
            slither_results: Parsed Slither JSON output.
            repo_path: Path to the analyzed repository.
            codeql_db: Optional path to CodeQL database for code lookup tools.
            function_tree: Optional path to FunctionTree.csv for function lookups.
            verbose: If True, print LLM thinking and tool calls in real-time.
        """
        self.repo_path = repo_path
        self.codeql_db = codeql_db
        self.function_tree = function_tree
        
        # Validate configuration
        if self.config is None:
            validate_and_exit_on_error()
        
        # Initialize LLM with verbose mode
        llm_analyzer = LLMAnalyzer(verbose=verbose)
        llm_analyzer.init_llm_client(config=self.config)
        
        # Parse Slither results
        parser = SlitherResultParser(repo_path)
        parser.load_results_from_dict(slither_results)
        
        if not parser.findings:
            logger.info("No findings to analyze")
            return
        
        # Group findings by detector type
        grouped = parser.group_findings_by_check()
        
        total_findings = len(parser.findings)
        logger.info("Total findings to analyze: %d", total_findings)
        if codeql_db and function_tree:
            logger.info("CodeQL database + function tree available - code lookup tools enabled")
        elif codeql_db:
            logger.info("CodeQL database available (function tree needed for full tool support)")
        logger.info("")
        
        # Process each group
        for check, findings in grouped.items():
            self.process_finding_group(check, findings, parser, llm_analyzer)


if __name__ == "__main__":
    import sys
    from src.utils.logger import setup_logging
    from src.slither.run_slither import clone_repo, run_slither_analysis, SlitherConfig, SECURITY_DETECTORS
    
    setup_logging()
    
    if len(sys.argv) < 2:
        print("Usage: python ouroborus_solidity.py <owner/repo>")
        sys.exit(1)
    
    repo_name = sys.argv[1]
    
    # Clone repo
    repo_path = clone_repo(repo_name, "output/repos/solidity")
    
    # Run Slither
    config = SlitherConfig(
        detectors=SECURITY_DETECTORS,
        filter_paths=["node_modules", "lib", "test", "tests", "mocks"]
    )
    results = run_slither_analysis(repo_path, "output/slither_results", config)
    
    # Analyze with LLM
    analyzer = SolidityIssueAnalyzer()
    analyzer.run(results, repo_path)
