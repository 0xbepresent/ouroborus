#!/usr/bin/env python3
"""
Parse Slither JSON output and extract relevant information for LLM analysis.

This module transforms Slither's raw JSON output into a structured format
that can be used to build prompts for the LLM analyzer.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger
from src.utils.exceptions import OuroborusError

logger = get_logger(__name__)


@dataclass
class SlitherFinding:
    """Represents a single Slither finding."""
    
    # Detector information
    check: str  # Detector name (e.g., "reentrancy-eth")
    impact: str  # high, medium, low, informational
    confidence: str  # high, medium, low
    
    # Finding details
    description: str
    markdown: str  # Detailed markdown description
    
    # Location information
    elements: List[Dict[str, Any]] = field(default_factory=list)
    
    # Additional metadata
    first_markdown_element: str = ""
    id: str = ""
    
    @property
    def severity_score(self) -> int:
        """Get numeric severity score for sorting."""
        impact_scores = {"high": 3, "medium": 2, "low": 1, "informational": 0}
        confidence_scores = {"high": 3, "medium": 2, "low": 1}
        return (
            impact_scores.get(self.impact, 0) * 10 +
            confidence_scores.get(self.confidence, 0)
        )
    
    def get_primary_location(self) -> Optional[Dict[str, Any]]:
        """Get the primary source location for this finding."""
        for elem in self.elements:
            if "source_mapping" in elem:
                return elem["source_mapping"]
        return None
    
    def get_affected_functions(self) -> List[str]:
        """Get list of affected function names."""
        functions = []
        for elem in self.elements:
            if elem.get("type") == "function":
                name = elem.get("name", "unknown")
                contract = elem.get("type_specific_fields", {}).get("parent", {}).get("name", "")
                if contract:
                    functions.append(f"{contract}.{name}")
                else:
                    functions.append(name)
        return functions
    
    def get_affected_contracts(self) -> List[str]:
        """Get list of affected contract names."""
        contracts = set()
        for elem in self.elements:
            if elem.get("type") == "contract":
                contracts.add(elem.get("name", "unknown"))
            elif "type_specific_fields" in elem:
                parent = elem["type_specific_fields"].get("parent", {})
                if parent.get("type") == "contract":
                    contracts.add(parent.get("name", "unknown"))
        return list(contracts)


class SlitherResultParser:
    """Parse and process Slither JSON results."""
    
    def __init__(self, repo_path: str):
        """
        Initialize the parser.
        
        Args:
            repo_path: Path to the analyzed repository.
        """
        self.repo_path = Path(repo_path)
        self.findings: List[SlitherFinding] = []
        self.raw_results: Dict[str, Any] = {}
    
    def _resolve_source_path(self, filename: str) -> Optional[Path]:
        """
        Resolve the source file path using multiple strategies.
        
        Slither stores paths relative to where it was run, which may differ
        from the repo_path. This method tries various path resolution strategies.
        
        Args:
            filename: The filename from Slither's source_mapping.
        
        Returns:
            Path to the source file, or None if not found.
        """
        if not filename:
            return None
        
        # Strategy 1: Check if filename is absolute path and exists
        if Path(filename).is_absolute() and Path(filename).exists():
            return Path(filename)
        
        # Strategy 2: Check if filename exists as-is (relative to cwd)
        if Path(filename).exists():
            return Path(filename)
        
        # Strategy 3: Prepend repo_path
        candidate = self.repo_path / filename
        if candidate.exists():
            return candidate
        
        # Strategy 4: If filename contains repo_path, strip it (avoid duplication)
        repo_path_str = str(self.repo_path)
        if filename.startswith(repo_path_str):
            relative_part = filename[len(repo_path_str):].lstrip("/")
            candidate = self.repo_path / relative_part
            if candidate.exists():
                return candidate
        
        # Strategy 5: Try with contracts/ prefix
        candidate = self.repo_path / "contracts" / filename
        if candidate.exists():
            return candidate
        
        # Strategy 6: Extract just the filename and search in common dirs
        basename = Path(filename).name
        for subdir in ["contracts", "src", "."]:
            subdir_path = self.repo_path / subdir
            if subdir_path.exists():
                for match in subdir_path.rglob(basename):
                    if match.is_file():
                        return match
        
        return None
    
    def load_results(self, json_path: str) -> None:
        """
        Load Slither JSON results from file.
        
        Args:
            json_path: Path to the Slither JSON output file.
        
        Raises:
            OuroborusError: If file cannot be read or parsed.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                self.raw_results = json.load(f)
        except FileNotFoundError as e:
            raise OuroborusError(f"Slither results file not found: {json_path}") from e
        except json.JSONDecodeError as e:
            raise OuroborusError(f"Invalid JSON in Slither results: {e}") from e
        
        self._parse_detectors()
    
    def load_results_from_dict(self, results: Dict[str, Any]) -> None:
        """
        Load Slither results from a dictionary.
        
        Args:
            results: Slither JSON results as a dictionary.
        """
        self.raw_results = results
        self._parse_detectors()
    
    def _parse_detectors(self) -> None:
        """Parse detector results into SlitherFinding objects."""
        self.findings = []
        
        detectors = self.raw_results.get("results", {}).get("detectors", [])
        
        for i, detector in enumerate(detectors):
            finding = SlitherFinding(
                check=detector.get("check", "unknown"),
                impact=detector.get("impact", "unknown"),
                confidence=detector.get("confidence", "unknown"),
                description=detector.get("description", ""),
                markdown=detector.get("markdown", ""),
                elements=detector.get("elements", []),
                first_markdown_element=detector.get("first_markdown_element", ""),
                id=f"{detector.get('check', 'unknown')}_{i}"
            )
            self.findings.append(finding)
        
        # Sort by severity
        self.findings.sort(key=lambda f: f.severity_score, reverse=True)
        
        logger.info("Parsed %d findings from Slither results", len(self.findings))
    
    def get_findings_by_impact(self, impact: str) -> List[SlitherFinding]:
        """Get findings filtered by impact level."""
        return [f for f in self.findings if f.impact == impact]
    
    def get_findings_by_check(self, check: str) -> List[SlitherFinding]:
        """Get findings filtered by detector check name."""
        return [f for f in self.findings if f.check == check]
    
    def group_findings_by_check(self) -> Dict[str, List[SlitherFinding]]:
        """Group findings by detector check name."""
        groups: Dict[str, List[SlitherFinding]] = {}
        for finding in self.findings:
            if finding.check not in groups:
                groups[finding.check] = []
            groups[finding.check].append(finding)
        return groups
    
    def extract_code_snippet(
        self,
        finding: SlitherFinding,
        context_lines: int = 5
    ) -> Tuple[str, str, int, int]:
        """
        Extract the relevant code snippet for a finding.
        
        Args:
            finding: The SlitherFinding to extract code for.
            context_lines: Number of context lines before/after.
        
        Returns:
            Tuple of (code_snippet, file_path, start_line, end_line)
        """
        location = finding.get_primary_location()
        if not location:
            return "", "", 0, 0
        
        filename = location.get("filename_relative") or location.get("filename_absolute", "")
        start_line = location.get("lines", [0])[0] if location.get("lines") else 0
        end_line = location.get("lines", [0])[-1] if location.get("lines") else start_line
        
        # Read the source file - try multiple path resolution strategies
        file_path = self._resolve_source_path(filename)
        
        if not file_path:
            logger.warning("Source file not found: %s", filename)
            return "", filename, start_line, end_line
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except (IOError, UnicodeDecodeError) as e:
            logger.warning("Failed to read source file %s: %s", filename, e)
            return "", filename, start_line, end_line
        
        # Extract snippet with context
        snippet_start = max(0, start_line - context_lines - 1)
        snippet_end = min(len(lines), end_line + context_lines)
        
        snippet_lines = []
        for i in range(snippet_start, snippet_end):
            line_num = i + 1
            line_content = lines[i].rstrip()
            snippet_lines.append(f"{line_num}: {line_content}")
        
        return "\n".join(snippet_lines), filename, start_line, end_line
    
    def extract_function_code(
        self,
        finding: SlitherFinding
    ) -> List[Tuple[str, str, str]]:
        """
        Extract full function code for all functions in a finding.
        
        Args:
            finding: The SlitherFinding to extract functions for.
        
        Returns:
            List of tuples: (function_name, file_path, code)
        """
        results = []
        
        for elem in finding.elements:
            if elem.get("type") != "function":
                continue
            
            source_mapping = elem.get("source_mapping", {})
            filename = (
                source_mapping.get("filename_relative") or 
                source_mapping.get("filename_absolute", "")
            )
            lines = source_mapping.get("lines", [])
            
            if not lines or not filename:
                continue
            
            func_name = elem.get("name", "unknown")
            contract = elem.get("type_specific_fields", {}).get("parent", {}).get("name", "")
            full_name = f"{contract}.{func_name}" if contract else func_name
            
            # Read source file - try multiple path resolution strategies
            file_path = self._resolve_source_path(filename)
            
            if not file_path:
                continue
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    all_lines = f.readlines()
                
                start = min(lines) - 1
                end = max(lines)
                
                code_lines = []
                for i in range(start, min(end, len(all_lines))):
                    line_num = i + 1
                    line_content = all_lines[i].rstrip()
                    code_lines.append(f"{line_num}: {line_content}")
                
                results.append((full_name, filename, "\n".join(code_lines)))
                
            except (IOError, UnicodeDecodeError) as e:
                logger.warning("Failed to read function code from %s: %s", filename, e)
        
        return results
    
    def get_compilation_info(self) -> Dict[str, Any]:
        """Get compilation information from results."""
        return self.raw_results.get("results", {}).get("compilation_unit", {})
    
    def get_contract_names(self) -> List[str]:
        """Get list of analyzed contract names."""
        contracts = set()
        for finding in self.findings:
            contracts.update(finding.get_affected_contracts())
        return sorted(contracts)


# Mapping of Slither detector names to human-readable descriptions
DETECTOR_DESCRIPTIONS = {
    "reentrancy-eth": "Reentrancy vulnerability that could lead to ETH theft",
    "reentrancy-no-eth": "Reentrancy vulnerability (no ETH involved)",
    "reentrancy-benign": "Reentrancy that appears benign but should be reviewed",
    "reentrancy-events": "Reentrancy that could cause event ordering issues",
    "unchecked-transfer": "Unchecked return value of transfer/transferFrom",
    "unchecked-lowlevel": "Unchecked low-level call return value",
    "unchecked-send": "Unchecked return value of send",
    "arbitrary-send-eth": "Arbitrary ETH send to user-controlled address",
    "arbitrary-send-erc20": "Arbitrary ERC20 transfer to user-controlled address",
    "suicidal": "Contract can be destroyed by anyone",
    "unprotected-upgrade": "Unprotected upgradeable contract",
    "delegatecall-loop": "Delegatecall inside a loop",
    "controlled-delegatecall": "Controlled delegatecall destination",
    "msg-value-loop": "msg.value used inside a loop",
    "tx-origin": "Dangerous usage of tx.origin for authentication",
    "uninitialized-state": "Uninitialized state variable",
    "uninitialized-storage": "Uninitialized storage pointer",
    "uninitialized-local": "Uninitialized local variable",
    "locked-ether": "Contract locks Ether without withdrawal mechanism",
    "shadowing-state": "State variable shadows another state variable",
    "weak-prng": "Weak pseudo-random number generator",
    "divide-before-multiply": "Division before multiplication causing precision loss",
    "incorrect-equality": "Dangerous strict equality comparison",
    "write-after-write": "Unused write to state variable",
}


def get_detector_description(check: str) -> str:
    """Get human-readable description for a detector."""
    return DETECTOR_DESCRIPTIONS.get(check, f"Detected by: {check}")
