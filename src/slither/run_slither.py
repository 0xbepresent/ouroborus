#!/usr/bin/env python3
"""
Run Slither static analysis on Solidity smart contracts.

This module handles:
- Cloning GitHub repositories
- Running Slither with configurable detectors
- Outputting results in JSON format for further processing

Example CLI usage:
    python run_slither.py owner/repo
"""

import subprocess
import json
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger
from src.utils.exceptions import OuroborusError

logger = get_logger(__name__)


class SlitherError(OuroborusError):
    """Raised when Slither execution fails."""
    pass


class SlitherConfigError(OuroborusError):
    """Raised when Slither configuration is invalid."""
    pass


@dataclass
class SlitherConfig:
    """Configuration for Slither analysis."""
    
    # Detectors to run (empty = all detectors)
    detectors: List[str] = field(default_factory=list)
    
    # Detectors to exclude
    exclude_detectors: List[str] = field(default_factory=list)
    
    # Filter paths (exclude results from these paths)
    filter_paths: List[str] = field(default_factory=lambda: ["node_modules", "lib", "test"])
    
    # Include paths regex pattern (e.g., "(src/|contracts/)" to include only these paths)
    # Uses regex with | as OR operator
    include_paths: Optional[str] = None
    
    # Contracts directory to analyze (e.g., "src", "contracts")
    # If None, auto-detects based on project structure
    contracts_dir: Optional[str] = None
    
    # Minimum severity: informational, low, medium, high
    min_severity: str = "low"
    
    # Timeout in seconds
    timeout: int = 300
    
    # Solc version to use (None = auto-detect)
    solc_version: Optional[str] = None
    
    # Whether to install dependencies before analysis (npm install, forge install)
    install_deps: bool = True


def check_slither_installed() -> bool:
    """
    Check if Slither is installed and accessible.
    
    Returns:
        bool: True if Slither is installed, False otherwise.
    """
    try:
        result = subprocess.run(
            ["slither", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def clone_repo(repo_name: str, target_dir: str, force: bool = False) -> str:
    """
    Clone a GitHub repository.
    
    Args:
        repo_name: Repository in 'owner/repo' format.
        target_dir: Directory to clone into.
        force: If True, remove existing directory and re-clone.
    
    Returns:
        str: Path to the cloned repository.
    
    Raises:
        SlitherError: If cloning fails.
    """
    repo_url = f"https://github.com/{repo_name}.git"
    repo_path = Path(target_dir) / repo_name.replace("/", "_")
    
    if repo_path.exists():
        if force:
            logger.info("Removing existing repository: %s", repo_path)
            shutil.rmtree(repo_path)
        else:
            logger.info("Repository already exists: %s", repo_path)
            return str(repo_path)
    
    logger.info("Cloning repository: %s", repo_name)
    
    try:
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120
        )
        return str(repo_path)
    except subprocess.CalledProcessError as e:
        raise SlitherError(f"Failed to clone repository {repo_name}: {e.stderr}") from e
    except subprocess.TimeoutExpired as e:
        raise SlitherError(f"Timeout while cloning repository {repo_name}") from e


def find_solidity_root(repo_path: str) -> str:
    """
    Find the root directory containing Solidity contracts.
    
    Looks for common patterns like:
    - contracts/
    - src/
    - Root directory with .sol files
    
    Args:
        repo_path: Path to the repository.
    
    Returns:
        str: Path to the Solidity root directory.
    """
    repo = Path(repo_path)
    
    # Common contract directories
    for subdir in ["contracts", "src", "."]:
        check_path = repo / subdir
        if check_path.exists():
            sol_files = list(check_path.rglob("*.sol"))
            if sol_files:
                return str(check_path)
    
    return repo_path


def _find_solidity_subdir(repo_path: str) -> Optional[str]:
    """
    Find the subdirectory name containing Solidity contracts.
    
    Args:
        repo_path: Path to the repository.
    
    Returns:
        Optional[str]: Subdirectory name (e.g., "contracts", "src") or None if .sol files are in root.
    """
    repo = Path(repo_path)
    
    # Check common contract directories
    for subdir in ["contracts", "src"]:
        check_path = repo / subdir
        if check_path.exists():
            sol_files = list(check_path.rglob("*.sol"))
            if sol_files:
                return subdir
    
    # No subdirectory found - .sol files might be in root
    return None


def _install_dependencies(repo_path: str) -> None:
    """
    Install project dependencies and build before running Slither.
    
    Runs:
    - `npm install` if package.json exists (for node_modules dependencies)
    - `forge install` if foundry.toml exists (for lib/ submodules)
    - `forge build --build-info` if foundry.toml exists (creates out/build-info for Slither)
    
    Args:
        repo_path: Path to the repository.
    """
    repo = Path(repo_path)
    
    # Check for package.json (npm dependencies)
    package_json = repo / "package.json"
    if package_json.exists():
        logger.info("Installing npm dependencies...")
        try:
            result = subprocess.run(
                ["npm", "install"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                logger.info("npm install completed")
            else:
                logger.warning("npm install had issues: %s", result.stderr[:500] if result.stderr else "")
        except FileNotFoundError:
            logger.warning("npm not found, skipping npm install")
        except subprocess.TimeoutExpired:
            logger.warning("npm install timed out")
    
    # Check for foundry.toml (forge dependencies and build)
    foundry_toml = repo / "foundry.toml"
    if foundry_toml.exists():
        # First, install forge dependencies
        logger.info("Installing forge dependencies...")
        try:
            result = subprocess.run(
                ["forge", "install", "--no-commit"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                logger.info("forge install completed")
            else:
                logger.warning("forge install had issues (may be ok): %s", result.stderr[:300] if result.stderr else "")
        except FileNotFoundError:
            logger.warning("forge not found, skipping forge install")
            return
        except subprocess.TimeoutExpired:
            logger.warning("forge install timed out")
        
        # Then, build with --build-info (required for Slither)
        logger.info("Building with forge (creates build-info for Slither)...")
        try:
            result = subprocess.run(
                ["forge", "build", "--build-info", "--skip", "test", "--skip", "script"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=600  # Build can take longer
            )
            if result.returncode == 0:
                logger.info("forge build completed")
            else:
                logger.warning("forge build failed: %s", result.stderr[:1000] if result.stderr else "")
                logger.warning("Slither may fail if build-info is missing")
        except FileNotFoundError:
            logger.warning("forge not found, skipping forge build")
        except subprocess.TimeoutExpired:
            logger.warning("forge build timed out")


def _fix_foundry_config(repo_path: str) -> None:
    """
    Fix Foundry configuration issues that may cause Slither to fail.
    
    Currently handles:
    - Unsupported EVM versions (e.g., "osaka" -> "cancun")
    
    Args:
        repo_path: Path to the repository.
    """
    foundry_toml = Path(repo_path) / "foundry.toml"
    
    if not foundry_toml.exists():
        return
    
    try:
        content = foundry_toml.read_text(encoding="utf-8")
        original_content = content
        
        # Fix unsupported EVM versions
        # osaka is a future EVM version not supported by older Foundry/solc
        unsupported_versions = ["osaka", "prague"]
        for version in unsupported_versions:
            if f'evm_version = "{version}"' in content:
                logger.warning(
                    "Fixing foundry.toml: replacing unsupported evm_version '%s' with 'cancun'",
                    version
                )
                content = content.replace(
                    f'evm_version = "{version}"',
                    'evm_version = "cancun"'
                )
        
        # Only write if changed
        if content != original_content:
            foundry_toml.write_text(content, encoding="utf-8")
            
    except (IOError, OSError) as e:
        logger.warning("Could not fix foundry.toml: %s", e)


def run_slither_analysis(
    repo_path: str,
    output_dir: str,
    config: Optional[SlitherConfig] = None
) -> Dict[str, Any]:
    """
    Run Slither analysis on a Solidity project.
    
    Args:
        repo_path: Path to the repository containing Solidity contracts.
        output_dir: Directory to store analysis results.
        config: Slither configuration options.
    
    Returns:
        Dict[str, Any]: Parsed Slither JSON output.
    
    Raises:
        SlitherConfigError: If Slither is not installed.
        SlitherError: If analysis fails.
    """
    if not check_slither_installed():
        raise SlitherConfigError(
            "Slither is not installed. Install with: pip install slither-analyzer"
        )
    
    config = config or SlitherConfig()
    
    # Use the repository root - Slither will auto-detect the framework (Foundry/Hardhat)
    repo_abs = str(Path(repo_path).resolve())
    
    # Fix any Foundry configuration issues FIRST (e.g., unsupported EVM versions)
    # This must happen before forge build
    _fix_foundry_config(repo_abs)
    
    # Install dependencies if enabled (npm install, forge install, forge build)
    if config.install_deps:
        _install_dependencies(repo_abs)
    
    # Determine the target path for Slither
    if config.contracts_dir:
        # User specified a contracts directory (e.g., "src" or "contracts")
        contracts_path = Path(repo_abs) / config.contracts_dir
        if contracts_path.exists():
            sol_root = str(contracts_path)
            logger.info("Using specified contracts directory: %s", config.contracts_dir)
        else:
            logger.warning(
                "Specified contracts directory '%s' not found, using repo root",
                config.contracts_dir
            )
            sol_root = repo_abs
    else:
        # Auto-detect or use repo root
        sol_root = repo_abs
    
    logger.info("Running Slither on: %s", sol_root)
    
    # Prepare output paths
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    repo_name = Path(repo_path).name
    json_output = output_path / f"{repo_name}_slither.json"
    
    # Build Slither command
    cmd = [
        "slither",
        sol_root,
        "--json", str(json_output),
    ]
    
    # Add detectors
    if config.detectors:
        cmd.extend(["--detect", ",".join(config.detectors)])
    
    # Exclude detectors
    if config.exclude_detectors:
        cmd.extend(["--exclude", ",".join(config.exclude_detectors)])
    
    # Include paths and filter paths are mutually exclusive in Slither
    # Prefer include_paths (whitelist) over filter_paths (blacklist) when specified
    if config.include_paths:
        # Use include paths (regex pattern with | for OR, e.g., "(src/|contracts/)")
        cmd.extend(["--include-paths", config.include_paths])
    elif config.filter_paths:
        # Fallback to filter paths (exclude) if no include paths specified
        for path in config.filter_paths:
            cmd.extend(["--filter-paths", path])
    
    # Solc version
    if config.solc_version:
        cmd.extend(["--solc-solcs-select", config.solc_version])
    
    logger.info("Running command: %s", " ".join(cmd))
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout,
            # cwd=repo_path
        )
        
        # Slither returns non-zero if it finds issues, but still produces valid JSON
        # Only fail if JSON wasn't produced
        if not json_output.exists():
            raise SlitherError(
                f"Slither failed to produce output. stderr: {result.stderr}"
            )
        
        # Parse JSON output
        with open(json_output, "r", encoding="utf-8") as f:
            slither_results = json.load(f)
        
        # Log summary
        detectors = slither_results.get("results", {}).get("detectors", [])
        logger.info("Slither found %d potential issues", len(detectors))
        
        return slither_results
        
    except subprocess.TimeoutExpired as e:
        raise SlitherError(f"Slither analysis timed out after {config.timeout}s") from e
    except json.JSONDecodeError as e:
        raise SlitherError(f"Failed to parse Slither JSON output: {e}") from e


def get_slither_detectors() -> List[Dict[str, str]]:
    """
    Get list of available Slither detectors with descriptions.
    
    Returns:
        List[Dict[str, str]]: List of detector info dictionaries.
    """
    try:
        result = subprocess.run(
            ["slither", "--list-detectors-json"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        pass
    
    return []


# High-impact security detectors for focused analysis
SECURITY_DETECTORS = [
    "reentrancy-eth",
    "reentrancy-no-eth", 
    "reentrancy-benign",
    "reentrancy-events",
    "unchecked-transfer",
    "unchecked-lowlevel",
    "unchecked-send",
    "arbitrary-send-eth",
    "arbitrary-send-erc20",
    "suicidal",
    "unprotected-upgrade",
    "delegatecall-loop",
    "controlled-delegatecall",
    "msg-value-loop",
    "tx-origin",
    "uninitialized-state",
    "uninitialized-storage",
    "uninitialized-local",
    "locked-ether",
    "shadowing-state",
    "weak-prng",
    "divide-before-multiply",
    "incorrect-equality",
    "write-after-write",
]


def main_cli() -> None:
    """CLI entry point for running Slither analysis."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python run_slither.py <owner/repo>")
        sys.exit(1)
    
    repo_name = sys.argv[1]
    
    # Clone and analyze
    repo_path = clone_repo(repo_name, "output/repos/solidity")
    
    config = SlitherConfig(
        detectors=SECURITY_DETECTORS,
        filter_paths=["node_modules", "lib", "test", "tests", "mocks", "mock"]
    )
    
    results = run_slither_analysis(
        repo_path,
        "output/slither_results",
        config
    )
    
    print(f"Analysis complete. Found {len(results.get('results', {}).get('detectors', []))} issues.")


if __name__ == "__main__":
    main_cli()
