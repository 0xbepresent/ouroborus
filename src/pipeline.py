#!/usr/bin/env python3
"""
Pipeline orchestration for Ouroborus.
This module coordinates the complete analysis pipeline:

For C/C++ (using CodeQL):
1. Fetch CodeQL databases
2. Run CodeQL queries
3. Classify results with LLM
4. Open UI (optional)

For Solidity (using Slither):
1. Clone GitHub repository
2. Run Slither analysis
3. Classify results with LLM
4. Open UI (optional)
"""
# Ignore pydantic warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.codeql.fetch_repos import fetch_codeql_dbs
from src.codeql.run_codeql_queries import compile_and_run_codeql_queries
from src.utils.config import get_codeql_path
from src.utils.config_validator import validate_and_exit_on_error
from src.utils.logger import setup_logging, get_logger
from src.utils.exceptions import (
    CodeQLError, CodeQLConfigError, CodeQLExecutionError,
    LLMError, LLMConfigError, LLMApiError,
    OuroborusError
)
from src.ouroborus import IssueAnalyzer
from src.ui.ui_app import main as ui_main

# Solidity/Slither imports (lazy loaded to avoid import errors if not installed)
SOLIDITY_AVAILABLE = False
try:
    from src.slither.run_slither import (
        clone_repo, run_slither_analysis, SlitherConfig, 
        SECURITY_DETECTORS, SlitherError, SlitherConfigError
    )
    from src.ouroborus_solidity import SolidityIssueAnalyzer
    SOLIDITY_AVAILABLE = True
except ImportError:
    pass

# Initialize logging
setup_logging()
logger = get_logger(__name__)


def _log_exception_cause(e: Exception) -> None:
    """
    Log the cause of an exception if available and not already included in the exception message.
    Checks both e.cause (if set via constructor) and e.__cause__ (if set via 'from e').
    """
    cause = getattr(e, 'cause', None) or getattr(e, '__cause__', None)
    if cause:
        # Only log cause if it's not already included in the exception message
        cause_str = str(cause)
        error_str = str(e)
        if cause_str not in error_str:
            logger.error("   Cause: %s", cause)


def step1_fetch_codeql_dbs(lang: str, threads: int, repo: str, force: bool = False) -> str:
    """
    Step 1: Fetch CodeQL databases from GitHub.
    
    Args:
        lang: Programming language code.
        threads: Number of threads for download operations.
        repo: Repository name (e.g., "redis/redis").
    
    Returns:
        str: Path to the directory containing downloaded databases.
    
    Raises:
        CodeQLConfigError: If configuration is invalid (e.g., missing GitHub token).
        CodeQLError: If database download or extraction fails.
    """
    logger.info("\nStep 1: Fetching CodeQL Databases")
    logger.info("-" * 60)
    logger.info("Fetching database for: %s", repo)
    
    try:
        dbs_dir = fetch_codeql_dbs(lang=lang, threads=threads, repo_name=repo, force=force)
        if not dbs_dir:
            raise CodeQLError(f"No CodeQL databases were downloaded/found for {repo}")
        return dbs_dir
    except CodeQLConfigError as e:
        logger.error("[-] Step 1: Configuration error while fetching CodeQL databases: %s", e)
        _log_exception_cause(e)
        logger.error("   Please check your GitHub token and permissions.")
        sys.exit(1)
    except CodeQLError as e:
        logger.error("[-] Step 1: Failed to fetch CodeQL databases: %s", e)
        _log_exception_cause(e)
        logger.error("   Please check file permissions, disk space, and GitHub API access.")
        sys.exit(1)


def step2_run_codeql_queries(dbs_dir: str, lang: str, threads: int) -> None:
    """
    Step 2: Run CodeQL queries on the downloaded databases.
    
    Args:
        dbs_dir: Path to the directory containing CodeQL databases.
        lang: Programming language code.
        threads: Number of threads for query execution.
    
    Raises:
        CodeQLConfigError: If CodeQL path configuration is invalid.
        CodeQLExecutionError: If query execution fails.
        CodeQLError: If other CodeQL-related errors occur (e.g., database access issues).
    """
    logger.info("\nStep 2: Running CodeQL Queries")
    logger.info("-" * 60)
    
    try:
        compile_and_run_codeql_queries(
            codeql_bin=get_codeql_path(),
            lang=lang,
            threads=threads,
            timeout=300,
            dbs_dir=dbs_dir
        )
    except CodeQLConfigError as e:
        logger.error("[-] Step 2: Configuration error while running CodeQL queries: %s", e)
        _log_exception_cause(e)
        logger.error("   Please check your CODEQL_PATH configuration.")
        sys.exit(1)
    except CodeQLExecutionError as e:
        logger.error("[-] Step 2: Failed to execute CodeQL queries: %s", e)
        _log_exception_cause(e)
        logger.error("   Please check your CodeQL installation and database files.")
        sys.exit(1)
    except CodeQLError as e:
        logger.error("[-] Step 2: CodeQL error while running queries: %s", e)
        _log_exception_cause(e)
        logger.error("   Please check your CodeQL database files and query syntax.")
        sys.exit(1)
    

def step3_classify_results_with_llm(dbs_dir: str, lang: str) -> None:
    """
    Step 3: Classify CodeQL results using LLM analysis.
    
    Args:
        dbs_dir: Path to the directory containing CodeQL databases.
        lang: Programming language code.
    
    Raises:
        LLMConfigError: If LLM configuration is invalid (e.g., missing API credentials).
        LLMApiError: If LLM API call fails (e.g., network issues, rate limits).
        LLMError: If other LLM-related errors occur.
        CodeQLError: If reading CodeQL database files fails (YAML, ZIP, CSV).
        OuroborusError: If saving analysis results to disk fails.
    """
    logger.info("\nStep 3: Classifying Results with LLM")
    logger.info("-" * 60)
    
    try:
        analyzer = IssueAnalyzer(lang=lang)
        analyzer.run(dbs_dir)
    except LLMConfigError as e:
        logger.error("[-] Step 3: LLM configuration error: %s", e)
        _log_exception_cause(e)
        logger.error("   Please check your LLM configuration and API credentials in .env file.")
        sys.exit(1)
    except LLMApiError as e:
        logger.error("[-] Step 3: LLM API error: %s", e)
        _log_exception_cause(e)
        logger.error("   Please check your API key, network connection, and rate limits.")
        sys.exit(1)
    except LLMError as e:
        logger.error("[-] Step 3: LLM error: %s", e)
        _log_exception_cause(e)
        logger.error("   Please check your LLM provider settings and API status.")
        sys.exit(1)
    except CodeQLError as e:
        logger.error("[-] Step 3: CodeQL error while reading database files: %s", e)
        _log_exception_cause(e)
        logger.error("   This step reads CodeQL database files (YAML, ZIP, CSV) to prepare data for LLM analysis.")
        logger.error("   Please check your CodeQL databases and files are accessible.")
        sys.exit(1)
    except OuroborusError as e:
        logger.error("[-] Step 3: File system error while saving results: %s", e)
        _log_exception_cause(e)
        logger.error("   This step writes analysis results to disk and creates output directories.")
        logger.error("   Please check file permissions and disk space.")
        sys.exit(1)


def step4_open_ui() -> None:
    """
    Step 4: Open the results UI (optional).

    Note:
        This function does not raise exceptions. UI errors are handled internally by the UI module.
    """
    logger.info("\n[4/4] Opening UI")
    logger.info("-" * 60)
    logger.info("[+] Pipeline completed successfully!")
    logger.info("Opening results UI...")
    ui_main()


def main_analyze() -> None:
    """
    CLI entry point for smart contract security analysis.
    
    Expected usage:
        ouroborus --repo-path /path/to/project [options]
        ouroborus <org/repo> [--force] [options]   # clone from GitHub then analyze
    """
    parser = argparse.ArgumentParser(
        prog="ouroborus",
        description="Ouroborus - Smart contract security analysis with LLM classification (Slither + LLM)"
    )
    parser.add_argument(
        "repo",
        nargs="?",
        default=None,
        help="GitHub repository in 'org/repo' format (optional; use --repo-path for local projects)"
    )
    parser.add_argument(
        "--repo-path",
        default=None,
        metavar="PATH",
        help="Path to local repository. Use this for local projects; Slither runs here (or use with --skip-slither + --slither-results)"
    )
    parser.add_argument("--force", "-f", action="store_true", help="Re-download even if repo exists (only when using repo)")
    parser.add_argument(
        "--contracts-dir", "-d",
        default=None,
        help="Contracts directory to analyze (e.g., 'src', 'contracts'). Auto-detects if not specified."
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Skip installing dependencies (npm install, forge install)"
    )
    parser.add_argument(
        "--skip-slither",
        action="store_true",
        help="Skip Slither and use existing results (requires --slither-results and --repo-path)"
    )
    parser.add_argument(
        "--slither-results",
        default=None,
        metavar="PATH",
        help="Path to existing Slither JSON results file (use with --skip-slither)"
    )
    parser.add_argument(
        "--codeql-db",
        default=None,
        metavar="PATH",
        help="Path to CodeQL database for Solidity (enables code lookup tools for LLM)"
    )
    parser.add_argument(
        "--function-tree",
        default=None,
        metavar="PATH",
        help="Path to FunctionTree.csv for function lookups (use with --codeql-db)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose mode to see LLM thinking and tool calls in real-time"
    )
    
    args = parser.parse_args()
    
    # Require either --repo-path or repo (org/repo)
    if not args.repo_path and not args.repo:
        parser.error("Either provide a local path (--repo-path PATH) or a GitHub repo (org/repo)")
    if args.repo_path and args.repo:
        parser.error("Use either --repo-path (local) or org/repo (clone), not both")
    
    if args.skip_slither:
        if not args.slither_results:
            parser.error("--skip-slither requires --slither-results PATH")
        if not args.repo_path:
            parser.error("--skip-slither requires --repo-path PATH (source location for results)")
    
    if args.repo and "/" not in args.repo:
        parser.error("Repository must be in format 'org/repo'")
    
    analyze_solidity_pipeline(
        repo=args.repo,
        force=args.force,
        contracts_dir=args.contracts_dir,
        install_deps=not args.no_install,
        skip_slither=args.skip_slither,
        slither_results_path=args.slither_results,
        repo_path=args.repo_path,
        codeql_db=args.codeql_db,
        function_tree=args.function_tree,
        verbose=args.verbose
    )


# ==============================================================================
# SOLIDITY PIPELINE (using Slither)
# ==============================================================================

def step1_clone_solidity_repo(repo: str, force: bool = False) -> str:
    """
    Step 1 (Solidity): Clone the GitHub repository.
    
    Args:
        repo: Repository name in 'owner/repo' format.
        force: If True, re-clone even if exists.
    
    Returns:
        str: Path to the cloned repository.
    """
    if not SOLIDITY_AVAILABLE:
        logger.error("[-] Solidity analysis is not available. Missing dependencies.")
        logger.error("   Install with: pip install slither-analyzer")
        sys.exit(1)
    
    logger.info("\nStep 1: Cloning Repository")
    logger.info("-" * 60)
    logger.info("Cloning: %s", repo)
    
    try:
        repo_path = clone_repo(repo, "output/repos/solidity", force=force)
        if not repo_path:
            raise SlitherError(f"Failed to clone repository: {repo}")
        return repo_path
    except SlitherError as e:
        logger.error("[-] Step 1: Failed to clone repository: %s", e)
        _log_exception_cause(e)
        sys.exit(1)


def step2_run_slither(
    repo_path: str,
    contracts_dir: Optional[str] = None,
    install_deps: bool = True
) -> dict:
    """
    Step 2 (Solidity): Run Slither static analysis.
    
    Args:
        repo_path: Path to the cloned repository.
        contracts_dir: Specific contracts directory to analyze (e.g., "src", "contracts").
        install_deps: Whether to install dependencies before analysis.
    
    Returns:
        dict: Slither analysis results.
    """
    logger.info("\nStep 2: Running Slither Analysis")
    logger.info("-" * 60)
    
    try:
        config = SlitherConfig(
            detectors=SECURITY_DETECTORS,
            filter_paths=["node_modules", "lib", "test", "tests", "mocks", "mock"],
            contracts_dir=contracts_dir,
            install_deps=install_deps
        )
        results = run_slither_analysis(repo_path, "output/slither_results", config)
        
        num_findings = len(results.get("results", {}).get("detectors", []))
        logger.info("[+] Slither found %d potential issues", num_findings)
        
        return results
    except SlitherConfigError as e:
        logger.error("[-] Step 2: Slither configuration error: %s", e)
        _log_exception_cause(e)
        logger.error("   Please ensure Slither is installed: pip install slither-analyzer")
        sys.exit(1)
    except SlitherError as e:
        logger.error("[-] Step 2: Slither analysis failed: %s", e)
        _log_exception_cause(e)
        sys.exit(1)


def step3_classify_solidity_with_llm(
    slither_results: dict, 
    repo_path: str,
    codeql_db: Optional[str] = None,
    function_tree: Optional[str] = None,
    verbose: bool = False
) -> None:
    """
    Step 3 (Solidity): Classify Slither results using LLM analysis.
    
    Args:
        slither_results: Slither JSON results.
        repo_path: Path to the analyzed repository.
        codeql_db: Optional path to CodeQL database for code lookup tools.
        function_tree: Optional path to FunctionTree.csv for function lookups.
        verbose: If True, print LLM thinking and tool calls in real-time.
    """
    logger.info("\nStep 3: Classifying Results with LLM")
    logger.info("-" * 60)
    
    try:
        analyzer = SolidityIssueAnalyzer()
        analyzer.run(slither_results, repo_path, codeql_db=codeql_db, function_tree=function_tree, verbose=verbose)
    except LLMConfigError as e:
        logger.error("[-] Step 3: LLM configuration error: %s", e)
        _log_exception_cause(e)
        logger.error("   Please check your LLM configuration and API credentials in .env file.")
        sys.exit(1)
    except LLMApiError as e:
        logger.error("[-] Step 3: LLM API error: %s", e)
        _log_exception_cause(e)
        logger.error("   Please check your API key, network connection, and rate limits.")
        sys.exit(1)
    except LLMError as e:
        logger.error("[-] Step 3: LLM error: %s", e)
        _log_exception_cause(e)
        sys.exit(1)
    except OuroborusError as e:
        logger.error("[-] Step 3: Analysis error: %s", e)
        _log_exception_cause(e)
        sys.exit(1)


def analyze_solidity_pipeline(
    repo: Optional[str] = None,
    force: bool = False,
    open_ui: bool = True,
    contracts_dir: Optional[str] = None,
    install_deps: bool = True,
    skip_slither: bool = False,
    slither_results_path: Optional[str] = None,
    repo_path: Optional[str] = None,
    codeql_db: Optional[str] = None,
    function_tree: Optional[str] = None,
    verbose: bool = False
) -> None:
    """
    Run the complete Ouroborus pipeline for Solidity contracts using Slither.
    
    Either repo (org/repo) or repo_path (local path) must be provided. When repo_path
    is used, no clone is performed; Slither runs on the local path (or existing results
    are used with --skip-slither).
    
    Args:
        repo: GitHub repository name (e.g., "OpenZeppelin/openzeppelin-contracts"). Optional if repo_path is set.
        force: If True, re-clone even if repository exists (only when using repo).
        open_ui: Whether to open the UI after completion.
        contracts_dir: Specific contracts directory to analyze (e.g., "src", "contracts").
        install_deps: Whether to install dependencies (npm install, forge install).
        skip_slither: If True, skip Slither and use existing results (requires slither_results_path and repo_path).
        slither_results_path: Path to existing Slither JSON results file.
        repo_path: Path to local repository (use for local projects or with skip_slither).
        codeql_db: Path to CodeQL database for Solidity (enables code lookup tools).
        function_tree: Path to FunctionTree.csv for function lookups.
        verbose: If True, print LLM thinking and tool calls in real-time.
    """
    logger.info("🚀 Starting Ouroborus Solidity Analysis Pipeline")
    logger.info("=" * 60)
    logger.info("Engine: Slither (static analysis for Solidity)")
    if repo_path:
        logger.info("Repository: local path %s", repo_path)
    else:
        logger.info("Repository: %s (clone from GitHub)", repo)
    if skip_slither:
        logger.info("Mode: Skip Slither (using existing results)")
    if codeql_db:
        logger.info("CodeQL database: %s", codeql_db)
        if function_tree:
            logger.info("Function tree: %s (code lookup tools enabled)", function_tree)
        else:
            logger.info("Note: --function-tree not provided, some tools may be limited")
    if contracts_dir:
        logger.info("Contracts directory: %s", contracts_dir)
    logger.info("")
    
    # Validate configuration before starting
    try:
        validate_and_exit_on_error()
    except (LLMConfigError, OuroborusError) as e:
        message = f"""
[-] Configuration Validation Failed
============================================================
{str(e)}
============================================================
Please fix the configuration errors above and try again.
See README.md for configuration reference.
"""
        logger.error(message)
        _log_exception_cause(e)
        sys.exit(1)
    
    if skip_slither:
        # Use existing Slither results and repo path (no clone, no Slither run)
        logger.info("[*] Step 1 & 2: SKIPPED (using existing Slither results)")
        logger.info("    Slither results: %s", slither_results_path)
        logger.info("    Repository path: %s", repo_path)
        
        try:
            with open(slither_results_path, "r", encoding="utf-8") as f:
                slither_results = json.load(f)
        except FileNotFoundError:
            logger.error("[-] Slither results file not found: %s", slither_results_path)
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error("[-] Failed to parse Slither results JSON: %s", e)
            sys.exit(1)
        
        local_repo_path = repo_path
    elif repo_path:
        # Local project: run Slither on repo_path (no clone)
        logger.info("[*] Step 1: SKIPPED (using local --repo-path)")
        logger.info("    Repository path: %s", repo_path)
        slither_results = step2_run_slither(repo_path, contracts_dir, install_deps)
        local_repo_path = repo_path
    else:
        # Clone from GitHub then run Slither
        assert repo is not None, "repo required when not using --repo-path or --skip-slither"
        local_repo_path = step1_clone_solidity_repo(repo, force)
        slither_results = step2_run_slither(local_repo_path, contracts_dir, install_deps)
    
    # Step 3: Classify with LLM
    step3_classify_solidity_with_llm(slither_results, local_repo_path, codeql_db, function_tree, verbose)
    
    # Step 4: Open UI (optional)
    if open_ui:
        step4_open_ui()


# ==============================================================================
# C/C++ PIPELINE (using CodeQL)
# ==============================================================================

def analyze_pipeline(repo: Optional[str] = None, lang: str = "c", threads: int = 16, open_ui: bool = True, force: bool = False) -> None:
    """
    Run the complete Ouroborus pipeline: fetch, analyze, classify, and optionally open UI.
    
    Args:
        repo: GitHub repository name (e.g., "redis/redis"). Required for fetching databases.
        lang: Programming language code. Defaults to "c".
        threads: Number of threads for CodeQL operations. Defaults to 16.
        open_ui: Whether to open the UI after completion. Defaults to True.
        force: If True, re-download even if database exists. Defaults to False.
    
    Note:
        This function catches and handles all exceptions internally, logging errors
        and exiting with code 1 on failure. It does not raise exceptions.
    """
    logger.info("🚀 Starting Ouroborus Analysis Pipeline")
    logger.info("=" * 60)
    
    # Validate configuration before starting
    try:
        validate_and_exit_on_error()
    except (CodeQLConfigError, LLMConfigError, OuroborusError) as e:
        # Format error message for display
        message = f"""
[-] Configuration Validation Failed
============================================================
{str(e)}
============================================================
Please fix the configuration errors above and try again.
See README.md for configuration reference.
"""
        logger.error(message)
        _log_exception_cause(e)
        sys.exit(1)
    
    # Step 1: Fetch CodeQL databases
    dbs_dir = step1_fetch_codeql_dbs(lang, threads, repo, force)
    
    # Step 2: Run CodeQL queries
    step2_run_codeql_queries(dbs_dir, lang, threads)
    
    # Step 3: Classify results with LLM
    step3_classify_results_with_llm(dbs_dir, lang)
    
    # Step 4: Open UI (optional)
    if open_ui:
        step4_open_ui()


def main_ui() -> None:
    """
    CLI entry point to open the UI without running analysis.
    
    Expected usage: ouroborus-ui
    """
    logger.info("Opening Ouroborus UI...")
    ui_main()


def main_validate() -> None:
    """
    CLI entry point to validate configuration.
    
    Expected usage: ouroborus-validate
    """
    from src.utils.config_validator import validate_all_config
    
    is_valid, errors = validate_all_config()
    
    if is_valid:
        logger.info("[+] All configurations are valid!")
    else:
        for error in errors:
            logger.error(error)
        sys.exit(1)


def main_list() -> None:
    """
    CLI entry point to list analyzed repositories.
    
    Expected usage: ouroborus-list
    """
    from src.ui.results_loader import ResultsLoader
    
    results_dir = Path("output/results")
    if not results_dir.exists():
        logger.info("No results found. Run 'ouroborus <org/repo>' first.")
        return
    
    loader = ResultsLoader()
    
    # Currently only 'c' language is supported
    lang = "c"
    issues, _ = loader.load_all_issues(lang)
    
    if not issues:
        logger.info("No analyzed repositories found.")
        return
    
    # Group issues by repo
    repos = {}
    for issue in issues:
        repo = issue.repo
        if repo not in repos:
            repos[repo] = {"true": 0, "false": 0, "needs_more_data to decide": 0}
        repos[repo][issue.status] += 1
    
    logger.info("Analyzed repositories:")
    logger.info("-" * 50)
    for repo, counts in sorted(repos.items()):
        total = counts["true"] + counts["false"] + counts["needs_more_data to decide"]
        logger.info(
            "  %-30s %3d issues (%d True positive, %d False positive, %d Needs more data to decide)",
            repo, total, counts["true"], counts["false"], counts["needs_more_data to decide"]
        )


def main_example() -> None:
    """
    CLI entry point to run the example pipeline.
    
    Expected usage: ouroborus-example
    """
    from examples.example import main as example_main
    example_main()


if __name__ == '__main__':
    main_analyze()