"""
Slither integration module for Solidity smart contract analysis.

This module provides functionality to:
- Clone GitHub repositories containing Solidity contracts
- Run Slither static analysis
- Parse and process Slither JSON output
- Extract code snippets for LLM analysis
"""

from src.slither.run_slither import (
    run_slither_analysis,
    clone_repo,
    SlitherConfig
)
from src.slither.slither_parser import SlitherResultParser

__all__ = [
    "run_slither_analysis",
    "clone_repo", 
    "SlitherConfig",
    "SlitherResultParser"
]
