# Ouroborus

**Smart contract security analysis with LLM classification.**

Ouroborus runs [Slither](https://github.com/crytic/slither) on Solidity repositories, then uses an LLM (OpenAI, Azure, Gemini, OpenRouter, etc.) to classify findings as true positives, false positives, or needing more data. Results can be browsed in the built-in UI.

---

## Install the Ouroborus CLI

### Prerequisites

- **Python 3.10 – 3.13** (3.11 or 3.12 recommended)
- **Poetry** (recommended) or pip
- **LLM API key** (OpenAI, Azure, Gemini, OpenRouter, or other provider — see `.env.example`)
- **Slither** (optional; install for full pipeline: `pip install slither-analyzer solc-select`)

### Install with Poetry (recommended)

```bash
git clone <your-ouroborus-repo-url>
cd Ouroborus
poetry install
```

Then run the CLI with:

```bash
poetry run ouroborus --repo-path /path/to/your/contracts   # local project
# or
poetry run ouroborus OpenZeppelin/openzeppelin-contracts     # clone from GitHub
poetry run ouroborus-ui
```

### Install with pip

From the project root:

```bash
pip install -e .
```

Then run:

```bash
ouroborus --repo-path /path/to/your/contracts
# or
ouroborus <org/repo>
ouroborus-ui
```

### One-time setup

1. Copy `.env.example` to `.env` and set your LLM provider and API key (see `.env.example` for variables).
2. (Optional) Run setup to validate config and install any optional dependencies:

   ```bash
   poetry run ouroborus-setup
   ```

---

## Usage

You can analyze **local projects** with `--repo-path` (no clone), or **GitHub repos** by passing `org/repo` (clone then analyze).

| Command | Description |
|--------|-------------|
| `ouroborus --repo-path PATH` | Analyze a local Solidity project; Slither runs at PATH |
| `ouroborus --repo-path PATH --skip-slither --slither-results FILE` | Use existing Slither JSON + local path (e.g. with your own CodeQL DB via `--codeql-db`) |
| `ouroborus --repo-path PATH --codeql-db PATH --function-tree PATH` | Local project + CodeQL DB for LLM code lookup |
| `ouroborus <org/repo>` | Clone from GitHub, run Slither, classify with LLM, then open UI |
| `ouroborus <org/repo> --force` | Re-clone and re-analyze even if repo/results exist |
| `ouroborus ... --contracts-dir src` | Contracts directory (e.g. `src`, `contracts`) |
| `ouroborus ... --no-install` | Skip `npm install` / `forge install` |
| `ouroborus ... --verbose` | Show LLM thinking and tool calls |
| `ouroborus-ui` | Open the results UI without running analysis |
| `ouroborus-validate` | Validate configuration (.env, LLM, etc.) |
| `ouroborus-list` | List analyzed repositories |

### CodeQL database and function tree (deterministic code lookup)

You can pass a **CodeQL database** and a **function tree** (e.g. a CSV of functions and locations) so the LLM looks up code through **deterministic tools** instead of inventing it. The agent calls tools such as `list_contract_functions` and `get_function_code` that query the CodeQL DB and function tree; the model reasons over real code retrieved this way and does not hallucinate source.

- `--codeql-db PATH` — path to a CodeQL database for the Solidity project (built with `codeql database create` or your usual workflow).
- `--function-tree PATH` — path to a function tree file (e.g. `FunctionTree.csv` or similar) used for resolving function names to code.

**CodeQL for Solidity (CyScout)**  
To build CodeQL databases and run queries (including the function tree) on Solidity, you can use [CyScout](https://github.com/0xbepresent/CyScout): a CodeQL extension for the Solidity language. Clone the repo and use its extractor and QL libraries as below.

**1. Create a CodeQL-Solidity database**

Point the CodeQL CLI at CyScout’s extractor and your project source (e.g. a cloned repo or `output/repos/solidity/<project>`):

```bash
codeql database create \
  --search-path <CyScout_path>/solidity/codeql/codeql/solidity/extractor-pack \
  -l solidity \
  ./<project>-solidity-db \
  -s /path/to/your/solidity-repo
```

**2. Run the FunctionTree query and export CSV**

Run the `FunctionTree.ql` query from CyScout and decode the result to CSV so Ouroborus can use it as `--function-tree`:

```bash
codeql query run <CyScout_path>/solidity/codeql/solidity/ql/lib/FunctionTree.ql \
  -d ./<project>-solidity-db \
  --output=FunctionTree.bqrs

codeql bqrs decode FunctionTree.bqrs --format=csv --output=FunctionTree.csv
```

**2b. (Optional) Run ModifierTree and build a combined code tree CSV**

To include modifiers in the same tree file, run `ModifierTree.ql`, decode to CSV, then append its rows (without the header) to a combined file. Use the combined CSV as `--function-tree` so the LLM can look up both functions and modifiers:

```bash
codeql query run <CyScout_path>/solidity/codeql/solidity/ql/lib/ModifierTree.ql \
  -d ./<project>-solidity-db \
  --output=Modifiers.bqrs

codeql bqrs decode Modifiers.bqrs --format=csv --output=Modifiers.csv

# Use FunctionTree.csv as the base, then append modifier rows (skip header with tail -n +2)
cp FunctionTree.csv CodeTree.csv
tail -n +2 Modifiers.csv >> CodeTree.csv
```

Then pass `--function-tree ./CodeTree.csv` when running Ouroborus (step 4).

**3. (Optional) Run other CodeQL-Solidity detectors**

CyScout includes detectors such as `slither-bad-prng.ql`, `slither-msg-value-in-loop.ql`, etc. Example:

```bash
codeql query run <CyScout_path>/solidity/codeql/solidity/ql/lib/slither-bad-prng.ql -d ./<project>-solidity-db
codeql query run <CyScout_path>/solidity/codeql/solidity/ql/lib/slither-msg-value-in-loop.ql -d ./<project>-solidity-db
```

**4. Run Ouroborus with the CodeQL DB and function tree**

Use the database directory and the generated function tree CSV with `--codeql-db` and `--function-tree`. For `--function-tree`, use `FunctionTree.csv` (functions only) or `CodeTree.csv` if you built the combined file in step 2b (functions + modifiers):

```bash
ouroborus --skip-slither \
  --slither-results output/slither_results/<project>_slither.json \
  --repo-path output/repos/solidity/<project> \
  --codeql-db ./<project>-solidity-db \
  --function-tree ./FunctionTree.csv \
  --verbose
```

Use this when you already have Slither results and a CodeQL DB (e.g. from CI or a separate build). Example: skip Slither, use existing results, and enable code lookup:

```bash
ouroborus --skip-slither \
  --slither-results output/slither_results/<project>_slither.json \
  --repo-path output/repos/solidity/<project> \
  --codeql-db /path/to/your/solidity-db \
  --function-tree output/databases/FunctionTree.csv \
  --verbose
```

**Examples (local projects):**

```bash
poetry run ouroborus --repo-path /path/to/my-contracts
poetry run ouroborus --repo-path ./my-contracts --contracts-dir src
poetry run ouroborus --repo-path . --skip-slither --slither-results out/slither.json --codeql-db ./codeql-db
```

**Examples (clone from GitHub):**

```bash
poetry run ouroborus OpenZeppelin/openzeppelin-contracts
poetry run ouroborus OpenZeppelin/openzeppelin-contracts --contracts-dir contracts
poetry run ouroborus Uniswap/v3-core --force
poetry run ouroborus-ui
```

---

## Example output

Running the CLI (e.g. clone from GitHub):

```text
(.venv) ➜  Ouroborus git:(main) ✗ ouroborus OpenZeppelin/openzeppelin-contracts
🚀 Starting Ouroborus Solidity Analysis Pipeline
============================================================
Engine: Slither (static analysis for Solidity)
Repository: OpenZeppelin/openzeppelin-contracts (clone from GitHub)


Step 1: Cloning Repository
------------------------------------------------------------
Cloning: OpenZeppelin/openzeppelin-contracts
Repository already exists: output/repos/solidity/OpenZeppelin_openzeppelin-contracts

Step 2: Running Slither Analysis
------------------------------------------------------------
Installing npm dependencies...
npm install completed
Installing forge dependencies...
WARNING - forge install had issues (may be ok): error: unexpected argument '--no-commit' found
  tip: a similar argument exists: '--commit'
Usage: forge install [OPTIONS] [DEPENDENCIES]...
Building with forge (creates build-info for Slither)...
forge build completed
Running Slither on: .../output/repos/solidity/OpenZeppelin_openzeppelin-contracts
Running command: slither ... --json output/slither_results/OpenZeppelin_openzeppelin-contracts_slither.json --detect reentrancy-eth,...
Slither found 41 potential issues
[+] Slither found 41 potential issues

Step 3: Classifying Results with LLM
------------------------------------------------------------
Parsed 41 findings from Slither results
Total findings to analyze: 41

Found 4 issues of type arbitrary-send-eth

Issue ID: 1, LLM decision: → False Positive
Issue ID: 2, LLM decision: → False Positive
...
```

Results are written under `output/results/solidity/<project_name>/slither-<detector>/` as `*_raw.json` (input to the LLM) and `*_final.json` (LLM response). Example structures:

**`<id>_raw.json`** (input sent to the LLM):

```json
{
  "check": "arbitrary-send-eth",
  "impact": "high",
  "confidence": "medium",
  "description": "Contract can send ETH to an arbitrary destination...",
  "affected_functions": ["MyContract.withdraw"],
  "affected_contracts": ["MyContract"],
  "prompt": "... full prompt with template, code snippet, hints ..."
}
```

**`<id>_final.json`** (LLM messages; decision is inferred from content, e.g. `1337` = true positive, `1007` = false positive):

```json
[
  {"role": "assistant", "content": "Analyzing the finding... [1337] True positive."}
]
```

---

## Example output with CodeQL + function tree (deterministic lookup)

When you pass `--codeql-db` and `--function-tree`, the agent fetches code via tools (e.g. `list_contract_functions`, `get_function_code`) so the LLM reasons over **real code** from the database instead of hallucinating. The following example uses placeholder paths and names; the important part is the **tool-call flow**: the model asks for contract functions and function code, and the responses come from the CodeQL DB/function tree.

```text
$ ouroborus --skip-slither \
  --slither-results output/slither_results/<project>_slither.json \
  --repo-path output/repos/solidity/<project> \
  --codeql-db /path/to/your/solidity-db \
  --function-tree output/databases/FunctionTree.csv \
  --verbose

🚀 Starting Ouroborus Solidity Analysis Pipeline
============================================================
Engine: Slither (static analysis for Solidity)
Repository: local path output/repos/solidity/<project>
Mode: Skip Slither (using existing results)
CodeQL database: /path/to/your/solidity-db
Function tree: output/databases/FunctionTree.csv (code lookup tools enabled)

[*] Step 1 & 2: SKIPPED (using existing Slither results)
    Slither results: output/slither_results/<project>_slither.json
    Repository path: output/repos/solidity/<project>

Step 3: Classifying Results with LLM
------------------------------------------------------------
Parsed 74 findings from Slither results
Total findings to analyze: 74
CodeQL database + function tree available - code lookup tools enabled

Found 6 issues of type arbitrary-send-erc20

[Agent Thinking]
────────────────────────────────────────────────────────────
  I need to understand the full context of this issue. Let me examine the
  contract's access controls and related functions.
────────────────────────────────────────────────────────────

[Tool Call] list_contract_functions
  Args: { "contract_name": "ExampleContract" }
[Tool Response]
  Available functions/modifiers (10 total):
  - addMapping, forward, requireAllowedEOA, requireAllowedSender,
    run, setEOAAllowStatus, setSenderAllowStatus, setUp, ...

[Tool Call] get_function_code
  Args: { "function_name": "requireAllowedEOA" }
[Tool Response]
  file: path/to/ExampleContract.sol
  43:     modifier requireAllowedEOA(address eoa) {
  44:         if (eoaAllowlist[eoa]) { _; }
  45:         else { revert NotAllowed(eoa); }
  46:     }

[Tool Call] get_function_code
  Args: { "function_name": "requireAllowedSender" }
[Tool Response]
  file: path/to/ExampleContract.sol
  54:     modifier requireAllowedSender() {
  55:         if (sendersAllowlist[msg.sender]) { _; }
  56:         else { revert SenderNotAllowed(msg.sender); }
  57:     }

... (further tool calls for setEOAAllowStatus, setSenderAllowStatus, etc.)

[Agent Thinking]
────────────────────────────────────────────────────────────
  Now I have a clear picture. The forward function transfers tokens in two
  steps; the second transfer is from an allowlisted address. Both sender
  and target are admin-allowlisted. The tokens at the intermediate address
  were just deposited in the same call, so net balance change is zero.
  No profit vector for an attacker.
  ...
  ## Verdict
  **Status: 1007**
  The code is secure. The transfer pattern is intentional; allowlists
  and net-zero flow prevent exploitation. False positive.
────────────────────────────────────────────────────────────

Issue ID: 6, LLM decision: → False Positive
```

The agent uses **deterministic** CodeQL/function-tree lookups for code; it does not invent or hallucinate source.

---

## License and attribution

This project is licensed under the **Apache License, Version 2.0**. See [LICENSE.txt](LICENSE.txt) for the full text.

Ouroborus is a derivative of [Vulnhalla](https://github.com/cyberark/Vulnhalla) by CyberArk Software Ltd., focused on smart contract audits. Original copyright and attribution are retained as required by the Apache License.
