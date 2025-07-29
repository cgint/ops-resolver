# Plan: Pydantic ReAct AI for Kubernetes Analysis

## 1. Goal

Create a ReAct-based AI agent that can use `kubectl` commands via a shell tool to analyze a Kubernetes cluster. The agent's reasoning and actions will be structured and validated using Pydantic models.

## 2. Core Components

-   **ReAct Agent:** The main loop that thinks, acts, and observes.
-   **Pydantic Models:** Define the structure for thoughts, actions, and observations.
-   **Shell Tool:** A single tool that allows the agent to execute shell commands (`kubectl`).
-   **LLM Integration:** Connect to an LLM (e.g., OpenAI) to power the agent's reasoning.

## 3. Project Structure

```
ops-resolver/
├── react_ai_k8s_analyst.md  # This plan
├── pyproject.toml           # Project metadata and dependencies
├── uv.lock                  # Lockfile for reproducible dependencies
├── .env.example             # Example environment variables
└── src/
    ├── __init__.py
    ├── agent.py             # The core ReAct agent logic
    ├── models.py            # Pydantic models
    ├── tools.py             # Shell command tool
    ├── prompts.py           # LLM prompts
    └── main.py              # Main script to run the agent
```

## 4. Implementation Steps

### Step 1: Setup Project and Dependencies

-   Create the directory structure.
-   Initialize a `uv` project with a `pyproject.toml`.
-   Add dependencies: `pydantic`, `openai`, `python-dotenv`, `uv`.

### Step 2: Define Pydantic Models (`src/models.py`)

-   Create models for `Thought`, `Action`, and `Observation`. The action model should specify the tool to use (always 'shell') and the command to run.

### Step 3: Implement the Shell Tool (`src/tools.py`)

-   Create a `ShellTool` class with a `run(command: str) -> str` method.
-   This method will use `subprocess.run` to execute the command.
-   It must capture and return `stdout` and `stderr`.

### Step 4: Define Prompts (`src/prompts.py`)

-   Create a system prompt that instructs the LLM on its role, the tools available, and the required output format (e.g., JSON with 'thought' and 'action' keys).
-   The prompt will explain the ReAct cycle: `Thought -> Action -> Observation`.

### Step 5: Build the ReAct Agent (`src/agent.py`)

-   Create an `ReActAgent` class.
-   The constructor will take an LLM client and the available tools.
-   Implement a `run(goal: str)` method.
-   Inside `run`, implement the ReAct loop:
    1.  Format the prompt with the current goal and history.
    2.  Call the LLM.
    3.  Parse the response into Pydantic models.
    4.  Execute the action using the `ShellTool`.
    5.  Record the observation.
    6.  Repeat until the LLM indicates it's finished or a max iteration limit is reached.

### Step 6: Create the Entrypoint (`src/main.py`)

-   Load environment variables (e.g., API keys).
-   Instantiate the LLM client, `ShellTool`, and `ReActAgent`.
-   Define an initial goal, for example: "Check the status of all pods in the 'default' namespace."
-   Call the agent's `run` method and print the final result.

## 5. Usage

1.  Install dependencies: `uv pip install -r requirements.txt` (or `uv sync`).
2.  Create a `.env` file with the `OPENAI_API_KEY`.
3.  Run the agent: `uv run python src/main.py`. 