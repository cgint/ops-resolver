import os
import subprocess
import logging
from datetime import datetime
from dotenv import load_dotenv
from pydantic_ai import Agent

def setup_logging():
    """Sets up a file-based logger for the application."""
    log_filename = f"react-log-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    
    # Configure the root logger to capture everything from pydantic_ai
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_filename,
        filemode='w'
    )
    
    # Add a stream handler to also print logs to the console for real-time feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    return log_filename

# Load environment variables from .env file
load_dotenv()

# It's good practice to handle the case where the API key is not set
if "OPENAI_API_KEY" not in os.environ:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit(1)

# Define the ReAct agent
# Using a more advanced model might yield better reasoning for kubectl commands
agent = Agent(
    "openai:gpt-4-turbo",
    system_prompt=(
        "You are a Kubernetes (k8s) expert assistant. "
        "Your task is to use the provided 'shell' tool to execute kubectl commands and analyze the cluster based on the user's request. "
        "Think step-by-step about what commands you need to run. "
        "After executing a command, observe the output and decide on the next step. "
        "When you have gathered enough information to answer the user's request, provide a final, comprehensive answer."
    ),
)


@agent.tool_plain
def shell(command: str) -> str:
    """
    Executes a shell command and returns its output.
    Only use this tool to run kubectl commands.
    """
    try:
        # Security Note: Running arbitrary shell commands can be dangerous.
        # In a real-world application, this should be heavily sandboxed and restricted.
        # For this example, we assume the commands are trusted.
        logging.info(f"Executing command: {command}")
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        output = f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
        logging.info(f"Command output:\n{output}")
        return output
    except subprocess.CalledProcessError as e:
        # If the command returns a non-zero exit code, it's an error
        error_message = f"Command failed with exit code {e.returncode}.\nStdout:\n{e.stdout}\nStderr:\n{e.stderr}"
        logging.error(error_message)
        return error_message
    except Exception as e:
        # Handle other potential exceptions
        error_message = f"An unexpected error occurred: {str(e)}"
        logging.error(error_message)
        return error_message


def main():
    """
    Main function to run the Kubernetes ReAct AI agent.
    """
    log_filename = setup_logging()
    
    # Example goal for the agent
    goal = "Check the status of all pods in all namespaces. Identify any pods that are not in a 'Running' or 'Completed' state."

    logging.info(f"Goal: {goal}\n")

    # Run the agent synchronously
    response = agent.run_sync(goal)

    # The final output is printed to the console
    logging.info(f"\nFinal Answer:\n{response.output}")
    print(f"\nFull conversation history logged to: {log_filename}")


if __name__ == "__main__":
    main() 