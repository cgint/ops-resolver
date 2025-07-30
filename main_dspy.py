import dspy
import subprocess
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

def setup_logging():
    """Sets up a file-based logger for the application."""
    log_filename = f"logs/dspy-log-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    
    # Configure the root logger to capture everything from DSPy
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
if "GEMINI_API_KEY" not in os.environ:
    print("Error: GEMINI_API_KEY environment variable not set.")
    exit(1)

# Configure DSPy with Gemini 2.5 Flash via Vertex AI (same as main.py)
gemini = dspy.LM('vertex_ai/gemini-2.5-flash', api_key=os.getenv("GEMINI_API_KEY"))
dspy.configure(lm=gemini)

def kubectl_shell(command: str) -> str:
    """
    Executes a kubectl command and returns its output.
    Only use this tool to run kubectl commands for Kubernetes cluster analysis.
    
    Args:
        command: The kubectl command to execute
        
    Returns:
        The command output including stdout and stderr
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

# Create DSPy tool from the shell function
shell_tool = dspy.Tool(kubectl_shell)

class K8sReActAgent(dspy.Module):
    """
    A Kubernetes ReAct AI agent using DSPy.
    
    This agent can analyze Kubernetes clusters by executing kubectl commands
    and reasoning about the results to answer user questions.
    """
    
    def __init__(self):
        super().__init__()
        
        # Define the signature with instructions
        signature = dspy.Signature(
            "question -> answer",
            instructions=(
                "You are a Kubernetes (k8s) expert assistant. "
                "Your task is to use the provided 'kubectl_shell' tool to execute kubectl commands and analyze the cluster based on the user's request. "
                "Think step-by-step about what commands you need to run. "
                "After executing a command, observe the output and decide on the next step. "
                "When you have gathered enough information to answer the user's request, provide a final, comprehensive answer."
            )
        )
        
        # Create ReAct module with the signature and tools
        self.react = dspy.ReAct(signature, tools=[shell_tool])
    
    def forward(self, question):
        """Forward pass through the ReAct agent."""
        return self.react(question=question)

def main():
    """
    Main function to run the Kubernetes ReAct AI agent using DSPy.
    """
    log_filename = setup_logging()
    
    # Initialize the agent
    agent = K8sReActAgent()
    
    # Example goals for the agent
    # goal = "Check the status of all pods in all namespaces. Identify any pods that are not in a 'Running' or 'Completed' state."
    # goal = "List all pods running on any node from node-pool master-pool from namespace 'default'."
    goal = "What are the max allowed pods per node in the cluster?"

    logging.info(f"Goal: {goal}\n")

    # Run the agent
    try:
        response = agent(question=goal)
        
        # Extract the final answer
        final_answer = response.answer
        
        # The final output is printed to the console
        logging.info(f"\nFinal Answer:\n{final_answer}")
        print(f"\nFinal Answer:\n{final_answer}")
        
    except Exception as e:
        error_msg = f"Error running agent: {str(e)}"
        logging.error(error_msg)
        print(error_msg)
    
    print(f"\nFull conversation history logged to: {log_filename}")

if __name__ == "__main__":
    main()