import subprocess
import logging
from main_dspy_prompt import allowed_commands_starts_with


def kubectl_shell(command: str) -> str:
    """
    Executes a kubectl command and returns its output.
    Only use this tool to run kubectl commands for Kubernetes cluster analysis.
    
    Args:
        command: The kubectl command to execute
        
    Returns:
        The command output including stdout and stderr
    """
    if not any(command.startswith(start) for start in allowed_commands_starts_with.keys()):
        error_message = f"Error: Command '{command}' not allowed! Allowed commands start with {', '.join(allowed_commands_starts_with.keys())}"
        logging.error(error_message)
        return error_message
    
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
        logging.info(f"Command output:\n{output[:10000]}\n\n  Output-Stats: (len={len(output)}, lines={len(output.splitlines())})")
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