import dspy  # type: ignore[import-untyped]
import subprocess
import logging
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv
from main_dspy_prompt import get_allowed_commands_starts_with_information, allowed_commands_starts_with
from tool_shell import kubectl_shell
from tool_gcloud_log_agent_filter import gcloud_logging_read_command

def setup_logging() -> str:
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
gemini = dspy.LM('vertex_ai/gemini-2.5-flash')
dspy.configure(lm=gemini)

def make_sure_we_are_using_the_correct_context(context_name: str | None = None, force_set_context: bool = False) -> None:
    """
    Make sure we are using the correct context.
    """
    if context_name is None:
        current_context = subprocess.run(["kubectl", "config", "current-context"], capture_output=True, text=True).stdout.strip()
    else:
        current_context = context_name
        
    print(f"\nCurrent context: {current_context}\n")
    if force_set_context:
        print(" -> Forcing selection of context.")
        selection_y_n_s_q = "y"
    else:   
        user_input = input("   Is this the correct context? y (yes)/n (no)/s (select context)/q (quit): ")
        selection_y_n_s_q = user_input.lower()
    if selection_y_n_s_q == "y":
        if context_name is not None:
            subprocess.run(["kubectl", "config", "use-context", current_context], capture_output=True, text=True)
        pass
    elif selection_y_n_s_q == "n" or selection_y_n_s_q == "q":
        print("Quitting on users request.")
        exit(1)
    elif selection_y_n_s_q == "s":
        print("Switching to the correct context.")
        # Get available contexts and let user choose
        contexts_output = subprocess.run(["kubectl", "config", "get-contexts", "-o", "name"], capture_output=True, text=True)
        available_contexts = contexts_output.stdout.strip().split('\n')
        
        print("Available contexts:")
        for i, context in enumerate(available_contexts, 1):
            print(f"{i}. {context}")
        
        context_input = input("   Enter a unique substring of the context name you want to switch to: ")
        
        # Find exactly matching context
        matching_contexts = [ctx for ctx in available_contexts if ctx == current_context]
        # Find matching contexts
        if len(matching_contexts) == 0:
            matching_contexts = [ctx for ctx in available_contexts if context_input in ctx]
        
        if len(matching_contexts) == 0:
            print(f"No contexts found matching '{context_input}'. Quitting.")
            exit(1)
        elif len(matching_contexts) > 1:
            print(f"Multiple contexts match '{context_input}':")
            for ctx in matching_contexts:
                print(f"  - {ctx}")
            print("Please be more specific. Quitting.")
            exit(1)
        else:
            current_context = matching_contexts[0]
            print(f"Selected context: {current_context}")
        subprocess.run(["kubectl", "config", "use-context", current_context], capture_output=True, text=True)
        make_sure_we_are_using_the_correct_context(current_context, force_set_context)
    else:
        print("Invalid selection. Quitting.")
        exit(1)

def main(goal: str, context_name: str | None = None, force_set_context: bool = False) -> None:
    """
    Main function to run the Kubernetes ReAct AI agent using DSPy.
    """
    log_filename = setup_logging()

    make_sure_we_are_using_the_correct_context(context_name, force_set_context)
    
    instructions = get_allowed_commands_starts_with_information(allowed_commands_starts_with)   
    signature=dspy.Signature(
            "question -> answer",
            instructions=instructions
        )
    tools = [dspy.Tool(kubectl_shell), dspy.Tool(gcloud_logging_read_command)]
    agent = dspy.ReAct(tools=tools, signature=signature)
    
    logging.info(f"Goal: {goal}\n")

    # Run the agent
    try:
        response = agent(question=goal)
        
        # Extract the final answer
        final_answer = response.answer
        
        # The final output is printed to the console
        logging.info(f"\nFinal Answer:\n{final_answer}")

        with open(log_filename.replace(".log", ".final_answer.md"), "a") as f:
            f.write(final_answer)
        
    except Exception as e:
        error_msg = f"Error running agent: {str(e)}"
        logging.error(error_msg)
        print(error_msg)
    
    print(f"\nFull conversation history logged to: {log_filename}")

if __name__ == "__main__":
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Kubernetes ReAct AI agent using DSPy")
    parser.add_argument("-c", "--context", type=str, help="Kubernetes context name to use")
    parser.add_argument("-f", "--force-set-context", action="store_true", help="Force context selection even if current context matches")
    parser.add_argument("-m", "--mission", type=str, help="Override the default goal/mission for the agent")
    args = parser.parse_args()
    
    # Define default goal and handle override
    default_goal = "What are the max allowed pods per node-pool in the cluster?"
    goal = args.mission if args.mission else default_goal
    
    main(goal, args.context, args.force_set_context)