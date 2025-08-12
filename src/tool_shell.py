import subprocess
import logging
import dspy # type: ignore[import-untyped]

allowed_commands_starts_with = {
    "kubectl get": "Get information about resources", 
    "kubectl describe": "Get detailed information about a resource",
    "kubectl config get-": "Get kubeconfig information (clusters, contexts, users)"
}
allowed_commands_starts_with_desc = "\n".join([f"- `{command}`: {description}" for command, description in allowed_commands_starts_with.items()])

KUBECTL_SHELL_INSTRUCTIONS = f"""

## Info on kubectl_shell tool

### Allowed Command Prefixes
The kubectl_shell tool only allows commands that start with:
{allowed_commands_starts_with_desc}

## KUBECTL COMMANDS REFERENCE

### Common kubectl get Commands

#### Viewing Resources:
- kubectl get pods (list all pods in current namespace)
- kubectl get pods --all-namespaces (list all pods in all namespaces)
- kubectl get pods -o wide (list pods with more details)
- kubectl get services (list all services)
- kubectl get namespaces (list all namespaces)
- kubectl get nodes (list all nodes)
- kubectl get deployment my-deployment (get specific deployment)
- kubectl get pods --show-labels (show labels for all pods)

#### Filtering and Sorting:
- kubectl get pods --field-selector=status.phase=Running (get running pods)
- kubectl get pods --selector=app=cassandra (get pods with specific label)
- kubectl get pods --sort-by='.status.containerStatuses[0].restartCount' (sort by restart count)
- kubectl get services --sort-by=.metadata.name (sort services by name)
- kubectl get node --selector='!node-role.kubernetes.io/control-plane' (get worker nodes)

#### Output Formats:
- kubectl get pod my-pod -o yaml (get pod YAML)
- kubectl get pod my-pod -o json (get pod JSON)
- kubectl get nodes -o jsonpath='{{.items[*].status.addresses[?({{@.type=="ExternalIP"}})].address}}' (get external IPs)
- kubectl get node -o custom-columns='NODE_NAME:.metadata.name,STATUS:.status.conditions[?({{@.type=="Ready"}})].status' (custom columns)

### kubectl describe Commands
- kubectl describe nodes my-node (describe nodes with verbose output)
- kubectl describe pods my-pod (describe pods with verbose output)
- kubectl describe deployment my-deployment (describe specific deployment)
- kubectl describe service my-service (describe service)
- kubectl describe namespace my-namespace (describe namespace)

### kubectl config get- Commands
- kubectl config get-clusters (display clusters defined in kubeconfig)
- kubectl config get-contexts (describe one or many contexts)
- kubectl config get-users (display users defined in kubeconfig)
- kubectl config get-contexts --no-headers (get contexts without headers)
- kubectl config get-contexts -o name (get only context names)

### Resource Types (common abbreviations):
pods (po), services (svc), deployments (deploy), replicasets (rs), daemonsets (ds), statefulsets (sts), jobs, cronjobs (cj), configmaps (cm), secrets, persistentvolumes (pv), persistentvolumeclaims (pvc), nodes (no), namespaces (ns), ingresses (ing)

### Useful Analysis Patterns:

#### Health Checks:
- kubectl get pods --all-namespaces --field-selector=status.phase!=Running (check unhealthy pods)
- kubectl get nodes -o custom-columns='NODE_NAME:.metadata.name,STATUS:.status.conditions[?({{@.type=="Ready"}})].status' (check node readiness)
- kubectl get events --sort-by=.metadata.creationTimestamp (get events by timestamp)
- kubectl get events --field-selector type=Warning (get warning events)

#### Resource Usage:
- kubectl get resourcequota --all-namespaces (get resource quotas)
- kubectl get limitrange --all-namespaces (get limit ranges)
- kubectl get pv --sort-by=.spec.capacity.storage (get persistent volumes by capacity)

#### Network Resources:
- kubectl get svc --all-namespaces (get all services)
- kubectl get ingress --all-namespaces (get ingresses)
- kubectl get networkpolicy --all-namespaces (get network policies)

### Analysis Tips:
1. Start broad with 'kubectl get all --all-namespaces' for overview
2. Use labels for filtering resources
3. Combine get and describe commands for deeper investigation
4. Check events with 'kubectl get events --sort-by=.metadata.creationTimestamp'
5. Use JSON/YAML output for complete resource information
6. Monitor logs in real-time for active debugging
"""

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
    

def get_kubectl_shell_tool() -> dspy.Tool: # type: ignore[no-any-unimported]
    return dspy.Tool(kubectl_shell, desc=KUBECTL_SHELL_INSTRUCTIONS)