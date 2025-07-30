import dspy  # type: ignore[import-untyped]
import json
import logging
from typing import Any, List
from pydantic import BaseModel
from google.cloud import logging as gcp_logging

class LogEntry(BaseModel):
    timestamp: str
    labels: dict[Any, Any]
    payload_json_str: str
    payload: dict[Any, Any]
    message: str

class LogEntryList(BaseModel):
    entries: List[LogEntry]

def to_entry(entry: gcp_logging.LogEntry) -> LogEntry:
    """Convert a Google Cloud Logging entry to our LogEntry model."""
    the_payload = entry.payload if entry.payload is not None else "-no-payload-"  # type: ignore[attr-defined]
    if isinstance(the_payload, str):
        msg = the_payload
        payload = {'payload': the_payload}
    else:
        msg = the_payload.get("message", "")
        payload = the_payload
    return LogEntry(
        timestamp=str(entry.timestamp),  # type: ignore[attr-defined]
        labels=entry.resource.labels,  # type: ignore[attr-defined]
        payload=payload, 
        payload_json_str=json.dumps(the_payload), 
        message=msg
    )

def gcloud_logging_read_command(
    project_id: str,
    query: str,
    start_time: str,
    end_time: str,
    ai_analyse_for_goal: str,
) -> str:
    """
    Fetches log entries from Google Cloud Logging and uses AI to filter for relevant information.
    
    Args:
        project_id: The Google Cloud project ID
        query: The log filter query
        start_time: Start time in ISO format (e.g., "2024-01-01T00:00:00Z")
        end_time: End time in ISO format (e.g., "2024-01-01T23:59:59Z")
        ai_analyse_for_goal: Description of what to look for in the logs
        
    Returns:
        String containing only relevant log information based on AI analysis
    """
    MAX_RESULTS = 250
    PAGE_SIZE = 100
    
    # Build the complete filter
    filter_str = f'({query}) AND timestamp>="{start_time}" AND timestamp<="{end_time}"'
    
    logging.info(f"Fetching logs for project {project_id} with filter: {filter_str}")
    logging.info(f"AI analysis goal: {ai_analyse_for_goal}")
    
    try:
        # Create Google Cloud Logging client
        client = gcp_logging.Client(project=project_id)  # type: ignore[no-untyped-call]
        
        # Get the configured DSPy LM instance
        lm = dspy.settings.lm
        if lm is None:
            return "Error: No language model configured in DSPy"
        
        relevant_chunks = []
        page_count = 0
        total_entries = 0
        
        # Iterate through log entries in batches
        entries_list = list(client.list_entries(  # type: ignore[no-untyped-call]
            filter_=filter_str,
            order_by="timestamp desc",
            max_results=MAX_RESULTS
        ))
        
        # Process entries in pages of PAGE_SIZE
        for i in range(0, len(entries_list), PAGE_SIZE):
            page_count += 1
            entries_in_page = entries_list[i:i + PAGE_SIZE]
            total_entries += len(entries_in_page)
            
            if not entries_in_page:
                continue
            
            # Convert entries to our model format
            entries_compact = [to_entry(entry).model_dump() for entry in entries_in_page]
            
            # Prepare prompt for AI analysis
            prompt = f"""You are a log analysis assistant.

Goal: {ai_analyse_for_goal}

Below is a JSON array with {len(entries_compact)} log entries from Google Cloud Logging.

Instructions:
- If none of the log entries are relevant to the goal, respond with exactly: NO_RELEVANT_INFO
- If some entries are relevant, extract ONLY the relevant facts and information
- Keep your response concise and focused on the goal
- Include timestamps and key details for relevant entries

LOG_ENTRIES_JSON:
{json.dumps(entries_compact, indent=2)}"""

            try:
                # Call the LM for analysis
                logging.info(f"Page {page_count} with {len(entries_in_page)} entries: Asking LLM for analysis using prompt: {prompt}")
                # Log the JSON entries for debugging
                logging.info(f"Page {page_count} entries JSON: {json.dumps(entries_compact, indent=2)}")
                response = lm(prompt)
                # Log the full response as JSON for debugging
                logging.info(f"Page {page_count} LLM response JSON: {json.dumps(str(response), indent=2)}")
                answer = response.strip() if hasattr(response, 'strip') else str(response).strip()
                
                # Check if the response indicates relevance
                if answer and answer.upper() != "NO_RELEVANT_INFO":
                    relevant_chunks.append(f"=== Page {page_count} Analysis ===\n{answer}")
                    # Log first 100 chars of the analysis for visibility
                    preview = answer.replace('\n', ' ')[:100] + "..." if len(answer) > 100 else answer.replace('\n', ' ')
                    logging.info(f"Page {page_count} with {len(entries_in_page)} entries: ✓ Found relevant info - {preview}")
                else:
                    logging.info(f"Page {page_count} with {len(entries_in_page)} entries: ✗ No relevant information found")
                    
            except Exception as e:
                error_msg = f"Error analyzing page {page_count}: {str(e)}"
                logging.error(f"Page {page_count} with {len(entries_in_page)} entries: ⚠ Error - {str(e)}")
                relevant_chunks.append(f"=== Page {page_count} Error ===\n{error_msg}")
        
        logging.info(f"Processed {page_count} pages with {total_entries} total entries")
        
        if relevant_chunks:
            result = "\n\n".join(relevant_chunks)
            logging.info(f"Found {len(relevant_chunks)} pages with relevant information")
            return result
        else:
            return "No relevant information found in the log entries."
            
    except Exception as e:
        error_message = f"Error fetching logs from project {project_id}: {str(e)}"
        logging.error(error_message)
        return error_message
