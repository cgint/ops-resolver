import dspy # type: ignore[import-untyped]
import json
import logging
from typing import Any, List
from pydantic import BaseModel
from google.cloud import logging as gcp_logging
from datetime import datetime

# Configuration constants - adjust these as needed
MAX_RESULTS = 1000
PAGE_SIZE = 100

class LogEntry(BaseModel):
    timestamp: str
    labels: dict[Any, Any]
    payload_json_str: str
    payload: dict[Any, Any]
    message: str

class LogEntryList(BaseModel):
    entries: List[LogEntry]

def _to_entry(entry: gcp_logging.LogEntry) -> LogEntry:
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

def _process_page(process_page_lm: dspy.LM, entries_in_page: List[gcp_logging.LogEntry], ai_analyse_for_goal: str, page_count: int) -> List[str]: # type: ignore[no-any-unimported]
    relevant_chunks = []
    # Convert entries to our model format
    entries_compact = [_to_entry(entry).model_dump() for entry in entries_in_page]
    
    # Prepare prompt for AI analysis
    prompt = f"""You are a log analysis expert.

Goal (I start with the goal so you know what to look for): {ai_analyse_for_goal}

Below is a JSON array with {len(entries_compact)} log entries from Google Cloud Logging.

Instructions:
- If none of the log entries are relevant to the goal, respond with exactly: NO_RELEVANT_INFO
- If some entries are relevant, extract ONLY the relevant facts and information
- Keep your response concise and focused on the goal
- Include timestamps and key details for relevant entries to support follow up analysis

LOG_ENTRIES_JSON:
{json.dumps(entries_compact, indent=2)}

GOAL (I repeat the goal now that you got all the details): {ai_analyse_for_goal}

"""

    try:
        # Call the LM for analysis
        logging.info(f"Page {page_count} with {len(entries_in_page)} entries: Asking LLM for analysis using prompt: {prompt}")
        # Log the JSON entries for debugging
        logging.info(f"Page {page_count} with {len(prompt)} char prompt: {prompt}")
        with dspy.context(lm=process_page_lm):
            response_answer = dspy.Predict(signature="question -> answer")(question=prompt).answer
        # Log the full response as JSON for debugging
        logging.info(f"Page {page_count} LLM response JSON: {json.dumps(response_answer, indent=2)}")
        answer = response_answer.strip() if hasattr(response_answer, 'strip') else str(response_answer).strip()
        
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

    return relevant_chunks

# https://cloud.google.com/logging/docs/view/logging-query-language
def gcloud_logging_read_command(
    project_id: str,
    query: str,
    start_time: str,
    end_time: str,
    ai_analyse_for_goal: str,
) -> str:
    """AI-POWERED LOG ANALYSIS TOOL

    The `gcloud_logging_read_command` tool provides intelligent log analysis capabilities:

    ### Usage
    ```python
    gcloud_logging_read_command(
        project_id="your-project-id",
        query="resource.type=k8s_container",
        start_time="2024-01-01T00:00:00Z", 
        end_time="2024-01-01T23:59:59Z",
        ai_analyse_for_goal="Find errors related to pod scheduling"
    )
    ```

    ### Features
    - Fetches logs from Google Cloud Logging based on query and time range
    - Processes logs in pages of 100 entries
    - Uses AI to analyze each page for relevance to your goal
    - Returns only relevant information, filtering out noise
    - Automatically handles log entry conversion and formatting

    ### Example Goals
    - "Find errors related to pod scheduling failures"
    - "Identify performance issues with container restarts"  
    - "Look for authentication or authorization problems"
    - "Find logs related to memory or CPU resource constraints"
        
    Args:
        project_id: The Google Cloud project ID
        query: The log filter query
        start_time: Start time in ISO format (e.g., "2024-01-01T00:00:00Z")
        end_time: End time in ISO format (e.g., "2024-01-01T23:59:59Z")
        ai_analyse_for_goal: Description of what to look for in the logs
        
    Returns:
        String containing only relevant log information based on AI analysis
    """
    
    # Build the complete filter
    ignore_ids = ["dagster - DEBUG", "dagster - INFO", "Failed to parse time", "Failed to parse latency field"]
    ignore_filter_part = " AND NOT (" + " OR ".join(f'"{id}"' for id in ignore_ids) + ")"
    filter_str = f'({query}) AND timestamp>="{start_time}" AND timestamp<="{end_time}" {ignore_filter_part}'
    
    logging.info(f"Fetching logs for project {project_id} with filter: {filter_str}")
    logging.info(f"AI analysis goal: {ai_analyse_for_goal}")
    
    try:
        # Create Google Cloud Logging client
        client = gcp_logging.Client(project=project_id)  # type: ignore[no-untyped-call]
        
        # Get the configured DSPy LM instance
        process_page_lm = dspy.settings.lm
        # process_page_lm = dspy.LM('vertex_ai/gemini-2.5-flash-lite', reasoning_effort="disable")
        
        relevant_chunks = []
        page_count = 0
        total_entries = 0
        
        # Iterate through log entries in batches
        list_entries_start_time = datetime.now()
        entries_list = list(client.list_entries(  # type: ignore[no-untyped-call]
            filter_=filter_str,
            order_by="timestamp desc",
            max_results=MAX_RESULTS
        ))
        list_entries_end_time = datetime.now()
        list_entries_duration_seconds = (list_entries_end_time - list_entries_start_time).total_seconds()
        logging.info(f"Fetched {len(entries_list)} log entries in {list_entries_duration_seconds} seconds.")
        
        # Process entries in pages of PAGE_SIZE
        for i in range(0, len(entries_list), PAGE_SIZE):
            page_count += 1
            entries_in_page = entries_list[i:i + PAGE_SIZE]
            total_entries += len(entries_in_page)
            
            if not entries_in_page:
                continue
            
            relevant_chunks.extend(_process_page(process_page_lm, entries_in_page, ai_analyse_for_goal, page_count))
        
        logging.info(f"Processed {page_count} pages with {total_entries} total entries and found {len(relevant_chunks)} pages with relevant information")
        would_there_have_been_more_pages: bool = total_entries >= MAX_RESULTS
        would_there_have_been_more_pages_text = "**There would be more pages** but MAX_RESULTS was reached" if would_there_have_been_more_pages else "There would be no more pages so we covered all the log messages with the query."
        page_fetching_statistics = f"""

# Page fetching statistics
Processed {page_count} pages with {total_entries} total entries MAX_RESULTS={MAX_RESULTS} and PAGE_SIZE={PAGE_SIZE}.

{would_there_have_been_more_pages_text}
"""
        output_prefix = page_fetching_statistics + "\n\n" + "# AI-filtered log entries" + "\n\n"
        
        if len(relevant_chunks) > 0:
            return output_prefix + "\n\n".join(relevant_chunks)
        else:
            return output_prefix + "No relevant information found in the log entries."
            
    except Exception as e:
        error_message = f"Error fetching logs from project {project_id}: {str(e)}"
        logging.error(error_message)
        return error_message

# Tried gcloud_logging_read_command_async once - immediately returned

# async def gcloud_logging_read_command_async(
#     project_id: str,
#     query: str,
#     start_time: str,
#     end_time: str,
#     ai_analyse_for_goal: str,
# ) -> AsyncGenerator[str, None]:
#     """
#     Async generator that fetches log entries from Google Cloud Logging and yields AI-filtered results.
    
#     Args:
#         project_id: The Google Cloud project ID
#         query: The log filter query
#         start_time: Start time in ISO format (e.g., "2024-01-01T00:00:00Z")
#         end_time: End time in ISO format (e.g., "2024-01-01T23:59:59Z")
#         ai_analyse_for_goal: Description of what to look for in the logs
        
#     Yields:
#         Strings containing relevant log information based on AI analysis, one per page
#     """
#     # Build the complete filter
#     filter_str = f'({query}) AND timestamp>="{start_time}" AND timestamp<="{end_time}"'
    
#     logging.info(f"Async fetching logs for project {project_id} with filter: {filter_str}")
#     logging.info(f"AI analysis goal: {ai_analyse_for_goal}")
    
#     try:
#         # Get the configured DSPy LM instance
#         lm = dspy.settings.lm
#         if lm is None:
#             yield "Error: No language model configured in DSPy"
#             return
        
#         # Create client in thread to avoid blocking
#         client = await asyncio.to_thread(gcp_logging.Client, project=project_id)
        
#         page_count = 0
#         total_entries = 0
        
#         # Fetch log entries using page iterator for streaming
#         def fetch_page_iterator():
#             return client.list_entries(
#                 filter_=filter_str,
#                 order_by="timestamp desc", 
#                 page_size=PAGE_SIZE
#             )
        
#         # Get the page iterator in a thread
#         page_iterator = await asyncio.to_thread(fetch_page_iterator)
        
#         # Process pages one by one
#         for page in page_iterator:
#             page_count += 1
#             entries_in_page = list(page)
#             total_entries += len(entries_in_page)
            
#             # Stop if we've reached the maximum results
#             if total_entries > MAX_RESULTS:
#                 entries_in_page = entries_in_page[:MAX_RESULTS - (total_entries - len(entries_in_page))]
#                 logging.info(f"Reached MAX_RESULTS limit of {MAX_RESULTS}, stopping")
            
#             if not entries_in_page:
#                 continue
            
#             # Convert entries to our model format in thread
#             def convert_entries():
#                 return [to_entry(entry).model_dump() for entry in entries_in_page]
            
#             entries_compact = await asyncio.to_thread(convert_entries)
            
#             # Prepare prompt for AI analysis
#             prompt = f"""You are a log analysis assistant.

# Goal: {ai_analyse_for_goal}

# Below is a JSON array with {len(entries_compact)} log entries from Google Cloud Logging.

# Instructions:
# - If none of the log entries are relevant to the goal, respond with exactly: NO_RELEVANT_INFO
# - If some entries are relevant, extract ONLY the relevant facts and information
# - Keep your response concise and focused on the goal
# - Include timestamps and key details for relevant entries

# LOG_ENTRIES_JSON:
# {json.dumps(entries_compact, indent=2)}"""

#             try:
#                 # Call the LM for analysis in thread
#                 logging.info(f"Page {page_count} with {len(entries_in_page)} entries: Asking LLM for analysis")
                
#                 def call_lm():
#                     return lm(prompt)
                
#                 response = await asyncio.to_thread(call_lm)
#                 answer = response.strip() if hasattr(response, 'strip') else str(response).strip()
                
#                 # Check if the response indicates relevance
#                 if answer and answer.upper() != "NO_RELEVANT_INFO":
#                     result = f"=== Page {page_count} Analysis ===\n{answer}"
#                     # Log first 100 chars of the analysis for visibility
#                     preview = answer.replace('\n', ' ')[:100] + "..." if len(answer) > 100 else answer.replace('\n', ' ')
#                     logging.info(f"Page {page_count} with {len(entries_in_page)} entries: ✓ Found relevant info - {preview}")
#                     yield result
#                 else:
#                     logging.info(f"Page {page_count} with {len(entries_in_page)} entries: ✗ No relevant information found")
                    
#             except Exception as e:
#                 error_msg = f"Error analyzing page {page_count}: {str(e)}"
#                 logging.error(f"Page {page_count} with {len(entries_in_page)} entries: ⚠ Error - {str(e)}")
#                 yield f"=== Page {page_count} Error ===\n{error_msg}"
            
#             # Stop if we've reached the maximum results
#             if total_entries >= MAX_RESULTS:
#                 break
        
#         logging.info(f"Processed {page_count} pages with {total_entries} total entries")
        
#         # If we didn't yield anything, indicate no relevant info found
#         if page_count == 0:
#             yield "No log entries found for the specified criteria."
            
#     except Exception as e:
#         error_message = f"Error fetching logs from project {project_id}: {str(e)}"
#         logging.error(error_message)
#         yield error_message


def get_gcloud_logging_read_command_tool() -> dspy.Tool: # type: ignore[no-any-unimported]
    return dspy.Tool(gcloud_logging_read_command)