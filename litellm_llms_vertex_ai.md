# Directory Structure
_Includes files where the actual content might be omitted. This way the LLM can still use the file structure to understand the project._
```
.
└── llms
    └── vertex_ai
        ├── batches
        │   ├── handler.py
        │   ├── Readme.md
        │   └── transformation.py
        ├── common_utils.py
        ├── context_caching
        │   ├── transformation.py
        │   └── vertex_ai_context_caching.py
        ├── cost_calculator.py
        ├── files
        │   ├── handler.py
        │   └── transformation.py
        ├── fine_tuning
        │   └── handler.py
        ├── gemini
        │   ├── cost_calculator.py
        │   ├── transformation.py
        │   └── vertex_and_google_ai_studio_gemini.py
        ├── gemini_embeddings
        │   ├── batch_embed_content_handler.py
        │   └── batch_embed_content_transformation.py
        ├── google_genai
        │   └── transformation.py
        ├── image_generation
        │   ├── cost_calculator.py
        │   └── image_generation_handler.py
        ├── multimodal_embeddings
        │   ├── embedding_handler.py
        │   └── transformation.py
        ├── text_to_speech
        │   └── text_to_speech_handler.py
        ├── vector_stores
        │   ├── __init__.py
        │   └── transformation.py
        ├── vertex_ai_non_gemini.py
        ├── vertex_ai_partner_models
        │   ├── __init__.py
        │   ├── ai21
        │   │   └── transformation.py
        │   ├── anthropic
        │   │   ├── experimental_pass_through
        │   │   │   └── transformation.py
        │   │   └── transformation.py
        │   ├── llama3
        │   │   └── transformation.py
        │   └── main.py
        ├── vertex_embeddings
        │   ├── embedding_handler.py
        │   ├── transformation.py
        │   └── types.py
        ├── vertex_llm_base.py
        └── vertex_model_garden
            └── main.py
```

# File Contents

## File: `llms/vertex_ai/batches/handler.py`
```
import json
from typing import Any, Coroutine, Dict, Optional, Union

import httpx

import litellm
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.types.llms.openai import CreateBatchRequest
from litellm.types.llms.vertex_ai import (
    VERTEX_CREDENTIALS_TYPES,
    VertexAIBatchPredictionJob,
)
from litellm.types.utils import LiteLLMBatch

from .transformation import VertexAIBatchTransformation


class VertexAIBatchPrediction(VertexLLM):
    def __init__(self, gcs_bucket_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gcs_bucket_name = gcs_bucket_name

    def create_batch(
        self,
        _is_async: bool,
        create_batch_data: CreateBatchRequest,
        api_base: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> Union[LiteLLMBatch, Coroutine[Any, Any, LiteLLMBatch]]:
        sync_handler = _get_httpx_client()

        access_token, project_id = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )

        default_api_base = self.create_vertex_batch_url(
            vertex_location=vertex_location or "us-central1",
            vertex_project=vertex_project or project_id,
        )

        if len(default_api_base.split(":")) > 1:
            endpoint = default_api_base.split(":")[-1]
        else:
            endpoint = ""

        _, api_base = self._check_custom_proxy(
            api_base=api_base,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint=endpoint,
            stream=None,
            auth_header=None,
            url=default_api_base,
        )

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {access_token}",
        }

        vertex_batch_request: VertexAIBatchPredictionJob = VertexAIBatchTransformation.transform_openai_batch_request_to_vertex_ai_batch_request(
            request=create_batch_data
        )

        if _is_async is True:
            return self._async_create_batch(
                vertex_batch_request=vertex_batch_request,
                api_base=api_base,
                headers=headers,
            )

        response = sync_handler.post(
            url=api_base,
            headers=headers,
            data=json.dumps(vertex_batch_request),
        )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        _json_response = response.json()
        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_json_response
        )
        return vertex_batch_response

    async def _async_create_batch(
        self,
        vertex_batch_request: VertexAIBatchPredictionJob,
        api_base: str,
        headers: Dict[str, str],
    ) -> LiteLLMBatch:
        client = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.VERTEX_AI,
        )
        response = await client.post(
            url=api_base,
            headers=headers,
            data=json.dumps(vertex_batch_request),
        )
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        _json_response = response.json()
        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_json_response
        )
        return vertex_batch_response

    def create_vertex_batch_url(
        self,
        vertex_location: str,
        vertex_project: str,
    ) -> str:
        """Return the base url for the vertex garden models"""
        #  POST https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/LOCATION/batchPredictionJobs
        return f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/batchPredictionJobs"

    def retrieve_batch(
        self,
        _is_async: bool,
        batch_id: str,
        api_base: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> Union[LiteLLMBatch, Coroutine[Any, Any, LiteLLMBatch]]:
        sync_handler = _get_httpx_client()

        access_token, project_id = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )

        default_api_base = self.create_vertex_batch_url(
            vertex_location=vertex_location or "us-central1",
            vertex_project=vertex_project or project_id,
        )

        # Append batch_id to the URL
        default_api_base = f"{default_api_base}/{batch_id}"

        if len(default_api_base.split(":")) > 1:
            endpoint = default_api_base.split(":")[-1]
        else:
            endpoint = ""

        _, api_base = self._check_custom_proxy(
            api_base=api_base,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint=endpoint,
            stream=None,
            auth_header=None,
            url=default_api_base,
        )

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {access_token}",
        }

        if _is_async is True:
            return self._async_retrieve_batch(
                api_base=api_base,
                headers=headers,
            )

        response = sync_handler.get(
            url=api_base,
            headers=headers,
        )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        _json_response = response.json()
        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_json_response
        )
        return vertex_batch_response

    async def _async_retrieve_batch(
        self,
        api_base: str,
        headers: Dict[str, str],
    ) -> LiteLLMBatch:
        client = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.VERTEX_AI,
        )
        response = await client.get(
            url=api_base,
            headers=headers,
        )
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        _json_response = response.json()
        vertex_batch_response = VertexAIBatchTransformation.transform_vertex_ai_batch_response_to_openai_batch_response(
            response=_json_response
        )
        return vertex_batch_response
```

## File: `llms/vertex_ai/batches/Readme.md`
```
# Vertex AI Batch Prediction Jobs

Implementation to call VertexAI Batch endpoints in OpenAI Batch API spec

Vertex Docs: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini

```

## File: `llms/vertex_ai/batches/transformation.py`
```
import uuid
from typing import Dict

from litellm.llms.vertex_ai.common_utils import (
    _convert_vertex_datetime_to_openai_datetime,
)
from litellm.types.llms.openai import BatchJobStatus, CreateBatchRequest
from litellm.types.llms.vertex_ai import *
from litellm.types.utils import LiteLLMBatch


class VertexAIBatchTransformation:
    """
    Transforms OpenAI Batch requests to Vertex AI Batch requests

    API Ref: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini
    """

    @classmethod
    def transform_openai_batch_request_to_vertex_ai_batch_request(
        cls,
        request: CreateBatchRequest,
    ) -> VertexAIBatchPredictionJob:
        """
        Transforms OpenAI Batch requests to Vertex AI Batch requests
        """
        request_display_name = f"litellm-vertex-batch-{uuid.uuid4()}"
        input_file_id = request.get("input_file_id")
        if input_file_id is None:
            raise ValueError("input_file_id is required, but not provided")
        input_config: InputConfig = InputConfig(
            gcsSource=GcsSource(uris=input_file_id), instancesFormat="jsonl"
        )
        model: str = cls._get_model_from_gcs_file(input_file_id)
        output_config: OutputConfig = OutputConfig(
            predictionsFormat="jsonl",
            gcsDestination=GcsDestination(
                outputUriPrefix=cls._get_gcs_uri_prefix_from_file(input_file_id)
            ),
        )
        return VertexAIBatchPredictionJob(
            inputConfig=input_config,
            outputConfig=output_config,
            model=model,
            displayName=request_display_name,
        )

    @classmethod
    def transform_vertex_ai_batch_response_to_openai_batch_response(
        cls, response: VertexBatchPredictionResponse
    ) -> LiteLLMBatch:
        return LiteLLMBatch(
            id=cls._get_batch_id_from_vertex_ai_batch_response(response),
            completion_window="24hrs",
            created_at=_convert_vertex_datetime_to_openai_datetime(
                vertex_datetime=response.get("createTime", "")
            ),
            endpoint="",
            input_file_id=cls._get_input_file_id_from_vertex_ai_batch_response(
                response
            ),
            object="batch",
            status=cls._get_batch_job_status_from_vertex_ai_batch_response(response),
            error_file_id=None,  # Vertex AI doesn't seem to have a direct equivalent
            output_file_id=cls._get_output_file_id_from_vertex_ai_batch_response(
                response
            ),
        )

    @classmethod
    def _get_batch_id_from_vertex_ai_batch_response(
        cls, response: VertexBatchPredictionResponse
    ) -> str:
        """
        Gets the batch id from the Vertex AI Batch response safely

        vertex response: `projects/510528649030/locations/us-central1/batchPredictionJobs/3814889423749775360`
        returns: `3814889423749775360`
        """
        _name = response.get("name", "")
        if not _name:
            return ""

        # Split by '/' and get the last part if it exists
        parts = _name.split("/")
        return parts[-1] if parts else _name

    @classmethod
    def _get_input_file_id_from_vertex_ai_batch_response(
        cls, response: VertexBatchPredictionResponse
    ) -> str:
        """
        Gets the input file id from the Vertex AI Batch response
        """
        input_file_id: str = ""
        input_config = response.get("inputConfig")
        if input_config is None:
            return input_file_id

        gcs_source = input_config.get("gcsSource")
        if gcs_source is None:
            return input_file_id

        uris = gcs_source.get("uris", "")
        if len(uris) == 0:
            return input_file_id

        return uris[0]

    @classmethod
    def _get_output_file_id_from_vertex_ai_batch_response(
        cls, response: VertexBatchPredictionResponse
    ) -> str:
        """
        Gets the output file id from the Vertex AI Batch response
        """
        output_file_id: str = ""
        output_config = response.get("outputConfig")
        if output_config is None:
            return output_file_id

        gcs_destination = output_config.get("gcsDestination")
        if gcs_destination is None:
            return output_file_id

        output_uri_prefix = gcs_destination.get("outputUriPrefix", "")
        return output_uri_prefix

    @classmethod
    def _get_batch_job_status_from_vertex_ai_batch_response(
        cls, response: VertexBatchPredictionResponse
    ) -> BatchJobStatus:
        """
        Gets the batch job status from the Vertex AI Batch response

        ref: https://cloud.google.com/vertex-ai/docs/reference/rest/v1/JobState
        """
        state_mapping: Dict[str, BatchJobStatus] = {
            "JOB_STATE_UNSPECIFIED": "failed",
            "JOB_STATE_QUEUED": "validating",
            "JOB_STATE_PENDING": "validating",
            "JOB_STATE_RUNNING": "in_progress",
            "JOB_STATE_SUCCEEDED": "completed",
            "JOB_STATE_FAILED": "failed",
            "JOB_STATE_CANCELLING": "cancelling",
            "JOB_STATE_CANCELLED": "cancelled",
            "JOB_STATE_PAUSED": "in_progress",
            "JOB_STATE_EXPIRED": "expired",
            "JOB_STATE_UPDATING": "in_progress",
            "JOB_STATE_PARTIALLY_SUCCEEDED": "completed",
        }

        vertex_state = response.get("state", "JOB_STATE_UNSPECIFIED")
        return state_mapping[vertex_state]

    @classmethod
    def _get_gcs_uri_prefix_from_file(cls, input_file_id: str) -> str:
        """
        Gets the gcs uri prefix from the input file id

        Example:
        input_file_id: "gs://litellm-testing-bucket/vtx_batch.jsonl"
        returns: "gs://litellm-testing-bucket"

        input_file_id: "gs://litellm-testing-bucket/batches/vtx_batch.jsonl"
        returns: "gs://litellm-testing-bucket/batches"
        """
        # Split the path and remove the filename
        path_parts = input_file_id.rsplit("/", 1)
        return path_parts[0]

    @classmethod
    def _get_model_from_gcs_file(cls, gcs_file_uri: str) -> str:
        """
        Extracts the model from the gcs file uri

        When files are uploaded using LiteLLM (/v1/files), the model is stored in the gcs file uri

        Why?
        - Because Vertex Requires the `model` param in create batch jobs request, but OpenAI does not require this


        gcs_file_uri format: gs://litellm-testing-bucket/litellm-vertex-files/publishers/google/models/gemini-1.5-flash-001/e9412502-2c91-42a6-8e61-f5c294cc0fc8
        returns: "publishers/google/models/gemini-1.5-flash-001"
        """
        from urllib.parse import unquote

        decoded_uri = unquote(gcs_file_uri)

        model_path = decoded_uri.split("publishers/")[1]
        parts = model_path.split("/")
        model = f"publishers/{'/'.join(parts[:3])}"
        return model
```

## File: `llms/vertex_ai/common_utils.py`
```
import re
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union, get_type_hints

import httpx

import litellm
from litellm import supports_response_schema, supports_system_messages, verbose_logger
from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.litellm_core_utils.prompt_templates.common_utils import unpack_defs
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.vertex_ai import PartType, Schema


class VertexAIError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[Union[Dict, httpx.Headers]] = None,
    ):
        super().__init__(message=message, status_code=status_code, headers=headers)


def get_supports_system_message(
    model: str, custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"]
) -> bool:
    try:
        _custom_llm_provider = custom_llm_provider
        if custom_llm_provider == "vertex_ai_beta":
            _custom_llm_provider = "vertex_ai"
        supports_system_message = supports_system_messages(
            model=model, custom_llm_provider=_custom_llm_provider
        )

        # Vertex Models called in the `/gemini` request/response format also support system messages
        if litellm.VertexGeminiConfig._is_model_gemini_spec_model(model):
            supports_system_message = True
    except Exception as e:
        verbose_logger.warning(
            "Unable to identify if system message supported. Defaulting to 'False'. Received error message - {}\nAdd it here - https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json".format(
                str(e)
            )
        )
        supports_system_message = False

    return supports_system_message


def get_supports_response_schema(
    model: str, custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"]
) -> bool:
    _custom_llm_provider = custom_llm_provider
    if custom_llm_provider == "vertex_ai_beta":
        _custom_llm_provider = "vertex_ai"

    _supports_response_schema = supports_response_schema(
        model=model, custom_llm_provider=_custom_llm_provider
    )

    return _supports_response_schema


from typing import Literal, Optional

all_gemini_url_modes = Literal[
    "chat", "embedding", "batch_embedding", "image_generation"
]


def _get_vertex_url(
    mode: all_gemini_url_modes,
    model: str,
    stream: Optional[bool],
    vertex_project: Optional[str],
    vertex_location: Optional[str],
    vertex_api_version: Literal["v1", "v1beta1"],
) -> Tuple[str, str]:
    url: Optional[str] = None
    endpoint: Optional[str] = None

    model = litellm.VertexGeminiConfig.get_model_for_vertex_ai_url(model=model)
    if mode == "chat":
        ### SET RUNTIME ENDPOINT ###
        endpoint = "generateContent"
        if stream is True:
            endpoint = "streamGenerateContent"
            if vertex_location == "global":
                url = f"https://aiplatform.googleapis.com/{vertex_api_version}/projects/{vertex_project}/locations/global/publishers/google/models/{model}:{endpoint}?alt=sse"
            else:
                url = f"https://{vertex_location}-aiplatform.googleapis.com/{vertex_api_version}/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model}:{endpoint}?alt=sse"
        else:
            if vertex_location == "global":
                url = f"https://aiplatform.googleapis.com/{vertex_api_version}/projects/{vertex_project}/locations/global/publishers/google/models/{model}:{endpoint}"
            else:
                url = f"https://{vertex_location}-aiplatform.googleapis.com/{vertex_api_version}/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model}:{endpoint}"

        # if model is only numeric chars then it's a fine tuned gemini model
        # model = 4965075652664360960
        # send to this url: url = f"https://{vertex_location}-aiplatform.googleapis.com/{version}/projects/{vertex_project}/locations/{vertex_location}/endpoints/{model}:{endpoint}"
        if model.isdigit():
            # It's a fine-tuned Gemini model
            url = f"https://{vertex_location}-aiplatform.googleapis.com/{vertex_api_version}/projects/{vertex_project}/locations/{vertex_location}/endpoints/{model}:{endpoint}"
            if stream is True:
                url += "?alt=sse"
    elif mode == "embedding":
        endpoint = "predict"
        url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model}:{endpoint}"
        if model.isdigit():
            # https://us-central1-aiplatform.googleapis.com/v1/projects/$PROJECT_ID/locations/us-central1/endpoints/$ENDPOINT_ID:predict
            url = f"https://{vertex_location}-aiplatform.googleapis.com/{vertex_api_version}/projects/{vertex_project}/locations/{vertex_location}/endpoints/{model}:{endpoint}"
    elif mode == "image_generation":
        endpoint = "predict"
        url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model}:{endpoint}"
        if model.isdigit():
            url = f"https://{vertex_location}-aiplatform.googleapis.com/{vertex_api_version}/projects/{vertex_project}/locations/{vertex_location}/endpoints/{model}:{endpoint}"
    if not url or not endpoint:
        raise ValueError(f"Unable to get vertex url/endpoint for mode: {mode}")
    return url, endpoint


def _get_gemini_url(
    mode: all_gemini_url_modes,
    model: str,
    stream: Optional[bool],
    gemini_api_key: Optional[str],
) -> Tuple[str, str]:
    _gemini_model_name = "models/{}".format(model)
    if mode == "chat":
        endpoint = "generateContent"
        if stream is True:
            endpoint = "streamGenerateContent"
            url = "https://generativelanguage.googleapis.com/v1beta/{}:{}?key={}&alt=sse".format(
                _gemini_model_name, endpoint, gemini_api_key
            )
        else:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/{}:{}?key={}".format(
                    _gemini_model_name, endpoint, gemini_api_key
                )
            )
    elif mode == "embedding":
        endpoint = "embedContent"
        url = "https://generativelanguage.googleapis.com/v1beta/{}:{}?key={}".format(
            _gemini_model_name, endpoint, gemini_api_key
        )
    elif mode == "batch_embedding":
        endpoint = "batchEmbedContents"
        url = "https://generativelanguage.googleapis.com/v1beta/{}:{}?key={}".format(
            _gemini_model_name, endpoint, gemini_api_key
        )
    elif mode == "image_generation":
        raise ValueError(
            "LiteLLM's `gemini/` route does not support image generation yet. Let us know if you need this feature by opening an issue at https://github.com/BerriAI/litellm/issues"
        )

    return url, endpoint


def _check_text_in_content(parts: List[PartType]) -> bool:
    """
    check that user_content has 'text' parameter.
        - Known Vertex Error: Unable to submit request because it must have a text parameter.
        - 'text' param needs to be len > 0
        - Relevant Issue: https://github.com/BerriAI/litellm/issues/5515
    """
    has_text_param = False
    for part in parts:
        if "text" in part and part.get("text"):
            has_text_param = True

    return has_text_param


def _build_vertex_schema(parameters: dict, add_property_ordering: bool = False):
    """
    This is a modified version of https://github.com/google-gemini/generative-ai-python/blob/8f77cc6ac99937cd3a81299ecf79608b91b06bbb/google/generativeai/types/content_types.py#L419

    Updates the input parameters, removing extraneous fields, adjusting types, unwinding $defs, and adding propertyOrdering if specified, returning the updated parameters.

    Parameters:
        parameters: dict - the json schema to build from
        add_property_ordering: bool - whether to add propertyOrdering to the schema. This is only applicable to schemas for structured outputs. See
          set_schema_property_ordering for more details.
    Returns:
        parameters: dict - the input parameters, modified in place
    """
    # Get valid fields from Schema TypedDict
    valid_schema_fields = set(get_type_hints(Schema).keys())

    defs = parameters.pop("$defs", {})
    # flatten the defs
    for name, value in defs.items():
        unpack_defs(value, defs)
    unpack_defs(parameters, defs)

    # 5. Nullable fields:
    #     * https://github.com/pydantic/pydantic/issues/1270
    #     * https://stackoverflow.com/a/58841311
    #     * https://github.com/pydantic/pydantic/discussions/4872
    convert_anyof_null_to_nullable(parameters)

    # Handle empty items objects
    process_items(parameters)
    add_object_type(parameters)
    # Postprocessing
    # Filter out fields that don't exist in Schema

    parameters = filter_schema_fields(parameters, valid_schema_fields)

    if add_property_ordering:
        set_schema_property_ordering(parameters)

    return parameters


def _filter_anyof_fields(schema_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    When anyof is present, only keep the anyof field and its contents - otherwise VertexAI will throw an error - https://github.com/BerriAI/litellm/issues/11164
    Filter out other fields in the same dict.

    E.g. {"anyOf": [{"type": "string"}, {"type": "null"}], "default": "test"} -> {"anyOf": [{"type": "string"}, {"type": "null"}]}

    Case 2: If additional metadata is present, try to keep it
    E.g. {"anyOf": [{"type": "string"}, {"type": "null"}], "default": "test", "title": "test"} -> {"anyOf": [{"type": "string", "title": "test"}, {"type": "null", "title": "test"}]}
    """
    title = schema_dict.get("title", None)
    description = schema_dict.get("description", None)

    if isinstance(schema_dict, dict) and schema_dict.get("anyOf"):
        any_of = schema_dict["anyOf"]
        if (
            (title or description)
            and isinstance(any_of, list)
            and all(isinstance(item, dict) for item in any_of)
        ):
            for item in any_of:
                if title:
                    item["title"] = title
                if description:
                    item["description"] = description
            return {"anyOf": any_of}
        else:
            return schema_dict
    return schema_dict


def process_items(schema, depth=0):
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        raise ValueError(
            f"Max depth of {DEFAULT_MAX_RECURSE_DEPTH} exceeded while processing schema. Please check the schema for excessive nesting."
        )
    if isinstance(schema, dict):
        if "items" in schema and schema["items"] == {}:
            schema["items"] = {"type": "object"}
        for key, value in schema.items():
            if isinstance(value, dict):
                process_items(value, depth + 1)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        process_items(item, depth + 1)


def set_schema_property_ordering(
    schema: Dict[str, Any], depth: int = 0
) -> Dict[str, Any]:
    """
    vertex ai and generativeai apis order output of fields alphabetically, unless you specify the order.
    python dicts retain order, so we just use that. Note that this field only applies to structured outputs, and not tools.
    Function tools are not afflicted by the same alphabetical ordering issue, (the order of keys returned seems to be arbitrary, up to the model)
    https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.cachedContents#Schema.FIELDS.property_ordering

    Args:
        schema: The schema dictionary to process
        depth: Current recursion depth to prevent infinite loops
    """
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        raise ValueError(
            f"Max depth of {DEFAULT_MAX_RECURSE_DEPTH} exceeded while processing schema. Please check the schema for excessive nesting."
        )

    if "properties" in schema and isinstance(schema["properties"], dict):
        # retain propertyOrdering as an escape hatch if user already specifies it
        if "propertyOrdering" not in schema:
            schema["propertyOrdering"] = [k for k, v in schema["properties"].items()]
        for k, v in schema["properties"].items():
            set_schema_property_ordering(v, depth + 1)
    if "items" in schema:
        set_schema_property_ordering(schema["items"], depth + 1)
    return schema


def filter_schema_fields(
    schema_dict: Dict[str, Any], valid_fields: Set[str], processed=None
) -> Dict[str, Any]:
    """
    Recursively filter a schema dictionary to keep only valid fields.
    """
    if processed is None:
        processed = set()

    # Handle circular references
    schema_id = id(schema_dict)
    if schema_id in processed:
        return schema_dict
    processed.add(schema_id)

    if not isinstance(schema_dict, dict):
        return schema_dict

    result = {}
    schema_dict = _filter_anyof_fields(schema_dict)
    for key, value in schema_dict.items():
        if key not in valid_fields:
            continue

        if key == "properties" and isinstance(value, dict):
            result[key] = {
                k: filter_schema_fields(v, valid_fields, processed)
                for k, v in value.items()
            }
        elif key == "format":
            if value in {"enum", "date-time"}:
                result[key] = value
            else:
                continue
        elif key == "items" and isinstance(value, dict):
            result[key] = filter_schema_fields(value, valid_fields, processed)
        elif key == "anyOf" and isinstance(value, list):
            result[key] = [
                filter_schema_fields(item, valid_fields, processed) for item in value  # type: ignore
            ]
        else:
            result[key] = value

    return result


def convert_anyof_null_to_nullable(schema, depth=0):
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        raise ValueError(
            f"Max depth of {DEFAULT_MAX_RECURSE_DEPTH} exceeded while processing schema. Please check the schema for excessive nesting."
        )
    """ Converts null objects within anyOf by removing them and adding nullable to all remaining objects """
    anyof = schema.get("anyOf", None)
    if anyof is not None:
        contains_null = False
        for atype in anyof:
            if atype == {"type": "null"}:
                # remove null type
                anyof.remove(atype)
                contains_null = True
            elif "type" not in atype and len(atype) == 0:
                # Handle empty object case
                atype["type"] = "object"

        if len(anyof) == 0:
            # Edge case: response schema with only null type present is invalid in Vertex AI
            raise ValueError(
                "Invalid input: AnyOf schema with only null type is not supported. "
                "Please provide a non-null type."
            )

        if contains_null:
            # set all types to nullable following guidance found here: https://cloud.google.com/vertex-ai/generative-ai/docs/samples/generativeaionvertexai-gemini-controlled-generation-response-schema-3#generativeaionvertexai_gemini_controlled_generation_response_schema_3-python
            for atype in anyof:
                # Remove items field if type is array and items is empty
                if (
                    atype.get("type") == "array"
                    and "items" in atype
                    and not atype["items"]
                ):
                    atype.pop("items")
                atype["nullable"] = True

    properties = schema.get("properties", None)
    if properties is not None:
        for name, value in properties.items():
            convert_anyof_null_to_nullable(value, depth=depth + 1)

    items = schema.get("items", None)
    if items is not None:
        convert_anyof_null_to_nullable(items, depth=depth + 1)


def add_object_type(schema):
    properties = schema.get("properties", None)
    if properties is not None:
        if "required" in schema and schema["required"] is None:
            schema.pop("required", None)
        schema["type"] = "object"
        for name, value in properties.items():
            add_object_type(value)

    items = schema.get("items", None)
    if items is not None:
        add_object_type(items)


def strip_field(schema, field_name: str):
    schema.pop(field_name, None)

    properties = schema.get("properties", None)
    if properties is not None:
        for name, value in properties.items():
            strip_field(value, field_name)

    items = schema.get("items", None)
    if items is not None:
        strip_field(items, field_name)


def _convert_vertex_datetime_to_openai_datetime(vertex_datetime: str) -> int:
    """
    Converts a Vertex AI datetime string to an OpenAI datetime integer

    vertex_datetime: str = "2024-12-04T21:53:12.120184Z"
    returns: int = 1722729192
    """
    from datetime import datetime

    # Parse the ISO format string to datetime object
    dt = datetime.strptime(vertex_datetime, "%Y-%m-%dT%H:%M:%S.%fZ")
    # Convert to Unix timestamp (seconds since epoch)
    return int(dt.timestamp())


def get_vertex_project_id_from_url(url: str) -> Optional[str]:
    """
    Get the vertex project id from the url

    `https://${LOCATION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL_ID}:streamGenerateContent`
    """
    match = re.search(r"/projects/([^/]+)", url)
    return match.group(1) if match else None


def get_vertex_location_from_url(url: str) -> Optional[str]:
    """
    Get the vertex location from the url

    `https://${LOCATION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL_ID}:streamGenerateContent`
    """
    match = re.search(r"/locations/([^/]+)", url)
    return match.group(1) if match else None


def replace_project_and_location_in_route(
    requested_route: str, vertex_project: str, vertex_location: str
) -> str:
    """
    Replace project and location values in the route with the provided values
    """
    # Replace project and location values while keeping route structure
    modified_route = re.sub(
        r"/projects/[^/]+/locations/[^/]+/",
        f"/projects/{vertex_project}/locations/{vertex_location}/",
        requested_route,
    )
    return modified_route


def construct_target_url(
    base_url: str,
    requested_route: str,
    vertex_location: Optional[str],
    vertex_project: Optional[str],
) -> httpx.URL:
    """
    Allow user to specify their own project id / location.

    If missing, use defaults

    Handle cachedContent scenario - https://github.com/BerriAI/litellm/issues/5460

    Constructed Url:
    POST https://LOCATION-aiplatform.googleapis.com/{version}/projects/PROJECT_ID/locations/LOCATION/cachedContents
    """

    new_base_url = httpx.URL(base_url)
    if "locations" in requested_route:  # contains the target project id + location
        if vertex_project and vertex_location:
            requested_route = replace_project_and_location_in_route(
                requested_route, vertex_project, vertex_location
            )
        return new_base_url.copy_with(path=requested_route)

    """
    - Add endpoint version (e.g. v1beta for cachedContent, v1 for rest)
    - Add default project id
    - Add default location
    """
    vertex_version: Literal["v1", "v1beta1"] = "v1"
    if "cachedContent" in requested_route:
        vertex_version = "v1beta1"

    base_requested_route = "{}/projects/{}/locations/{}".format(
        vertex_version, vertex_project, vertex_location
    )

    updated_requested_route = "/" + base_requested_route + requested_route

    updated_url = new_base_url.copy_with(path=updated_requested_route)
    return updated_url


def is_global_only_vertex_model(model: str) -> bool:
    """
    Check if a model is only available in the global region.

    Args:
        model: The model name to check

    Returns:
        True if the model is only available in global region, False otherwise
    """
    from litellm.utils import get_supported_regions

    supported_regions = get_supported_regions(
        model=model, custom_llm_provider="vertex_ai"
    )
    if supported_regions is None:
        return False
    return "global" in supported_regions
```

## File: `llms/vertex_ai/context_caching/transformation.py`
```
"""
Transformation logic for context caching. 

Why separate file? Make it easy to see how transformation works
"""

import re
from typing import List, Optional, Tuple

from litellm.types.llms.openai import AllMessageValues
from litellm.types.llms.vertex_ai import CachedContentRequestBody
from litellm.utils import is_cached_message

from ..common_utils import get_supports_system_message
from ..gemini.transformation import (
    _gemini_convert_messages_with_history,
    _transform_system_message,
)


def get_first_continuous_block_idx(
    filtered_messages: List[Tuple[int, AllMessageValues]]  # (idx, message)
) -> int:
    """
    Find the array index that ends the first continuous sequence of message blocks.

    Args:
        filtered_messages: List of tuples containing (index, message) pairs

    Returns:
        int: The array index where the first continuous sequence ends
    """
    if not filtered_messages:
        return -1

    if len(filtered_messages) == 1:
        return 0

    current_value = filtered_messages[0][0]

    # Search forward through the array indices
    for i in range(1, len(filtered_messages)):
        if filtered_messages[i][0] != current_value + 1:
            return i - 1
        current_value = filtered_messages[i][0]

    # If we made it through the whole list, return the last index
    return len(filtered_messages) - 1


def extract_ttl_from_cached_messages(messages: List[AllMessageValues]) -> Optional[str]:
    """
    Extract TTL from cached messages. Returns the first valid TTL found.
    
    Args:
        messages: List of messages to extract TTL from
        
    Returns:
        Optional[str]: TTL string in format "3600s" or None if not found/invalid
    """
    for message in messages:
        if not is_cached_message(message):
            continue
            
        content = message.get("content")
        if not content or isinstance(content, str):
            continue
            
        for content_item in content:
            # Type check to ensure content_item is a dictionary before calling .get()
            if not isinstance(content_item, dict):
                continue
                
            cache_control = content_item.get("cache_control")
            if not cache_control or not isinstance(cache_control, dict):
                continue
                
            if cache_control.get("type") != "ephemeral":
                continue
                
            ttl = cache_control.get("ttl")
            if ttl and _is_valid_ttl_format(ttl):
                return str(ttl)
    
    return None


def _is_valid_ttl_format(ttl: str) -> bool:
    """
    Validate TTL format. Should be a string ending with 's' for seconds.
    Examples: "3600s", "7200s", "1.5s"
    
    Args:
        ttl: TTL string to validate
        
    Returns:
        bool: True if valid format, False otherwise
    """
    if not isinstance(ttl, str):
        return False
    
    # TTL should end with 's' and contain a valid number before it
    pattern = r'^([0-9]*\.?[0-9]+)s$'
    match = re.match(pattern, ttl)
    
    if not match:
        return False
    
    try:
        # Ensure the numeric part is valid and positive
        numeric_part = float(match.group(1))
        return numeric_part > 0
    except ValueError:
        return False


def separate_cached_messages(
    messages: List[AllMessageValues],
) -> Tuple[List[AllMessageValues], List[AllMessageValues]]:
    """
    Returns separated cached and non-cached messages.

    Args:
        messages: List of messages to be separated.

    Returns:
        Tuple containing:
        - cached_messages: List of cached messages.
        - non_cached_messages: List of non-cached messages.
    """
    cached_messages: List[AllMessageValues] = []
    non_cached_messages: List[AllMessageValues] = []

    # Extract cached messages and their indices
    filtered_messages: List[Tuple[int, AllMessageValues]] = []
    for idx, message in enumerate(messages):
        if is_cached_message(message=message):
            filtered_messages.append((idx, message))

    # Validate only one block of continuous cached messages
    last_continuous_block_idx = get_first_continuous_block_idx(filtered_messages)
    # Separate messages based on the block of cached messages
    if filtered_messages and last_continuous_block_idx is not None:
        first_cached_idx = filtered_messages[0][0]
        last_cached_idx = filtered_messages[last_continuous_block_idx][0]

        cached_messages = messages[first_cached_idx : last_cached_idx + 1]
        non_cached_messages = (
            messages[:first_cached_idx] + messages[last_cached_idx + 1 :]
        )
    else:
        non_cached_messages = messages

    return cached_messages, non_cached_messages


def transform_openai_messages_to_gemini_context_caching(
    model: str, messages: List[AllMessageValues], cache_key: str
) -> CachedContentRequestBody:
    # Extract TTL from cached messages BEFORE system message transformation
    ttl = extract_ttl_from_cached_messages(messages)
    
    supports_system_message = get_supports_system_message(
        model=model, custom_llm_provider="gemini"
    )

    transformed_system_messages, new_messages = _transform_system_message(
        supports_system_message=supports_system_message, messages=messages
    )

    transformed_messages = _gemini_convert_messages_with_history(messages=new_messages)
    
    data = CachedContentRequestBody(
        contents=transformed_messages,
        model="models/{}".format(model),
        displayName=cache_key,
    )
    
    # Add TTL if present and valid
    if ttl:
        data["ttl"] = ttl
    
    if transformed_system_messages is not None:
        data["system_instruction"] = transformed_system_messages

    return data
```

## File: `llms/vertex_ai/context_caching/vertex_ai_context_caching.py`
```
from typing import List, Literal, Optional, Tuple, Union

import httpx

import litellm
from litellm.caching.caching import Cache, LiteLLMCacheType
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.llms.openai.openai import AllMessageValues
from litellm.types.llms.vertex_ai import (
    CachedContentListAllResponseBody,
    VertexAICachedContentResponseObject,
)

from ..common_utils import VertexAIError
from ..vertex_llm_base import VertexBase
from .transformation import (
    separate_cached_messages,
    transform_openai_messages_to_gemini_context_caching,
)

local_cache_obj = Cache(
    type=LiteLLMCacheType.LOCAL
)  # only used for calling 'get_cache_key' function


class ContextCachingEndpoints(VertexBase):
    """
    Covers context caching endpoints for Vertex AI + Google AI Studio

    v0: covers Google AI Studio
    """

    def __init__(self) -> None:
        pass

    def _get_token_and_url_context_caching(
        self,
        gemini_api_key: Optional[str],
        custom_llm_provider: Literal["gemini"],
        api_base: Optional[str],
    ) -> Tuple[Optional[str], str]:
        """
        Internal function. Returns the token and url for the call.

        Handles logic if it's google ai studio vs. vertex ai.

        Returns
            token, url
        """
        if custom_llm_provider == "gemini":
            auth_header = None
            endpoint = "cachedContents"
            url = "https://generativelanguage.googleapis.com/v1beta/{}?key={}".format(
                endpoint, gemini_api_key
            )

        else:
            raise NotImplementedError

        return self._check_custom_proxy(
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            gemini_api_key=gemini_api_key,
            endpoint=endpoint,
            stream=None,
            auth_header=auth_header,
            url=url,
        )

    def check_cache(
        self,
        cache_key: str,
        client: HTTPHandler,
        headers: dict,
        api_key: str,
        api_base: Optional[str],
        logging_obj: Logging,
    ) -> Optional[str]:
        """
        Checks if content already cached.

        Currently, checks cache list, for cache key == displayName, since Google doesn't let us set the name of the cache (their API docs are out of sync with actual implementation).

        Returns
        - cached_content_name - str - cached content name stored on google. (if found.)
        OR
        - None
        """

        _, url = self._get_token_and_url_context_caching(
            gemini_api_key=api_key,
            custom_llm_provider="gemini",
            api_base=api_base,
        )
        try:
            ## LOGGING
            logging_obj.pre_call(
                input="",
                api_key="",
                additional_args={
                    "complete_input_dict": {},
                    "api_base": url,
                    "headers": headers,
                },
            )

            resp = client.get(url=url, headers=headers)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                return None
            raise VertexAIError(
                status_code=e.response.status_code, message=e.response.text
            )
        except Exception as e:
            raise VertexAIError(status_code=500, message=str(e))
        raw_response = resp.json()
        logging_obj.post_call(original_response=raw_response)

        if "cachedContents" not in raw_response:
            return None

        all_cached_items = CachedContentListAllResponseBody(**raw_response)

        if "cachedContents" not in all_cached_items:
            return None

        for cached_item in all_cached_items["cachedContents"]:
            display_name = cached_item.get("displayName")
            if display_name is not None and display_name == cache_key:
                return cached_item.get("name")

        return None

    async def async_check_cache(
        self,
        cache_key: str,
        client: AsyncHTTPHandler,
        headers: dict,
        api_key: str,
        api_base: Optional[str],
        logging_obj: Logging,
    ) -> Optional[str]:
        """
        Checks if content already cached.

        Currently, checks cache list, for cache key == displayName, since Google doesn't let us set the name of the cache (their API docs are out of sync with actual implementation).

        Returns
        - cached_content_name - str - cached content name stored on google. (if found.)
        OR
        - None
        """

        _, url = self._get_token_and_url_context_caching(
            gemini_api_key=api_key,
            custom_llm_provider="gemini",
            api_base=api_base,
        )
        try:
            ## LOGGING
            logging_obj.pre_call(
                input="",
                api_key="",
                additional_args={
                    "complete_input_dict": {},
                    "api_base": url,
                    "headers": headers,
                },
            )

            resp = await client.get(url=url, headers=headers)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                return None
            raise VertexAIError(
                status_code=e.response.status_code, message=e.response.text
            )
        except Exception as e:
            raise VertexAIError(status_code=500, message=str(e))
        raw_response = resp.json()
        logging_obj.post_call(original_response=raw_response)

        if "cachedContents" not in raw_response:
            return None

        all_cached_items = CachedContentListAllResponseBody(**raw_response)

        if "cachedContents" not in all_cached_items:
            return None

        for cached_item in all_cached_items["cachedContents"]:
            display_name = cached_item.get("displayName")
            if display_name is not None and display_name == cache_key:
                return cached_item.get("name")

        return None

    def check_and_create_cache(
        self,
        messages: List[AllMessageValues],  # receives openai format messages
        optional_params: dict,  # cache the tools if present, in case cache content exists in messages
        api_key: str,
        api_base: Optional[str],
        model: str,
        client: Optional[HTTPHandler],
        timeout: Optional[Union[float, httpx.Timeout]],
        logging_obj: Logging,
        extra_headers: Optional[dict] = None,
        cached_content: Optional[str] = None,
    ) -> Tuple[List[AllMessageValues], dict, Optional[str]]:
        """
        Receives
        - messages: List of dict - messages in the openai format

        Returns
        - messages - List[dict] - filtered list of messages in the openai format.
        - cached_content - str - the cache content id, to be passed in the gemini request body

        Follows - https://ai.google.dev/api/caching#request-body
        """
        if cached_content is not None:
            return messages, optional_params, cached_content

        cached_messages, non_cached_messages = separate_cached_messages(
            messages=messages
        )

        if len(cached_messages) == 0:
            return messages, optional_params, None

        tools = optional_params.pop("tools", None)

        ## AUTHORIZATION ##
        token, url = self._get_token_and_url_context_caching(
            gemini_api_key=api_key,
            custom_llm_provider="gemini",
            api_base=api_base,
        )

        headers = {
            "Content-Type": "application/json",
        }
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        if extra_headers is not None:
            headers.update(extra_headers)

        if client is None or not isinstance(client, HTTPHandler):
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            client = HTTPHandler(**_params)  # type: ignore
        else:
            client = client

        ## CHECK IF CACHED ALREADY
        generated_cache_key = local_cache_obj.get_cache_key(
            messages=cached_messages, tools=tools
        )
        google_cache_name = self.check_cache(
            cache_key=generated_cache_key,
            client=client,
            headers=headers,
            api_key=api_key,
            api_base=api_base,
            logging_obj=logging_obj,
        )
        if google_cache_name:
            return non_cached_messages, optional_params, google_cache_name

        ## TRANSFORM REQUEST
        cached_content_request_body = (
            transform_openai_messages_to_gemini_context_caching(
                model=model, messages=cached_messages, cache_key=generated_cache_key
            )
        )

        cached_content_request_body["tools"] = tools

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": cached_content_request_body,
                "api_base": url,
                "headers": headers,
            },
        )

        try:
            response = client.post(
                url=url, headers=headers, json=cached_content_request_body  # type: ignore
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise VertexAIError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException:
            raise VertexAIError(status_code=408, message="Timeout error occurred.")

        raw_response_cached = response.json()
        cached_content_response_obj = VertexAICachedContentResponseObject(
            name=raw_response_cached.get("name"), model=raw_response_cached.get("model")
        )
        return (
            non_cached_messages,
            optional_params,
            cached_content_response_obj["name"],
        )

    async def async_check_and_create_cache(
        self,
        messages: List[AllMessageValues],  # receives openai format messages
        optional_params: dict,  # cache the tools if present, in case cache content exists in messages
        api_key: str,
        api_base: Optional[str],
        model: str,
        client: Optional[AsyncHTTPHandler],
        timeout: Optional[Union[float, httpx.Timeout]],
        logging_obj: Logging,
        extra_headers: Optional[dict] = None,
        cached_content: Optional[str] = None,
    ) -> Tuple[List[AllMessageValues], dict, Optional[str]]:
        """
        Receives
        - messages: List of dict - messages in the openai format

        Returns
        - messages - List[dict] - filtered list of messages in the openai format.
        - cached_content - str - the cache content id, to be passed in the gemini request body

        Follows - https://ai.google.dev/api/caching#request-body
        """
        if cached_content is not None:
            return messages, optional_params, cached_content

        cached_messages, non_cached_messages = separate_cached_messages(
            messages=messages
        )

        if len(cached_messages) == 0:
            return messages, optional_params, None

        tools = optional_params.pop("tools", None)

        ## AUTHORIZATION ##
        token, url = self._get_token_and_url_context_caching(
            gemini_api_key=api_key,
            custom_llm_provider="gemini",
            api_base=api_base,
        )

        headers = {
            "Content-Type": "application/json",
        }
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        if extra_headers is not None:
            headers.update(extra_headers)

        if client is None or not isinstance(client, AsyncHTTPHandler):
            client = get_async_httpx_client(
                params={"timeout": timeout}, llm_provider=litellm.LlmProviders.VERTEX_AI
            )
        else:
            client = client

        ## CHECK IF CACHED ALREADY
        generated_cache_key = local_cache_obj.get_cache_key(
            messages=cached_messages, tools=tools
        )
        google_cache_name = await self.async_check_cache(
            cache_key=generated_cache_key,
            client=client,
            headers=headers,
            api_key=api_key,
            api_base=api_base,
            logging_obj=logging_obj,
        )

        if google_cache_name:
            return non_cached_messages, optional_params, google_cache_name

        ## TRANSFORM REQUEST
        cached_content_request_body = (
            transform_openai_messages_to_gemini_context_caching(
                model=model, messages=cached_messages, cache_key=generated_cache_key
            )
        )

        cached_content_request_body["tools"] = tools

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": cached_content_request_body,
                "api_base": url,
                "headers": headers,
            },
        )

        try:
            response = await client.post(
                url=url, headers=headers, json=cached_content_request_body  # type: ignore
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise VertexAIError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException:
            raise VertexAIError(status_code=408, message="Timeout error occurred.")

        raw_response_cached = response.json()
        cached_content_response_obj = VertexAICachedContentResponseObject(
            name=raw_response_cached.get("name"), model=raw_response_cached.get("model")
        )
        return (
            non_cached_messages,
            optional_params,
            cached_content_response_obj["name"],
        )

    def get_cache(self):
        pass

    async def async_get_cache(self):
        pass
```

## File: `llms/vertex_ai/cost_calculator.py`
```
# What is this?
## Cost calculation for Google AI Studio / Vertex AI models
from typing import Literal, Optional, Tuple, Union

import litellm
from litellm import verbose_logger
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    _is_above_128k,
    generic_cost_per_token,
)
from litellm.types.utils import ModelInfo, Usage

"""
Gemini pricing covers: 
- token
- image
- audio
- video
"""

"""
Vertex AI -> character based pricing 

Google AI Studio -> token based pricing
"""

models_without_dynamic_pricing = ["gemini-1.0-pro", "gemini-pro", "gemini-2"]


def cost_router(
    model: str,
    custom_llm_provider: str,
    call_type: Union[Literal["embedding", "aembedding"], str],
) -> Literal["cost_per_character", "cost_per_token"]:
    """
    Route the cost calc to the right place, based on model/call_type/etc.

    Returns
        - str, the specific google cost calc function it should route to.
    """
    if custom_llm_provider == "vertex_ai" and (
        "claude" in model
        or "llama" in model
        or "mistral" in model
        or "jamba" in model
        or "codestral" in model
    ):
        return "cost_per_token"
    elif custom_llm_provider == "vertex_ai" and (
        call_type == "embedding" or call_type == "aembedding"
    ):
        return "cost_per_token"
    elif custom_llm_provider == "vertex_ai" and ("gemini-2" in model):
        return "cost_per_token"
    return "cost_per_character"


def cost_per_character(
    model: str,
    custom_llm_provider: str,
    usage: Usage,
    prompt_characters: Optional[float] = None,
    completion_characters: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Calculates the cost per character for a given VertexAI model, input messages, and response object.

    Input:
        - model: str, the model name without provider prefix
        - custom_llm_provider: str, "vertex_ai-*"
        - prompt_characters: float, the number of input characters
        - completion_characters: float, the number of output characters

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd

    Raises:
        Exception if model requires >128k pricing, but model cost not mapped
    """
    model_info = litellm.get_model_info(
        model=model, custom_llm_provider=custom_llm_provider
    )

    ## GET MODEL INFO
    model_info = litellm.get_model_info(
        model=model, custom_llm_provider=custom_llm_provider
    )

    ## CALCULATE INPUT COST
    if prompt_characters is None:
        prompt_cost, _ = cost_per_token(
            model=model,
            custom_llm_provider=custom_llm_provider,
            usage=usage,
        )
    else:
        try:
            if (
                _is_above_128k(tokens=prompt_characters * 4)  # 1 token = 4 char
                and model not in models_without_dynamic_pricing
            ):
                ## check if character pricing, else default to token pricing
                assert (
                    "input_cost_per_character_above_128k_tokens" in model_info
                    and model_info["input_cost_per_character_above_128k_tokens"]
                    is not None
                ), "model info for model={} does not have 'input_cost_per_character_above_128k_tokens'-pricing for > 128k tokens\nmodel_info={}".format(
                    model, model_info
                )
                prompt_cost = (
                    prompt_characters
                    * model_info["input_cost_per_character_above_128k_tokens"]
                )
            else:
                assert (
                    "input_cost_per_character" in model_info
                    and model_info["input_cost_per_character"] is not None
                ), "model info for model={} does not have 'input_cost_per_character'-pricing\nmodel_info={}".format(
                    model, model_info
                )
                prompt_cost = prompt_characters * model_info["input_cost_per_character"]
        except Exception as e:
            verbose_logger.debug(
                "litellm.litellm_core_utils.llm_cost_calc.google.py::cost_per_character(): Exception occured - {}\nDefaulting to None".format(
                    str(e)
                )
            )
            prompt_cost, _ = cost_per_token(
                model=model,
                custom_llm_provider=custom_llm_provider,
                usage=usage,
            )

    ## CALCULATE OUTPUT COST
    if completion_characters is None:
        _, completion_cost = cost_per_token(
            model=model,
            custom_llm_provider=custom_llm_provider,
            usage=usage,
        )
    else:
        completion_tokens = usage.completion_tokens
        try:
            if (
                _is_above_128k(tokens=completion_characters * 4)  # 1 token = 4 char
                and model not in models_without_dynamic_pricing
            ):
                assert (
                    "output_cost_per_character_above_128k_tokens" in model_info
                    and model_info["output_cost_per_character_above_128k_tokens"]
                    is not None
                ), "model info for model={} does not have 'output_cost_per_character_above_128k_tokens' pricing\nmodel_info={}".format(
                    model, model_info
                )
                completion_cost = (
                    completion_tokens
                    * model_info["output_cost_per_character_above_128k_tokens"]
                )
            else:
                assert (
                    "output_cost_per_character" in model_info
                    and model_info["output_cost_per_character"] is not None
                ), "model info for model={} does not have 'output_cost_per_character'-pricing\nmodel_info={}".format(
                    model, model_info
                )
                completion_cost = (
                    completion_characters * model_info["output_cost_per_character"]
                )
        except Exception as e:
            verbose_logger.debug(
                "litellm.litellm_core_utils.llm_cost_calc.google.py::cost_per_character(): Exception occured - {}\nDefaulting to None".format(
                    str(e)
                )
            )
            _, completion_cost = cost_per_token(
                model=model,
                custom_llm_provider=custom_llm_provider,
                usage=usage,
            )

    return prompt_cost, completion_cost


def _handle_128k_pricing(
    model_info: ModelInfo,
    usage: Usage,
) -> Tuple[float, float]:
    ## CALCULATE INPUT COST
    input_cost_per_token_above_128k_tokens = model_info.get(
        "input_cost_per_token_above_128k_tokens"
    )
    output_cost_per_token_above_128k_tokens = model_info.get(
        "output_cost_per_token_above_128k_tokens"
    )

    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens

    if (
        _is_above_128k(tokens=prompt_tokens)
        and input_cost_per_token_above_128k_tokens is not None
    ):
        prompt_cost = prompt_tokens * input_cost_per_token_above_128k_tokens
    else:
        prompt_cost = prompt_tokens * model_info["input_cost_per_token"]

    ## CALCULATE OUTPUT COST
    output_cost_per_token_above_128k_tokens = model_info.get(
        "output_cost_per_token_above_128k_tokens"
    )
    if (
        _is_above_128k(tokens=completion_tokens)
        and output_cost_per_token_above_128k_tokens is not None
    ):
        completion_cost = completion_tokens * output_cost_per_token_above_128k_tokens
    else:
        completion_cost = completion_tokens * model_info["output_cost_per_token"]

    return prompt_cost, completion_cost


def cost_per_token(
    model: str,
    custom_llm_provider: str,
    usage: Usage,
) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - custom_llm_provider: str, either "vertex_ai-*" or "gemini"
        - prompt_tokens: float, the number of input tokens
        - completion_tokens: float, the number of output tokens

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd

    Raises:
        Exception if model requires >128k pricing, but model cost not mapped
    """

    ## GET MODEL INFO
    model_info = litellm.get_model_info(
        model=model, custom_llm_provider=custom_llm_provider
    )

    ## HANDLE 128k+ PRICING
    input_cost_per_token_above_128k_tokens = model_info.get(
        "input_cost_per_token_above_128k_tokens"
    )
    output_cost_per_token_above_128k_tokens = model_info.get(
        "output_cost_per_token_above_128k_tokens"
    )
    if (
        input_cost_per_token_above_128k_tokens is not None
        or output_cost_per_token_above_128k_tokens is not None
    ):
        return _handle_128k_pricing(
            model_info=model_info,
            usage=usage,
        )

    return generic_cost_per_token(
        model=model,
        custom_llm_provider=custom_llm_provider,
        usage=usage,
    )
```

## File: `llms/vertex_ai/files/handler.py`
```
import asyncio
from typing import Any, Coroutine, Optional, Union

import httpx

from litellm import LlmProviders
from litellm.integrations.gcs_bucket.gcs_bucket_base import (
    GCSBucketBase,
    GCSLoggingConfig,
)
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.openai import CreateFileRequest, OpenAIFileObject
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES

from .transformation import VertexAIJsonlFilesTransformation

vertex_ai_files_transformation = VertexAIJsonlFilesTransformation()


class VertexAIFilesHandler(GCSBucketBase):
    """
    Handles Calling VertexAI in OpenAI Files API format v1/files/*

    This implementation uploads files on GCS Buckets
    """

    def __init__(self):
        super().__init__()
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=LlmProviders.VERTEX_AI,
        )

    async def async_create_file(
        self,
        create_file_data: CreateFileRequest,
        api_base: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> OpenAIFileObject:
        gcs_logging_config: GCSLoggingConfig = await self.get_gcs_logging_config(
            kwargs={}
        )
        headers = await self.construct_request_headers(
            vertex_instance=gcs_logging_config["vertex_instance"],
            service_account_json=gcs_logging_config["path_service_account"],
        )
        bucket_name = gcs_logging_config["bucket_name"]
        (
            logging_payload,
            object_name,
        ) = vertex_ai_files_transformation.transform_openai_file_content_to_vertex_ai_file_content(
            openai_file_content=create_file_data.get("file")
        )
        gcs_upload_response = await self._log_json_data_on_gcs(
            headers=headers,
            bucket_name=bucket_name,
            object_name=object_name,
            logging_payload=logging_payload,
        )

        return vertex_ai_files_transformation.transform_gcs_bucket_response_to_openai_file_object(
            create_file_data=create_file_data,
            gcs_upload_response=gcs_upload_response,
        )

    def create_file(
        self,
        _is_async: bool,
        create_file_data: CreateFileRequest,
        api_base: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> Union[OpenAIFileObject, Coroutine[Any, Any, OpenAIFileObject]]:
        """
        Creates a file on VertexAI GCS Bucket

        Only supported for Async litellm.acreate_file
        """

        if _is_async:
            return self.async_create_file(
                create_file_data=create_file_data,
                api_base=api_base,
                vertex_credentials=vertex_credentials,
                vertex_project=vertex_project,
                vertex_location=vertex_location,
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            return asyncio.run(
                self.async_create_file(
                    create_file_data=create_file_data,
                    api_base=api_base,
                    vertex_credentials=vertex_credentials,
                    vertex_project=vertex_project,
                    vertex_location=vertex_location,
                    timeout=timeout,
                    max_retries=max_retries,
                )
            )
```

## File: `llms/vertex_ai/files/transformation.py`
```
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from httpx import Headers, Response

from litellm.litellm_core_utils.prompt_templates.common_utils import extract_file_data
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.files.transformation import (
    BaseFilesConfig,
    LiteLLMLoggingObj,
)
from litellm.llms.vertex_ai.common_utils import (
    _convert_vertex_datetime_to_openai_datetime,
)
from litellm.llms.vertex_ai.gemini.transformation import _transform_request_body
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    CreateFileRequest,
    FileTypes,
    OpenAICreateFileRequestOptionalParams,
    OpenAIFileObject,
    PathLike,
)
from litellm.types.llms.vertex_ai import GcsBucketResponse
from litellm.types.utils import ExtractedFileData, LlmProviders

from ..common_utils import VertexAIError
from ..vertex_llm_base import VertexBase


class VertexAIFilesConfig(VertexBase, BaseFilesConfig):
    """
    Config for VertexAI Files
    """

    def __init__(self):
        self.jsonl_transformation = VertexAIJsonlFilesTransformation()
        super().__init__()

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.VERTEX_AI

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        if not api_key:
            api_key, _ = self.get_access_token(
                credentials=litellm_params.get("vertex_credentials"),
                project_id=litellm_params.get("vertex_project"),
            )
            if not api_key:
                raise ValueError("api_key is required")
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _get_content_from_openai_file(self, openai_file_content: FileTypes) -> str:
        """
        Helper to extract content from various OpenAI file types and return as string.

        Handles:
        - Direct content (str, bytes, IO[bytes])
        - Tuple formats: (filename, content, [content_type], [headers])
        - PathLike objects
        """
        content: Union[str, bytes] = b""
        # Extract file content from tuple if necessary
        if isinstance(openai_file_content, tuple):
            # Take the second element which is always the file content
            file_content = openai_file_content[1]
        else:
            file_content = openai_file_content

        # Handle different file content types
        if isinstance(file_content, str):
            # String content can be used directly
            content = file_content
        elif isinstance(file_content, bytes):
            # Bytes content can be decoded
            content = file_content
        elif isinstance(file_content, PathLike):  # PathLike
            with open(str(file_content), "rb") as f:
                content = f.read()
        elif hasattr(file_content, "read"):  # IO[bytes]
            # File-like objects need to be read
            content = file_content.read()

        # Ensure content is string
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        return content

    def _get_gcs_object_name_from_batch_jsonl(
        self,
        openai_jsonl_content: List[Dict[str, Any]],
    ) -> str:
        """
        Gets a unique GCS object name for the VertexAI batch prediction job

        named as: litellm-vertex-{model}-{uuid}
        """
        _model = openai_jsonl_content[0].get("body", {}).get("model", "")
        if "publishers/google/models" not in _model:
            _model = f"publishers/google/models/{_model}"
        object_name = f"litellm-vertex-files/{_model}/{uuid.uuid4()}"
        return object_name

    def get_object_name(
        self, extracted_file_data: ExtractedFileData, purpose: str
    ) -> str:
        """
        Get the object name for the request
        """
        extracted_file_data_content = extracted_file_data.get("content")

        if extracted_file_data_content is None:
            raise ValueError("file content is required")

        if purpose == "batch":
            ## 1. If jsonl, check if there's a model name
            file_content = self._get_content_from_openai_file(
                extracted_file_data_content
            )

            # Split into lines and parse each line as JSON
            openai_jsonl_content = [
                json.loads(line) for line in file_content.splitlines() if line.strip()
            ]
            if len(openai_jsonl_content) > 0:
                return self._get_gcs_object_name_from_batch_jsonl(openai_jsonl_content)

        ## 2. If not jsonl, return the filename
        filename = extracted_file_data.get("filename")
        if filename:
            return filename
        ## 3. If no file name, return timestamp
        return str(int(time.time()))

    def get_complete_file_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: Dict,
        litellm_params: Dict,
        data: CreateFileRequest,
    ) -> str:
        """
        Get the complete url for the request
        """
        bucket_name = litellm_params.get("bucket_name") or os.getenv("GCS_BUCKET_NAME")
        if not bucket_name:
            raise ValueError("GCS bucket_name is required")
        file_data = data.get("file")
        purpose = data.get("purpose")
        if file_data is None:
            raise ValueError("file is required")
        if purpose is None:
            raise ValueError("purpose is required")
        extracted_file_data = extract_file_data(file_data)
        object_name = self.get_object_name(extracted_file_data, purpose)
        endpoint = (
            f"upload/storage/v1/b/{bucket_name}/o?uploadType=media&name={object_name}"
        )
        api_base = api_base or "https://storage.googleapis.com"
        if not api_base:
            raise ValueError("api_base is required")

        return f"{api_base}/{endpoint}"

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAICreateFileRequestOptionalParams]:
        return []

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return optional_params

    def _map_openai_to_vertex_params(
        self,
        openai_request_body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        wrapper to call VertexGeminiConfig.map_openai_params
        """
        from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
            VertexGeminiConfig,
        )

        config = VertexGeminiConfig()
        _model = openai_request_body.get("model", "")
        vertex_params = config.map_openai_params(
            model=_model,
            non_default_params=openai_request_body,
            optional_params={},
            drop_params=False,
        )
        return vertex_params

    def _transform_openai_jsonl_content_to_vertex_ai_jsonl_content(
        self, openai_jsonl_content: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Transforms OpenAI JSONL content to VertexAI JSONL content

        jsonl body for vertex is {"request": <request_body>}
        Example Vertex jsonl
        {"request":{"contents": [{"role": "user", "parts": [{"text": "What is the relation between the following video and image samples?"}, {"fileData": {"fileUri": "gs://cloud-samples-data/generative-ai/video/animals.mp4", "mimeType": "video/mp4"}}, {"fileData": {"fileUri": "gs://cloud-samples-data/generative-ai/image/cricket.jpeg", "mimeType": "image/jpeg"}}]}]}}
        {"request":{"contents": [{"role": "user", "parts": [{"text": "Describe what is happening in this video."}, {"fileData": {"fileUri": "gs://cloud-samples-data/generative-ai/video/another_video.mov", "mimeType": "video/mov"}}]}]}}
        """

        vertex_jsonl_content = []
        for _openai_jsonl_content in openai_jsonl_content:
            openai_request_body = _openai_jsonl_content.get("body") or {}
            vertex_request_body = _transform_request_body(
                messages=openai_request_body.get("messages", []),
                model=openai_request_body.get("model", ""),
                optional_params=self._map_openai_to_vertex_params(openai_request_body),
                custom_llm_provider="vertex_ai",
                litellm_params={},
                cached_content=None,
            )
            vertex_jsonl_content.append({"request": vertex_request_body})
        return vertex_jsonl_content

    def transform_create_file_request(
        self,
        model: str,
        create_file_data: CreateFileRequest,
        optional_params: dict,
        litellm_params: dict,
    ) -> Union[bytes, str, dict]:
        """
        2 Cases:
        1. Handle basic file upload
        2. Handle batch file upload (.jsonl)
        """
        file_data = create_file_data.get("file")
        if file_data is None:
            raise ValueError("file is required")
        extracted_file_data = extract_file_data(file_data)
        extracted_file_data_content = extracted_file_data.get("content")
        if (
            create_file_data.get("purpose") == "batch"
            and extracted_file_data.get("content_type") == "application/jsonl"
            and extracted_file_data_content is not None
        ):
            ## 1. If jsonl, check if there's a model name
            file_content = self._get_content_from_openai_file(
                extracted_file_data_content
            )

            # Split into lines and parse each line as JSON
            openai_jsonl_content = [
                json.loads(line) for line in file_content.splitlines() if line.strip()
            ]
            vertex_jsonl_content = (
                self._transform_openai_jsonl_content_to_vertex_ai_jsonl_content(
                    openai_jsonl_content
                )
            )
            return json.dumps(vertex_jsonl_content)
        elif isinstance(extracted_file_data_content, bytes):
            return extracted_file_data_content
        else:
            raise ValueError("Unsupported file content type")

    def transform_create_file_response(
        self,
        model: Optional[str],
        raw_response: Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> OpenAIFileObject:
        """
        Transform VertexAI File upload response into OpenAI-style FileObject
        """
        response_json = raw_response.json()

        try:
            response_object = GcsBucketResponse(**response_json)  # type: ignore
        except Exception as e:
            raise VertexAIError(
                status_code=raw_response.status_code,
                message=f"Error reading GCS bucket response: {e}",
                headers=raw_response.headers,
            )

        gcs_id = response_object.get("id", "")
        # Remove the last numeric ID from the path
        gcs_id = "/".join(gcs_id.split("/")[:-1]) if gcs_id else ""

        return OpenAIFileObject(
            purpose=response_object.get("purpose", "batch"),
            id=f"gs://{gcs_id}",
            filename=response_object.get("name", ""),
            created_at=_convert_vertex_datetime_to_openai_datetime(
                vertex_datetime=response_object.get("timeCreated", "")
            ),
            status="uploaded",
            bytes=int(response_object.get("size", 0)),
            object="file",
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[Dict, Headers]
    ) -> BaseLLMException:
        return VertexAIError(
            status_code=status_code, message=error_message, headers=headers
        )


class VertexAIJsonlFilesTransformation(VertexGeminiConfig):
    """
    Transforms OpenAI /v1/files/* requests to VertexAI /v1/files/* requests
    """

    def transform_openai_file_content_to_vertex_ai_file_content(
        self, openai_file_content: Optional[FileTypes] = None
    ) -> Tuple[str, str]:
        """
        Transforms OpenAI FileContentRequest to VertexAI FileContentRequest
        """

        if openai_file_content is None:
            raise ValueError("contents of file are None")
        # Read the content of the file
        file_content = self._get_content_from_openai_file(openai_file_content)

        # Split into lines and parse each line as JSON
        openai_jsonl_content = [
            json.loads(line) for line in file_content.splitlines() if line.strip()
        ]
        vertex_jsonl_content = (
            self._transform_openai_jsonl_content_to_vertex_ai_jsonl_content(
                openai_jsonl_content
            )
        )
        vertex_jsonl_string = "\n".join(
            json.dumps(item) for item in vertex_jsonl_content
        )
        object_name = self._get_gcs_object_name(
            openai_jsonl_content=openai_jsonl_content
        )
        return vertex_jsonl_string, object_name

    def _transform_openai_jsonl_content_to_vertex_ai_jsonl_content(
        self, openai_jsonl_content: List[Dict[str, Any]]
    ):
        """
        Transforms OpenAI JSONL content to VertexAI JSONL content

        jsonl body for vertex is {"request": <request_body>}
        Example Vertex jsonl
        {"request":{"contents": [{"role": "user", "parts": [{"text": "What is the relation between the following video and image samples?"}, {"fileData": {"fileUri": "gs://cloud-samples-data/generative-ai/video/animals.mp4", "mimeType": "video/mp4"}}, {"fileData": {"fileUri": "gs://cloud-samples-data/generative-ai/image/cricket.jpeg", "mimeType": "image/jpeg"}}]}]}}
        {"request":{"contents": [{"role": "user", "parts": [{"text": "Describe what is happening in this video."}, {"fileData": {"fileUri": "gs://cloud-samples-data/generative-ai/video/another_video.mov", "mimeType": "video/mov"}}]}]}}
        """

        vertex_jsonl_content = []
        for _openai_jsonl_content in openai_jsonl_content:
            openai_request_body = _openai_jsonl_content.get("body") or {}
            vertex_request_body = _transform_request_body(
                messages=openai_request_body.get("messages", []),
                model=openai_request_body.get("model", ""),
                optional_params=self._map_openai_to_vertex_params(openai_request_body),
                custom_llm_provider="vertex_ai",
                litellm_params={},
                cached_content=None,
            )
            vertex_jsonl_content.append({"request": vertex_request_body})
        return vertex_jsonl_content

    def _get_gcs_object_name(
        self,
        openai_jsonl_content: List[Dict[str, Any]],
    ) -> str:
        """
        Gets a unique GCS object name for the VertexAI batch prediction job

        named as: litellm-vertex-{model}-{uuid}
        """
        _model = openai_jsonl_content[0].get("body", {}).get("model", "")
        if "publishers/google/models" not in _model:
            _model = f"publishers/google/models/{_model}"
        object_name = f"litellm-vertex-files/{_model}/{uuid.uuid4()}"
        return object_name

    def _map_openai_to_vertex_params(
        self,
        openai_request_body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        wrapper to call VertexGeminiConfig.map_openai_params
        """
        _model = openai_request_body.get("model", "")
        vertex_params = self.map_openai_params(
            model=_model,
            non_default_params=openai_request_body,
            optional_params={},
            drop_params=False,
        )
        return vertex_params

    def _get_content_from_openai_file(self, openai_file_content: FileTypes) -> str:
        """
        Helper to extract content from various OpenAI file types and return as string.

        Handles:
        - Direct content (str, bytes, IO[bytes])
        - Tuple formats: (filename, content, [content_type], [headers])
        - PathLike objects
        """
        content: Union[str, bytes] = b""
        # Extract file content from tuple if necessary
        if isinstance(openai_file_content, tuple):
            # Take the second element which is always the file content
            file_content = openai_file_content[1]
        else:
            file_content = openai_file_content

        # Handle different file content types
        if isinstance(file_content, str):
            # String content can be used directly
            content = file_content
        elif isinstance(file_content, bytes):
            # Bytes content can be decoded
            content = file_content
        elif isinstance(file_content, PathLike):  # PathLike
            with open(str(file_content), "rb") as f:
                content = f.read()
        elif hasattr(file_content, "read"):  # IO[bytes]
            # File-like objects need to be read
            content = file_content.read()

        # Ensure content is string
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        return content

    def transform_gcs_bucket_response_to_openai_file_object(
        self, create_file_data: CreateFileRequest, gcs_upload_response: Dict[str, Any]
    ) -> OpenAIFileObject:
        """
        Transforms GCS Bucket upload file response to OpenAI FileObject
        """
        gcs_id = gcs_upload_response.get("id", "")
        # Remove the last numeric ID from the path
        gcs_id = "/".join(gcs_id.split("/")[:-1]) if gcs_id else ""

        return OpenAIFileObject(
            purpose=create_file_data.get("purpose", "batch"),
            id=f"gs://{gcs_id}",
            filename=gcs_upload_response.get("name", ""),
            created_at=_convert_vertex_datetime_to_openai_datetime(
                vertex_datetime=gcs_upload_response.get("timeCreated", "")
            ),
            status="uploaded",
            bytes=gcs_upload_response.get("size", 0),
            object="file",
        )
```

## File: `llms/vertex_ai/fine_tuning/handler.py`
```
import json
import traceback
from datetime import datetime
from typing import Any, Coroutine, Literal, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import HTTPHandler, get_async_httpx_client
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.types.fine_tuning import OpenAIFineTuningHyperparameters
from litellm.types.llms.openai import FineTuningJobCreate
from litellm.types.llms.vertex_ai import (
    VERTEX_CREDENTIALS_TYPES,
    FineTuneHyperparameters,
    FineTuneJobCreate,
    FineTunesupervisedTuningSpec,
    ResponseSupervisedTuningSpec,
    ResponseTuningJob,
)
from litellm.types.utils import LiteLLMFineTuningJob


class VertexFineTuningAPI(VertexLLM):
    """
    Vertex methods to support for batches
    """

    def __init__(self) -> None:
        super().__init__()
        self.async_handler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.VERTEX_AI,
            params={"timeout": 600.0},
        )

    def convert_response_created_at(self, response: ResponseTuningJob):
        try:
            create_time_str = response.get("createTime", "") or ""
            create_time_datetime = datetime.fromisoformat(
                create_time_str.replace("Z", "+00:00")
            )
            # Convert to Unix timestamp (seconds since epoch)
            created_at = int(create_time_datetime.timestamp())

            return created_at
        except Exception:
            return 0

    def convert_openai_request_to_vertex(
        self,
        create_fine_tuning_job_data: FineTuningJobCreate,
        original_hyperparameters: dict = {},
        kwargs: Optional[dict] = None,
    ) -> FineTuneJobCreate:
        """
        convert request from OpenAI format to Vertex format
        https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/tuning
        supervised_tuning_spec = FineTunesupervisedTuningSpec(
        """

        supervised_tuning_spec = FineTunesupervisedTuningSpec(
            training_dataset_uri=create_fine_tuning_job_data.training_file,
        )

        if create_fine_tuning_job_data.validation_file:
            supervised_tuning_spec[
                "validation_dataset"
            ] = create_fine_tuning_job_data.validation_file

        _vertex_hyperparameters = (
            self._transform_openai_hyperparameters_to_vertex_hyperparameters(
                create_fine_tuning_job_data=create_fine_tuning_job_data,
                kwargs=kwargs,
                original_hyperparameters=original_hyperparameters,
            )
        )

        if _vertex_hyperparameters and len(_vertex_hyperparameters) > 0:
            supervised_tuning_spec["hyperParameters"] = _vertex_hyperparameters

        fine_tune_job = FineTuneJobCreate(
            baseModel=create_fine_tuning_job_data.model,
            supervisedTuningSpec=supervised_tuning_spec,
            tunedModelDisplayName=create_fine_tuning_job_data.suffix,
        )

        return fine_tune_job

    def _transform_openai_hyperparameters_to_vertex_hyperparameters(
        self,
        create_fine_tuning_job_data: FineTuningJobCreate,
        original_hyperparameters: dict = {},
        kwargs: Optional[dict] = None,
    ) -> FineTuneHyperparameters:
        _oai_hyperparameters = create_fine_tuning_job_data.hyperparameters
        _vertex_hyperparameters = FineTuneHyperparameters()
        if _oai_hyperparameters:
            if _oai_hyperparameters.n_epochs:
                _vertex_hyperparameters["epoch_count"] = int(
                    _oai_hyperparameters.n_epochs
                )
            if _oai_hyperparameters.learning_rate_multiplier:
                _vertex_hyperparameters["learning_rate_multiplier"] = float(
                    _oai_hyperparameters.learning_rate_multiplier
                )

        _adapter_size = original_hyperparameters.get("adapter_size", None)
        if _adapter_size:
            _vertex_hyperparameters["adapter_size"] = _adapter_size

        return _vertex_hyperparameters

    def convert_vertex_response_to_open_ai_response(
        self, response: ResponseTuningJob
    ) -> LiteLLMFineTuningJob:
        status: Literal[
            "validating_files", "queued", "running", "succeeded", "failed", "cancelled"
        ] = "queued"
        if response["state"] == "JOB_STATE_PENDING":
            status = "queued"
        if response["state"] == "JOB_STATE_SUCCEEDED":
            status = "succeeded"
        if response["state"] == "JOB_STATE_FAILED":
            status = "failed"
        if response["state"] == "JOB_STATE_CANCELLED":
            status = "cancelled"
        if response["state"] == "JOB_STATE_RUNNING":
            status = "running"

        created_at = self.convert_response_created_at(response)

        _supervisedTuningSpec: ResponseSupervisedTuningSpec = (
            response.get("supervisedTuningSpec", None) or {}
        )
        training_uri: str = _supervisedTuningSpec.get("trainingDatasetUri", "") or ""
        return LiteLLMFineTuningJob(
            id=response.get("name", "") or "",
            created_at=created_at,
            fine_tuned_model=response.get("tunedModelDisplayName", ""),
            finished_at=None,
            hyperparameters=self._translate_vertex_response_hyperparameters(
                vertex_hyper_parameters=_supervisedTuningSpec.get("hyperParameters", {})
                or {}
            ),
            model=response.get("baseModel", "") or "",
            object="fine_tuning.job",
            organization_id="",
            result_files=[],
            seed=0,
            status=status,
            trained_tokens=None,
            training_file=training_uri,
            validation_file=None,
            estimated_finish=None,
            integrations=[],
        )

    def _translate_vertex_response_hyperparameters(
        self, vertex_hyper_parameters: FineTuneHyperparameters
    ) -> OpenAIFineTuningHyperparameters:
        """
        translate vertex responsehyperparameters to openai hyperparameters
        """
        _dict_remaining_hyperparameters: dict = dict(vertex_hyper_parameters)
        return OpenAIFineTuningHyperparameters(
            n_epochs=_dict_remaining_hyperparameters.pop("epoch_count", 0),
            **_dict_remaining_hyperparameters,
        )

    async def acreate_fine_tuning_job(
        self,
        fine_tuning_url: str,
        headers: dict,
        request_data: FineTuneJobCreate,
    ):
        try:
            verbose_logger.debug(
                "about to create fine tuning job: %s, request_data: %s",
                fine_tuning_url,
                json.dumps(request_data, indent=4),
            )
            if self.async_handler is None:
                raise ValueError(
                    "VertexAI Fine Tuning - async_handler is not initialized"
                )
            response = await self.async_handler.post(
                headers=headers,
                url=fine_tuning_url,
                json=request_data,  # type: ignore
            )

            if response.status_code != 200:
                raise Exception(
                    f"Error creating fine tuning job. Status code: {response.status_code}. Response: {response.text}"
                )

            verbose_logger.debug(
                "got response from creating fine tuning job: %s", response.json()
            )

            vertex_response = ResponseTuningJob(  # type: ignore
                **response.json(),
            )

            verbose_logger.debug("vertex_response %s", vertex_response)
            open_ai_response = self.convert_vertex_response_to_open_ai_response(
                vertex_response
            )
            return open_ai_response

        except Exception as e:
            verbose_logger.error("asyncerror creating fine tuning job %s", e)
            trace_back_str = traceback.format_exc()
            verbose_logger.error(trace_back_str)
            raise e

    def create_fine_tuning_job(
        self,
        _is_async: bool,
        create_fine_tuning_job_data: FineTuningJobCreate,
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        kwargs: Optional[dict] = None,
        original_hyperparameters: Optional[dict] = {},
    ) -> Union[LiteLLMFineTuningJob, Coroutine[Any, Any, LiteLLMFineTuningJob]]:
        verbose_logger.debug(
            "creating fine tuning job, args= %s", create_fine_tuning_job_data
        )
        _auth_header, vertex_project = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai_beta",
        )

        auth_header, _ = self._get_token_and_url(
            model="",
            auth_header=_auth_header,
            gemini_api_key=None,
            vertex_credentials=vertex_credentials,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            stream=False,
            custom_llm_provider="vertex_ai_beta",
            api_base=api_base,
        )

        headers = {
            "Authorization": f"Bearer {auth_header}",
            "Content-Type": "application/json",
        }

        fine_tune_job = self.convert_openai_request_to_vertex(
            create_fine_tuning_job_data=create_fine_tuning_job_data,
            kwargs=kwargs,
            original_hyperparameters=original_hyperparameters or {},
        )

        fine_tuning_url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/tuningJobs"
        if _is_async is True:
            return self.acreate_fine_tuning_job(  # type: ignore
                fine_tuning_url=fine_tuning_url,
                headers=headers,
                request_data=fine_tune_job,
            )
        sync_handler = HTTPHandler(timeout=httpx.Timeout(timeout=600.0, connect=5.0))

        verbose_logger.debug(
            "about to create fine tuning job: %s, request_data: %s",
            fine_tuning_url,
            fine_tune_job,
        )
        response = sync_handler.post(
            headers=headers,
            url=fine_tuning_url,
            json=fine_tune_job,  # type: ignore
        )

        if response.status_code != 200:
            raise Exception(
                f"Error creating fine tuning job. Status code: {response.status_code}. Response: {response.text}"
            )

        verbose_logger.debug(
            "got response from creating fine tuning job: %s", response.json()
        )
        vertex_response = ResponseTuningJob(  # type: ignore
            **response.json(),
        )

        verbose_logger.debug("vertex_response %s", vertex_response)
        open_ai_response = self.convert_vertex_response_to_open_ai_response(
            vertex_response
        )
        return open_ai_response

    async def pass_through_vertex_ai_POST_request(
        self,
        request_data: dict,
        vertex_project: str,
        vertex_location: str,
        vertex_credentials: str,
        request_route: str,
    ):
        _auth_header, vertex_project = await self._ensure_access_token_async(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai_beta",
        )
        auth_header, _ = self._get_token_and_url(
            model="",
            auth_header=_auth_header,
            gemini_api_key=None,
            vertex_credentials=vertex_credentials,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            stream=False,
            custom_llm_provider="vertex_ai_beta",
            api_base="",
        )

        headers = {
            "Authorization": f"Bearer {auth_header}",
            "Content-Type": "application/json",
        }

        url = None
        if request_route == "/tuningJobs":
            url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/tuningJobs"
        elif "/tuningJobs/" in request_route and "cancel" in request_route:
            url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/tuningJobs{request_route}"
        elif "generateContent" in request_route:
            url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}{request_route}"
        elif "predict" in request_route:
            url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}{request_route}"
        elif "/batchPredictionJobs" in request_route:
            url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}{request_route}"
        elif "countTokens" in request_route:
            url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}{request_route}"
        elif "cachedContents" in request_route:
            _model = request_data.get("model")
            if _model is not None and "/publishers/google/models/" not in _model:
                request_data[
                    "model"
                ] = f"projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{_model}"

            url = f"https://{vertex_location}-aiplatform.googleapis.com/v1beta1/projects/{vertex_project}/locations/{vertex_location}{request_route}"
        else:
            raise ValueError(f"Unsupported Vertex AI request route: {request_route}")
        if self.async_handler is None:
            raise ValueError("VertexAI Fine Tuning - async_handler is not initialized")

        response = await self.async_handler.post(
            headers=headers,
            url=url,
            json=request_data,  # type: ignore
        )

        if response.status_code != 200:
            raise Exception(
                f"Error creating fine tuning job. Status code: {response.status_code}. Response: {response.text}"
            )

        response_json = response.json()
        return response_json
```

## File: `llms/vertex_ai/gemini_embeddings/batch_embed_content_handler.py`
```
"""
Google AI Studio /batchEmbedContents Embeddings Endpoint
"""

import json
from typing import Any, Literal, Optional, Union

import httpx

import litellm
from litellm import EmbeddingResponse
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.types.llms.openai import EmbeddingInput
from litellm.types.llms.vertex_ai import (
    VertexAIBatchEmbeddingsRequestBody,
    VertexAIBatchEmbeddingsResponseObject,
)

from ..gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from .batch_embed_content_transformation import (
    process_response,
    transform_openai_input_gemini_content,
)


class GoogleBatchEmbeddings(VertexLLM):
    def batch_embeddings(
        self,
        model: str,
        input: EmbeddingInput,
        print_verbose,
        model_response: EmbeddingResponse,
        custom_llm_provider: Literal["gemini", "vertex_ai"],
        optional_params: dict,
        logging_obj: Any,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        encoding=None,
        vertex_project=None,
        vertex_location=None,
        vertex_credentials=None,
        aembedding=False,
        timeout=300,
        client=None,
    ) -> EmbeddingResponse:
        _auth_header, vertex_project = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider=custom_llm_provider,
        )

        auth_header, url = self._get_token_and_url(
            model=model,
            auth_header=_auth_header,
            gemini_api_key=api_key,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=None,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            should_use_v1beta1_features=False,
            mode="batch_embedding",
        )

        if client is None:
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    _httpx_timeout = httpx.Timeout(timeout)
                    _params["timeout"] = _httpx_timeout
            else:
                _params["timeout"] = httpx.Timeout(timeout=600.0, connect=5.0)

            sync_handler: HTTPHandler = HTTPHandler(**_params)  # type: ignore
        else:
            sync_handler = client  # type: ignore

        optional_params = optional_params or {}

        ### TRANSFORMATION ###
        request_data = transform_openai_input_gemini_content(
            input=input, model=model, optional_params=optional_params
        )

        headers = {
            "Content-Type": "application/json; charset=utf-8",
        }

        ## LOGGING
        logging_obj.pre_call(
            input=input,
            api_key="",
            additional_args={
                "complete_input_dict": request_data,
                "api_base": url,
                "headers": headers,
            },
        )

        if aembedding is True:
            return self.async_batch_embeddings(  # type: ignore
                model=model,
                api_base=api_base,
                url=url,
                data=request_data,
                model_response=model_response,
                timeout=timeout,
                headers=headers,
                input=input,
            )

        response = sync_handler.post(
            url=url,
            headers=headers,
            data=json.dumps(request_data),
        )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        _json_response = response.json()
        _predictions = VertexAIBatchEmbeddingsResponseObject(**_json_response)  # type: ignore

        return process_response(
            model=model,
            model_response=model_response,
            _predictions=_predictions,
            input=input,
        )

    async def async_batch_embeddings(
        self,
        model: str,
        api_base: Optional[str],
        url: str,
        data: VertexAIBatchEmbeddingsRequestBody,
        model_response: EmbeddingResponse,
        input: EmbeddingInput,
        timeout: Optional[Union[float, httpx.Timeout]],
        headers={},
        client: Optional[AsyncHTTPHandler] = None,
    ) -> EmbeddingResponse:
        if client is None:
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    _httpx_timeout = httpx.Timeout(timeout)
                    _params["timeout"] = _httpx_timeout
            else:
                _params["timeout"] = httpx.Timeout(timeout=600.0, connect=5.0)

            async_handler: AsyncHTTPHandler = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.VERTEX_AI,
                params={"timeout": timeout},
            )
        else:
            async_handler = client  # type: ignore

        response = await async_handler.post(
            url=url,
            headers=headers,
            data=json.dumps(data),
        )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        _json_response = response.json()
        _predictions = VertexAIBatchEmbeddingsResponseObject(**_json_response)  # type: ignore

        return process_response(
            model=model,
            model_response=model_response,
            _predictions=_predictions,
            input=input,
        )
```

## File: `llms/vertex_ai/gemini_embeddings/batch_embed_content_transformation.py`
```
"""
Transformation logic from OpenAI /v1/embeddings format to Google AI Studio /batchEmbedContents format. 

Why separate file? Make it easy to see how transformation works
"""

from typing import List

from litellm import EmbeddingResponse
from litellm.types.llms.openai import EmbeddingInput
from litellm.types.llms.vertex_ai import (
    ContentType,
    EmbedContentRequest,
    PartType,
    VertexAIBatchEmbeddingsRequestBody,
    VertexAIBatchEmbeddingsResponseObject,
)
from litellm.types.utils import Embedding, Usage
from litellm.utils import get_formatted_prompt, token_counter


def transform_openai_input_gemini_content(
    input: EmbeddingInput, model: str, optional_params: dict
) -> VertexAIBatchEmbeddingsRequestBody:
    """
    The content to embed. Only the parts.text fields will be counted.
    """
    gemini_model_name = "models/{}".format(model)
    requests: List[EmbedContentRequest] = []
    if isinstance(input, str):
        request = EmbedContentRequest(
            model=gemini_model_name,
            content=ContentType(parts=[PartType(text=input)]),
            **optional_params
        )
        requests.append(request)
    else:
        for i in input:
            request = EmbedContentRequest(
                model=gemini_model_name,
                content=ContentType(parts=[PartType(text=i)]),
                **optional_params
            )
            requests.append(request)

    return VertexAIBatchEmbeddingsRequestBody(requests=requests)


def process_response(
    input: EmbeddingInput,
    model_response: EmbeddingResponse,
    model: str,
    _predictions: VertexAIBatchEmbeddingsResponseObject,
) -> EmbeddingResponse:
    openai_embeddings: List[Embedding] = []
    for embedding in _predictions["embeddings"]:
        openai_embedding = Embedding(
            embedding=embedding["values"],
            index=0,
            object="embedding",
        )
        openai_embeddings.append(openai_embedding)

    model_response.data = openai_embeddings
    model_response.model = model

    input_text = get_formatted_prompt(data={"input": input}, call_type="embedding")
    prompt_tokens = token_counter(model=model, text=input_text)
    model_response.usage = Usage(
        prompt_tokens=prompt_tokens, total_tokens=prompt_tokens
    )

    return model_response
```

## File: `llms/vertex_ai/gemini/cost_calculator.py`
```
"""
Cost calculator for Vertex AI Gemini.

Used because there are differences in how Google AI Studio and Vertex AI Gemini handle web search requests.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo, Usage


def cost_per_web_search_request(usage: "Usage", model_info: "ModelInfo") -> float:
    """
    Calculate the cost of a web search request for Vertex AI Gemini.

    Vertex AI charges $35/1000 prompts, independent of the number of web search requests.

    For a single call, this is $35e-3 USD.

    Args:
        usage: The usage object for the web search request.
        model_info: The model info for the web search request.

    Returns:
        The cost of the web search request.
    """
    from litellm.types.utils import PromptTokensDetailsWrapper

    # check if usage object has web search requests
    cost_per_llm_call_with_web_search = 35e-3

    makes_web_search_request = False
    if (
        usage is not None
        and usage.prompt_tokens_details is not None
        and isinstance(usage.prompt_tokens_details, PromptTokensDetailsWrapper)
    ):
        makes_web_search_request = True

    # Calculate total cost
    if makes_web_search_request:
        return cost_per_llm_call_with_web_search
    else:
        return 0.0
```

## File: `llms/vertex_ai/gemini/transformation.py`
```
"""
Transformation logic from OpenAI format to Gemini format.

Why separate file? Make it easy to see how transformation works
"""

import os
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Union, cast

import httpx
from pydantic import BaseModel

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    _get_image_mime_type_from_url,
)
from litellm.litellm_core_utils.prompt_templates.factory import (
    convert_generic_image_chunk_to_openai_image_obj,
    convert_to_anthropic_image_obj,
    convert_to_gemini_tool_call_invoke,
    convert_to_gemini_tool_call_result,
    response_schema_prompt,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.files import (
    get_file_mime_type_for_file_type,
    get_file_type_from_extension,
    is_gemini_1_5_accepted_file_type,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionAudioObject,
    ChatCompletionFileObject,
    ChatCompletionImageObject,
    ChatCompletionTextObject,
)
from litellm.types.llms.vertex_ai import *
from litellm.types.llms.vertex_ai import (
    GenerationConfig,
    PartType,
    RequestBody,
    SafetSettingsConfig,
    SystemInstructions,
    ToolConfig,
    Tools,
)
from litellm.types.utils import GenericImageParsingChunk

from ..common_utils import (
    _check_text_in_content,
    get_supports_response_schema,
    get_supports_system_message,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


def _process_gemini_image(image_url: str, format: Optional[str] = None) -> PartType:
    """
    Given an image URL, return the appropriate PartType for Gemini
    """

    try:
        # GCS URIs
        if "gs://" in image_url:
            # Figure out file type
            extension_with_dot = os.path.splitext(image_url)[-1]  # Ex: ".png"
            extension = extension_with_dot[1:]  # Ex: "png"

            if not format:
                file_type = get_file_type_from_extension(extension)

                # Validate the file type is supported by Gemini
                if not is_gemini_1_5_accepted_file_type(file_type):
                    raise Exception(f"File type not supported by gemini - {file_type}")

                mime_type = get_file_mime_type_for_file_type(file_type)
            else:
                mime_type = format
            file_data = FileDataType(mime_type=mime_type, file_uri=image_url)

            return PartType(file_data=file_data)
        elif (
            "https://" in image_url
            and (image_type := format or _get_image_mime_type_from_url(image_url))
            is not None
        ):
            file_data = FileDataType(file_uri=image_url, mime_type=image_type)
            return PartType(file_data=file_data)
        elif "http://" in image_url or "https://" in image_url or "base64" in image_url:
            # https links for unsupported mime types and base64 images
            image = convert_to_anthropic_image_obj(image_url, format=format)
            _blob = BlobType(data=image["data"], mime_type=image["media_type"])
            return PartType(inline_data=_blob)
        raise Exception("Invalid image received - {}".format(image_url))
    except Exception as e:
        raise e


def _gemini_convert_messages_with_history(  # noqa: PLR0915
    messages: List[AllMessageValues],
) -> List[ContentType]:
    """
    Converts given messages from OpenAI format to Gemini format

    - Parts must be iterable
    - Roles must alternate b/w 'user' and 'model' (same as anthropic -> merge consecutive roles)
    - Please ensure that function response turn comes immediately after a function call turn
    """
    user_message_types = {"user", "system"}
    contents: List[ContentType] = []

    last_message_with_tool_calls = None

    msg_i = 0
    tool_call_responses = []
    try:
        while msg_i < len(messages):
            user_content: List[PartType] = []
            init_msg_i = msg_i
            ## MERGE CONSECUTIVE USER CONTENT ##
            while (
                msg_i < len(messages) and messages[msg_i]["role"] in user_message_types
            ):
                _message_content = messages[msg_i].get("content")
                if _message_content is not None and isinstance(_message_content, list):
                    _parts: List[PartType] = []
                    for element_idx, element in enumerate(_message_content):
                        if (
                            element["type"] == "text"
                            and "text" in element
                            and len(element["text"]) > 0
                        ):
                            element = cast(ChatCompletionTextObject, element)
                            _part = PartType(text=element["text"])
                            _parts.append(_part)
                        elif element["type"] == "image_url":
                            element = cast(ChatCompletionImageObject, element)
                            img_element = element
                            format: Optional[str] = None
                            if isinstance(img_element["image_url"], dict):
                                image_url = img_element["image_url"]["url"]
                                format = img_element["image_url"].get("format")
                            else:
                                image_url = img_element["image_url"]
                            _part = _process_gemini_image(
                                image_url=image_url, format=format
                            )
                            _parts.append(_part)
                        elif element["type"] == "input_audio":
                            audio_element = cast(ChatCompletionAudioObject, element)
                            audio_data = audio_element["input_audio"].get("data")
                            audio_format = audio_element["input_audio"].get("format")
                            if audio_data is not None and audio_format is not None:
                                audio_format_modified = (
                                    "audio/" + audio_format
                                    if audio_format.startswith("audio/") is False
                                    else audio_format
                                )  # Gemini expects audio/wav, audio/mp3, etc.
                                openai_image_str = (
                                    convert_generic_image_chunk_to_openai_image_obj(
                                        image_chunk=GenericImageParsingChunk(
                                            type="base64",
                                            media_type=audio_format_modified,
                                            data=audio_data,
                                        )
                                    )
                                )
                                _part = _process_gemini_image(
                                    image_url=openai_image_str,
                                    format=audio_format_modified,
                                )
                                _parts.append(_part)
                        elif element["type"] == "file":
                            file_element = cast(ChatCompletionFileObject, element)
                            file_id = file_element["file"].get("file_id")
                            format = file_element["file"].get("format")
                            file_data = file_element["file"].get("file_data")
                            passed_file = file_id or file_data
                            if passed_file is None:
                                raise Exception(
                                    "Unknown file type. Please pass in a file_id or file_data"
                                )
                            try:
                                _part = _process_gemini_image(
                                    image_url=passed_file, format=format
                                )
                                _parts.append(_part)
                            except Exception:
                                raise Exception(
                                    "Unable to determine mime type for file_id: {}, set this explicitly using message[{}].content[{}].file.format".format(
                                        file_id, msg_i, element_idx
                                    )
                                )
                    user_content.extend(_parts)
                elif (
                    _message_content is not None
                    and isinstance(_message_content, str)
                    and len(_message_content) > 0
                ):
                    _part = PartType(text=_message_content)
                    user_content.append(_part)

                msg_i += 1

            if user_content:
                """
                check that user_content has 'text' parameter.
                    - Known Vertex Error: Unable to submit request because it must have a text parameter.
                    - Relevant Issue: https://github.com/BerriAI/litellm/issues/5515
                """
                has_text_in_content = _check_text_in_content(user_content)
                if has_text_in_content is False:
                    verbose_logger.warning(
                        "No text in user content. Adding a blank text to user content, to ensure Gemini doesn't fail the request. Relevant Issue - https://github.com/BerriAI/litellm/issues/5515"
                    )
                    user_content.append(
                        PartType(text=" ")
                    )  # add a blank text, to ensure Gemini doesn't fail the request.
                contents.append(ContentType(role="user", parts=user_content))
            assistant_content = []
            ## MERGE CONSECUTIVE ASSISTANT CONTENT ##
            while msg_i < len(messages) and messages[msg_i]["role"] == "assistant":
                if isinstance(messages[msg_i], BaseModel):
                    msg_dict: Union[ChatCompletionAssistantMessage, dict] = messages[msg_i].model_dump()  # type: ignore
                else:
                    msg_dict = messages[msg_i]  # type: ignore
                assistant_msg = ChatCompletionAssistantMessage(**msg_dict)  # type: ignore
                _message_content = assistant_msg.get("content", None)
                reasoning_content = assistant_msg.get("reasoning_content", None)
                if reasoning_content is not None:
                    assistant_content.append(
                        PartType(thought=True, text=reasoning_content)
                    )
                if _message_content is not None and isinstance(_message_content, list):
                    _parts = []
                    for element in _message_content:
                        if isinstance(element, dict):
                            if element["type"] == "text":
                                _part = PartType(text=element["text"])
                                _parts.append(_part)

                    assistant_content.extend(_parts)
                elif (
                    _message_content is not None
                    and isinstance(_message_content, str)
                    and _message_content
                ):
                    assistant_text = _message_content  # either string or none
                    assistant_content.append(PartType(text=assistant_text))  # type: ignore

                ## HANDLE ASSISTANT FUNCTION CALL
                if (
                    assistant_msg.get("tool_calls", []) is not None
                    or assistant_msg.get("function_call") is not None
                ):  # support assistant tool invoke conversion
                    assistant_content.extend(
                        convert_to_gemini_tool_call_invoke(assistant_msg)
                    )
                    last_message_with_tool_calls = assistant_msg

                msg_i += 1

            if assistant_content:
                contents.append(ContentType(role="model", parts=assistant_content))

            ## APPEND TOOL CALL MESSAGES ##
            tool_call_message_roles = ["tool", "function"]
            if (
                msg_i < len(messages)
                and messages[msg_i]["role"] in tool_call_message_roles
            ):
                _part = convert_to_gemini_tool_call_result(
                    messages[msg_i], last_message_with_tool_calls  # type: ignore
                )
                msg_i += 1
                tool_call_responses.append(_part)
            if msg_i < len(messages) and (
                messages[msg_i]["role"] not in tool_call_message_roles
            ):
                if len(tool_call_responses) > 0:
                    contents.append(ContentType(parts=tool_call_responses))
                    tool_call_responses = []

            if msg_i == init_msg_i:  # prevent infinite loops
                raise Exception(
                    "Invalid Message passed in - {}. File an issue https://github.com/BerriAI/litellm/issues".format(
                        messages[msg_i]
                    )
                )
        if len(tool_call_responses) > 0:
            contents.append(ContentType(parts=tool_call_responses))
        return contents
    except Exception as e:
        raise e


def _transform_request_body(
    messages: List[AllMessageValues],
    model: str,
    optional_params: dict,
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    litellm_params: dict,
    cached_content: Optional[str],
) -> RequestBody:
    """
    Common transformation logic across sync + async Gemini /generateContent calls.
    """
    # Separate system prompt from rest of message
    supports_system_message = get_supports_system_message(
        model=model, custom_llm_provider=custom_llm_provider
    )
    system_instructions, messages = _transform_system_message(
        supports_system_message=supports_system_message, messages=messages
    )
    # Checks for 'response_schema' support - if passed in
    if "response_schema" in optional_params:
        supports_response_schema = get_supports_response_schema(
            model=model, custom_llm_provider=custom_llm_provider
        )
        if supports_response_schema is False:
            user_response_schema_message = response_schema_prompt(
                model=model, response_schema=optional_params.get("response_schema")  # type: ignore
            )
            messages.append({"role": "user", "content": user_response_schema_message})
            optional_params.pop("response_schema")

    # Check for any 'litellm_param_*' set during optional param mapping

    remove_keys = []
    for k, v in optional_params.items():
        if k.startswith("litellm_param_"):
            litellm_params.update({k: v})
            remove_keys.append(k)

    optional_params = {k: v for k, v in optional_params.items() if k not in remove_keys}

    try:
        if custom_llm_provider == "gemini":
            content = litellm.GoogleAIStudioGeminiConfig()._transform_messages(
                messages=messages
            )
        else:
            content = litellm.VertexGeminiConfig()._transform_messages(
                messages=messages
            )
        tools: Optional[Tools] = optional_params.pop("tools", None)
        tool_choice: Optional[ToolConfig] = optional_params.pop("tool_choice", None)
        safety_settings: Optional[List[SafetSettingsConfig]] = optional_params.pop(
            "safety_settings", None
        )  # type: ignore
        config_fields = GenerationConfig.__annotations__.keys()

        filtered_params = {
            k: v for k, v in optional_params.items() if k in config_fields
        }

        generation_config: Optional[GenerationConfig] = GenerationConfig(
            **filtered_params
        )
        data = RequestBody(contents=content)
        if system_instructions is not None:
            data["system_instruction"] = system_instructions
        if tools is not None:
            data["tools"] = tools
        if tool_choice is not None:
            data["toolConfig"] = tool_choice
        if safety_settings is not None:
            data["safetySettings"] = safety_settings
        if generation_config is not None:
            data["generationConfig"] = generation_config
        if cached_content is not None:
            data["cachedContent"] = cached_content
    except Exception as e:
        raise e

    return data


def sync_transform_request_body(
    gemini_api_key: Optional[str],
    messages: List[AllMessageValues],
    api_base: Optional[str],
    model: str,
    client: Optional[HTTPHandler],
    timeout: Optional[Union[float, httpx.Timeout]],
    extra_headers: Optional[dict],
    optional_params: dict,
    logging_obj: LiteLLMLoggingObj,
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    litellm_params: dict,
) -> RequestBody:
    from ..context_caching.vertex_ai_context_caching import ContextCachingEndpoints

    context_caching_endpoints = ContextCachingEndpoints()

    if gemini_api_key is not None:
        messages, optional_params, cached_content = (
            context_caching_endpoints.check_and_create_cache(
                messages=messages,
                optional_params=optional_params,
                api_key=gemini_api_key,
                api_base=api_base,
                model=model,
                client=client,
                timeout=timeout,
                extra_headers=extra_headers,
                cached_content=optional_params.pop("cached_content", None),
                logging_obj=logging_obj,
            )
        )
    else:  # [TODO] implement context caching for gemini as well
        cached_content = optional_params.pop("cached_content", None)

    return _transform_request_body(
        messages=messages,
        model=model,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        cached_content=cached_content,
        optional_params=optional_params,
    )


async def async_transform_request_body(
    gemini_api_key: Optional[str],
    messages: List[AllMessageValues],
    api_base: Optional[str],
    model: str,
    client: Optional[AsyncHTTPHandler],
    timeout: Optional[Union[float, httpx.Timeout]],
    extra_headers: Optional[dict],
    optional_params: dict,
    logging_obj: litellm.litellm_core_utils.litellm_logging.Logging,  # type: ignore
    custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
    litellm_params: dict,
) -> RequestBody:
    from ..context_caching.vertex_ai_context_caching import ContextCachingEndpoints

    context_caching_endpoints = ContextCachingEndpoints()

    if gemini_api_key is not None:
        (
            messages,
            optional_params,
            cached_content,
        ) = await context_caching_endpoints.async_check_and_create_cache(
            messages=messages,
            optional_params=optional_params,
            api_key=gemini_api_key,
            api_base=api_base,
            model=model,
            client=client,
            timeout=timeout,
            extra_headers=extra_headers,
            cached_content=optional_params.pop("cached_content", None),
            logging_obj=logging_obj,
        )
    else:  # [TODO] implement context caching for gemini as well
        cached_content = optional_params.pop("cached_content", None)

    return _transform_request_body(
        messages=messages,
        model=model,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        cached_content=cached_content,
        optional_params=optional_params,
    )


def _transform_system_message(
    supports_system_message: bool, messages: List[AllMessageValues]
) -> Tuple[Optional[SystemInstructions], List[AllMessageValues]]:
    """
    Extracts the system message from the openai message list.

    Converts the system message to Gemini format

    Returns
    - system_content_blocks: Optional[SystemInstructions] - the system message list in Gemini format.
    - messages: List[AllMessageValues] - filtered list of messages in OpenAI format (transformed separately)
    """
    # Separate system prompt from rest of message
    system_prompt_indices = []
    system_content_blocks: List[PartType] = []
    if supports_system_message is True:
        for idx, message in enumerate(messages):
            if message["role"] == "system":
                _system_content_block: Optional[PartType] = None
                if isinstance(message["content"], str):
                    _system_content_block = PartType(text=message["content"])
                elif isinstance(message["content"], list):
                    system_text = ""
                    for content in message["content"]:
                        system_text += content.get("text") or ""
                    _system_content_block = PartType(text=system_text)
                if _system_content_block is not None:
                    system_content_blocks.append(_system_content_block)
                    system_prompt_indices.append(idx)
        if len(system_prompt_indices) > 0:
            for idx in reversed(system_prompt_indices):
                messages.pop(idx)

    if len(system_content_blocks) > 0:
        return SystemInstructions(parts=system_content_blocks), messages

    return None, messages
```

## File: `llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py`
```
# What is this?
## httpx client for vertex ai calls
## Initial implementation - covers gemini + image gen calls
import json
import time
import uuid
from copy import deepcopy
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

import httpx  # type: ignore

import litellm
import litellm.litellm_core_utils
import litellm.litellm_core_utils.litellm_logging
from litellm import verbose_logger
from litellm.constants import (
    DEFAULT_REASONING_EFFORT_DISABLE_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
)
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.llms.anthropic import AnthropicThinkingParam
from litellm.types.llms.gemini import BidiGenerateContentServerMessage
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionResponseMessage,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolParamFunctionChunk,
    OpenAIChatCompletionFinishReason,
)
from litellm.types.llms.vertex_ai import (
    VERTEX_CREDENTIALS_TYPES,
    Candidates,
    ContentType,
    FunctionCallingConfig,
    FunctionDeclaration,
    GeminiThinkingConfig,
    GenerateContentResponseBody,
    HttpxPartType,
    LogprobsResult,
    ToolConfig,
    Tools,
    UsageMetadata,
)
from litellm.types.utils import (
    ChatCompletionAudioResponse,
    ChatCompletionTokenLogprob,
    ChoiceLogprobs,
    CompletionTokensDetailsWrapper,
    PromptTokensDetailsWrapper,
    TopLogprob,
    Usage,
)
from litellm.utils import (
    CustomStreamWrapper,
    ModelResponse,
    is_base64_encoded,
    supports_reasoning,
)

from ....utils import _remove_additional_properties, _remove_strict_from_schema
from ..common_utils import VertexAIError, _build_vertex_schema
from ..vertex_llm_base import VertexBase
from .transformation import (
    _gemini_convert_messages_with_history,
    async_transform_request_body,
    sync_transform_request_body,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import ModelResponseStream

    LoggingClass = LiteLLMLoggingObj
else:
    LoggingClass = Any


class VertexAIBaseConfig:
    def get_mapped_special_auth_params(self) -> dict:
        """
        Common auth params across bedrock/vertex_ai/azure/watsonx
        """
        return {"project": "vertex_project", "region_name": "vertex_location"}

    def map_special_auth_params(self, non_default_params: dict, optional_params: dict):
        mapped_params = self.get_mapped_special_auth_params()

        for param, value in non_default_params.items():
            if param in mapped_params:
                optional_params[mapped_params[param]] = value
        return optional_params

    def get_eu_regions(self) -> List[str]:
        """
        Source: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#available-regions
        """
        return [
            "europe-central2",
            "europe-north1",
            "europe-southwest1",
            "europe-west1",
            "europe-west2",
            "europe-west3",
            "europe-west4",
            "europe-west6",
            "europe-west8",
            "europe-west9",
        ]

    def get_us_regions(self) -> List[str]:
        """
        Source: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#available-regions
        """
        return [
            "us-central1",
            "us-east1",
            "us-east4",
            "us-east5",
            "us-south1",
            "us-west1",
            "us-west4",
            "us-west5",
        ]


class VertexGeminiConfig(VertexAIBaseConfig, BaseConfig):
    """
    Reference: https://cloud.google.com/vertex-ai/docs/generative-ai/chat/test-chat-prompts
    Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference

    The class `VertexAIConfig` provides configuration for the VertexAI's API interface. Below are the parameters:

    - `temperature` (float): This controls the degree of randomness in token selection.

    - `max_output_tokens` (integer): This sets the limitation for the maximum amount of token in the text output. In this case, the default value is 256.

    - `top_p` (float): The tokens are selected from the most probable to the least probable until the sum of their probabilities equals the `top_p` value. Default is 0.95.

    - `top_k` (integer): The value of `top_k` determines how many of the most probable tokens are considered in the selection. For example, a `top_k` of 1 means the selected token is the most probable among all tokens. The default value is 40.

    - `response_mime_type` (str): The MIME type of the response. The default value is 'text/plain'.

    - `candidate_count` (int): Number of generated responses to return.

    - `stop_sequences` (List[str]): The set of character sequences (up to 5) that will stop output generation. If specified, the API will stop at the first appearance of a stop sequence. The stop sequence will not be included as part of the response.

    - `frequency_penalty` (float): This parameter is used to penalize the model from repeating the same output. The default value is 0.0.

    - `presence_penalty` (float): This parameter is used to penalize the model from generating the same output as the input. The default value is 0.0.

    - `seed` (int): The seed value is used to help generate the same output for the same input. The default value is None.

    Note: Please make sure to modify the default parameters as required for your use case.
    """

    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    response_mime_type: Optional[str] = None
    candidate_count: Optional[int] = None
    stop_sequences: Optional[list] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None

    def __init__(
        self,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        response_mime_type: Optional[str] = None,
        candidate_count: Optional[int] = None,
        stop_sequences: Optional[list] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> List[str]:
        supported_params = [
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "tools",
            "functions",
            "tool_choice",
            "response_format",
            "n",
            "stop",
            "frequency_penalty",
            "presence_penalty",
            "extra_headers",
            "seed",
            "logprobs",
            "top_logprobs",
            "modalities",
            "parallel_tool_calls",
            "web_search_options",
        ]
        if supports_reasoning(model):
            supported_params.append("reasoning_effort")
            supported_params.append("thinking")
        return supported_params

    def map_tool_choice_values(
        self, model: str, tool_choice: Union[str, dict]
    ) -> Optional[ToolConfig]:
        if tool_choice == "none":
            return ToolConfig(functionCallingConfig=FunctionCallingConfig(mode="NONE"))
        elif tool_choice == "required":
            return ToolConfig(functionCallingConfig=FunctionCallingConfig(mode="ANY"))
        elif tool_choice == "auto":
            return ToolConfig(functionCallingConfig=FunctionCallingConfig(mode="AUTO"))
        elif isinstance(tool_choice, dict):
            # only supported for anthropic + mistral models - https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
            name = tool_choice.get("function", {}).get("name", "")
            return ToolConfig(
                functionCallingConfig=FunctionCallingConfig(
                    mode="ANY", allowed_function_names=[name]
                )
            )
        else:
            raise litellm.utils.UnsupportedParamsError(
                message="VertexAI doesn't support tool_choice={}. Supported tool_choice values=['auto', 'required', json object]. To drop it from the call, set `litellm.drop_params = True.".format(
                    tool_choice
                ),
                status_code=400,
            )

    def _map_web_search_options(self, value: dict) -> Tools:
        """
        Base Case: empty dict

        Google doesn't support user_location or search_context_size params
        """
        return Tools(googleSearch={})

    def _map_function(self, value: List[dict]) -> List[Tools]:  # noqa: PLR0915
        gtool_func_declarations = []
        googleSearch: Optional[dict] = None
        googleSearchRetrieval: Optional[dict] = None
        enterpriseWebSearch: Optional[dict] = None
        urlContext: Optional[dict] = None
        code_execution: Optional[dict] = None
        # remove 'additionalProperties' from tools
        value = _remove_additional_properties(value)
        # remove 'strict' from tools
        value = _remove_strict_from_schema(value)

        def get_tool_value(tool: dict, tool_name: str) -> Optional[dict]:
            """
            Helper function to get tool value handling both camelCase and underscore_case variants

            Args:
                tool (dict): The tool dictionary
                tool_name (str): The base tool name (e.g. "codeExecution")

            Returns:
                Optional[dict]: The tool value if found, None otherwise
            """
            # Convert camelCase to underscore_case
            underscore_name = "".join(
                ["_" + c.lower() if c.isupper() else c for c in tool_name]
            ).lstrip("_")
            # Try both camelCase and underscore_case variants

            if tool.get(tool_name) is not None:
                return tool.get(tool_name)
            elif tool.get(underscore_name) is not None:
                return tool.get(underscore_name)
            else:
                return None

        for tool in value:
            openai_function_object: Optional[
                ChatCompletionToolParamFunctionChunk
            ] = None
            if "function" in tool:  # tools list
                _openai_function_object = ChatCompletionToolParamFunctionChunk(  # type: ignore
                    **tool["function"]
                )

                if (
                    "parameters" in _openai_function_object
                    and _openai_function_object["parameters"] is not None
                    and isinstance(_openai_function_object["parameters"], dict)
                ):  # OPENAI accepts JSON Schema, Google accepts OpenAPI schema.
                    _openai_function_object["parameters"] = _build_vertex_schema(
                        _openai_function_object["parameters"]
                    )

                openai_function_object = _openai_function_object

            elif "name" in tool:  # functions list
                openai_function_object = ChatCompletionToolParamFunctionChunk(**tool)  # type: ignore

            tool_name = list(tool.keys())[0] if len(tool.keys()) == 1 else None
            if tool_name and (
                tool_name == "codeExecution" or tool_name == "code_execution"
            ):  # code_execution maintained for backwards compatibility
                code_execution = get_tool_value(tool, "codeExecution")
            elif tool_name and tool_name == "googleSearch":
                googleSearch = get_tool_value(tool, "googleSearch")
            elif tool_name and tool_name == "googleSearchRetrieval":
                googleSearchRetrieval = get_tool_value(tool, "googleSearchRetrieval")
            elif tool_name and tool_name == "enterpriseWebSearch":
                enterpriseWebSearch = get_tool_value(tool, "enterpriseWebSearch")
            elif tool_name and tool_name == "urlContext":
                urlContext = get_tool_value(tool, "urlContext")
            elif openai_function_object is not None:
                gtool_func_declaration = FunctionDeclaration(
                    name=openai_function_object["name"],
                )
                _description = openai_function_object.get("description", None)
                _parameters = openai_function_object.get("parameters", None)
                if isinstance(_parameters, str) and len(_parameters) == 0:
                    _parameters = {
                        "type": "object",
                    }
                if _description is not None:
                    gtool_func_declaration["description"] = _description
                if _parameters is not None:
                    gtool_func_declaration["parameters"] = _parameters
                gtool_func_declarations.append(gtool_func_declaration)
            else:
                # assume it's a provider-specific param
                verbose_logger.warning(
                    "Invalid tool={}. Use `litellm.set_verbose` or `litellm --detailed_debug` to see raw request."
                )

        _tools = Tools(
            function_declarations=gtool_func_declarations,
        )
        if googleSearch is not None:
            _tools["googleSearch"] = googleSearch
        if googleSearchRetrieval is not None:
            _tools["googleSearchRetrieval"] = googleSearchRetrieval
        if enterpriseWebSearch is not None:
            _tools["enterpriseWebSearch"] = enterpriseWebSearch
        if code_execution is not None:
            _tools["code_execution"] = code_execution
        if urlContext is not None:
            _tools["url_context"] = urlContext
        return [_tools]

    def _map_response_schema(self, value: dict) -> dict:
        old_schema = deepcopy(value)
        if isinstance(old_schema, list):
            for item in old_schema:
                if isinstance(item, dict):
                    item = _build_vertex_schema(
                        parameters=item, add_property_ordering=True
                    )

        elif isinstance(old_schema, dict):
            old_schema = _build_vertex_schema(
                parameters=old_schema, add_property_ordering=True
            )
        return old_schema

    def apply_response_schema_transformation(self, value: dict, optional_params: dict):
        new_value = deepcopy(value)
        # remove 'additionalProperties' from json schema
        new_value = _remove_additional_properties(new_value)
        # remove 'strict' from json schema
        new_value = _remove_strict_from_schema(new_value)
        if new_value["type"] == "json_object":
            optional_params["response_mime_type"] = "application/json"
        elif new_value["type"] == "text":
            optional_params["response_mime_type"] = "text/plain"
        if "response_schema" in new_value:
            optional_params["response_mime_type"] = "application/json"
            optional_params["response_schema"] = new_value["response_schema"]
        elif new_value["type"] == "json_schema":  # type: ignore
            if "json_schema" in new_value and "schema" in new_value["json_schema"]:  # type: ignore
                optional_params["response_mime_type"] = "application/json"
                optional_params["response_schema"] = new_value["json_schema"]["schema"]  # type: ignore

        if "response_schema" in optional_params and isinstance(
            optional_params["response_schema"], dict
        ):
            optional_params["response_schema"] = self._map_response_schema(
                value=optional_params["response_schema"]
            )

    @staticmethod
    def _map_reasoning_effort_to_thinking_budget(
        reasoning_effort: str,
    ) -> GeminiThinkingConfig:
        if reasoning_effort == "low":
            return {
                "thinkingBudget": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
                "includeThoughts": True,
            }
        elif reasoning_effort == "medium":
            return {
                "thinkingBudget": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
                "includeThoughts": True,
            }
        elif reasoning_effort == "high":
            return {
                "thinkingBudget": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
                "includeThoughts": True,
            }
        elif reasoning_effort == "disable":
            return {
                "thinkingBudget": DEFAULT_REASONING_EFFORT_DISABLE_THINKING_BUDGET,
                "includeThoughts": False,
            }
        else:
            raise ValueError(f"Invalid reasoning effort: {reasoning_effort}")

    @staticmethod
    def _is_thinking_budget_zero(thinking_budget: Optional[int]) -> bool:
        return thinking_budget is not None and thinking_budget == 0

    @staticmethod
    def _map_thinking_param(
        thinking_param: AnthropicThinkingParam,
    ) -> GeminiThinkingConfig:
        thinking_enabled = thinking_param.get("type") == "enabled"
        thinking_budget = thinking_param.get("budget_tokens")

        params: GeminiThinkingConfig = {}
        if thinking_enabled and not VertexGeminiConfig._is_thinking_budget_zero(
            thinking_budget
        ):
            params["includeThoughts"] = True
        if thinking_budget is not None and isinstance(thinking_budget, int):
            params["thinkingBudget"] = thinking_budget

        return params

    def map_response_modalities(self, value: list) -> list:
        response_modalities = []
        for modality in value:
            if modality == "text":
                response_modalities.append("TEXT")
            elif modality == "image":
                response_modalities.append("IMAGE")
            elif modality == "audio":
                response_modalities.append("AUDIO")
            else:
                response_modalities.append("MODALITY_UNSPECIFIED")
        return response_modalities

    def validate_parallel_tool_calls(self, value: bool, non_default_params: dict):
        tools = non_default_params.get("tools", non_default_params.get("functions"))
        num_function_declarations = len(tools) if isinstance(tools, list) else 0
        if num_function_declarations > 1:
            raise litellm.utils.UnsupportedParamsError(
                message=(
                    "`parallel_tool_calls=False` is not supported by Gemini when multiple tools are "
                    "provided. Specify a single tool, or set "
                    "`parallel_tool_calls=True`. If you want to drop this param, set `litellm.drop_params = True` or pass in `(.., drop_params=True)` in the requst - https://docs.litellm.ai/docs/completion/drop_params"
                ),
                status_code=400,
            )

    def _map_audio_params(self, value: dict) -> dict:
        """
        Expected input:
        {
            "voice": "alloy",
            "format": "mp3",
        }

        Expected output:
        speechConfig = {
            voiceConfig: {
                prebuiltVoiceConfig: {
                    voiceName: "alloy",
                }
            }
        }
        """
        from litellm.types.llms.vertex_ai import (
            PrebuiltVoiceConfig,
            SpeechConfig,
            VoiceConfig,
        )

        # Validate audio format - Gemini TTS only supports pcm16
        audio_format = value.get("format")
        if audio_format is not None and audio_format != "pcm16":
            raise ValueError(
                f"Unsupported audio format for Gemini TTS models: {audio_format}. "
                f"Gemini TTS models only support 'pcm16' format as they return audio data in L16 PCM format. "
                f"Please set audio format to 'pcm16'."
            )

        # Map OpenAI audio parameter to Gemini speech config
        speech_config: SpeechConfig = {}

        if "voice" in value:
            prebuilt_voice_config: PrebuiltVoiceConfig = {"voiceName": value["voice"]}
            voice_config: VoiceConfig = {"prebuiltVoiceConfig": prebuilt_voice_config}
            speech_config["voiceConfig"] = voice_config

        return cast(dict, speech_config)

    def map_openai_params(  # noqa: PLR0915
        self,
        non_default_params: Dict,
        optional_params: Dict,
        model: str,
        drop_params: bool,
    ) -> Dict:
        for param, value in non_default_params.items():
            if param == "temperature":
                optional_params["temperature"] = value
            elif param == "top_p":
                optional_params["top_p"] = value
            elif (
                param == "stream" and value is True
            ):  # sending stream = False, can cause it to get passed unchecked and raise issues
                optional_params["stream"] = value
            elif param == "n":
                optional_params["candidate_count"] = value
            elif param == "audio" and isinstance(value, dict):
                optional_params["speechConfig"] = self._map_audio_params(value)
            elif param == "stop":
                if isinstance(value, str):
                    optional_params["stop_sequences"] = [value]
                elif isinstance(value, list):
                    optional_params["stop_sequences"] = value
            elif param == "max_tokens" or param == "max_completion_tokens":
                optional_params["max_output_tokens"] = value
            elif param == "response_format" and isinstance(value, dict):  # type: ignore
                self.apply_response_schema_transformation(
                    value=value, optional_params=optional_params
                )
            elif param == "frequency_penalty":
                optional_params["frequency_penalty"] = value
            elif param == "presence_penalty":
                optional_params["presence_penalty"] = value
            elif param == "logprobs":
                optional_params["responseLogprobs"] = value
            elif param == "top_logprobs":
                optional_params["logprobs"] = value
            elif (
                (param == "tools" or param == "functions")
                and isinstance(value, list)
                and value
            ):
                optional_params = self._add_tools_to_optional_params(
                    optional_params, self._map_function(value=value)
                )
            elif param == "tool_choice" and (
                isinstance(value, str) or isinstance(value, dict)
            ):
                _tool_choice_value = self.map_tool_choice_values(
                    model=model, tool_choice=value  # type: ignore
                )
                if _tool_choice_value is not None:
                    optional_params["tool_choice"] = _tool_choice_value
            elif param == "parallel_tool_calls":
                if value is False and not (
                    drop_params or litellm.drop_params
                ):  # if drop params is True, then we should just ignore this
                    self.validate_parallel_tool_calls(value, non_default_params)
                else:
                    optional_params["parallel_tool_calls"] = value
            elif param == "seed":
                optional_params["seed"] = value
            elif param == "reasoning_effort" and isinstance(value, str):
                optional_params[
                    "thinkingConfig"
                ] = VertexGeminiConfig._map_reasoning_effort_to_thinking_budget(value)
            elif param == "thinking":
                optional_params[
                    "thinkingConfig"
                ] = VertexGeminiConfig._map_thinking_param(
                    cast(AnthropicThinkingParam, value)
                )
            elif param == "modalities" and isinstance(value, list):
                response_modalities = self.map_response_modalities(value)
                optional_params["responseModalities"] = response_modalities
            elif param == "web_search_options" and value and isinstance(value, dict):
                _tools = self._map_web_search_options(value)
                optional_params = self._add_tools_to_optional_params(
                    optional_params, [_tools]
                )
        if litellm.vertex_ai_safety_settings is not None:
            optional_params["safety_settings"] = litellm.vertex_ai_safety_settings

        # if audio param is set, ensure responseModalities is set to AUDIO
        audio_param = optional_params.get("speechConfig")
        if audio_param is not None:
            if "responseModalities" not in optional_params:
                optional_params["responseModalities"] = ["AUDIO"]
            elif "AUDIO" not in optional_params["responseModalities"]:
                optional_params["responseModalities"].append("AUDIO")

        return optional_params

    def get_mapped_special_auth_params(self) -> dict:
        """
        Common auth params across bedrock/vertex_ai/azure/watsonx
        """
        return {"project": "vertex_project", "region_name": "vertex_location"}

    def map_special_auth_params(self, non_default_params: dict, optional_params: dict):
        mapped_params = self.get_mapped_special_auth_params()

        for param, value in non_default_params.items():
            if param in mapped_params:
                optional_params[mapped_params[param]] = value
        return optional_params

    def get_eu_regions(self) -> List[str]:
        """
        Source: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#available-regions
        """
        return [
            "europe-central2",
            "europe-north1",
            "europe-southwest1",
            "europe-west1",
            "europe-west2",
            "europe-west3",
            "europe-west4",
            "europe-west6",
            "europe-west8",
            "europe-west9",
        ]

    @staticmethod
    def get_model_for_vertex_ai_url(model: str) -> str:
        """
        Returns the model name to use in the request to Vertex AI

        Handles 2 cases:
        1. User passed `model="vertex_ai/gemini/ft-uuid"`, we need to return `ft-uuid` for the request to Vertex AI
        2. User passed `model="vertex_ai/gemini-2.0-flash-001"`, we need to return `gemini-2.0-flash-001` for the request to Vertex AI

        Args:
            model (str): The model name to use in the request to Vertex AI

        Returns:
            str: The model name to use in the request to Vertex AI
        """
        if VertexGeminiConfig._is_model_gemini_spec_model(model):
            return VertexGeminiConfig._get_model_name_from_gemini_spec_model(model)
        return model

    @staticmethod
    def _is_model_gemini_spec_model(model: Optional[str]) -> bool:
        """
        Returns true if user is trying to call custom model in `/gemini` request/response format
        """
        if model is None:
            return False
        if "gemini/" in model:
            return True
        return False

    @staticmethod
    def _get_model_name_from_gemini_spec_model(model: str) -> str:
        """
        Returns the model name if model="vertex_ai/gemini/<unique_id>"

        Example:
        - model = "gemini/1234567890"
        - returns "1234567890"
        """
        if "gemini/" in model:
            return model.split("/")[-1]
        return model

    def get_flagged_finish_reasons(self) -> Dict[str, str]:
        """
        Return Dictionary of finish reasons which indicate response was flagged

        and what it means
        """
        return {
            "SAFETY": "The token generation was stopped as the response was flagged for safety reasons. NOTE: When streaming the Candidate.content will be empty if content filters blocked the output.",
            "RECITATION": "The token generation was stopped as the response was flagged for unauthorized citations.",
            "BLOCKLIST": "The token generation was stopped as the response was flagged for the terms which are included from the terminology blocklist.",
            "PROHIBITED_CONTENT": "The token generation was stopped as the response was flagged for the prohibited contents.",
            "SPII": "The token generation was stopped as the response was flagged for Sensitive Personally Identifiable Information (SPII) contents.",
            "IMAGE_SAFETY": "The token generation was stopped as the response was flagged for image safety reasons.",
        }

    @staticmethod
    def get_finish_reason_mapping() -> Dict[str, OpenAIChatCompletionFinishReason]:
        """
        Return Dictionary of finish reasons which indicate response was flagged

        and what it means
        """
        return {
            "FINISH_REASON_UNSPECIFIED": "stop",  # openai doesn't have a way of representing this
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter",
            "LANGUAGE": "content_filter",
            "OTHER": "content_filter",
            "BLOCKLIST": "content_filter",
            "PROHIBITED_CONTENT": "content_filter",
            "SPII": "content_filter",
            "MALFORMED_FUNCTION_CALL": "stop",  # openai doesn't have a way of representing this
            "IMAGE_SAFETY": "content_filter",
        }

    def translate_exception_str(self, exception_string: str):
        if (
            "GenerateContentRequest.tools[0].function_declarations[0].parameters.properties: should be non-empty for OBJECT type"
            in exception_string
        ):
            return "'properties' field in tools[0]['function']['parameters'] cannot be empty if 'type' == 'object'. Received error from provider - {}".format(
                exception_string
            )
        return exception_string

    def get_assistant_content_message(
        self, parts: List[HttpxPartType]
    ) -> Tuple[Optional[str], Optional[str]]:
        content_str: Optional[str] = None
        reasoning_content_str: Optional[str] = None

        for part in parts:
            _content_str = ""
            if "text" in part:
                text_content = part["text"]
                # Check if text content is audio data URI - if so, exclude from text content
                if text_content.startswith("data:audio") and ";base64," in text_content:
                    try:
                        if is_base64_encoded(text_content):
                            media_type, _ = text_content.split("data:")[1].split(
                                ";base64,"
                            )
                            if media_type.startswith("audio/"):
                                continue
                    except (ValueError, IndexError):
                        # If parsing fails, treat as regular text
                        pass
                _content_str += text_content
            elif "inlineData" in part:
                mime_type = part["inlineData"]["mimeType"]
                data = part["inlineData"]["data"]
                # Check if inline data is audio - if so, exclude from text content
                if mime_type.startswith("audio/"):
                    continue
                _content_str += "data:{};base64,{}".format(mime_type, data)

            if len(_content_str) > 0:
                if part.get("thought") is True:
                    if reasoning_content_str is None:
                        reasoning_content_str = ""
                    reasoning_content_str += _content_str
                else:
                    if content_str is None:
                        content_str = ""
                    content_str += _content_str

        return content_str, reasoning_content_str

    def _extract_audio_response_from_parts(
        self, parts: List[HttpxPartType]
    ) -> Optional[ChatCompletionAudioResponse]:
        """Extract audio response from parts if present"""
        for part in parts:
            if "text" in part:
                text_content = part["text"]
                # Check if text content contains audio data URI
                if text_content.startswith("data:audio") and ";base64," in text_content:
                    try:
                        if is_base64_encoded(text_content):
                            media_type, audio_data = text_content.split("data:")[
                                1
                            ].split(";base64,")

                            if media_type.startswith("audio/"):
                                expires_at = int(time.time()) + (24 * 60 * 60)
                                transcript = ""  # Gemini doesn't provide transcript

                                return ChatCompletionAudioResponse(
                                    data=audio_data,
                                    expires_at=expires_at,
                                    transcript=transcript,
                                )
                    except (ValueError, IndexError):
                        pass

            elif "inlineData" in part:
                mime_type = part["inlineData"]["mimeType"]
                data = part["inlineData"]["data"]

                if mime_type.startswith("audio/"):
                    expires_at = int(time.time()) + (24 * 60 * 60)
                    transcript = ""  # Gemini doesn't provide transcript

                    return ChatCompletionAudioResponse(
                        data=data, expires_at=expires_at, transcript=transcript
                    )

        return None

    @staticmethod
    def _transform_parts(
        parts: List[HttpxPartType],
        cumulative_tool_call_idx: int,
        is_function_call: Optional[bool],
    ) -> Tuple[
        Optional[ChatCompletionToolCallFunctionChunk],
        Optional[List[ChatCompletionToolCallChunk]],
        int,
    ]:
        function: Optional[ChatCompletionToolCallFunctionChunk] = None
        _tools: List[ChatCompletionToolCallChunk] = []
        for part in parts:
            if "functionCall" in part:
                _function_chunk = ChatCompletionToolCallFunctionChunk(
                    name=part["functionCall"]["name"],
                    arguments=json.dumps(part["functionCall"]["args"]),
                )
                if is_function_call is True:
                    function = _function_chunk
                else:
                    _tool_response_chunk = ChatCompletionToolCallChunk(
                        id=f"call_{uuid.uuid4().hex[:28]}",
                        type="function",
                        function=_function_chunk,
                        index=cumulative_tool_call_idx,
                    )
                    _tools.append(_tool_response_chunk)
                cumulative_tool_call_idx += 1
        if len(_tools) == 0:
            tools: Optional[List[ChatCompletionToolCallChunk]] = None
        else:
            tools = _tools
        return function, tools, cumulative_tool_call_idx

    @staticmethod
    def _transform_logprobs(
        logprobs_result: Optional[LogprobsResult],
    ) -> Optional[ChoiceLogprobs]:
        if logprobs_result is None:
            return None
        if "chosenCandidates" not in logprobs_result:
            return None
        logprobs_list: List[ChatCompletionTokenLogprob] = []
        for index, candidate in enumerate(logprobs_result["chosenCandidates"]):
            top_logprobs: List[TopLogprob] = []
            if "topCandidates" in logprobs_result and index < len(
                logprobs_result["topCandidates"]
            ):
                top_candidates_for_index = logprobs_result["topCandidates"][index][
                    "candidates"
                ]

                for options in top_candidates_for_index:
                    top_logprobs.append(
                        TopLogprob(
                            token=options["token"], logprob=options["logProbability"]
                        )
                    )
            logprobs_list.append(
                ChatCompletionTokenLogprob(
                    token=candidate["token"],
                    logprob=candidate["logProbability"],
                    top_logprobs=top_logprobs,
                )
            )
        return ChoiceLogprobs(content=logprobs_list)

    def _handle_blocked_response(
        self,
        model_response: ModelResponse,
        completion_response: GenerateContentResponseBody,
    ) -> ModelResponse:
        # If set, the prompt was blocked and no candidates are returned. Rephrase your prompt
        model_response.choices[0].finish_reason = "content_filter"

        chat_completion_message: ChatCompletionResponseMessage = {
            "role": "assistant",
            "content": None,
        }

        choice = litellm.Choices(
            finish_reason="content_filter",
            index=0,
            message=chat_completion_message,  # type: ignore
            logprobs=None,
            enhancements=None,
        )

        model_response.choices = [choice]

        ## GET USAGE ##
        usage = Usage(
            prompt_tokens=completion_response["usageMetadata"].get(
                "promptTokenCount", 0
            ),
            completion_tokens=completion_response["usageMetadata"].get(
                "candidatesTokenCount", 0
            ),
            total_tokens=completion_response["usageMetadata"].get("totalTokenCount", 0),
        )

        setattr(model_response, "usage", usage)

        return model_response

    def _handle_content_policy_violation(
        self,
        model_response: ModelResponse,
        completion_response: GenerateContentResponseBody,
    ) -> ModelResponse:
        ## CONTENT POLICY VIOLATION ERROR
        model_response.choices[0].finish_reason = "content_filter"

        _chat_completion_message = {
            "role": "assistant",
            "content": None,
        }

        choice = litellm.Choices(
            finish_reason="content_filter",
            index=0,
            message=_chat_completion_message,
            logprobs=None,
            enhancements=None,
        )

        model_response.choices = [choice]

        ## GET USAGE ##
        usage = Usage(
            prompt_tokens=completion_response["usageMetadata"].get(
                "promptTokenCount", 0
            ),
            completion_tokens=completion_response["usageMetadata"].get(
                "candidatesTokenCount", 0
            ),
            total_tokens=completion_response["usageMetadata"].get("totalTokenCount", 0),
        )

        setattr(model_response, "usage", usage)

        return model_response

    @staticmethod
    def is_candidate_token_count_inclusive(usage_metadata: UsageMetadata) -> bool:
        """
        Check if the candidate token count is inclusive of the thinking token count

        if prompttokencount + candidatesTokenCount == totalTokenCount, then the candidate token count is inclusive of the thinking token count

        else the candidate token count is exclusive of the thinking token count

        Addresses - https://github.com/BerriAI/litellm/pull/10141#discussion_r2052272035
        """
        if usage_metadata.get("promptTokenCount", 0) + usage_metadata.get(
            "candidatesTokenCount", 0
        ) == usage_metadata.get("totalTokenCount", 0):
            return True
        else:
            return False

    @staticmethod
    def _calculate_usage(
        completion_response: Union[
            GenerateContentResponseBody, BidiGenerateContentServerMessage
        ],
    ) -> Usage:
        if (
            completion_response is not None
            and "usageMetadata" not in completion_response
        ):
            raise ValueError(
                f"usageMetadata not found in completion_response. Got={completion_response}"
            )
        cached_tokens: Optional[int] = None
        audio_tokens: Optional[int] = None
        text_tokens: Optional[int] = None
        prompt_tokens_details: Optional[PromptTokensDetailsWrapper] = None
        reasoning_tokens: Optional[int] = None
        response_tokens: Optional[int] = None
        response_tokens_details: Optional[CompletionTokensDetailsWrapper] = None
        usage_metadata = completion_response["usageMetadata"]
        if "cachedContentTokenCount" in usage_metadata:
            cached_tokens = usage_metadata["cachedContentTokenCount"]

        ## GEMINI LIVE API ONLY PARAMS ##
        if "responseTokenCount" in usage_metadata:
            response_tokens = usage_metadata["responseTokenCount"]
        if "responseTokensDetails" in usage_metadata:
            response_tokens_details = CompletionTokensDetailsWrapper()
            for detail in usage_metadata["responseTokensDetails"]:
                if detail["modality"] == "TEXT":
                    response_tokens_details.text_tokens = detail.get("tokenCount", 0)
                elif detail["modality"] == "AUDIO":
                    response_tokens_details.audio_tokens = detail.get("tokenCount", 0)
        #########################################################

        if "promptTokensDetails" in usage_metadata:
            for detail in usage_metadata["promptTokensDetails"]:
                if detail["modality"] == "AUDIO":
                    audio_tokens = detail.get("tokenCount", 0)
                elif detail["modality"] == "TEXT":
                    text_tokens = detail.get("tokenCount", 0)
        if "thoughtsTokenCount" in usage_metadata:
            reasoning_tokens = usage_metadata["thoughtsTokenCount"]
        prompt_tokens_details = PromptTokensDetailsWrapper(
            cached_tokens=cached_tokens,
            audio_tokens=audio_tokens,
            text_tokens=text_tokens,
        )

        completion_tokens = response_tokens or completion_response["usageMetadata"].get(
            "candidatesTokenCount", 0
        )
        if (
            not VertexGeminiConfig.is_candidate_token_count_inclusive(usage_metadata)
            and reasoning_tokens
        ):
            completion_tokens = reasoning_tokens + completion_tokens
        ## GET USAGE ##
        usage = Usage(
            prompt_tokens=usage_metadata.get("promptTokenCount", 0),
            completion_tokens=completion_tokens,
            total_tokens=usage_metadata.get("totalTokenCount", 0),
            prompt_tokens_details=prompt_tokens_details,
            reasoning_tokens=reasoning_tokens,
            completion_tokens_details=response_tokens_details,
        )

        return usage

    @staticmethod
    def _check_finish_reason(
        chat_completion_message: Optional[ChatCompletionResponseMessage],
        finish_reason: Optional[str],
    ) -> OpenAIChatCompletionFinishReason:
        mapped_finish_reason = VertexGeminiConfig.get_finish_reason_mapping()
        if chat_completion_message and chat_completion_message.get("function_call"):
            return "function_call"
        elif chat_completion_message and chat_completion_message.get("tool_calls"):
            return "tool_calls"
        elif (
            finish_reason and finish_reason in mapped_finish_reason.keys()
        ):  # vertex ai
            return mapped_finish_reason[finish_reason]
        else:
            return "stop"

    @staticmethod
    def _calculate_web_search_requests(grounding_metadata: List[dict]) -> Optional[int]:
        web_search_requests: Optional[int] = None

        if (
            grounding_metadata
            and isinstance(grounding_metadata, list)
            and len(grounding_metadata) > 0
        ):
            for grounding_metadata_item in grounding_metadata:
                web_search_queries = grounding_metadata_item.get("webSearchQueries")
                if web_search_queries and web_search_requests:
                    web_search_requests += len(web_search_queries)
                elif web_search_queries:
                    web_search_requests = len(grounding_metadata)
        return web_search_requests

    @staticmethod
    def _process_candidates(
        _candidates: List[Candidates],
        model_response: Union[ModelResponse, "ModelResponseStream"],
        standard_optional_params: dict,
    ) -> Tuple[List[dict], List[dict], List, List]:
        """
        Helper method to process candidates and extract metadata

        Returns:
            grounding_metadata: List[dict]
            url_context_metadata: List[dict]
            safety_ratings: List
            citation_metadata: List
        """
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            is_function_call,
        )
        from litellm.types.utils import ModelResponseStream

        grounding_metadata: List[dict] = []
        url_context_metadata: List[dict] = []
        safety_ratings: List = []
        citation_metadata: List = []
        chat_completion_message: ChatCompletionResponseMessage = {"role": "assistant"}
        chat_completion_logprobs: Optional[ChoiceLogprobs] = None
        tools: Optional[List[ChatCompletionToolCallChunk]] = []
        functions: Optional[ChatCompletionToolCallFunctionChunk] = None
        cumulative_tool_call_index: int = 0

        for idx, candidate in enumerate(_candidates):
            if "content" not in candidate:
                continue

            if "groundingMetadata" in candidate:
                if isinstance(candidate["groundingMetadata"], list):
                    grounding_metadata.extend(candidate["groundingMetadata"])  # type: ignore
                else:
                    grounding_metadata.append(candidate["groundingMetadata"])  # type: ignore

            if "safetyRatings" in candidate:
                safety_ratings.append(candidate["safetyRatings"])

            if "citationMetadata" in candidate:
                citation_metadata.append(candidate["citationMetadata"])

            if "urlContextMetadata" in candidate:
                # Add URL context metadata to grounding metadata
                url_context_metadata.append(cast(dict, candidate["urlContextMetadata"]))

            if "parts" in candidate["content"]:
                (
                    content,
                    reasoning_content,
                ) = VertexGeminiConfig().get_assistant_content_message(
                    parts=candidate["content"]["parts"]
                )

                audio_response = (
                    VertexGeminiConfig()._extract_audio_response_from_parts(
                        parts=candidate["content"]["parts"]
                    )
                )

                if audio_response is not None:
                    cast(Dict[str, Any], chat_completion_message)[
                        "audio"
                    ] = audio_response
                    chat_completion_message["content"] = None  # OpenAI spec
                elif content is not None:
                    chat_completion_message["content"] = content

                if reasoning_content is not None:
                    chat_completion_message["reasoning_content"] = reasoning_content

                (
                    functions,
                    tools,
                    cumulative_tool_call_index,
                ) = VertexGeminiConfig._transform_parts(
                    parts=candidate["content"]["parts"],
                    cumulative_tool_call_idx=cumulative_tool_call_index,
                    is_function_call=is_function_call(standard_optional_params),
                )

            if "logprobsResult" in candidate:
                chat_completion_logprobs = VertexGeminiConfig._transform_logprobs(
                    logprobs_result=candidate["logprobsResult"]
                )

            if tools:
                chat_completion_message["tool_calls"] = tools

            if functions is not None:
                chat_completion_message["function_call"] = functions

            if isinstance(model_response, ModelResponseStream):
                from litellm.types.utils import Delta, StreamingChoices

                # create a streaming choice object
                choice = StreamingChoices(
                    finish_reason=VertexGeminiConfig._check_finish_reason(
                        chat_completion_message, candidate.get("finishReason")
                    ),
                    index=candidate.get("index", idx),
                    delta=Delta(
                        content=chat_completion_message.get("content"),
                        reasoning_content=chat_completion_message.get(
                            "reasoning_content"
                        ),
                        tool_calls=tools,
                        function_call=functions,
                    ),
                    logprobs=chat_completion_logprobs,
                    enhancements=None,
                )
                model_response.choices.append(choice)
            elif isinstance(model_response, ModelResponse):
                choice = litellm.Choices(
                    finish_reason=VertexGeminiConfig._check_finish_reason(
                        chat_completion_message, candidate.get("finishReason")
                    ),
                    index=candidate.get("index", idx),
                    message=chat_completion_message,  # type: ignore
                    logprobs=chat_completion_logprobs,
                    enhancements=None,
                )
                model_response.choices.append(choice)

        return (
            grounding_metadata,
            url_context_metadata,
            safety_ratings,
            citation_metadata,
        )

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LoggingClass,
        request_data: Dict,
        messages: List[AllMessageValues],
        optional_params: Dict,
        litellm_params: Dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )

        ## RESPONSE OBJECT
        try:
            completion_response = GenerateContentResponseBody(**raw_response.json())  # type: ignore
        except Exception as e:
            raise VertexAIError(
                message="Received={}, Error converting to valid response block={}. File an issue if litellm error - https://github.com/BerriAI/litellm/issues".format(
                    raw_response.text, str(e)
                ),
                status_code=422,
                headers=raw_response.headers,
            )

        return self._transform_google_generate_content_to_openai_model_response(
            completion_response=completion_response,
            model_response=model_response,
            model=model,
            logging_obj=logging_obj,
            raw_response=raw_response,
        )

    def _transform_google_generate_content_to_openai_model_response(
        self,
        completion_response: Union[GenerateContentResponseBody, dict],
        model_response: ModelResponse,
        model: str,
        logging_obj: LoggingClass,
        raw_response: httpx.Response,
    ) -> ModelResponse:
        """
        Transforms a Google GenAI generate content response to an OpenAI model response.
        """
        if isinstance(completion_response, dict):
            completion_response = GenerateContentResponseBody(**completion_response)  # type: ignore

        ## GET MODEL ##
        model_response.model = model

        ## CHECK IF RESPONSE FLAGGED
        if (
            "promptFeedback" in completion_response
            and "blockReason" in completion_response["promptFeedback"]
        ):
            return self._handle_blocked_response(
                model_response=model_response,
                completion_response=completion_response,
            )

        _candidates = completion_response.get("candidates")
        if _candidates and len(_candidates) > 0:
            content_policy_violations = (
                VertexGeminiConfig().get_flagged_finish_reasons()
            )
            if (
                "finishReason" in _candidates[0]
                and _candidates[0]["finishReason"] in content_policy_violations.keys()
            ):
                return self._handle_content_policy_violation(
                    model_response=model_response,
                    completion_response=completion_response,
                )

        model_response.choices = []
        response_id = completion_response.get("responseId")
        if response_id:
            model_response.id = response_id
        url_context_metadata: List[dict] = []
        try:
            grounding_metadata: List[dict] = []
            safety_ratings: List[dict] = []
            citation_metadata: List[dict] = []
            if _candidates:
                (
                    grounding_metadata,
                    url_context_metadata,
                    safety_ratings,
                    citation_metadata,
                ) = VertexGeminiConfig._process_candidates(
                    _candidates, model_response, logging_obj.optional_params
                )

            usage = VertexGeminiConfig._calculate_usage(
                completion_response=completion_response
            )
            setattr(model_response, "usage", usage)

            ## ADD METADATA TO RESPONSE ##

            setattr(model_response, "vertex_ai_grounding_metadata", grounding_metadata)
            model_response._hidden_params[
                "vertex_ai_grounding_metadata"
            ] = grounding_metadata

            setattr(
                model_response, "vertex_ai_url_context_metadata", url_context_metadata
            )

            model_response._hidden_params[
                "vertex_ai_url_context_metadata"
            ] = url_context_metadata

            setattr(model_response, "vertex_ai_safety_results", safety_ratings)
            model_response._hidden_params[
                "vertex_ai_safety_results"
            ] = safety_ratings  # older approach - maintaining to prevent regressions

            ## ADD CITATION METADATA ##
            setattr(model_response, "vertex_ai_citation_metadata", citation_metadata)
            model_response._hidden_params[
                "vertex_ai_citation_metadata"
            ] = citation_metadata  # older approach - maintaining to prevent regressions

        except Exception as e:
            raise VertexAIError(
                message="Received={}, Error converting to valid response block={}. File an issue if litellm error - https://github.com/BerriAI/litellm/issues".format(
                    completion_response, str(e)
                ),
                status_code=422,
                headers=raw_response.headers,
            )

        return model_response

    def _transform_messages(
        self, messages: List[AllMessageValues]
    ) -> List[ContentType]:
        return _gemini_convert_messages_with_history(messages=messages)

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[Dict, httpx.Headers]
    ) -> BaseLLMException:
        return VertexAIError(
            message=error_message, status_code=status_code, headers=headers
        )

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: Dict,
        litellm_params: Dict,
        headers: Dict,
    ) -> Dict:
        raise NotImplementedError(
            "Vertex AI has a custom implementation of transform_request. Needs sync + async."
        )

    def validate_environment(
        self,
        headers: Optional[Dict],
        model: str,
        messages: List[AllMessageValues],
        optional_params: Dict,
        litellm_params: Dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict:
        default_headers = {
            "Content-Type": "application/json",
        }
        if api_key is not None:
            default_headers["Authorization"] = f"Bearer {api_key}"
        if headers is not None:
            default_headers.update(headers)

        return default_headers


async def make_call(
    client: Optional[AsyncHTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj,
):
    if client is None:
        client = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.VERTEX_AI,
        )

    try:
        response = await client.post(api_base, headers=headers, data=data, stream=True)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        exception_string = str(await e.response.aread())
        raise VertexAIError(
            status_code=e.response.status_code,
            message=VertexGeminiConfig().translate_exception_str(exception_string),
            headers=e.response.headers,
        )
    if response.status_code != 200 and response.status_code != 201:
        raise VertexAIError(
            status_code=response.status_code,
            message=response.text,
            headers=response.headers,
        )

    completion_stream = ModelResponseIterator(
        streaming_response=response.aiter_lines(),
        sync_stream=False,
        logging_obj=logging_obj,
    )
    # LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response="first stream response received",
        additional_args={"complete_input_dict": data},
    )

    return completion_stream


def make_sync_call(
    client: Optional[HTTPHandler],  # module-level client
    gemini_client: Optional[HTTPHandler],  # if passed by user
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj,
):
    if gemini_client is not None:
        client = gemini_client
    if client is None:
        client = HTTPHandler()  # Create a new client if none provided

    response = client.post(api_base, headers=headers, data=data, stream=True)

    if response.status_code != 200 and response.status_code != 201:
        raise VertexAIError(
            status_code=response.status_code,
            message=str(response.read()),
            headers=response.headers,
        )

    completion_stream = ModelResponseIterator(
        streaming_response=response.iter_lines(),
        sync_stream=True,
        logging_obj=logging_obj,
    )

    # LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response="first stream response received",
        additional_args={"complete_input_dict": data},
    )

    return completion_stream


class VertexLLM(VertexBase):
    def __init__(self) -> None:
        super().__init__()

    async def async_streaming(
        self,
        model: str,
        custom_llm_provider: Literal[
            "vertex_ai", "vertex_ai_beta", "gemini"
        ],  # if it's vertex_ai or gemini (google ai studio)
        messages: list,
        model_response: ModelResponse,
        print_verbose: Callable,
        data: dict,
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj,
        stream,
        optional_params: dict,
        litellm_params: dict,
        logger_fn=None,
        api_base: Optional[str] = None,
        client: Optional[AsyncHTTPHandler] = None,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES] = None,
        gemini_api_key: Optional[str] = None,
        extra_headers: Optional[dict] = None,
    ) -> CustomStreamWrapper:
        request_body = await async_transform_request_body(**data)  # type: ignore

        should_use_v1beta1_features = self.is_using_v1beta1_features(
            optional_params=optional_params
        )

        _auth_header, vertex_project = await self._ensure_access_token_async(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider=custom_llm_provider,
        )

        auth_header, api_base = self._get_token_and_url(
            model=model,
            gemini_api_key=gemini_api_key,
            auth_header=_auth_header,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=stream,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            should_use_v1beta1_features=should_use_v1beta1_features,
        )

        headers = VertexGeminiConfig().validate_environment(
            api_key=auth_header,
            headers=extra_headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": request_body,
                "api_base": api_base,
                "headers": headers,
            },
        )

        request_body_str = json.dumps(request_body)
        streaming_response = CustomStreamWrapper(
            completion_stream=None,
            make_call=partial(
                make_call,
                client=client,
                api_base=api_base,
                headers=headers,
                data=request_body_str,
                model=model,
                messages=messages,
                logging_obj=logging_obj,
            ),
            model=model,
            custom_llm_provider="vertex_ai_beta",
            logging_obj=logging_obj,
        )
        return streaming_response

    async def async_completion(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        print_verbose: Callable,
        data: dict,
        custom_llm_provider: Literal[
            "vertex_ai", "vertex_ai_beta", "gemini"
        ],  # if it's vertex_ai or gemini (google ai studio)
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj,
        stream,
        optional_params: dict,
        litellm_params: dict,
        logger_fn=None,
        api_base: Optional[str] = None,
        client: Optional[AsyncHTTPHandler] = None,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES] = None,
        gemini_api_key: Optional[str] = None,
        extra_headers: Optional[dict] = None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        should_use_v1beta1_features = self.is_using_v1beta1_features(
            optional_params=optional_params
        )

        _auth_header, vertex_project = await self._ensure_access_token_async(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider=custom_llm_provider,
        )

        auth_header, api_base = self._get_token_and_url(
            model=model,
            gemini_api_key=gemini_api_key,
            auth_header=_auth_header,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=stream,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            should_use_v1beta1_features=should_use_v1beta1_features,
        )

        headers = VertexGeminiConfig().validate_environment(
            api_key=auth_header,
            headers=extra_headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        request_body = await async_transform_request_body(**data)  # type: ignore
        _async_client_params = {}
        if timeout:
            _async_client_params["timeout"] = timeout
        if client is None or not isinstance(client, AsyncHTTPHandler):
            client = get_async_httpx_client(
                params=_async_client_params, llm_provider=litellm.LlmProviders.VERTEX_AI
            )
        else:
            client = client  # type: ignore
        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": request_body,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = await client.post(
                api_base, headers=headers, json=cast(dict, request_body)
            )  # type: ignore
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise VertexAIError(
                status_code=error_code,
                message=err.response.text,
                headers=err.response.headers,
            )
        except httpx.TimeoutException:
            raise VertexAIError(
                status_code=408,
                message="Timeout error occurred.",
                headers=None,
            )

        return VertexGeminiConfig().transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key="",
            request_data=cast(dict, request_body),
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
        )

    def completion(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        print_verbose: Callable,
        custom_llm_provider: Literal[
            "vertex_ai", "vertex_ai_beta", "gemini"
        ],  # if it's vertex_ai or gemini (google ai studio)
        encoding,
        logging_obj,
        optional_params: dict,
        acompletion: bool,
        timeout: Optional[Union[float, httpx.Timeout]],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        gemini_api_key: Optional[str],
        litellm_params: dict,
        logger_fn=None,
        extra_headers: Optional[dict] = None,
        client: Optional[Union[AsyncHTTPHandler, HTTPHandler]] = None,
        api_base: Optional[str] = None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        stream: Optional[bool] = optional_params.pop("stream", None)  # type: ignore

        transform_request_params = {
            "gemini_api_key": gemini_api_key,
            "messages": messages,
            "api_base": api_base,
            "model": model,
            "client": client,
            "timeout": timeout,
            "extra_headers": extra_headers,
            "optional_params": optional_params,
            "logging_obj": logging_obj,
            "custom_llm_provider": custom_llm_provider,
            "litellm_params": litellm_params,
        }

        ### ROUTING (ASYNC, STREAMING, SYNC)
        if acompletion:
            ### ASYNC STREAMING
            if stream is True:
                return self.async_streaming(
                    model=model,
                    messages=messages,
                    api_base=api_base,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    encoding=encoding,
                    logging_obj=logging_obj,
                    optional_params=optional_params,
                    stream=stream,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    timeout=timeout,
                    client=client,  # type: ignore
                    data=transform_request_params,
                    vertex_project=vertex_project,
                    vertex_location=vertex_location,
                    vertex_credentials=vertex_credentials,
                    gemini_api_key=gemini_api_key,
                    custom_llm_provider=custom_llm_provider,
                    extra_headers=extra_headers,
                )
            ### ASYNC COMPLETION
            return self.async_completion(
                model=model,
                messages=messages,
                data=transform_request_params,  # type: ignore
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                encoding=encoding,
                logging_obj=logging_obj,
                optional_params=optional_params,
                stream=stream,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                timeout=timeout,
                client=client,  # type: ignore
                vertex_project=vertex_project,
                vertex_location=vertex_location,
                vertex_credentials=vertex_credentials,
                gemini_api_key=gemini_api_key,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
            )

        should_use_v1beta1_features = self.is_using_v1beta1_features(
            optional_params=optional_params
        )

        _auth_header, vertex_project = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider=custom_llm_provider,
        )

        auth_header, url = self._get_token_and_url(
            model=model,
            gemini_api_key=gemini_api_key,
            auth_header=_auth_header,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=stream,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            should_use_v1beta1_features=should_use_v1beta1_features,
        )
        headers = VertexGeminiConfig().validate_environment(
            api_key=auth_header,
            headers=extra_headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        ## TRANSFORMATION ##
        data = sync_transform_request_body(**transform_request_params)

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": url,
                "headers": headers,
            },
        )

        ## SYNC STREAMING CALL ##
        if stream is True:
            request_data_str = json.dumps(data)
            streaming_response = CustomStreamWrapper(
                completion_stream=None,
                make_call=partial(
                    make_sync_call,
                    gemini_client=(
                        client
                        if client is not None and isinstance(client, HTTPHandler)
                        else None
                    ),
                    api_base=url,
                    data=request_data_str,
                    model=model,
                    messages=messages,
                    logging_obj=logging_obj,
                    headers=headers,
                ),
                model=model,
                custom_llm_provider="vertex_ai_beta",
                logging_obj=logging_obj,
            )

            return streaming_response
        ## COMPLETION CALL ##

        if client is None or isinstance(client, AsyncHTTPHandler):
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            client = _get_httpx_client(params=_params)
        else:
            client = client

        try:
            response = client.post(url=url, headers=headers, json=data)  # type: ignore
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise VertexAIError(
                status_code=error_code,
                message=err.response.text,
                headers=err.response.headers,
            )
        except httpx.TimeoutException:
            raise VertexAIError(
                status_code=408,
                message="Timeout error occurred.",
                headers=None,
            )

        return VertexGeminiConfig().transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key="",
            request_data=data,  # type: ignore
            messages=messages,
            encoding=encoding,
        )


class ModelResponseIterator:
    def __init__(
        self, streaming_response, sync_stream: bool, logging_obj: LoggingClass
    ):
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            check_is_function_call,
        )

        self.streaming_response = streaming_response
        self.chunk_type: Literal["valid_json", "accumulated_json"] = "valid_json"
        self.accumulated_json = ""
        self.sent_first_chunk = False
        self.logging_obj = logging_obj
        self.is_function_call = check_is_function_call(logging_obj)

    def chunk_parser(self, chunk: dict) -> Optional["ModelResponseStream"]:
        try:
            verbose_logger.debug(f"RAW GEMINI CHUNK: {chunk}")
            from litellm.types.utils import ModelResponseStream

            processed_chunk = GenerateContentResponseBody(**chunk)  # type: ignore
            response_id = processed_chunk.get("responseId")
            model_response = ModelResponseStream(choices=[], id=response_id)
            usage: Optional[Usage] = None
            _candidates: Optional[List[Candidates]] = processed_chunk.get("candidates")
            grounding_metadata: List[dict] = []
            url_context_metadata: List[dict] = []
            safety_ratings: List[dict] = []
            citation_metadata: List[dict] = []
            if _candidates:
                (
                    grounding_metadata,
                    url_context_metadata,
                    safety_ratings,
                    citation_metadata,
                ) = VertexGeminiConfig._process_candidates(
                    _candidates, model_response, self.logging_obj.optional_params
                )

                setattr(model_response, "vertex_ai_grounding_metadata", grounding_metadata)  # type: ignore
                setattr(model_response, "vertex_ai_url_context_metadata", url_context_metadata)  # type: ignore
                setattr(model_response, "vertex_ai_safety_ratings", safety_ratings)  # type: ignore
                setattr(model_response, "vertex_ai_citation_metadata", citation_metadata)  # type: ignore

            if "usageMetadata" in processed_chunk:
                usage = VertexGeminiConfig._calculate_usage(
                    completion_response=processed_chunk,
                )

                web_search_requests = VertexGeminiConfig._calculate_web_search_requests(
                    grounding_metadata
                )
                if web_search_requests is not None:
                    cast(
                        PromptTokensDetailsWrapper, usage.prompt_tokens_details
                    ).web_search_requests = web_search_requests

            setattr(model_response, "usage", usage)  # type: ignore

            model_response._hidden_params["is_finished"] = False
            return model_response

        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from chunk: {chunk}")

    # Sync iterator
    def __iter__(self):
        self.response_iterator = self.streaming_response
        return self

    def handle_valid_json_chunk(self, chunk: str) -> Optional["ModelResponseStream"]:
        chunk = chunk.strip()
        try:
            json_chunk = json.loads(chunk)

        except json.JSONDecodeError as e:
            if (
                self.sent_first_chunk is False
            ):  # only check for accumulated json, on first chunk, else raise error. Prevent real errors from being masked.
                self.chunk_type = "accumulated_json"
                return self.handle_accumulated_json_chunk(chunk=chunk)
            raise e

        if self.sent_first_chunk is False:
            self.sent_first_chunk = True

        return self.chunk_parser(chunk=json_chunk)

    def handle_accumulated_json_chunk(
        self, chunk: str
    ) -> Optional["ModelResponseStream"]:
        chunk = litellm.CustomStreamWrapper._strip_sse_data_from_chunk(chunk) or ""
        message = chunk.replace("\n\n", "")

        # Accumulate JSON data
        self.accumulated_json += message

        # Try to parse the accumulated JSON
        try:
            _data = json.loads(self.accumulated_json)
            self.accumulated_json = ""  # reset after successful parsing
            return self.chunk_parser(chunk=_data)
        except json.JSONDecodeError:
            # If it's not valid JSON yet, continue to the next event
            return None

    def _common_chunk_parsing_logic(
        self, chunk: str
    ) -> Optional["ModelResponseStream"]:
        try:
            chunk = litellm.CustomStreamWrapper._strip_sse_data_from_chunk(chunk) or ""
            if len(chunk) > 0:
                """
                Check if initial chunk valid json
                - if partial json -> enter accumulated json logic
                - if valid - continue
                """
                if self.chunk_type == "valid_json":
                    return self.handle_valid_json_chunk(chunk=chunk)
                elif self.chunk_type == "accumulated_json":
                    return self.handle_accumulated_json_chunk(chunk=chunk)

            return None
        except Exception:
            raise

    def __next__(self):
        try:
            chunk = self.response_iterator.__next__()
        except StopIteration:
            if self.chunk_type == "accumulated_json" and self.accumulated_json:
                return self.handle_accumulated_json_chunk(chunk="")
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            return self._common_chunk_parsing_logic(chunk=chunk)
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")

    # Async iterator
    def __aiter__(self):
        self.async_response_iterator = self.streaming_response.__aiter__()
        return self

    async def __anext__(self):
        try:
            chunk = await self.async_response_iterator.__anext__()
        except StopAsyncIteration:
            if self.chunk_type == "accumulated_json" and self.accumulated_json:
                return self.handle_accumulated_json_chunk(chunk="")
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            return self._common_chunk_parsing_logic(chunk=chunk)
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")
```

## File: `llms/vertex_ai/google_genai/transformation.py`
```
"""
Transformation for Calling Google models in their native format.
"""
from typing import Literal

from litellm.llms.gemini.google_genai.transformation import GoogleGenAIConfig


class VertexAIGoogleGenAIConfig(GoogleGenAIConfig):
    """
    Configuration for calling Google models in their native format.
    """
    @property
    def custom_llm_provider(self) -> Literal["gemini", "vertex_ai"]:
        return "vertex_ai"
    ```

## File: `llms/vertex_ai/image_generation/cost_calculator.py`
```
"""
Vertex AI Image Generation Cost Calculator
"""

import litellm
from litellm.types.utils import ImageResponse


def cost_calculator(
    model: str,
    image_response: ImageResponse,
) -> float:
    """
    Vertex AI Image Generation Cost Calculator
    """
    _model_info = litellm.get_model_info(
        model=model,
        custom_llm_provider="vertex_ai",
    )

    output_cost_per_image: float = _model_info.get("output_cost_per_image") or 0.0
    num_images: int = 0
    if image_response.data:
        num_images = len(image_response.data)
    return output_cost_per_image * num_images
```

## File: `llms/vertex_ai/image_generation/image_generation_handler.py`
```
import json
from typing import Any, Dict, List, Optional

import httpx
from openai.types.image import Image

import litellm
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from litellm.types.utils import ImageResponse


class VertexImageGeneration(VertexLLM):
    def process_image_generation_response(
        self,
        json_response: Dict[str, Any],
        model_response: ImageResponse,
        model: Optional[str] = None,
    ) -> ImageResponse:
        if "predictions" not in json_response:
            raise litellm.InternalServerError(
                message=f"image generation response does not contain 'predictions', got {json_response}",
                llm_provider="vertex_ai",
                model=model,
            )

        predictions = json_response["predictions"]
        response_data: List[Image] = []

        for prediction in predictions:
            bytes_base64_encoded = prediction["bytesBase64Encoded"]
            image_object = Image(b64_json=bytes_base64_encoded)
            response_data.append(image_object)

        model_response.data = response_data
        return model_response

    def transform_optional_params(self, optional_params: Optional[dict]) -> dict:
        """
        Transform the optional params to the format expected by the Vertex AI API.
        For example, "aspect_ratio" is transformed to "aspectRatio".
        """
        if optional_params is None:
            return {
                "sampleCount": 1,
            }

        def snake_to_camel(snake_str: str) -> str:
            """Convert snake_case to camelCase"""
            components = snake_str.split("_")
            return components[0] + "".join(word.capitalize() for word in components[1:])

        transformed_params = {}
        for key, value in optional_params.items():
            if "_" in key:
                camel_case_key = snake_to_camel(key)
                transformed_params[camel_case_key] = value
            else:
                transformed_params[key] = value

        return transformed_params

    def image_generation(
        self,
        prompt: str,
        api_base: Optional[str],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        model_response: ImageResponse,
        logging_obj: Any,
        model: str = "imagegeneration",  # vertex ai uses imagegeneration as the default model
        client: Optional[Any] = None,
        optional_params: Optional[dict] = None,
        timeout: Optional[int] = None,
        aimg_generation=False,
        extra_headers: Optional[dict] = None,
    ) -> ImageResponse:
        if aimg_generation is True:
            return self.aimage_generation(  # type: ignore
                prompt=prompt,
                api_base=api_base,
                vertex_project=vertex_project,
                vertex_location=vertex_location,
                vertex_credentials=vertex_credentials,
                model=model,
                client=client,
                optional_params=optional_params,
                timeout=timeout,
                logging_obj=logging_obj,
                model_response=model_response,
            )

        if client is None:
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    _httpx_timeout = httpx.Timeout(timeout)
                    _params["timeout"] = _httpx_timeout
            else:
                _params["timeout"] = httpx.Timeout(timeout=600.0, connect=5.0)

            sync_handler: HTTPHandler = HTTPHandler(**_params)  # type: ignore
        else:
            sync_handler = client  # type: ignore

        # url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model}:predict"

        auth_header: Optional[str] = None
        auth_header, _ = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )
        auth_header, api_base = self._get_token_and_url(
            model=model,
            gemini_api_key=None,
            auth_header=auth_header,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=False,
            custom_llm_provider="vertex_ai",
            api_base=api_base,
            should_use_v1beta1_features=False,
            mode="image_generation",
        )
        optional_params = optional_params or {
            "sampleCount": 1
        }  # default optional params

        # Transform optional params to camelCase format
        optional_params = self.transform_optional_params(optional_params)

        request_data = {
            "instances": [{"prompt": prompt}],
            "parameters": optional_params,
        }

        headers = self.set_headers(auth_header=auth_header, extra_headers=extra_headers)

        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": optional_params,
                "api_base": api_base,
                "headers": headers,
            },
        )

        response = sync_handler.post(
            url=api_base,
            headers=headers,
            data=json.dumps(request_data),
        )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        json_response = response.json()
        return self.process_image_generation_response(
            json_response, model_response, model
        )

    async def aimage_generation(
        self,
        prompt: str,
        api_base: Optional[str],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        model_response: litellm.ImageResponse,
        logging_obj: Any,
        model: str = "imagegeneration",  # vertex ai uses imagegeneration as the default model
        client: Optional[AsyncHTTPHandler] = None,
        optional_params: Optional[dict] = None,
        timeout: Optional[int] = None,
        extra_headers: Optional[dict] = None,
    ):
        response = None
        if client is None:
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    _httpx_timeout = httpx.Timeout(timeout)
                    _params["timeout"] = _httpx_timeout
            else:
                _params["timeout"] = httpx.Timeout(timeout=600.0, connect=5.0)

            self.async_handler = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.VERTEX_AI,
                params={"timeout": timeout},
            )
        else:
            self.async_handler = client  # type: ignore

        # make POST request to
        # https://us-central1-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/us-central1/publishers/google/models/imagegeneration:predict

        """
        Docs link: https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/imagegeneration?project=adroit-crow-413218
        curl -X POST \
        -H "Authorization: Bearer $(gcloud auth print-access-token)" \
        -H "Content-Type: application/json; charset=utf-8" \
        -d {
            "instances": [
                {
                    "prompt": "a cat"
                }
            ],
            "parameters": {
                "sampleCount": 1
            }
        } \
        "https://us-central1-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/us-central1/publishers/google/models/imagegeneration:predict"
        """
        auth_header: Optional[str] = None
        auth_header, _ = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )
        auth_header, api_base = self._get_token_and_url(
            model=model,
            gemini_api_key=None,
            auth_header=auth_header,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=False,
            custom_llm_provider="vertex_ai",
            api_base=api_base,
            should_use_v1beta1_features=False,
            mode="image_generation",
        )

        # Transform optional params to camelCase format
        optional_params = self.transform_optional_params(optional_params)

        request_data = {
            "instances": [{"prompt": prompt}],
            "parameters": optional_params,
        }

        headers = self.set_headers(auth_header=auth_header, extra_headers=extra_headers)

        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": optional_params,
                "api_base": api_base,
                "headers": headers,
            },
        )

        response = await self.async_handler.post(
            url=api_base,
            headers=headers,
            data=json.dumps(request_data),
        )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        json_response = response.json()
        return self.process_image_generation_response(
            json_response, model_response, model
        )

    def is_image_generation_response(self, json_response: Dict[str, Any]) -> bool:
        if "predictions" in json_response:
            if "bytesBase64Encoded" in json_response["predictions"][0]:
                return True
        return False
```

## File: `llms/vertex_ai/multimodal_embeddings/embedding_handler.py`
```
import json
from typing import Literal, Optional, Union

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexAIError,
    VertexLLM,
)
from litellm.types.utils import EmbeddingResponse

from .transformation import VertexAIMultimodalEmbeddingConfig

vertex_multimodal_embedding_handler = VertexAIMultimodalEmbeddingConfig()


class VertexMultimodalEmbedding(VertexLLM):
    def __init__(self) -> None:
        super().__init__()
        self.SUPPORTED_MULTIMODAL_EMBEDDING_MODELS = [
            "multimodalembedding",
            "multimodalembedding@001",
        ]

    def multimodal_embedding(
        self,
        model: str,
        input: Union[list, str],
        print_verbose,
        model_response: EmbeddingResponse,
        custom_llm_provider: Literal["gemini", "vertex_ai"],
        optional_params: dict,
        litellm_params: dict,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        headers: dict = {},
        encoding=None,
        vertex_project=None,
        vertex_location=None,
        vertex_credentials=None,
        aembedding=False,
        timeout=300,
        client=None,
    ) -> EmbeddingResponse:
        _auth_header, vertex_project = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider=custom_llm_provider,
        )

        auth_header, url = self._get_token_and_url(
            model=model,
            auth_header=_auth_header,
            gemini_api_key=api_key,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=None,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            should_use_v1beta1_features=False,
            mode="embedding",
        )

        if client is None:
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    _httpx_timeout = httpx.Timeout(timeout)
                    _params["timeout"] = _httpx_timeout
            else:
                _params["timeout"] = httpx.Timeout(timeout=600.0, connect=5.0)

            sync_handler: HTTPHandler = HTTPHandler(**_params)  # type: ignore
        else:
            sync_handler = client  # type: ignore

        request_data = vertex_multimodal_embedding_handler.transform_embedding_request(
            model, input, optional_params, headers
        )

        headers = vertex_multimodal_embedding_handler.validate_environment(
            headers=headers,
            model=model,
            messages=[],
            optional_params=optional_params,
            api_key=auth_header,
            api_base=api_base,
            litellm_params=litellm_params,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=input,
            api_key="",
            additional_args={
                "complete_input_dict": request_data,
                "api_base": url,
                "headers": headers,
            },
        )

        if aembedding is True:
            return self.async_multimodal_embedding(  # type: ignore
                model=model,
                api_base=url,
                data=request_data,
                timeout=timeout,
                headers=headers,
                client=client,
                model_response=model_response,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                api_key=api_key,
            )

        response = sync_handler.post(
            url=url,
            headers=headers,
            data=json.dumps(request_data),
        )

        return vertex_multimodal_embedding_handler.transform_embedding_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=request_data,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

    async def async_multimodal_embedding(
        self,
        model: str,
        api_base: str,
        optional_params: dict,
        litellm_params: dict,
        data: dict,
        model_response: litellm.EmbeddingResponse,
        timeout: Optional[Union[float, httpx.Timeout]],
        logging_obj: LiteLLMLoggingObj,
        headers={},
        client: Optional[AsyncHTTPHandler] = None,
        api_key: Optional[str] = None,
    ) -> litellm.EmbeddingResponse:
        if client is None:
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.VERTEX_AI,
                params={"timeout": timeout},
            )
        else:
            client = client  # type: ignore

        try:
            response = await client.post(api_base, headers=headers, json=data)  # type: ignore
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise VertexAIError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException:
            raise VertexAIError(status_code=408, message="Timeout error occurred.")

        return vertex_multimodal_embedding_handler.transform_embedding_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data=data,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )
```

## File: `llms/vertex_ai/multimodal_embeddings/transformation.py`
```
from typing import List, Optional, Union, cast

from httpx import Headers, Response

from litellm.exceptions import InternalServerError
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import LiteLLMLoggingObj
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.llms.vertex_ai import (
    Instance,
    InstanceImage,
    InstanceVideo,
    MultimodalPredictions,
    VertexMultimodalEmbeddingRequest,
)
from litellm.types.utils import (
    Embedding,
    EmbeddingResponse,
    PromptTokensDetailsWrapper,
    Usage,
)
from litellm.utils import _count_characters, is_base64_encoded

from ...base_llm.embedding.transformation import BaseEmbeddingConfig
from ..common_utils import VertexAIError


class VertexAIMultimodalEmbeddingConfig(BaseEmbeddingConfig):
    def get_supported_openai_params(self, model: str) -> list:
        return ["dimensions"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "dimensions":
                optional_params["outputDimensionality"] = value
        return optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        default_headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {api_key}",
        }
        headers.update(default_headers)
        return headers

    def _process_input_element(self, input_element: str) -> Instance:
        """
        Process the input element for multimodal embedding requests. checks if the if the input is gcs uri, base64 encoded image or plain text.

        Args:
            input_element (str): The input element to process.

        Returns:
            Dict[str, Any]: A dictionary representing the processed input element.
        """
        if len(input_element) == 0:
            return Instance(text=input_element)
        elif "gs://" in input_element:
            if "mp4" in input_element:
                return Instance(video=InstanceVideo(gcsUri=input_element))
            else:
                return Instance(image=InstanceImage(gcsUri=input_element))
        elif is_base64_encoded(s=input_element):
            return Instance(
                image=InstanceImage(
                    bytesBase64Encoded=(
                        input_element.split(",")[1]
                        if "," in input_element
                        else input_element
                    )
                )
            )
        else:
            return Instance(text=input_element)

    def process_openai_embedding_input(
        self, _input: Union[list, str]
    ) -> List[Instance]:
        """
        Process the input for multimodal embedding requests.

        Args:
            _input (Union[list, str]): The input data to process.

        Returns:
            Union[Instance, List[Instance]]: Either a single Instance or list of Instance objects.
        """
        _input_list = [_input] if not isinstance(_input, list) else _input
        processed_instances = []

        i = 0
        while i < len(_input_list):
            current = _input_list[i]

            # Look ahead for potential media elements
            next_elem = _input_list[i + 1] if i + 1 < len(_input_list) else None

            # If current is a text and next is a GCS URI, or current is a GCS URI
            if isinstance(current, str):
                instance_args: Instance = {}

                # Process current element
                if "gs://" not in current:
                    instance_args["text"] = current
                elif "mp4" in current:
                    instance_args["video"] = InstanceVideo(gcsUri=current)
                else:
                    instance_args["image"] = InstanceImage(gcsUri=current)

                # Check next element if it's a GCS URI
                if next_elem and isinstance(next_elem, str) and "gs://" in next_elem:
                    if "mp4" in next_elem:
                        instance_args["video"] = InstanceVideo(gcsUri=next_elem)
                    else:
                        instance_args["image"] = InstanceImage(gcsUri=next_elem)
                    i += 2  # Skip next element since we processed it
                else:
                    i += 1  # Move to next element

                processed_instances.append(instance_args)
                continue

            # Handle dict or other types
            if isinstance(current, dict):
                instance = Instance(**current)
                processed_instances.append(instance)
            else:
                raise ValueError(f"Unsupported input type: {type(current)}")
            i += 1

        return processed_instances

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        optional_params = optional_params or {}

        request_data = VertexMultimodalEmbeddingRequest(instances=[])

        if "instances" in optional_params:
            request_data["instances"] = optional_params["instances"]
        elif isinstance(input, list):
            vertex_instances: List[Instance] = self.process_openai_embedding_input(
                _input=input
            )
            request_data["instances"] = vertex_instances

        else:
            # construct instances
            vertex_request_instance = Instance(**optional_params)

            if isinstance(input, str):
                vertex_request_instance = self._process_input_element(input)

            request_data["instances"] = [vertex_request_instance]

        return cast(dict, request_data)

    def transform_embedding_response(
        self,
        model: str,
        raw_response: Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
    ) -> EmbeddingResponse:
        if raw_response.status_code != 200:
            raise Exception(f"Error: {raw_response.status_code} {raw_response.text}")

        _json_response = raw_response.json()
        if "predictions" not in _json_response:
            raise InternalServerError(
                message=f"embedding response does not contain 'predictions', got {_json_response}",
                llm_provider="vertex_ai",
                model=model,
            )
        _predictions = _json_response["predictions"]
        vertex_predictions = MultimodalPredictions(predictions=_predictions)
        model_response.data = self.transform_embedding_response_to_openai(
            predictions=vertex_predictions
        )
        model_response.model = model

        model_response.usage = self.calculate_usage(
            request_data=cast(VertexMultimodalEmbeddingRequest, request_data),
            vertex_predictions=vertex_predictions,
        )

        return model_response

    def calculate_usage(
        self,
        request_data: VertexMultimodalEmbeddingRequest,
        vertex_predictions: MultimodalPredictions,
    ) -> Usage:
        ## Calculate text embeddings usage
        prompt: Optional[str] = None
        character_count: Optional[int] = None

        for instance in request_data["instances"]:
            text = instance.get("text")
            if text:
                if prompt is None:
                    prompt = text
                else:
                    prompt += text

        if prompt is not None:
            character_count = _count_characters(prompt)

        ## Calculate image embeddings usage
        image_count = 0
        for instance in request_data["instances"]:
            if instance.get("image"):
                image_count += 1

        ## Calculate video embeddings usage
        video_length_seconds = 0
        for prediction in vertex_predictions["predictions"]:
            video_embeddings = prediction.get("videoEmbeddings")
            if video_embeddings:
                for embedding in video_embeddings:
                    duration = embedding["endOffsetSec"] - embedding["startOffsetSec"]
                    video_length_seconds += duration

        prompt_tokens_details = PromptTokensDetailsWrapper(
            character_count=character_count,
            image_count=image_count,
            video_length_seconds=video_length_seconds,
        )

        return Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=prompt_tokens_details,
        )

    def transform_embedding_response_to_openai(
        self, predictions: MultimodalPredictions
    ) -> List[Embedding]:
        openai_embeddings: List[Embedding] = []
        if "predictions" in predictions:
            for idx, _prediction in enumerate(predictions["predictions"]):
                if _prediction:
                    if "textEmbedding" in _prediction:
                        openai_embedding_object = Embedding(
                            embedding=_prediction["textEmbedding"],
                            index=idx,
                            object="embedding",
                        )
                        openai_embeddings.append(openai_embedding_object)
                    elif "imageEmbedding" in _prediction:
                        openai_embedding_object = Embedding(
                            embedding=_prediction["imageEmbedding"],
                            index=idx,
                            object="embedding",
                        )
                        openai_embeddings.append(openai_embedding_object)
                    elif "videoEmbeddings" in _prediction:
                        for video_embedding in _prediction["videoEmbeddings"]:
                            openai_embedding_object = Embedding(
                                embedding=video_embedding["embedding"],
                                index=idx,
                                object="embedding",
                            )
                            openai_embeddings.append(openai_embedding_object)
        return openai_embeddings

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return VertexAIError(
            status_code=status_code, message=error_message, headers=headers
        )
```

## File: `llms/vertex_ai/text_to_speech/text_to_speech_handler.py`
```
from typing import Optional, TypedDict, Union

import httpx

import litellm
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.openai.openai import HttpxBinaryResponseContent
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES


class VertexInput(TypedDict, total=False):
    text: Optional[str]
    ssml: Optional[str]


class VertexVoice(TypedDict, total=False):
    languageCode: str
    name: str


class VertexAudioConfig(TypedDict, total=False):
    audioEncoding: str
    speakingRate: str


class VertexTextToSpeechRequest(TypedDict, total=False):
    input: VertexInput
    voice: VertexVoice
    audioConfig: Optional[VertexAudioConfig]


class VertexTextToSpeechAPI(VertexLLM):
    """
    Vertex methods to support for batches
    """

    def __init__(self) -> None:
        super().__init__()

    def audio_speech(
        self,
        logging_obj,
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        api_base: Optional[str],
        timeout: Union[float, httpx.Timeout],
        model: str,
        input: str,
        voice: Optional[dict] = None,
        _is_async: Optional[bool] = False,
        optional_params: Optional[dict] = None,
        kwargs: Optional[dict] = None,
    ) -> HttpxBinaryResponseContent:
        import base64

        ####### Authenticate with Vertex AI ########
        _auth_header, vertex_project = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai_beta",
        )

        auth_header, _ = self._get_token_and_url(
            model="",
            auth_header=_auth_header,
            gemini_api_key=None,
            vertex_credentials=vertex_credentials,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            stream=False,
            custom_llm_provider="vertex_ai_beta",
            api_base=api_base,
        )

        headers = {
            "Authorization": f"Bearer {auth_header}",
            "x-goog-user-project": vertex_project,
            "Content-Type": "application/json",
            "charset": "UTF-8",
        }

        ######### End of Authentication ###########

        ####### Build the request ################
        # API Ref: https://cloud.google.com/text-to-speech/docs/reference/rest/v1/text/synthesize
        kwargs = kwargs or {}
        optional_params = optional_params or {}

        vertex_input = VertexInput(text=input)
        validate_vertex_input(vertex_input, kwargs, optional_params)

        # required param
        if voice is not None:
            vertex_voice = VertexVoice(**voice)
        elif "voice" in kwargs:
            vertex_voice = VertexVoice(**kwargs["voice"])
        else:
            # use defaults to not fail the request
            vertex_voice = VertexVoice(
                languageCode="en-US",
                name="en-US-Studio-O",
            )

        if "audioConfig" in kwargs:
            vertex_audio_config = VertexAudioConfig(**kwargs["audioConfig"])
        else:
            # use defaults to not fail the request
            vertex_audio_config = VertexAudioConfig(
                audioEncoding="LINEAR16",
                speakingRate="1",
            )

        request = VertexTextToSpeechRequest(
            input=vertex_input,
            voice=vertex_voice,
            audioConfig=vertex_audio_config,
        )

        url = "https://texttospeech.googleapis.com/v1/text:synthesize"
        ########## End of building request ############

        ########## Log the request for debugging / logging ############
        logging_obj.pre_call(
            input=[],
            api_key="",
            additional_args={
                "complete_input_dict": request,
                "api_base": url,
                "headers": headers,
            },
        )

        ########## End of logging ############
        ####### Send the request ###################
        if _is_async is True:
            return self.async_audio_speech(  # type:ignore
                logging_obj=logging_obj, url=url, headers=headers, request=request
            )
        sync_handler = _get_httpx_client()

        response = sync_handler.post(
            url=url,
            headers=headers,
            json=request,  # type: ignore
        )
        if response.status_code != 200:
            raise Exception(
                f"Request failed with status code {response.status_code}, {response.text}"
            )
        ############ Process the response ############
        _json_response = response.json()

        response_content = _json_response["audioContent"]

        # Decode base64 to get binary content
        binary_data = base64.b64decode(response_content)

        # Create an httpx.Response object
        response = httpx.Response(
            status_code=200,
            content=binary_data,
        )

        # Initialize the HttpxBinaryResponseContent instance
        http_binary_response = HttpxBinaryResponseContent(response)
        return http_binary_response

    async def async_audio_speech(
        self,
        logging_obj,
        url: str,
        headers: dict,
        request: VertexTextToSpeechRequest,
    ) -> HttpxBinaryResponseContent:
        import base64

        async_handler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.VERTEX_AI
        )

        response = await async_handler.post(
            url=url,
            headers=headers,
            json=request,  # type: ignore
        )

        if response.status_code != 200:
            raise Exception(
                f"Request did not return a 200 status code: {response.status_code}, {response.text}"
            )

        _json_response = response.json()

        response_content = _json_response["audioContent"]

        # Decode base64 to get binary content
        binary_data = base64.b64decode(response_content)

        # Create an httpx.Response object
        response = httpx.Response(
            status_code=200,
            content=binary_data,
        )

        # Initialize the HttpxBinaryResponseContent instance
        http_binary_response = HttpxBinaryResponseContent(response)
        return http_binary_response


def validate_vertex_input(
    input_data: VertexInput, kwargs: dict, optional_params: dict
) -> None:
    # Remove None values
    if input_data.get("text") is None:
        input_data.pop("text", None)
    if input_data.get("ssml") is None:
        input_data.pop("ssml", None)

    # Check if use_ssml is set
    use_ssml = kwargs.get("use_ssml", optional_params.get("use_ssml", False))

    if use_ssml:
        if "text" in input_data:
            input_data["ssml"] = input_data.pop("text")
        elif "ssml" not in input_data:
            raise ValueError("SSML input is required when use_ssml is True.")
    else:
        # LiteLLM will auto-detect if text is in ssml format
        # check if "text" is an ssml - in this case we should pass it as ssml instead of text
        if input_data:
            _text = input_data.get("text", None) or ""
            if "<speak>" in _text:
                input_data["ssml"] = input_data.pop("text")

    if not input_data:
        raise ValueError("Either 'text' or 'ssml' must be provided.")
    if "text" in input_data and "ssml" in input_data:
        raise ValueError("Only one of 'text' or 'ssml' should be provided, not both.")
```

## File: `llms/vertex_ai/vector_stores/__init__.py`
```
from .transformation import VertexVectorStoreConfig

__all__ = ["VertexVectorStoreConfig"] ```

## File: `llms/vertex_ai/vector_stores/transformation.py`
```
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm.llms.base_llm.vector_store.transformation import BaseVectorStoreConfig
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_stores import (
    VectorStoreCreateOptionalRequestParams,
    VectorStoreCreateResponse,
    VectorStoreResultContent,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class VertexVectorStoreConfig(BaseVectorStoreConfig, VertexBase):
    """
    Configuration for Vertex AI Vector Store RAG API
    
    This implementation uses the Vertex AI RAG Engine API for vector store operations.
    """

    def __init__(self):
        super().__init__()

    def validate_environment(
        self, headers: dict, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        """
        Validate and set up authentication for Vertex AI RAG API
        """
        litellm_params = litellm_params or GenericLiteLLMParams()
        
        # Get credentials and project info
        vertex_credentials = self.get_vertex_ai_credentials(dict(litellm_params))
        vertex_project = self.get_vertex_ai_project(dict(litellm_params))
        
        # Get access token using the base class method
        access_token, project_id = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )
        
        headers.update({
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        })
        
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the Base endpoint for Vertex AI RAG API
        """
        vertex_location = self.get_vertex_ai_location(litellm_params)
        vertex_project = self.get_vertex_ai_project(litellm_params)
        
        if api_base:
            return api_base.rstrip("/")
        
        # Vertex AI RAG API endpoint for retrieveContexts
        return f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}"

    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: Union[str, List[str]],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Transform search request for Vertex AI RAG API
        """
        # Convert query to string if it's a list
        if isinstance(query, list):
            query = " ".join(query)
        
        # Vertex AI RAG API endpoint for retrieving contexts
        url = f"{api_base}:retrieveContexts"
        
        # Use helper methods to get project and location, then construct full rag corpus path
        vertex_project = self.get_vertex_ai_project(litellm_params)
        vertex_location = self.get_vertex_ai_location(litellm_params)
        
        # Construct full rag corpus path
        full_rag_corpus = f"projects/{vertex_project}/locations/{vertex_location}/ragCorpora/{vector_store_id}"
        
        # Build the request body for Vertex AI RAG API
        request_body: Dict[str, Any] = {
            "vertex_rag_store": {
                "rag_resources": [
                    {
                        "rag_corpus": full_rag_corpus
                    }
                ]
            },
            "query": {
                "text": query
            }
        }

        #########################################################
        # Update logging object with details of the request
        #########################################################
        litellm_logging_obj.model_call_details["query"] = query
        
        # Add optional parameters
        max_num_results = vector_store_search_optional_params.get("max_num_results")
        if max_num_results is not None:
            request_body["query"]["rag_retrieval_config"] = {
                "top_k": max_num_results
            }
        
        # Add filters if provided
        filters = vector_store_search_optional_params.get("filters")
        if filters is not None:
            if "rag_retrieval_config" not in request_body["query"]:
                request_body["query"]["rag_retrieval_config"] = {}
            request_body["query"]["rag_retrieval_config"]["filter"] = filters
        
        # Add ranking options if provided
        ranking_options = vector_store_search_optional_params.get("ranking_options")
        if ranking_options is not None:
            if "rag_retrieval_config" not in request_body["query"]:
                request_body["query"]["rag_retrieval_config"] = {}
            request_body["query"]["rag_retrieval_config"]["ranking"] = ranking_options
        
        return url, request_body

    def transform_search_vector_store_response(self, response: httpx.Response, litellm_logging_obj: LiteLLMLoggingObj) -> VectorStoreSearchResponse:
        """
        Transform Vertex AI RAG API response to standard vector store search response
        """
        try:

            response_json = response.json()
            # Extract contexts from Vertex AI response - handle nested structure
            contexts = response_json.get("contexts", {}).get("contexts", [])
            
            # Transform contexts to standard format
            search_results = []
            for context in contexts:
                content = [
                    VectorStoreResultContent(
                        text=context.get("text", ""),
                        type="text",
                    )
                ]
                
                # Extract file information
                source_uri = context.get("sourceUri", "")
                source_display_name = context.get("sourceDisplayName", "")
                
                # Generate file_id from source URI or use display name as fallback
                file_id = source_uri if source_uri else source_display_name
                filename = source_display_name if source_display_name else "Unknown Document"
                
                # Build attributes with available metadata
                attributes = {}
                if source_uri:
                    attributes["sourceUri"] = source_uri
                if source_display_name:
                    attributes["sourceDisplayName"] = source_display_name
                
                # Add page span information if available
                page_span = context.get("pageSpan", {})
                if page_span:
                    attributes["pageSpan"] = page_span
                
                result = VectorStoreSearchResult(
                    score=context.get("score", 0.0),
                    content=content,
                    file_id=file_id,
                    filename=filename,
                    attributes=attributes,
                )
                search_results.append(result)
            
            return VectorStoreSearchResponse(
                object="vector_store.search_results.page",
                search_query=litellm_logging_obj.model_call_details.get("query", ""),
                data=search_results
            )
            
        except Exception as e:
            raise self.get_error_class(
                error_message=str(e),
                status_code=response.status_code,
                headers=response.headers
            )

    def transform_create_vector_store_request(
        self,
        vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams,
        api_base: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Transform create request for Vertex AI RAG Corpus
        """
        url = f"{api_base}/ragCorpora"  # Base URL for creating RAG corpus
        
        # Build the request body for Vertex AI RAG Corpus creation
        request_body: Dict[str, Any] = {
            "display_name": vector_store_create_optional_params.get("name", "litellm-vector-store"),
            "description": "Vector store created via LiteLLM"
        }
        
        # Add metadata if provided
        metadata = vector_store_create_optional_params.get("metadata")
        if metadata is not None:
            request_body["labels"] = metadata
        
        return url, request_body

    def transform_create_vector_store_response(self, response: httpx.Response) -> VectorStoreCreateResponse:
        """
        Transform Vertex AI RAG Corpus creation response to standard vector store response
        """
        try:
            response_json = response.json()
            
            # Extract the corpus ID from the response name
            corpus_name = response_json.get("name", "")
            corpus_id = corpus_name.split("/")[-1] if "/" in corpus_name else corpus_name
            
            # Handle createTime conversion
            create_time = response_json.get("createTime", 0)
            if isinstance(create_time, str):
                # Convert ISO timestamp to Unix timestamp
                from datetime import datetime
                try:
                    dt = datetime.fromisoformat(create_time.replace('Z', '+00:00'))
                    create_time = int(dt.timestamp())
                except ValueError:
                    create_time = 0
            elif not isinstance(create_time, int):
                create_time = 0
            
            # Handle labels safely
            labels = response_json.get("labels", {})
            metadata = labels if isinstance(labels, dict) else {}
            
            return VectorStoreCreateResponse(
                id=corpus_id,
                object="vector_store",
                created_at=create_time,
                name=response_json.get("display_name", ""),
                bytes=0,  # Vertex AI doesn't provide byte count in the same way
                file_counts={
                    "in_progress": 0,
                    "completed": 0,
                    "failed": 0,
                    "cancelled": 0,
                    "total": 0
                },
                status="completed",  # Vertex AI corpus creation is typically synchronous
                expires_after=None,
                expires_at=None,
                last_active_at=None,
                metadata=metadata
            )
            
        except Exception as e:
            raise self.get_error_class(
                error_message=str(e),
                status_code=response.status_code,
                headers=response.headers
            ) ```

## File: `llms/vertex_ai/vertex_ai_non_gemini.py`
```
import json
import os
import time
from typing import Any, Callable, Optional, cast

import httpx

import litellm
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.bedrock.common_utils import ModelResponseIterator
from litellm.llms.custom_httpx.http_handler import _DEFAULT_TTL_FOR_HTTPX_CLIENTS
from litellm.types.llms.vertex_ai import *
from litellm.utils import CustomStreamWrapper, ModelResponse, Usage


class VertexAIError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url=" https://cloud.google.com/vertex-ai/"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class TextStreamer:
    """
    Fake streaming iterator for Vertex AI Model Garden calls
    """

    def __init__(self, text):
        self.text = text.split()  # let's assume words as a streaming unit
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.text):
            result = self.text[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index < len(self.text):
            result = self.text[self.index]
            self.index += 1
            return result
        else:
            raise StopAsyncIteration  # once we run out of data to stream, we raise this error


def _get_client_cache_key(
    model: str, vertex_project: Optional[str], vertex_location: Optional[str]
):
    _cache_key = f"{model}-{vertex_project}-{vertex_location}"
    return _cache_key


def _get_client_from_cache(client_cache_key: str):
    return litellm.in_memory_llm_clients_cache.get_cache(client_cache_key)


def _set_client_in_cache(client_cache_key: str, vertex_llm_model: Any):
    litellm.in_memory_llm_clients_cache.set_cache(
        key=client_cache_key,
        value=vertex_llm_model,
        ttl=_DEFAULT_TTL_FOR_HTTPX_CLIENTS,
    )


def completion(  # noqa: PLR0915
    model: str,
    messages: list,
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    logging_obj,
    optional_params: dict,
    vertex_project=None,
    vertex_location=None,
    vertex_credentials=None,
    litellm_params=None,
    logger_fn=None,
    acompletion: bool = False,
):
    """
    NON-GEMINI/ANTHROPIC CALLS.

    This is the handler for OLDER PALM MODELS and VERTEX AI MODEL GARDEN

    For Vertex AI Anthropic: `vertex_anthropic.py`
    For Gemini: `vertex_httpx.py`
    """
    try:
        import vertexai
    except Exception:
        raise VertexAIError(
            status_code=400,
            message="vertexai import failed please run `pip install google-cloud-aiplatform`. This is required for the 'vertex_ai/' route on LiteLLM",
        )

    if not (
        hasattr(vertexai, "preview") or hasattr(vertexai.preview, "language_models")
    ):
        raise VertexAIError(
            status_code=400,
            message="""Upgrade vertex ai. Run `pip install "google-cloud-aiplatform>=1.38"`""",
        )
    try:
        import google.auth  # type: ignore
        from google.cloud import aiplatform  # type: ignore
        from google.cloud.aiplatform_v1beta1.types import (
            content as gapic_content_types,  # type: ignore
        )
        from google.protobuf import json_format  # type: ignore
        from google.protobuf.struct_pb2 import Value  # type: ignore
        from vertexai.language_models import CodeGenerationModel, TextGenerationModel
        from vertexai.preview.generative_models import GenerativeModel
        from vertexai.preview.language_models import ChatModel, CodeChatModel

        ## Load credentials with the correct quota project ref: https://github.com/googleapis/python-aiplatform/issues/2557#issuecomment-1709284744
        print_verbose(
            f"VERTEX AI: vertex_project={vertex_project}; vertex_location={vertex_location}"
        )

        _cache_key = _get_client_cache_key(
            model=model, vertex_project=vertex_project, vertex_location=vertex_location
        )
        _vertex_llm_model_object = _get_client_from_cache(client_cache_key=_cache_key)

        if _vertex_llm_model_object is None:
            from google.auth.credentials import Credentials

            if vertex_credentials is not None and isinstance(vertex_credentials, str):
                import google.oauth2.service_account

                json_obj = json.loads(vertex_credentials)

                creds = (
                    google.oauth2.service_account.Credentials.from_service_account_info(
                        json_obj,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                )
            else:
                creds, _ = google.auth.default(quota_project_id=vertex_project)
            print_verbose(
                f"VERTEX AI: creds={creds}; google application credentials: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}"
            )
            vertexai.init(
                project=vertex_project,
                location=vertex_location,
                credentials=cast(Credentials, creds),
            )

        ## Load Config
        config = litellm.VertexAIConfig.get_config()
        for k, v in config.items():
            if k not in optional_params:
                optional_params[k] = v

        ## Process safety settings into format expected by vertex AI
        safety_settings = None
        if "safety_settings" in optional_params:
            safety_settings = optional_params.pop("safety_settings")
            if not isinstance(safety_settings, list):
                raise ValueError("safety_settings must be a list")
            if len(safety_settings) > 0 and not isinstance(safety_settings[0], dict):
                raise ValueError("safety_settings must be a list of dicts")
            safety_settings = [
                gapic_content_types.SafetySetting(x) for x in safety_settings
            ]

        # vertexai does not use an API key, it looks for credentials.json in the environment

        prompt = " ".join(
            [
                message.get("content")
                for message in messages
                if isinstance(message.get("content", None), str)
            ]
        )

        mode = ""

        request_str = ""
        response_obj = None
        instances = None
        client_options = {
            "api_endpoint": f"{vertex_location}-aiplatform.googleapis.com"
        }
        fake_stream = False
        if (
            model in litellm.vertex_language_models
            or model in litellm.vertex_vision_models
        ):
            llm_model: Any = _vertex_llm_model_object or GenerativeModel(model)
            mode = "vision"
            request_str += f"llm_model = GenerativeModel({model})\n"
        elif model in litellm.vertex_chat_models:
            llm_model = _vertex_llm_model_object or ChatModel.from_pretrained(model)
            mode = "chat"
            request_str += f"llm_model = ChatModel.from_pretrained({model})\n"
        elif model in litellm.vertex_text_models:
            llm_model = _vertex_llm_model_object or TextGenerationModel.from_pretrained(
                model
            )
            mode = "text"
            request_str += f"llm_model = TextGenerationModel.from_pretrained({model})\n"
        elif model in litellm.vertex_code_text_models:
            llm_model = _vertex_llm_model_object or CodeGenerationModel.from_pretrained(
                model
            )
            mode = "text"
            request_str += f"llm_model = CodeGenerationModel.from_pretrained({model})\n"
            fake_stream = True
        elif model in litellm.vertex_code_chat_models:  # vertex_code_llm_models
            llm_model = _vertex_llm_model_object or CodeChatModel.from_pretrained(model)
            mode = "chat"
            request_str += f"llm_model = CodeChatModel.from_pretrained({model})\n"
        elif model == "private":
            mode = "private"
            model = optional_params.pop("model_id", None)
            # private endpoint requires a dict instead of JSON
            instances = [optional_params.copy()]
            instances[0]["prompt"] = prompt
            llm_model = aiplatform.PrivateEndpoint(
                endpoint_name=model,
                project=vertex_project,
                location=vertex_location,
            )
            request_str += f"llm_model = aiplatform.PrivateEndpoint(endpoint_name={model}, project={vertex_project}, location={vertex_location})\n"
        else:  # assume vertex model garden on public endpoint
            mode = "custom"

            instances = [optional_params.copy()]
            instances[0]["prompt"] = prompt
            instances = [
                json_format.ParseDict(instance_dict, Value())
                for instance_dict in instances
            ]
            # Will determine the API used based on async parameter
            llm_model = None

        # NOTE: async prediction and streaming under "private" mode isn't supported by aiplatform right now
        if acompletion is True:
            data = {
                "llm_model": llm_model,
                "mode": mode,
                "prompt": prompt,
                "logging_obj": logging_obj,
                "request_str": request_str,
                "model": model,
                "model_response": model_response,
                "encoding": encoding,
                "messages": messages,
                "print_verbose": print_verbose,
                "client_options": client_options,
                "instances": instances,
                "vertex_location": vertex_location,
                "vertex_project": vertex_project,
                "safety_settings": safety_settings,
                **optional_params,
            }
            if optional_params.get("stream", False) is True:
                # async streaming
                return async_streaming(**data)

            return async_completion(**data)

        completion_response = None

        stream = optional_params.pop(
            "stream", None
        )  # See note above on handling streaming for vertex ai
        if mode == "chat":
            chat = llm_model.start_chat()
            request_str += "chat = llm_model.start_chat()\n"

            if fake_stream is not True and stream is True:
                # NOTE: VertexAI does not accept stream=True as a param and raises an error,
                # we handle this by removing 'stream' from optional params and sending the request
                # after we get the response we add optional_params["stream"] = True, since main.py needs to know it's a streaming response to then transform it for the OpenAI format
                optional_params.pop(
                    "stream", None
                )  # vertex ai raises an error when passing stream in optional params

                request_str += (
                    f"chat.send_message_streaming({prompt}, **{optional_params})\n"
                )
                ## LOGGING
                logging_obj.pre_call(
                    input=prompt,
                    api_key=None,
                    additional_args={
                        "complete_input_dict": optional_params,
                        "request_str": request_str,
                    },
                )

                model_response = chat.send_message_streaming(prompt, **optional_params)

                return model_response

            request_str += f"chat.send_message({prompt}, **{optional_params}).text\n"
            ## LOGGING
            logging_obj.pre_call(
                input=prompt,
                api_key=None,
                additional_args={
                    "complete_input_dict": optional_params,
                    "request_str": request_str,
                },
            )
            completion_response = chat.send_message(prompt, **optional_params).text
        elif mode == "text":
            if fake_stream is not True and stream is True:
                request_str += (
                    f"llm_model.predict_streaming({prompt}, **{optional_params})\n"
                )
                ## LOGGING
                logging_obj.pre_call(
                    input=prompt,
                    api_key=None,
                    additional_args={
                        "complete_input_dict": optional_params,
                        "request_str": request_str,
                    },
                )
                model_response = llm_model.predict_streaming(prompt, **optional_params)

                return model_response

            request_str += f"llm_model.predict({prompt}, **{optional_params}).text\n"
            ## LOGGING
            logging_obj.pre_call(
                input=prompt,
                api_key=None,
                additional_args={
                    "complete_input_dict": optional_params,
                    "request_str": request_str,
                },
            )
            completion_response = llm_model.predict(prompt, **optional_params).text
        elif mode == "custom":
            """
            Vertex AI Model Garden
            """

            if vertex_project is None or vertex_location is None:
                raise ValueError(
                    "Vertex project and location are required for custom endpoint"
                )

            ## LOGGING
            logging_obj.pre_call(
                input=prompt,
                api_key=None,
                additional_args={
                    "complete_input_dict": optional_params,
                    "request_str": request_str,
                },
            )
            llm_model = aiplatform.gapic.PredictionServiceClient(
                client_options=client_options
            )
            request_str += f"llm_model = aiplatform.gapic.PredictionServiceClient(client_options={client_options})\n"
            endpoint_path = llm_model.endpoint_path(
                project=vertex_project, location=vertex_location, endpoint=model
            )
            request_str += (
                f"llm_model.predict(endpoint={endpoint_path}, instances={instances})\n"
            )
            response = llm_model.predict(
                endpoint=endpoint_path, instances=instances
            ).predictions

            completion_response = response[0]
            if (
                isinstance(completion_response, str)
                and "\nOutput:\n" in completion_response
            ):
                completion_response = completion_response.split("\nOutput:\n", 1)[1]
            if stream is True:
                response = TextStreamer(completion_response)
                return response
        elif mode == "private":
            """
            Vertex AI Model Garden deployed on private endpoint
            """
            if instances is None:
                raise ValueError("instances are required for private endpoint")
            if llm_model is None:
                raise ValueError("Unable to pick client for private endpoint")
            ## LOGGING
            logging_obj.pre_call(
                input=prompt,
                api_key=None,
                additional_args={
                    "complete_input_dict": optional_params,
                    "request_str": request_str,
                },
            )
            request_str += f"llm_model.predict(instances={instances})\n"
            response = llm_model.predict(instances=instances).predictions

            completion_response = response[0]
            if (
                isinstance(completion_response, str)
                and "\nOutput:\n" in completion_response
            ):
                completion_response = completion_response.split("\nOutput:\n", 1)[1]
            if stream is True:
                response = TextStreamer(completion_response)
                return response

        ## LOGGING
        logging_obj.post_call(
            input=prompt, api_key=None, original_response=completion_response
        )

        ## RESPONSE OBJECT
        if isinstance(completion_response, litellm.Message):
            model_response.choices[0].message = completion_response  # type: ignore
        elif len(str(completion_response)) > 0:
            model_response.choices[0].message.content = str(completion_response)  # type: ignore
        model_response.created = int(time.time())
        model_response.model = model
        ## CALCULATING USAGE
        if model in litellm.vertex_language_models and response_obj is not None:
            model_response.choices[0].finish_reason = map_finish_reason(
                response_obj.candidates[0].finish_reason.name
            )
            usage = Usage(
                prompt_tokens=response_obj.usage_metadata.prompt_token_count,
                completion_tokens=response_obj.usage_metadata.candidates_token_count,
                total_tokens=response_obj.usage_metadata.total_token_count,
            )
        else:
            # init prompt tokens
            # this block attempts to get usage from response_obj if it exists, if not it uses the litellm token counter
            prompt_tokens, completion_tokens, _ = 0, 0, 0
            if response_obj is not None:
                if hasattr(response_obj, "usage_metadata") and hasattr(
                    response_obj.usage_metadata, "prompt_token_count"
                ):
                    prompt_tokens = response_obj.usage_metadata.prompt_token_count
                    completion_tokens = (
                        response_obj.usage_metadata.candidates_token_count
                    )
            else:
                prompt_tokens = len(encoding.encode(prompt))
                completion_tokens = len(
                    encoding.encode(
                        model_response["choices"][0]["message"].get("content", "")
                    )
                )

            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
        setattr(model_response, "usage", usage)

        if fake_stream is True and stream is True:
            return ModelResponseIterator(model_response)
        return model_response
    except Exception as e:
        if isinstance(e, VertexAIError):
            raise e
        raise litellm.APIConnectionError(
            message=str(e), llm_provider="vertex_ai", model=model
        )


async def async_completion(  # noqa: PLR0915
    llm_model,
    mode: str,
    prompt: str,
    model: str,
    messages: list,
    model_response: ModelResponse,
    request_str: str,
    print_verbose: Callable,
    logging_obj,
    encoding,
    client_options=None,
    instances=None,
    vertex_project=None,
    vertex_location=None,
    safety_settings=None,
    **optional_params,
):
    """
    Add support for acompletion calls for gemini-pro
    """
    try:
        response_obj = None
        completion_response = None
        if mode == "chat":
            # chat-bison etc.
            chat = llm_model.start_chat()
            ## LOGGING
            logging_obj.pre_call(
                input=prompt,
                api_key=None,
                additional_args={
                    "complete_input_dict": optional_params,
                    "request_str": request_str,
                },
            )
            response_obj = await chat.send_message_async(prompt, **optional_params)
            completion_response = response_obj.text
        elif mode == "text":
            # gecko etc.
            request_str += f"llm_model.predict({prompt}, **{optional_params}).text\n"
            ## LOGGING
            logging_obj.pre_call(
                input=prompt,
                api_key=None,
                additional_args={
                    "complete_input_dict": optional_params,
                    "request_str": request_str,
                },
            )
            response_obj = await llm_model.predict_async(prompt, **optional_params)
            completion_response = response_obj.text
        elif mode == "custom":
            """
            Vertex AI Model Garden
            """
            from google.cloud import aiplatform  # type: ignore

            if vertex_project is None or vertex_location is None:
                raise ValueError(
                    "Vertex project and location are required for custom endpoint"
                )

            ## LOGGING
            logging_obj.pre_call(
                input=prompt,
                api_key=None,
                additional_args={
                    "complete_input_dict": optional_params,
                    "request_str": request_str,
                },
            )

            llm_model = aiplatform.gapic.PredictionServiceAsyncClient(
                client_options=client_options
            )
            request_str += f"llm_model = aiplatform.gapic.PredictionServiceAsyncClient(client_options={client_options})\n"
            endpoint_path = llm_model.endpoint_path(
                project=vertex_project, location=vertex_location, endpoint=model
            )
            request_str += (
                f"llm_model.predict(endpoint={endpoint_path}, instances={instances})\n"
            )
            response_obj = await llm_model.predict(
                endpoint=endpoint_path,
                instances=instances,
            )
            response = response_obj.predictions
            completion_response = response[0]
            if (
                isinstance(completion_response, str)
                and "\nOutput:\n" in completion_response
            ):
                completion_response = completion_response.split("\nOutput:\n", 1)[1]

        elif mode == "private":
            request_str += f"llm_model.predict_async(instances={instances})\n"
            response_obj = await llm_model.predict_async(
                instances=instances,
            )

            response = response_obj.predictions
            completion_response = response[0]
            if (
                isinstance(completion_response, str)
                and "\nOutput:\n" in completion_response
            ):
                completion_response = completion_response.split("\nOutput:\n", 1)[1]

        ## LOGGING
        logging_obj.post_call(
            input=prompt, api_key=None, original_response=completion_response
        )

        ## RESPONSE OBJECT
        if isinstance(completion_response, litellm.Message):
            model_response.choices[0].message = completion_response  # type: ignore
        elif len(str(completion_response)) > 0:
            model_response.choices[0].message.content = str(  # type: ignore
                completion_response
            )
        model_response.created = int(time.time())
        model_response.model = model
        ## CALCULATING USAGE
        if model in litellm.vertex_language_models and response_obj is not None:
            model_response.choices[0].finish_reason = map_finish_reason(
                response_obj.candidates[0].finish_reason.name
            )
            usage = Usage(
                prompt_tokens=response_obj.usage_metadata.prompt_token_count,
                completion_tokens=response_obj.usage_metadata.candidates_token_count,
                total_tokens=response_obj.usage_metadata.total_token_count,
            )
        else:
            # init prompt tokens
            # this block attempts to get usage from response_obj if it exists, if not it uses the litellm token counter
            prompt_tokens, completion_tokens, _ = 0, 0, 0
            if response_obj is not None and (
                hasattr(response_obj, "usage_metadata")
                and hasattr(response_obj.usage_metadata, "prompt_token_count")
            ):
                prompt_tokens = response_obj.usage_metadata.prompt_token_count
                completion_tokens = response_obj.usage_metadata.candidates_token_count
            else:
                prompt_tokens = len(encoding.encode(prompt))
                completion_tokens = len(
                    encoding.encode(
                        model_response["choices"][0]["message"].get("content", "")
                    )
                )

            # set usage
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
        setattr(model_response, "usage", usage)
        return model_response
    except Exception as e:
        raise VertexAIError(status_code=500, message=str(e))


async def async_streaming(  # noqa: PLR0915
    llm_model,
    mode: str,
    prompt: str,
    model: str,
    model_response: ModelResponse,
    messages: list,
    print_verbose: Callable,
    logging_obj,
    request_str: str,
    encoding=None,
    client_options=None,
    instances=None,
    vertex_project=None,
    vertex_location=None,
    safety_settings=None,
    **optional_params,
):
    """
    Add support for async streaming calls for gemini-pro
    """
    response: Any = None
    if mode == "chat":
        chat = llm_model.start_chat()
        optional_params.pop(
            "stream", None
        )  # vertex ai raises an error when passing stream in optional params
        request_str += (
            f"chat.send_message_streaming_async({prompt}, **{optional_params})\n"
        )
        ## LOGGING
        logging_obj.pre_call(
            input=prompt,
            api_key=None,
            additional_args={
                "complete_input_dict": optional_params,
                "request_str": request_str,
            },
        )
        response = chat.send_message_streaming_async(prompt, **optional_params)

    elif mode == "text":
        optional_params.pop(
            "stream", None
        )  # See note above on handling streaming for vertex ai
        request_str += (
            f"llm_model.predict_streaming_async({prompt}, **{optional_params})\n"
        )
        ## LOGGING
        logging_obj.pre_call(
            input=prompt,
            api_key=None,
            additional_args={
                "complete_input_dict": optional_params,
                "request_str": request_str,
            },
        )
        response = llm_model.predict_streaming_async(prompt, **optional_params)
    elif mode == "custom":
        from google.cloud import aiplatform  # type: ignore

        if vertex_project is None or vertex_location is None:
            raise ValueError(
                "Vertex project and location are required for custom endpoint"
            )

        stream = optional_params.pop("stream", None)

        ## LOGGING
        logging_obj.pre_call(
            input=prompt,
            api_key=None,
            additional_args={
                "complete_input_dict": optional_params,
                "request_str": request_str,
            },
        )
        llm_model = aiplatform.gapic.PredictionServiceAsyncClient(
            client_options=client_options
        )
        request_str += f"llm_model = aiplatform.gapic.PredictionServiceAsyncClient(client_options={client_options})\n"
        endpoint_path = llm_model.endpoint_path(
            project=vertex_project, location=vertex_location, endpoint=model
        )
        request_str += (
            f"client.predict(endpoint={endpoint_path}, instances={instances})\n"
        )
        response_obj = await llm_model.predict(
            endpoint=endpoint_path,
            instances=instances,
        )

        response = response_obj.predictions
        completion_response = response[0]
        if (
            isinstance(completion_response, str)
            and "\nOutput:\n" in completion_response
        ):
            completion_response = completion_response.split("\nOutput:\n", 1)[1]
        if stream:
            response = TextStreamer(completion_response)

    elif mode == "private":
        if instances is None:
            raise ValueError("Instances are required for private endpoint")
        stream = optional_params.pop("stream", None)
        _ = instances[0].pop("stream", None)
        request_str += f"llm_model.predict_async(instances={instances})\n"
        response_obj = await llm_model.predict_async(
            instances=instances,
        )
        response = response_obj.predictions
        completion_response = response[0]
        if (
            isinstance(completion_response, str)
            and "\nOutput:\n" in completion_response
        ):
            completion_response = completion_response.split("\nOutput:\n", 1)[1]
        if stream:
            response = TextStreamer(completion_response)

    if response is None:
        raise ValueError("Unable to generate response")

    logging_obj.post_call(input=prompt, api_key=None, original_response=response)

    streamwrapper = CustomStreamWrapper(
        completion_stream=response,
        model=model,
        custom_llm_provider="vertex_ai",
        logging_obj=logging_obj,
    )

    return streamwrapper
```

## File: `llms/vertex_ai/vertex_ai_partner_models/__init__.py`
```
from litellm.llms.base_llm.chat.transformation import BaseConfig


def get_vertex_ai_partner_model_config(
    model: str, vertex_publisher_or_api_spec: str
) -> BaseConfig:
    """Return config for handling response transformation for vertex ai partner models"""
    if vertex_publisher_or_api_spec == "anthropic":
        from .anthropic.transformation import VertexAIAnthropicConfig

        return VertexAIAnthropicConfig()
    elif vertex_publisher_or_api_spec == "ai21":
        from .ai21.transformation import VertexAIAi21Config

        return VertexAIAi21Config()
    elif (
        vertex_publisher_or_api_spec == "openapi"
        or vertex_publisher_or_api_spec == "mistralai"
    ):
        from .llama3.transformation import VertexAILlama3Config

        return VertexAILlama3Config()
    else:
        raise ValueError(f"Unsupported model: {model}")
```

## File: `llms/vertex_ai/vertex_ai_partner_models/ai21/transformation.py`
```
import types
from typing import Optional

import litellm
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class VertexAIAi21Config(OpenAIGPTConfig):
    """
    Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/ai21

    The class `VertexAIAi21Config` provides configuration for the VertexAI's AI21 API interface

    -> Supports all OpenAI parameters
    """

    def __init__(
        self,
        max_tokens: Optional[int] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ):
        if "max_completion_tokens" in non_default_params:
            non_default_params["max_tokens"] = non_default_params.pop(
                "max_completion_tokens"
            )
        return litellm.OpenAIConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )
```

## File: `llms/vertex_ai/vertex_ai_partner_models/anthropic/experimental_pass_through/transformation.py`
```
from typing import Any, Dict, List, Optional, Tuple

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.types.llms.vertex_ai import VertexPartnerProvider
from litellm.types.router import GenericLiteLLMParams

from ....vertex_llm_base import VertexBase


class VertexAIPartnerModelsAnthropicMessagesConfig(AnthropicMessagesConfig, VertexBase):
    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[dict, Optional[str]]:
        """
        OPTIONAL

        Validate the environment for the request
        """
        if "Authorization" not in headers:
            vertex_ai_project = VertexBase.get_vertex_ai_project(litellm_params)
            vertex_credentials = VertexBase.get_vertex_ai_credentials(litellm_params)
            vertex_ai_location = VertexBase.get_vertex_ai_location(litellm_params)

            access_token, project_id = self._ensure_access_token(
                credentials=vertex_credentials,
                project_id=vertex_ai_project,
                custom_llm_provider="vertex_ai",
            )

            headers["Authorization"] = f"Bearer {access_token}"

            api_base = self.get_complete_vertex_url(
                custom_api_base=api_base,
                vertex_location=vertex_ai_location,
                vertex_project=vertex_ai_project,
                project_id=project_id,
                partner=VertexPartnerProvider.claude,
                stream=optional_params.get("stream", False),
                model=model,
            )

        headers["content-type"] = "application/json"
        return headers, api_base

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if api_base is None:
            raise ValueError(
                "api_base is required. Unable to determine the correct api_base for the request."
            )
        return api_base  # no transformation is needed - handled in validate_environment

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        anthropic_messages_request = super().transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        anthropic_messages_request["anthropic_version"] = "vertex-2023-10-16"

        anthropic_messages_request.pop(
            "model", None
        )  # do not pass model in request body to vertex ai
        return anthropic_messages_request
```

## File: `llms/vertex_ai/vertex_ai_partner_models/anthropic/transformation.py`
```
# What is this?
## Handler file for calling claude-3 on vertex ai
from typing import Any, List, Optional

import httpx

import litellm
from litellm.llms.base_llm.chat.transformation import LiteLLMLoggingObj
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ....anthropic.chat.transformation import AnthropicConfig


class VertexAIError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url=" https://cloud.google.com/vertex-ai/"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class VertexAIAnthropicConfig(AnthropicConfig):
    """
    Reference:https://docs.anthropic.com/claude/reference/messages_post

    Note that the API for Claude on Vertex differs from the Anthropic API documentation in the following ways:

    - `model` is not a valid parameter. The model is instead specified in the Google Cloud endpoint URL.
    - `anthropic_version` is a required parameter and must be set to "vertex-2023-10-16".

    The class `VertexAIAnthropicConfig` provides configuration for the VertexAI's Anthropic API interface. Below are the parameters:

    - `max_tokens` Required (integer) max tokens,
    - `anthropic_version` Required (string) version of anthropic for bedrock - e.g. "bedrock-2023-05-31"
    - `system` Optional (string) the system prompt, conversion from openai format to this is handled in factory.py
    - `temperature` Optional (float) The amount of randomness injected into the response
    - `top_p` Optional (float) Use nucleus sampling.
    - `top_k` Optional (int) Only sample from the top K options for each subsequent token
    - `stop_sequences` Optional (List[str]) Custom text sequences that cause the model to stop generating

    Note: Please make sure to modify the default parameters as required for your use case.
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "vertex_ai"

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        data = super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        data.pop("model", None)  # vertex anthropic doesn't accept 'model' parameter
        return data

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        response = super().transform_response(
            model,
            raw_response,
            model_response,
            logging_obj,
            request_data,
            messages,
            optional_params,
            litellm_params,
            encoding,
            api_key,
            json_mode,
        )
        response.model = model

        return response

    @classmethod
    def is_supported_model(cls, model: str, custom_llm_provider: str) -> bool:
        """
        Check if the model is supported by the VertexAI Anthropic API.
        """
        if (
            custom_llm_provider != "vertex_ai"
            and custom_llm_provider != "vertex_ai_beta"
        ):
            return False
        if "claude" in model.lower():
            return True
        elif model in litellm.vertex_anthropic_models:
            return True
        return False
```

## File: `llms/vertex_ai/vertex_ai_partner_models/llama3/transformation.py`
```
import types
from typing import Any, List, Optional

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.llms.openai import AllMessageValues, OpenAIChatCompletionResponse
from litellm.types.utils import ModelResponse, Usage

from ...common_utils import VertexAIError


class VertexAILlama3Config(OpenAIGPTConfig):
    """
    Reference:https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/llama#streaming

    The class `VertexAILlama3Config` provides configuration for the VertexAI's Llama API interface. Below are the parameters:

    - `max_tokens` Required (integer) max tokens,

    Note: Please make sure to modify the default parameters as required for your use case.
    """

    max_tokens: Optional[int] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key == "max_tokens" and value is None:
                value = self.max_tokens
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    def get_supported_openai_params(self, model: str):
        supported_params = super().get_supported_openai_params(model=model)
        try:
            supported_params.remove("max_retries")
        except KeyError:
            pass
        return supported_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ):
        if "max_completion_tokens" in non_default_params:
            non_default_params["max_tokens"] = non_default_params.pop(
                "max_completion_tokens"
            )
        return super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )

        ## RESPONSE OBJECT
        try:
            completion_response = OpenAIChatCompletionResponse(**raw_response.json())  # type: ignore
        except Exception as e:
            response_headers = getattr(raw_response, "headers", None)
            raise VertexAIError(
                message="Unable to get json response - {}, Original Response: {}".format(
                    str(e), raw_response.text
                ),
                status_code=raw_response.status_code,
                headers=response_headers,
            )
        model_response.model = completion_response["model"]
        model_response.id = completion_response["id"]
        model_response.created = completion_response["created"]
        setattr(model_response, "usage", Usage(**completion_response["usage"]))

        model_response.choices = self._transform_choices(  # type: ignore
            choices=completion_response["choices"],
            json_mode=json_mode,
        )

        return model_response
```

## File: `llms/vertex_ai/vertex_ai_partner_models/main.py`
```
# What is this?
## API Handler for calling Vertex AI Partner Models
from typing import Callable, Optional, Union

import httpx  # type: ignore

import litellm
from litellm import LlmProviders
from litellm.types.llms.vertex_ai import VertexPartnerProvider
from litellm.utils import ModelResponse

from ...custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from ..vertex_llm_base import VertexBase

base_llm_http_handler = BaseLLMHTTPHandler()


class VertexAIError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url=" https://cloud.google.com/vertex-ai/"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class VertexAIPartnerModels(VertexBase):
    def __init__(self) -> None:
        pass

    @staticmethod
    def is_vertex_partner_model(model: str):
        """
        Check if the model string is a Vertex AI Partner Model
        Only use this once you have confirmed that custom_llm_provider is vertex_ai

        Returns:
            bool: True if the model string is a Vertex AI Partner Model, False otherwise
        """
        if (
            model.startswith("meta/")
            or model.startswith("deepseek-ai")
            or model.startswith("mistral")
            or model.startswith("codestral")
            or model.startswith("jamba")
            or model.startswith("claude")
        ):
            return True
        return False

    def completion(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        logging_obj,
        api_base: Optional[str],
        optional_params: dict,
        custom_prompt_dict: dict,
        headers: Optional[dict],
        timeout: Union[float, httpx.Timeout],
        litellm_params: dict,
        vertex_project=None,
        vertex_location=None,
        vertex_credentials=None,
        logger_fn=None,
        acompletion: bool = False,
        client=None,
    ):
        try:
            import vertexai

            from litellm.llms.anthropic.chat import AnthropicChatCompletion
            from litellm.llms.codestral.completion.handler import (
                CodestralTextCompletion,
            )
            from litellm.llms.openai_like.chat.handler import OpenAILikeChatHandler
            from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
                VertexLLM,
            )
        except Exception as e:
            raise VertexAIError(
                status_code=400,
                message=f"""vertexai import failed please run `pip install -U "google-cloud-aiplatform>=1.38"`. Got error: {e}""",
            )

        if not (
            hasattr(vertexai, "preview") or hasattr(vertexai.preview, "language_models")
        ):
            raise VertexAIError(
                status_code=400,
                message="""Upgrade vertex ai. Run `pip install "google-cloud-aiplatform>=1.38"`""",
            )
        try:
            vertex_httpx_logic = VertexLLM()

            access_token, project_id = vertex_httpx_logic._ensure_access_token(
                credentials=vertex_credentials,
                project_id=vertex_project,
                custom_llm_provider="vertex_ai",
            )

            openai_like_chat_completions = OpenAILikeChatHandler()
            codestral_fim_completions = CodestralTextCompletion()
            anthropic_chat_completions = AnthropicChatCompletion()

            ## CONSTRUCT API BASE
            stream: bool = optional_params.get("stream", False) or False

            optional_params["stream"] = stream

            if "llama" in model or "deepseek-ai" in model:
                partner = VertexPartnerProvider.llama
            elif "mistral" in model or "codestral" in model:
                partner = VertexPartnerProvider.mistralai
            elif "jamba" in model:
                partner = VertexPartnerProvider.ai21
            elif "claude" in model:
                partner = VertexPartnerProvider.claude
            else:
                raise ValueError(f"Unknown partner model: {model}")

            api_base = self.get_complete_vertex_url(
                custom_api_base=api_base,
                vertex_location=vertex_location,
                vertex_project=vertex_project,
                project_id=project_id,
                partner=partner,
                stream=stream,
                model=model,
            )

            if "codestral" in model or "mistral" in model:
                model = model.split("@")[0]

            if "codestral" in model and litellm_params.get("text_completion") is True:
                optional_params["model"] = model
                text_completion_model_response = litellm.TextCompletionResponse(
                    stream=stream
                )
                return codestral_fim_completions.completion(
                    model=model,
                    messages=messages,
                    api_base=api_base,
                    api_key=access_token,
                    custom_prompt_dict=custom_prompt_dict,
                    model_response=text_completion_model_response,
                    print_verbose=print_verbose,
                    logging_obj=logging_obj,
                    optional_params=optional_params,
                    acompletion=acompletion,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    timeout=timeout,
                    encoding=encoding,
                )
            elif "claude" in model:
                if headers is None:
                    headers = {}
                headers.update({"Authorization": "Bearer {}".format(access_token)})

                optional_params.update(
                    {
                        "anthropic_version": "vertex-2023-10-16",
                        "is_vertex_request": True,
                    }
                )

                return anthropic_chat_completions.completion(
                    model=model,
                    messages=messages,
                    api_base=api_base,
                    acompletion=acompletion,
                    custom_prompt_dict=litellm.custom_prompt_dict,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    encoding=encoding,  # for calculating input/output tokens
                    api_key=access_token,
                    logging_obj=logging_obj,
                    headers=headers,
                    timeout=timeout,
                    client=client,
                    custom_llm_provider=LlmProviders.VERTEX_AI.value,
                )
            elif "llama" in model:
                return base_llm_http_handler.completion(
                    model=model,
                    stream=stream,
                    messages=messages,
                    acompletion=acompletion,
                    api_base=api_base,
                    model_response=model_response,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    custom_llm_provider="vertex_ai",
                    timeout=timeout,
                    headers=headers,
                    encoding=encoding,
                    api_key=access_token,
                    logging_obj=logging_obj,  # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
                    client=client,
                )
            return openai_like_chat_completions.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                api_key=access_token,
                custom_prompt_dict=custom_prompt_dict,
                model_response=model_response,
                print_verbose=print_verbose,
                logging_obj=logging_obj,
                optional_params=optional_params,
                acompletion=acompletion,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                client=client,
                timeout=timeout,
                encoding=encoding,
                custom_llm_provider="vertex_ai",
                custom_endpoint=True,
            )

        except Exception as e:
            if hasattr(e, "status_code"):
                raise e
            raise VertexAIError(status_code=500, message=str(e))
```

## File: `llms/vertex_ai/vertex_embeddings/embedding_handler.py`
```
from typing import Literal, Optional, Union

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObject
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.vertex_ai.vertex_ai_non_gemini import VertexAIError
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.llms.vertex_ai import *
from litellm.types.utils import EmbeddingResponse

from .types import *


class VertexEmbedding(VertexBase):
    def __init__(self) -> None:
        super().__init__()

    def embedding(
        self,
        model: str,
        input: Union[list, str],
        print_verbose,
        model_response: EmbeddingResponse,
        optional_params: dict,
        logging_obj: LiteLLMLoggingObject,
        custom_llm_provider: Literal[
            "vertex_ai", "vertex_ai_beta", "gemini"
        ],  # if it's vertex_ai or gemini (google ai studio)
        timeout: Optional[Union[float, httpx.Timeout]],
        api_key: Optional[str] = None,
        encoding=None,
        aembedding=False,
        api_base: Optional[str] = None,
        client: Optional[Union[AsyncHTTPHandler, HTTPHandler]] = None,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES] = None,
        gemini_api_key: Optional[str] = None,
        extra_headers: Optional[dict] = None,
    ) -> EmbeddingResponse:
        if aembedding is True:
            return self.async_embedding(  # type: ignore
                model=model,
                input=input,
                logging_obj=logging_obj,
                model_response=model_response,
                optional_params=optional_params,
                encoding=encoding,
                custom_llm_provider=custom_llm_provider,
                timeout=timeout,
                api_base=api_base,
                vertex_project=vertex_project,
                vertex_location=vertex_location,
                vertex_credentials=vertex_credentials,
                gemini_api_key=gemini_api_key,
                extra_headers=extra_headers,
            )

        should_use_v1beta1_features = self.is_using_v1beta1_features(
            optional_params=optional_params
        )

        _auth_header, vertex_project = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider=custom_llm_provider,
        )
        auth_header, api_base = self._get_token_and_url(
            model=model,
            gemini_api_key=gemini_api_key,
            auth_header=_auth_header,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=False,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            should_use_v1beta1_features=should_use_v1beta1_features,
            mode="embedding",
        )
        headers = self.set_headers(auth_header=auth_header, extra_headers=extra_headers)
        vertex_request: VertexEmbeddingRequest = litellm.vertexAITextEmbeddingConfig.transform_openai_request_to_vertex_embedding_request(
            input=input, optional_params=optional_params, model=model
        )

        _client_params = {}
        if timeout:
            _client_params["timeout"] = timeout
        if client is None or not isinstance(client, HTTPHandler):
            client = _get_httpx_client(params=_client_params)
        else:
            client = client  # type: ignore
        ## LOGGING
        logging_obj.pre_call(
            input=vertex_request,
            api_key="",
            additional_args={
                "complete_input_dict": vertex_request,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = client.post(url=api_base, headers=headers, json=vertex_request)  # type: ignore
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise VertexAIError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException:
            raise VertexAIError(status_code=408, message="Timeout error occurred.")

        _json_response = response.json()
        ## LOGGING POST-CALL
        logging_obj.post_call(
            input=input, api_key=None, original_response=_json_response
        )

        model_response = (
            litellm.vertexAITextEmbeddingConfig.transform_vertex_response_to_openai(
                response=_json_response, model=model, model_response=model_response
            )
        )

        return model_response

    async def async_embedding(
        self,
        model: str,
        input: Union[list, str],
        model_response: litellm.EmbeddingResponse,
        logging_obj: LiteLLMLoggingObject,
        optional_params: dict,
        custom_llm_provider: Literal[
            "vertex_ai", "vertex_ai_beta", "gemini"
        ],  # if it's vertex_ai or gemini (google ai studio)
        timeout: Optional[Union[float, httpx.Timeout]],
        api_base: Optional[str] = None,
        client: Optional[AsyncHTTPHandler] = None,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES] = None,
        gemini_api_key: Optional[str] = None,
        extra_headers: Optional[dict] = None,
        encoding=None,
    ) -> litellm.EmbeddingResponse:
        """
        Async embedding implementation
        """
        should_use_v1beta1_features = self.is_using_v1beta1_features(
            optional_params=optional_params
        )
        _auth_header, vertex_project = await self._ensure_access_token_async(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider=custom_llm_provider,
        )
        auth_header, api_base = self._get_token_and_url(
            model=model,
            gemini_api_key=gemini_api_key,
            auth_header=_auth_header,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=False,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            should_use_v1beta1_features=should_use_v1beta1_features,
            mode="embedding",
        )
        headers = self.set_headers(auth_header=auth_header, extra_headers=extra_headers)
        vertex_request: VertexEmbeddingRequest = litellm.vertexAITextEmbeddingConfig.transform_openai_request_to_vertex_embedding_request(
            input=input, optional_params=optional_params, model=model
        )

        _async_client_params = {}
        if timeout:
            _async_client_params["timeout"] = timeout
        if client is None or not isinstance(client, AsyncHTTPHandler):
            client = get_async_httpx_client(
                params=_async_client_params, llm_provider=litellm.LlmProviders.VERTEX_AI
            )
        else:
            client = client  # type: ignore
        ## LOGGING
        logging_obj.pre_call(
            input=vertex_request,
            api_key="",
            additional_args={
                "complete_input_dict": vertex_request,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = await client.post(api_base, headers=headers, json=vertex_request)  # type: ignore
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise VertexAIError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException:
            raise VertexAIError(status_code=408, message="Timeout error occurred.")

        _json_response = response.json()
        ## LOGGING POST-CALL
        logging_obj.post_call(
            input=input, api_key=None, original_response=_json_response
        )

        model_response = (
            litellm.vertexAITextEmbeddingConfig.transform_vertex_response_to_openai(
                response=_json_response, model=model, model_response=model_response
            )
        )

        return model_response
```

## File: `llms/vertex_ai/vertex_embeddings/transformation.py`
```
import types
from typing import List, Literal, Optional, Union

from pydantic import BaseModel

from litellm.types.utils import EmbeddingResponse, Usage

from .types import *


class VertexAITextEmbeddingConfig(BaseModel):
    """
    Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#TextEmbeddingInput

    Args:
        auto_truncate: Optional(bool) If True, will truncate input text to fit within the model's max input length.
        task_type: Optional(str) The type of task to be performed. The default is "RETRIEVAL_QUERY".
        title: Optional(str) The title of the document to be embedded. (only valid with task_type=RETRIEVAL_DOCUMENT).
    """

    auto_truncate: Optional[bool] = None
    task_type: Optional[
        Literal[
            "RETRIEVAL_QUERY",
            "RETRIEVAL_DOCUMENT",
            "SEMANTIC_SIMILARITY",
            "CLASSIFICATION",
            "CLUSTERING",
            "QUESTION_ANSWERING",
            "FACT_VERIFICATION",
        ]
    ] = None
    title: Optional[str] = None

    def __init__(
        self,
        auto_truncate: Optional[bool] = None,
        task_type: Optional[
            Literal[
                "RETRIEVAL_QUERY",
                "RETRIEVAL_DOCUMENT",
                "SEMANTIC_SIMILARITY",
                "CLASSIFICATION",
                "CLUSTERING",
                "QUESTION_ANSWERING",
                "FACT_VERIFICATION",
            ]
        ] = None,
        title: Optional[str] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    def get_supported_openai_params(self):
        return ["dimensions"]

    def map_openai_params(
        self, non_default_params: dict, optional_params: dict, kwargs: dict
    ):
        for param, value in non_default_params.items():
            if param == "dimensions":
                optional_params["outputDimensionality"] = value

        if "input_type" in kwargs:
            optional_params["task_type"] = kwargs.pop("input_type")
        return optional_params, kwargs

    def get_mapped_special_auth_params(self) -> dict:
        """
        Common auth params across bedrock/vertex_ai/azure/watsonx
        """
        return {"project": "vertex_project", "region_name": "vertex_location"}

    def map_special_auth_params(self, non_default_params: dict, optional_params: dict):
        mapped_params = self.get_mapped_special_auth_params()

        for param, value in non_default_params.items():
            if param in mapped_params:
                optional_params[mapped_params[param]] = value
        return optional_params

    def transform_openai_request_to_vertex_embedding_request(
        self, input: Union[list, str], optional_params: dict, model: str
    ) -> VertexEmbeddingRequest:
        """
        Transforms an openai request to a vertex embedding request.
        """
        if model.isdigit():
            return self._transform_openai_request_to_fine_tuned_embedding_request(
                input, optional_params, model
            )

        vertex_request: VertexEmbeddingRequest = VertexEmbeddingRequest()
        vertex_text_embedding_input_list: List[TextEmbeddingInput] = []
        task_type: Optional[TaskType] = optional_params.get("task_type")
        title = optional_params.get("title")

        if isinstance(input, str):
            input = [input]  # Convert single string to list for uniform processing

        for text in input:
            embedding_input = self.create_embedding_input(
                content=text, task_type=task_type, title=title
            )
            vertex_text_embedding_input_list.append(embedding_input)

        vertex_request["instances"] = vertex_text_embedding_input_list
        vertex_request["parameters"] = EmbeddingParameters(**optional_params)

        return vertex_request

    def _transform_openai_request_to_fine_tuned_embedding_request(
        self, input: Union[list, str], optional_params: dict, model: str
    ) -> VertexEmbeddingRequest:
        """
        Transforms an openai request to a vertex fine-tuned embedding request.

        Vertex Doc: https://console.cloud.google.com/vertex-ai/model-garden?hl=en&project=adroit-crow-413218&pageState=(%22galleryStateKey%22:(%22f%22:(%22g%22:%5B%5D,%22o%22:%5B%5D),%22s%22:%22%22))
        Sample Request:

        ```json
        {
            "instances" : [
                {
                "inputs": "How would the Future of AI in 10 Years look?",
                "parameters": {
                    "max_new_tokens": 128,
                    "temperature": 1.0,
                    "top_p": 0.9,
                    "top_k": 10
                }
                }
            ]
        }
        ```
        """
        vertex_request: VertexEmbeddingRequest = VertexEmbeddingRequest()
        vertex_text_embedding_input_list: List[TextEmbeddingFineTunedInput] = []
        if isinstance(input, str):
            input = [input]  # Convert single string to list for uniform processing

        for text in input:
            embedding_input = TextEmbeddingFineTunedInput(inputs=text)
            vertex_text_embedding_input_list.append(embedding_input)

        vertex_request["instances"] = vertex_text_embedding_input_list
        vertex_request["parameters"] = TextEmbeddingFineTunedParameters(
            **optional_params
        )

        return vertex_request

    def create_embedding_input(
        self,
        content: str,
        task_type: Optional[TaskType] = None,
        title: Optional[str] = None,
    ) -> TextEmbeddingInput:
        """
        Creates a TextEmbeddingInput object.

        Vertex requires a List of TextEmbeddingInput objects. This helper function creates a single TextEmbeddingInput object.

        Args:
            content (str): The content to be embedded.
            task_type (Optional[TaskType]): The type of task to be performed".
            title (Optional[str]): The title of the document to be embedded

        Returns:
            TextEmbeddingInput: A TextEmbeddingInput object.
        """
        text_embedding_input = TextEmbeddingInput(content=content)
        if task_type is not None:
            text_embedding_input["task_type"] = task_type
        if title is not None:
            text_embedding_input["title"] = title
        return text_embedding_input

    def transform_vertex_response_to_openai(
        self, response: dict, model: str, model_response: EmbeddingResponse
    ) -> EmbeddingResponse:
        """
        Transforms a vertex embedding response to an openai response.
        """
        if model.isdigit():
            return self._transform_vertex_response_to_openai_for_fine_tuned_models(
                response, model, model_response
            )

        _predictions = response["predictions"]

        embedding_response = []
        input_tokens: int = 0
        for idx, element in enumerate(_predictions):
            embedding = element["embeddings"]
            embedding_response.append(
                {
                    "object": "embedding",
                    "index": idx,
                    "embedding": embedding["values"],
                }
            )
            input_tokens += embedding["statistics"]["token_count"]

        model_response.object = "list"
        model_response.data = embedding_response
        model_response.model = model
        usage = Usage(
            prompt_tokens=input_tokens, completion_tokens=0, total_tokens=input_tokens
        )
        setattr(model_response, "usage", usage)
        return model_response

    def _transform_vertex_response_to_openai_for_fine_tuned_models(
        self, response: dict, model: str, model_response: EmbeddingResponse
    ) -> EmbeddingResponse:
        """
        Transforms a vertex fine-tuned model embedding response to an openai response format.
        """
        _predictions = response["predictions"]

        embedding_response = []
        # For fine-tuned models, we don't get token counts in the response
        input_tokens = 0

        for idx, embedding_values in enumerate(_predictions):
            embedding_response.append(
                {
                    "object": "embedding",
                    "index": idx,
                    "embedding": embedding_values[
                        0
                    ],  # The embedding values are nested one level deeper
                }
            )

        model_response.object = "list"
        model_response.data = embedding_response
        model_response.model = model
        usage = Usage(
            prompt_tokens=input_tokens, completion_tokens=0, total_tokens=input_tokens
        )
        setattr(model_response, "usage", usage)
        return model_response
```

## File: `llms/vertex_ai/vertex_embeddings/types.py`
```
"""
Types for Vertex Embeddings Requests
"""

from enum import Enum
from typing import List, Optional, TypedDict, Union


class TaskType(str, Enum):
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    CLASSIFICATION = "CLASSIFICATION"
    CLUSTERING = "CLUSTERING"
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    FACT_VERIFICATION = "FACT_VERIFICATION"
    CODE_RETRIEVAL_QUERY = "CODE_RETRIEVAL_QUERY"


class TextEmbeddingInput(TypedDict, total=False):
    content: str
    task_type: Optional[TaskType]
    title: Optional[str]


# Fine-tuned models require a different input format
# Ref: https://console.cloud.google.com/vertex-ai/model-garden?hl=en&project=adroit-crow-413218&pageState=(%22galleryStateKey%22:(%22f%22:(%22g%22:%5B%5D,%22o%22:%5B%5D),%22s%22:%22%22))
class TextEmbeddingFineTunedInput(TypedDict, total=False):
    inputs: str


class TextEmbeddingFineTunedParameters(TypedDict, total=False):
    max_new_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]


class EmbeddingParameters(TypedDict, total=False):
    auto_truncate: Optional[bool]
    output_dimensionality: Optional[int]


class VertexEmbeddingRequest(TypedDict, total=False):
    instances: Union[List[TextEmbeddingInput], List[TextEmbeddingFineTunedInput]]
    parameters: Optional[Union[EmbeddingParameters, TextEmbeddingFineTunedParameters]]


# Example usage:
# example_request: VertexEmbeddingRequest = {
#     "instances": [
#         {
#             "content": "I would like embeddings for this text!",
#             "task_type": "RETRIEVAL_DOCUMENT",
#             "title": "document title"
#         }
#     ],
#     "parameters": {
#         "auto_truncate": True,
#         "output_dimensionality": None
#     }
# }
```

## File: `llms/vertex_ai/vertex_llm_base.py`
```
"""
Base Vertex, Google AI Studio LLM Class

Handles Authentication and generating request urls for Vertex AI and Google AI Studio
"""

import json
import os
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.asyncify import asyncify
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES, VertexPartnerProvider

from .common_utils import (
    _get_gemini_url,
    _get_vertex_url,
    all_gemini_url_modes,
    is_global_only_vertex_model,
)

if TYPE_CHECKING:
    from google.auth.credentials import Credentials as GoogleCredentialsObject
else:
    GoogleCredentialsObject = Any


class VertexBase:
    def __init__(self) -> None:
        super().__init__()
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self._credentials: Optional[GoogleCredentialsObject] = None
        self._credentials_project_mapping: Dict[
            Tuple[Optional[VERTEX_CREDENTIALS_TYPES], Optional[str]],
            Tuple[GoogleCredentialsObject, str],
        ] = {}
        self.project_id: Optional[str] = None
        self.async_handler: Optional[AsyncHTTPHandler] = None

    def get_vertex_region(self, vertex_region: Optional[str], model: str) -> str:
        if is_global_only_vertex_model(model):
            return "global"
        return vertex_region or "us-central1"

    def load_auth(
        self, credentials: Optional[VERTEX_CREDENTIALS_TYPES], project_id: Optional[str]
    ) -> Tuple[Any, str]:
        if credentials is not None:
            if isinstance(credentials, str):
                verbose_logger.debug(
                    "Vertex: Loading vertex credentials from %s", credentials
                )
                verbose_logger.debug(
                    "Vertex: checking if credentials is a valid path, os.path.exists(%s)=%s, current dir %s",
                    credentials,
                    os.path.exists(credentials),
                    os.getcwd(),
                )

                try:
                    if os.path.exists(credentials):
                        json_obj = json.load(open(credentials))
                    else:
                        json_obj = json.loads(credentials)
                except Exception:
                    raise Exception(
                        "Unable to load vertex credentials from environment. Got={}".format(
                            credentials
                        )
                    )
            elif isinstance(credentials, dict):
                json_obj = credentials
            else:
                raise ValueError(
                    "Invalid credentials type: {}".format(type(credentials))
                )

            # Check if the JSON object contains Workload Identity Federation configuration
            if "type" in json_obj and json_obj["type"] == "external_account":
                # If environment_id key contains "aws" value it corresponds to an AWS config file
                credential_source = json_obj.get("credential_source", {})
                environment_id = (
                    credential_source.get("environment_id", "")
                    if isinstance(credential_source, dict)
                    else ""
                )
                if isinstance(environment_id, str) and "aws" in environment_id:
                    creds = self._credentials_from_identity_pool_with_aws(json_obj)
                else:
                    creds = self._credentials_from_identity_pool(json_obj)
            # Check if the JSON object contains Authorized User configuration (via gcloud auth application-default login)
            elif "type" in json_obj and json_obj["type"] == "authorized_user":
                creds = self._credentials_from_authorized_user(
                    json_obj,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                if project_id is None:
                    project_id = (
                        creds.quota_project_id
                    )  # authorized user credentials don't have a project_id, only quota_project_id
            else:
                creds = self._credentials_from_service_account(
                    json_obj,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )

            if project_id is None:
                project_id = getattr(creds, "project_id", None)
        else:
            creds, creds_project_id = self._credentials_from_default_auth(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            if project_id is None:
                project_id = creds_project_id

        self.refresh_auth(creds)

        if not project_id:
            raise ValueError("Could not resolve project_id")

        if not isinstance(project_id, str):
            raise TypeError(
                f"Expected project_id to be a str but got {type(project_id)}"
            )

        return creds, project_id

    # Google Auth Helpers -- extracted for mocking purposes in tests
    def _credentials_from_identity_pool(self, json_obj):
        from google.auth import identity_pool

        return identity_pool.Credentials.from_info(json_obj)

    def _credentials_from_identity_pool_with_aws(self, json_obj):
        from google.auth import aws

        return aws.Credentials.from_info(json_obj)

    def _credentials_from_authorized_user(self, json_obj, scopes):
        import google.oauth2.credentials

        return google.oauth2.credentials.Credentials.from_authorized_user_info(
            json_obj, scopes=scopes
        )

    def _credentials_from_service_account(self, json_obj, scopes):
        import google.oauth2.service_account

        return google.oauth2.service_account.Credentials.from_service_account_info(
            json_obj, scopes=scopes
        )

    def _credentials_from_default_auth(self, scopes):
        import google.auth as google_auth

        return google_auth.default(scopes=scopes)

    def get_default_vertex_location(self) -> str:
        return "us-central1"

    def get_api_base(
        self, api_base: Optional[str], vertex_location: Optional[str]
    ) -> str:
        if api_base:
            return api_base
        elif vertex_location == "global":
            return "https://aiplatform.googleapis.com"
        elif vertex_location:
            return f"https://{vertex_location}-aiplatform.googleapis.com"
        else:
            return f"https://{self.get_default_vertex_location()}-aiplatform.googleapis.com"

    @staticmethod
    def create_vertex_url(
        vertex_location: str,
        vertex_project: str,
        partner: VertexPartnerProvider,
        stream: Optional[bool],
        model: str,
        api_base: Optional[str] = None,
    ) -> str:
        """Return the base url for the vertex partner models"""

        api_base = api_base or f"https://{vertex_location}-aiplatform.googleapis.com"
        if partner == VertexPartnerProvider.llama:
            return f"{api_base}/v1/projects/{vertex_project}/locations/{vertex_location}/endpoints/openapi/chat/completions"
        elif partner == VertexPartnerProvider.mistralai:
            if stream:
                return f"{api_base}/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/mistralai/models/{model}:streamRawPredict"
            else:
                return f"{api_base}/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/mistralai/models/{model}:rawPredict"
        elif partner == VertexPartnerProvider.ai21:
            if stream:
                return f"{api_base}/v1beta1/projects/{vertex_project}/locations/{vertex_location}/publishers/ai21/models/{model}:streamRawPredict"
            else:
                return f"{api_base}/v1beta1/projects/{vertex_project}/locations/{vertex_location}/publishers/ai21/models/{model}:rawPredict"
        elif partner == VertexPartnerProvider.claude:
            if stream:
                return f"{api_base}/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/anthropic/models/{model}:streamRawPredict"
            else:
                return f"{api_base}/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/anthropic/models/{model}:rawPredict"

    def get_complete_vertex_url(
        self,
        custom_api_base: Optional[str],
        vertex_location: Optional[str],
        vertex_project: Optional[str],
        project_id: str,
        partner: VertexPartnerProvider,
        stream: Optional[bool],
        model: str,
    ) -> str:
        api_base = self.get_api_base(
            api_base=custom_api_base, vertex_location=vertex_location
        )
        default_api_base = VertexBase.create_vertex_url(
            vertex_location=vertex_location or "us-central1",
            vertex_project=vertex_project or project_id,
            partner=partner,
            stream=stream,
            model=model,
            api_base=api_base,
        )

        if len(default_api_base.split(":")) > 1:
            endpoint = default_api_base.split(":")[-1]
        else:
            endpoint = ""

        _, api_base = self._check_custom_proxy(
            api_base=custom_api_base,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint=endpoint,
            stream=stream,
            auth_header=None,
            url=default_api_base,
        )
        return api_base

    def refresh_auth(self, credentials: Any) -> None:
        from google.auth.transport.requests import (
            Request,  # type: ignore[import-untyped]
        )

        credentials.refresh(Request())

    def _ensure_access_token(
        self,
        credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        project_id: Optional[str],
        custom_llm_provider: Literal[
            "vertex_ai", "vertex_ai_beta", "gemini"
        ],  # if it's vertex_ai or gemini (google ai studio)
    ) -> Tuple[str, str]:
        """
        Returns auth token and project id
        """
        if custom_llm_provider == "gemini":
            return "", ""
        else:
            return self.get_access_token(
                credentials=credentials,
                project_id=project_id,
            )

    def is_using_v1beta1_features(self, optional_params: dict) -> bool:
        """
        VertexAI only supports ContextCaching on v1beta1

        use this helper to decide if request should be sent to v1 or v1beta1

        Returns v1beta1 if context caching is enabled
        Returns v1 in all other cases
        """
        if "cached_content" in optional_params:
            return True
        if "CachedContent" in optional_params:
            return True
        return False

    def _check_custom_proxy(
        self,
        api_base: Optional[str],
        custom_llm_provider: str,
        gemini_api_key: Optional[str],
        endpoint: str,
        stream: Optional[bool],
        auth_header: Optional[str],
        url: str,
    ) -> Tuple[Optional[str], str]:
        """
        for cloudflare ai gateway - https://github.com/BerriAI/litellm/issues/4317

        ## Returns
        - (auth_header, url) - Tuple[Optional[str], str]
        """
        if api_base:
            if custom_llm_provider == "gemini":
                url = "{}:{}".format(api_base, endpoint)
                if gemini_api_key is None:
                    raise ValueError(
                        "Missing gemini_api_key, please set `GEMINI_API_KEY`"
                    )
                auth_header = (
                    gemini_api_key  # cloudflare expects api key as bearer token
                )
            else:
                url = "{}:{}".format(api_base, endpoint)

            if stream is True:
                url = url + "?alt=sse"
        return auth_header, url

    def _get_token_and_url(
        self,
        model: str,
        auth_header: Optional[str],
        gemini_api_key: Optional[str],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        stream: Optional[bool],
        custom_llm_provider: Literal["vertex_ai", "vertex_ai_beta", "gemini"],
        api_base: Optional[str],
        should_use_v1beta1_features: Optional[bool] = False,
        mode: all_gemini_url_modes = "chat",
    ) -> Tuple[Optional[str], str]:
        """
        Internal function. Returns the token and url for the call.

        Handles logic if it's google ai studio vs. vertex ai.

        Returns
            token, url
        """
        if custom_llm_provider == "gemini":
            url, endpoint = _get_gemini_url(
                mode=mode,
                model=model,
                stream=stream,
                gemini_api_key=gemini_api_key,
            )
            auth_header = None  # this field is not used for gemin
        else:
            vertex_location = self.get_vertex_region(
                vertex_region=vertex_location,
                model=model,
            )

            ### SET RUNTIME ENDPOINT ###
            version: Literal["v1beta1", "v1"] = (
                "v1beta1" if should_use_v1beta1_features is True else "v1"
            )
            url, endpoint = _get_vertex_url(
                mode=mode,
                model=model,
                stream=stream,
                vertex_project=vertex_project,
                vertex_location=vertex_location,
                vertex_api_version=version,
            )

        return self._check_custom_proxy(
            api_base=api_base,
            auth_header=auth_header,
            custom_llm_provider=custom_llm_provider,
            gemini_api_key=gemini_api_key,
            endpoint=endpoint,
            stream=stream,
            url=url,
        )

    def get_access_token(
        self,
        credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        project_id: Optional[str],
    ) -> Tuple[str, str]:
        """
        Get access token and project id

        1. Check if credentials are already in self._credentials_project_mapping
        2. If not, load credentials and add to self._credentials_project_mapping
        3. Check if loaded credentials have expired
        4. If expired, refresh credentials
        5. Return access token and project id
        """

        # Convert dict credentials to string for caching
        cache_credentials = (
            json.dumps(credentials) if isinstance(credentials, dict) else credentials
        )
        credential_cache_key = (cache_credentials, project_id)
        _credentials: Optional[GoogleCredentialsObject] = None

        verbose_logger.debug(
            f"Checking cached credentials for project_id: {project_id}"
        )

        if credential_cache_key in self._credentials_project_mapping:
            verbose_logger.debug(
                f"Cached credentials found for project_id: {project_id}."
            )
            # Retrieve both credentials and cached project_id
            cached_entry = self._credentials_project_mapping[credential_cache_key]
            verbose_logger.debug("cached_entry: %s", cached_entry)
            if isinstance(cached_entry, tuple):
                _credentials, credential_project_id = cached_entry
            else:
                # Backward compatibility with old cache format
                _credentials = cached_entry
                credential_project_id = _credentials.quota_project_id or getattr(
                    _credentials, "project_id", None
                )
            verbose_logger.debug(
                "Using cached credentials for project_id: %s",
                credential_project_id,
            )

        else:
            verbose_logger.debug(
                f"Credential cache key not found for project_id: {project_id}, loading new credentials"
            )

            try:
                _credentials, credential_project_id = self.load_auth(
                    credentials=credentials, project_id=project_id
                )
            except Exception as e:
                verbose_logger.exception(
                    f"Failed to load vertex credentials. Check to see if credentials containing partial/invalid information. Error: {str(e)}"
                )
                raise e

            if _credentials is None:
                raise ValueError(
                    "Could not resolve credentials - either dynamically or from environment, for project_id: {}".format(
                        project_id
                    )
                )
            # Cache the project_id and credentials from load_auth result (resolved project_id)
            self._credentials_project_mapping[credential_cache_key] = (
                _credentials,
                credential_project_id,
            )

        ## VALIDATE CREDENTIALS
        verbose_logger.debug(f"Validating credentials for project_id: {project_id}")
        if (
            project_id is None
            and credential_project_id is not None
            and isinstance(credential_project_id, str)
        ):
            project_id = credential_project_id
            # Update cache with resolved project_id for future lookups
            resolved_cache_key = (cache_credentials, project_id)
            if resolved_cache_key not in self._credentials_project_mapping:
                self._credentials_project_mapping[resolved_cache_key] = (
                    _credentials,
                    credential_project_id,
                )

        # Check if credentials are None before accessing attributes
        if _credentials is None:
            raise ValueError("Credentials are None after loading")

        if _credentials.expired:
            verbose_logger.debug(
                f"Credentials expired, refreshing for project_id: {project_id}"
            )
            self.refresh_auth(_credentials)
            self._credentials_project_mapping[credential_cache_key] = (
                _credentials,
                credential_project_id,
            )

        ## VALIDATION STEP
        if _credentials.token is None or not isinstance(_credentials.token, str):
            raise ValueError(
                "Could not resolve credentials token. Got None or non-string token - {}".format(
                    _credentials.token
                )
            )

        if project_id is None:
            raise ValueError("Could not resolve project_id")

        return _credentials.token, project_id

    async def _ensure_access_token_async(
        self,
        credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        project_id: Optional[str],
        custom_llm_provider: Literal[
            "vertex_ai", "vertex_ai_beta", "gemini"
        ],  # if it's vertex_ai or gemini (google ai studio)
    ) -> Tuple[str, str]:
        """
        Async version of _ensure_access_token
        """
        if custom_llm_provider == "gemini":
            return "", ""
        else:
            try:
                return await asyncify(self.get_access_token)(
                    credentials=credentials,
                    project_id=project_id,
                )
            except Exception as e:
                raise e

    def set_headers(
        self, auth_header: Optional[str], extra_headers: Optional[dict]
    ) -> dict:
        headers = {
            "Content-Type": "application/json",
        }
        if auth_header is not None:
            headers["Authorization"] = f"Bearer {auth_header}"
        if extra_headers is not None:
            headers.update(extra_headers)

        return headers

    @staticmethod
    def get_vertex_ai_project(litellm_params: dict) -> Optional[str]:
        return (
            litellm_params.pop("vertex_project", None)
            or litellm_params.pop("vertex_ai_project", None)
            or litellm.vertex_project
            or get_secret_str("VERTEXAI_PROJECT")
        )

    @staticmethod
    def get_vertex_ai_credentials(litellm_params: dict) -> Optional[str]:
        return (
            litellm_params.pop("vertex_credentials", None)
            or litellm_params.pop("vertex_ai_credentials", None)
            or get_secret_str("VERTEXAI_CREDENTIALS")
        )

    @staticmethod
    def get_vertex_ai_location(litellm_params: dict) -> Optional[str]:
        return (
            litellm_params.pop("vertex_location", None)
            or litellm_params.pop("vertex_ai_location", None)
            or litellm.vertex_location
            or get_secret_str("VERTEXAI_LOCATION")
            or get_secret_str("VERTEX_LOCATION")
        )
```

## File: `llms/vertex_ai/vertex_model_garden/main.py`
```
"""
API Handler for calling Vertex AI Model Garden Models

Most Vertex Model Garden Models are OpenAI compatible - so this handler calls `openai_like_chat_completions`

Usage:

response = litellm.completion(
    model="vertex_ai/openai/5464397967697903616",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
)

Sent to this route when `model` is in the format `vertex_ai/openai/{MODEL_ID}`


Vertex Documentation for using the OpenAI /chat/completions endpoint: https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/model_garden/model_garden_pytorch_llama3_deployment.ipynb
"""

from typing import Callable, Optional, Union

import httpx  # type: ignore

from litellm.utils import ModelResponse

from ..common_utils import VertexAIError
from ..vertex_llm_base import VertexBase


def create_vertex_url(
    vertex_location: str,
    vertex_project: str,
    stream: Optional[bool],
    model: str,
    api_base: Optional[str] = None,
) -> str:
    """Return the base url for the vertex garden models"""
    #  f"https://{self.endpoint.location}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{self.endpoint.location}"
    return f"https://{vertex_location}-aiplatform.googleapis.com/v1beta1/projects/{vertex_project}/locations/{vertex_location}/endpoints/{model}"


class VertexAIModelGardenModels(VertexBase):
    def __init__(self) -> None:
        pass

    def completion(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        logging_obj,
        api_base: Optional[str],
        optional_params: dict,
        custom_prompt_dict: dict,
        headers: Optional[dict],
        timeout: Union[float, httpx.Timeout],
        litellm_params: dict,
        vertex_project=None,
        vertex_location=None,
        vertex_credentials=None,
        logger_fn=None,
        acompletion: bool = False,
        client=None,
    ):
        """
        Handles calling Vertex AI Model Garden Models in OpenAI compatible format

        Sent to this route when `model` is in the format `vertex_ai/openai/{MODEL_ID}`
        """
        try:
            import vertexai

            from litellm.llms.openai_like.chat.handler import OpenAILikeChatHandler
            from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
                VertexLLM,
            )
        except Exception as e:
            raise VertexAIError(
                status_code=400,
                message=f"""vertexai import failed please run `pip install -U "google-cloud-aiplatform>=1.38"`. Got error: {e}""",
            )

        if not (
            hasattr(vertexai, "preview") or hasattr(vertexai.preview, "language_models")
        ):
            raise VertexAIError(
                status_code=400,
                message="""Upgrade vertex ai. Run `pip install "google-cloud-aiplatform>=1.38"`""",
            )
        try:
            model = model.replace("openai/", "")
            vertex_httpx_logic = VertexLLM()

            access_token, project_id = vertex_httpx_logic._ensure_access_token(
                credentials=vertex_credentials,
                project_id=vertex_project,
                custom_llm_provider="vertex_ai",
            )

            openai_like_chat_completions = OpenAILikeChatHandler()

            ## CONSTRUCT API BASE
            stream: bool = optional_params.get("stream", False) or False
            optional_params["stream"] = stream
            default_api_base = create_vertex_url(
                vertex_location=vertex_location or "us-central1",
                vertex_project=vertex_project or project_id,
                stream=stream,
                model=model,
            )

            if len(default_api_base.split(":")) > 1:
                endpoint = default_api_base.split(":")[-1]
            else:
                endpoint = ""

            _, api_base = self._check_custom_proxy(
                api_base=api_base,
                custom_llm_provider="vertex_ai",
                gemini_api_key=None,
                endpoint=endpoint,
                stream=stream,
                auth_header=None,
                url=default_api_base,
            )
            model = ""
            return openai_like_chat_completions.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                api_key=access_token,
                custom_prompt_dict=custom_prompt_dict,
                model_response=model_response,
                print_verbose=print_verbose,
                logging_obj=logging_obj,
                optional_params=optional_params,
                acompletion=acompletion,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                client=client,
                timeout=timeout,
                encoding=encoding,
                custom_llm_provider="vertex_ai",
            )

        except Exception as e:
            raise VertexAIError(status_code=500, message=str(e))
```
