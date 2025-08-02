# Directory Structure
_Includes files where the actual content might be omitted. This way the LLM can still use the file structure to understand the project._
```
.
└── clients
    ├── __init__.py
    ├── base_lm.py
    ├── cache.py
    ├── databricks.py
    ├── embedding.py
    ├── lm_local_arbor.py
    ├── lm_local.py
    ├── lm.py
    ├── openai.py
    ├── provider.py
    └── utils_finetune.py
```

# File Contents

## File: `clients/__init__.py`
```
import logging
import os
from pathlib import Path
from typing import Optional

import litellm
from litellm.caching.caching import Cache as LitellmCache

from dspy.clients.base_lm import BaseLM, inspect_history
from dspy.clients.cache import Cache
from dspy.clients.embedding import Embedder
from dspy.clients.lm import LM
from dspy.clients.provider import Provider, TrainingJob

logger = logging.getLogger(__name__)

DISK_CACHE_DIR = os.environ.get("DSPY_CACHEDIR") or os.path.join(Path.home(), ".dspy_cache")
DISK_CACHE_LIMIT = int(os.environ.get("DSPY_CACHE_LIMIT", 3e10))  # 30 GB default


def _litellm_track_cache_hit_callback(kwargs, completion_response, start_time, end_time):
    # Access the cache_hit information
    completion_response.cache_hit = kwargs.get("cache_hit", False)


litellm.success_callback = [_litellm_track_cache_hit_callback]


def configure_cache(
    enable_disk_cache: bool | None = True,
    enable_memory_cache: bool | None = True,
    disk_cache_dir: str | None = DISK_CACHE_DIR,
    disk_size_limit_bytes: int | None = DISK_CACHE_LIMIT,
    memory_max_entries: int | None = 1000000,
    enable_litellm_cache: bool = False,
):
    """Configure the cache for DSPy.

    Args:
        enable_disk_cache: Whether to enable on-disk cache.
        enable_memory_cache: Whether to enable in-memory cache.
        disk_cache_dir: The directory to store the on-disk cache.
        disk_size_limit_bytes: The size limit of the on-disk cache.
        memory_max_entries: The maximum number of entries in the in-memory cache.
        enable_litellm_cache: Whether to enable LiteLLM cache.
    """
    if enable_disk_cache and enable_litellm_cache:
        raise ValueError(
            "Cannot enable both LiteLLM and DSPy on-disk cache, please set at most one of `enable_disk_cache` or "
            "`enable_litellm_cache` to True."
        )

    if enable_litellm_cache:
        try:
            litellm.cache = LitellmCache(disk_cache_dir=DISK_CACHE_DIR, type="disk")

            if litellm.cache.cache.disk_cache.size_limit != DISK_CACHE_LIMIT:
                litellm.cache.cache.disk_cache.reset("size_limit", DISK_CACHE_LIMIT)
        except Exception as e:
            # It's possible that users don't have the write permissions to the cache directory.
            # In that case, we'll just disable the cache.
            logger.warning("Failed to initialize LiteLLM cache: %s", e)
            litellm.cache = None
    else:
        litellm.cache = None

    import dspy

    dspy.cache = Cache(
        enable_disk_cache,
        enable_memory_cache,
        disk_cache_dir,
        disk_size_limit_bytes,
        memory_max_entries,
    )


litellm.telemetry = False
litellm.cache = None  # By default we disable litellm cache and use DSPy on-disk cache.

DSPY_CACHE = Cache(
    enable_disk_cache=True,
    enable_memory_cache=True,
    disk_cache_dir=DISK_CACHE_DIR,
    disk_size_limit_bytes=DISK_CACHE_LIMIT,
    memory_max_entries=1000000,
)

# Turn off by default to avoid LiteLLM logging during every LM call.
litellm.suppress_debug_info = True

if "LITELLM_LOCAL_MODEL_COST_MAP" not in os.environ:
    # Accessed at run time by litellm; i.e., fine to keep after import
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


def enable_litellm_logging():
    litellm.suppress_debug_info = False


def disable_litellm_logging():
    litellm.suppress_debug_info = True


__all__ = [
    "BaseLM",
    "LM",
    "Provider",
    "TrainingJob",
    "inspect_history",
    "Embedder",
    "enable_litellm_logging",
    "disable_litellm_logging",
    "configure_cache",
]
```

## File: `clients/base_lm.py`
```
import datetime
import uuid

from dspy.dsp.utils import settings
from dspy.utils.callback import with_callbacks
from dspy.utils.inspect_history import pretty_print_history

MAX_HISTORY_SIZE = 10_000
GLOBAL_HISTORY = []


class BaseLM:
    """Base class for handling LLM calls.

    Most users can directly use the `dspy.LM` class, which is a subclass of `BaseLM`. Users can also implement their
    own subclasses of `BaseLM` to support custom LLM providers and inject custom logic. To do so, simply override the
    `forward` method and make sure the return format is identical to the
    [OpenAI response format](https://platform.openai.com/docs/api-reference/responses/object).

    Example:

    ```python
    from openai import OpenAI

    import dspy


    class MyLM(dspy.BaseLM):
        def forward(self, prompt, messages=None, **kwargs):
            client = OpenAI()
            return client.chat.completions.create(
                model=self.model,
                messages=messages or [{"role": "user", "content": prompt}],
                **self.kwargs,
            )


    lm = MyLM(model="gpt-4o-mini")
    dspy.configure(lm=lm)
    print(dspy.Predict("q->a")(q="Why did the chicken cross the kitchen?"))
    ```
    """

    def __init__(self, model, model_type="chat", temperature=0.0, max_tokens=1000, cache=True, **kwargs):
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history = []

    def _process_lm_response(self, response, prompt, messages, **kwargs):
        merged_kwargs = {**self.kwargs, **kwargs}

        outputs = []
        for c in response.choices:
            output = {}
            output["text"] = c.message.content if hasattr(c, "message") else c["text"]
            if merged_kwargs.get("logprobs"):
                output["logprobs"] = c.logprobs if hasattr(c, "logprobs") else c["logprobs"]
            if hasattr(c, "message") and getattr(c.message, "tool_calls", None):
                output["tool_calls"] = c.message.tool_calls
            outputs.append(output)

        if all(len(output) == 1 for output in outputs):
            # Return a list if every output only has "text" key
            outputs = [output["text"] for output in outputs]

        if settings.disable_history:
            return outputs

        # Logging, with removed api key & where `cost` is None on cache hit.
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("api_")}
        entry = {
            "prompt": prompt,
            "messages": messages,
            "kwargs": kwargs,
            "response": response,
            "outputs": outputs,
            "usage": dict(response.usage),
            "cost": getattr(response, "_hidden_params", {}).get("response_cost"),
            "timestamp": datetime.datetime.now().isoformat(),
            "uuid": str(uuid.uuid4()),
            "model": self.model,
            "response_model": response.model,
            "model_type": self.model_type,
        }
        self.history.append(entry)
        self.update_global_history(entry)
        caller_modules = settings.caller_modules or []
        for module in caller_modules:
            module.history.append(entry)
        return outputs

    @with_callbacks
    def __call__(self, prompt=None, messages=None, **kwargs):
        response = self.forward(prompt=prompt, messages=messages, **kwargs)
        outputs = self._process_lm_response(response, prompt, messages, **kwargs)

        return outputs

    @with_callbacks
    async def acall(self, prompt=None, messages=None, **kwargs):
        response = await self.aforward(prompt=prompt, messages=messages, **kwargs)
        outputs = self._process_lm_response(response, prompt, messages, **kwargs)
        return outputs

    def forward(self, prompt=None, messages=None, **kwargs):
        """Forward pass for the language model.

        Subclasses must implement this method, and the response should be identical to
        [OpenAI response format](https://platform.openai.com/docs/api-reference/responses/object).
        """
        raise NotImplementedError("Subclasses must implement this method.")

    async def aforward(self, prompt=None, messages=None, **kwargs):
        """Async forward pass for the language model.

        Subclasses that support async should implement this method, and the response should be identical to
        [OpenAI response format](https://platform.openai.com/docs/api-reference/responses/object).
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def copy(self, **kwargs):
        """Returns a copy of the language model with possibly updated parameters."""

        import copy

        new_instance = copy.deepcopy(self)
        new_instance.history = []

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(new_instance, key, value)
            if (key in self.kwargs) or (not hasattr(self, key)):
                new_instance.kwargs[key] = value

        return new_instance

    def inspect_history(self, n: int = 1):
        return pretty_print_history(self.history, n)

    def update_global_history(self, entry):
        if settings.disable_history:
            return

        if len(GLOBAL_HISTORY) >= MAX_HISTORY_SIZE:
            GLOBAL_HISTORY.pop(0)

        GLOBAL_HISTORY.append(entry)


def inspect_history(n: int = 1):
    """The global history shared across all LMs."""
    return pretty_print_history(GLOBAL_HISTORY, n)
```

## File: `clients/cache.py`
```
import copy
import inspect
import logging
import threading
from functools import wraps
from hashlib import sha256
from typing import Any

import cloudpickle
import pydantic
import ujson
from cachetools import LRUCache
from diskcache import FanoutCache

logger = logging.getLogger(__name__)


class Cache:
    """DSPy Cache

    `Cache` provides 2 levels of caching (in the given order):
        1. In-memory cache - implemented with cachetools.LRUCache
        2. On-disk cache - implemented with diskcache.FanoutCache
    """

    def __init__(
        self,
        enable_disk_cache: bool,
        enable_memory_cache: bool,
        disk_cache_dir: str,
        disk_size_limit_bytes: int | None = 1024 * 1024 * 10,
        memory_max_entries: int | None = 1000000,
    ):
        """
        Args:
            enable_disk_cache: Whether to enable on-disk cache.
            enable_memory_cache: Whether to enable in-memory cache.
            disk_cache_dir: The directory where the disk cache is stored.
            disk_size_limit_bytes: The maximum size of the disk cache (in bytes).
            memory_max_entries: The maximum size of the in-memory cache (in number of items).
        """

        self.enable_disk_cache = enable_disk_cache
        self.enable_memory_cache = enable_memory_cache
        if self.enable_memory_cache:
            self.memory_cache = LRUCache(maxsize=memory_max_entries)
        else:
            self.memory_cache = {}
        if self.enable_disk_cache:
            self.disk_cache = FanoutCache(
                shards=16,
                timeout=10,
                directory=disk_cache_dir,
                size_limit=disk_size_limit_bytes,
            )
        else:
            self.disk_cache = {}

        self._lock = threading.RLock()

    def __contains__(self, key: str) -> bool:
        """Check if a key is in the cache."""
        return key in self.memory_cache or key in self.disk_cache

    def cache_key(self, request: dict[str, Any], ignored_args_for_cache_key: list[str] | None = None) -> str:
        """
        Obtain a unique cache key for the given request dictionary by hashing its JSON
        representation. For request fields having types that are known to be JSON-incompatible,
        convert them to a JSON-serializable format before hashing.
        """

        ignored_args_for_cache_key = ignored_args_for_cache_key or []

        def transform_value(value):
            if isinstance(value, type) and issubclass(value, pydantic.BaseModel):
                return value.model_json_schema()
            elif isinstance(value, pydantic.BaseModel):
                return value.model_dump()
            elif callable(value):
                # Try to get the source code of the callable if available
                import inspect

                try:
                    # For regular functions, we can get the source code
                    return f"<callable_source:{inspect.getsource(value)}>"
                except (TypeError, OSError):
                    # For lambda functions or other callables where source isn't available,
                    # use a string representation
                    return f"<callable:{value.__name__ if hasattr(value, '__name__') else 'lambda'}>"
            elif isinstance(value, dict):
                return {k: transform_value(v) for k, v in value.items()}
            else:
                return value

        params = {k: transform_value(v) for k, v in request.items() if k not in ignored_args_for_cache_key}
        return sha256(ujson.dumps(params, sort_keys=True).encode()).hexdigest()

    def get(self, request: dict[str, Any], ignored_args_for_cache_key: list[str] | None = None) -> Any:
        try:
            key = self.cache_key(request, ignored_args_for_cache_key)
        except Exception:
            logger.debug(f"Failed to generate cache key for request: {request}")
            return None

        if self.enable_memory_cache and key in self.memory_cache:
            with self._lock:
                response = self.memory_cache[key]
        elif self.enable_disk_cache and key in self.disk_cache:
            # Found on disk but not in memory cache, add to memory cache
            response = self.disk_cache[key]
            if self.enable_memory_cache:
                with self._lock:
                    self.memory_cache[key] = response
        else:
            return None

        response = copy.deepcopy(response)
        if hasattr(response, "usage"):
            # Clear the usage data when cache is hit, because no LM call is made
            response.usage = {}
        return response

    def put(
        self,
        request: dict[str, Any],
        value: Any,
        ignored_args_for_cache_key: list[str] | None = None,
        enable_memory_cache: bool = True,
    ) -> None:
        try:
            key = self.cache_key(request, ignored_args_for_cache_key)
        except Exception:
            logger.debug(f"Failed to generate cache key for request: {request}")
            return

        if self.enable_memory_cache and enable_memory_cache:
            with self._lock:
                self.memory_cache[key] = value

        if self.enable_disk_cache:
            try:
                self.disk_cache[key] = value
            except Exception as e:
                # Disk cache writing can fail for different reasons, e.g. disk full or the `value` is not picklable.
                logger.debug(f"Failed to put value in disk cache: {value}, {e}")

    def reset_memory_cache(self) -> None:
        if not self.enable_memory_cache:
            return

        with self._lock:
            self.memory_cache.clear()

    def save_memory_cache(self, filepath: str) -> None:
        if not self.enable_memory_cache:
            return

        with self._lock:
            with open(filepath, "wb") as f:
                cloudpickle.dump(self.memory_cache, f)

    def load_memory_cache(self, filepath: str) -> None:
        if not self.enable_memory_cache:
            return

        with self._lock:
            with open(filepath, "rb") as f:
                self.memory_cache = cloudpickle.load(f)


def request_cache(
    cache_arg_name: str | None = None,
    ignored_args_for_cache_key: list[str] | None = None,
    enable_memory_cache: bool = True,
    *,  # everything after this is keyword-only
    maxsize: int | None = None,  # legacy / no-op
):
    """
    Decorator for applying caching to a function based on the request argument.

    Args:
        cache_arg_name: The name of the argument that contains the request. If not provided, the entire kwargs is used
            as the request.
        ignored_args_for_cache_key: A list of arguments to ignore when computing the cache key from the request.
        enable_memory_cache: Whether to enable in-memory cache at call time. If False, the memory cache will not be
            written to on new data.
    """
    ignored_args_for_cache_key = ignored_args_for_cache_key or ["api_key", "api_base", "base_url"]
    # Deprecation notice
    if maxsize is not None:
        logger.warning(
            "[DEPRECATION] `maxsize` is deprecated and no longer does anything; "
            "the cache is now handled internally by `dspy.cache`. "
            "This parameter will be removed in a future release.",
        )

    def decorator(fn):
        @wraps(fn)
        def process_request(args, kwargs):
            # Use fully qualified function name for uniqueness
            fn_identifier = f"{fn.__module__}.{fn.__qualname__}"

            # Create a modified request that includes the function identifier so that it's incorporated into the cache
            # key. Deep copy is required because litellm sometimes modifies the kwargs in place.
            if cache_arg_name:
                # When `cache_arg_name` is provided, use the value of the argument with this name as the request for
                # caching.
                modified_request = copy.deepcopy(kwargs[cache_arg_name])
            else:
                # When `cache_arg_name` is not provided, use the entire kwargs as the request for caching.
                modified_request = copy.deepcopy(kwargs)
                for i, arg in enumerate(args):
                    modified_request[f"positional_arg_{i}"] = arg
            modified_request["_fn_identifier"] = fn_identifier

            return modified_request

        @wraps(fn)
        def sync_wrapper(*args, **kwargs):
            import dspy

            cache = dspy.cache
            modified_request = process_request(args, kwargs)

            # Retrieve from cache if available
            cached_result = cache.get(modified_request, ignored_args_for_cache_key)

            if cached_result is not None:
                return cached_result

            # Otherwise, compute and store the result
            # Make a copy of the original request in case it's modified in place, e.g., deleting some fields
            original_request = copy.deepcopy(modified_request)
            result = fn(*args, **kwargs)
            # `enable_memory_cache` can be provided at call time to avoid indefinite growth.
            cache.put(original_request, result, ignored_args_for_cache_key, enable_memory_cache)

            return result

        @wraps(fn)
        async def async_wrapper(*args, **kwargs):
            import dspy

            cache = dspy.cache
            modified_request = process_request(args, kwargs)

            # Retrieve from cache if available
            cached_result = cache.get(modified_request, ignored_args_for_cache_key)
            if cached_result is not None:
                return cached_result

            # Otherwise, compute and store the result
            # Make a copy of the original request in case it's modified in place, e.g., deleting some fields
            original_request = copy.deepcopy(modified_request)
            result = await fn(*args, **kwargs)
            cache.put(original_request, result, ignored_args_for_cache_key, enable_memory_cache)

            return result

        if inspect.iscoroutinefunction(fn):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
```

## File: `clients/databricks.py`
```
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any

import requests
import ujson

from dspy.clients.provider import Provider, TrainingJob
from dspy.clients.utils_finetune import TrainDataFormat, get_finetune_directory

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)


class TrainingJobDatabricks(TrainingJob):
    def __init__(self, finetuning_run=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finetuning_run = finetuning_run
        self.launch_started = False
        self.launch_completed = False
        self.endpoint_name = None

    def status(self):
        if not self.finetuning_run:
            return None
        try:
            from databricks.model_training import foundation_model as fm
        except ImportError:
            raise ImportError(
                "To use Databricks finetuning, please install the databricks_genai package via "
                "`pip install databricks_genai`."
            )
        run = fm.get(self.finetuning_run)
        return run.status


class DatabricksProvider(Provider):
    finetunable = True
    TrainingJob = TrainingJobDatabricks

    @staticmethod
    def is_provider_model(model: str) -> bool:
        # We don't automatically infer Databricks models because Databricks is not a proprietary model provider.
        return False

    @staticmethod
    def deploy_finetuned_model(
        model: str,
        data_format: TrainDataFormat | None = None,
        databricks_host: str | None = None,
        databricks_token: str | None = None,
        deploy_timeout: int = 900,
    ):
        workspace_client = _get_workspace_client()
        model_version = next(workspace_client.model_versions.list(model)).version

        # Allow users to override the host and token. This is useful on Databricks hosted runtime.
        databricks_host = databricks_host or workspace_client.config.host
        databricks_token = databricks_token or workspace_client.config.token

        headers = {"Context-Type": "text/json", "Authorization": f"Bearer {databricks_token}"}

        optimizable_info = requests.get(
            url=f"{databricks_host}/api/2.0/serving-endpoints/get-model-optimization-info/{model}/{model_version}",
            headers=headers,
        ).json()

        if "optimizable" not in optimizable_info or not optimizable_info["optimizable"]:
            raise ValueError(f"Model is not eligible for provisioned throughput: {optimizable_info}")

        chunk_size = optimizable_info["throughput_chunk_size"]

        # Minimum desired provisioned throughput
        min_provisioned_throughput = 0

        # Maximum desired provisioned throughput
        max_provisioned_throughput = chunk_size

        # Databricks serving endpoint names cannot contain ".".
        model_name = model.replace(".", "_")

        get_endpoint_response = requests.get(
            url=f"{databricks_host}/api/2.0/serving-endpoints/{model_name}", json={"name": model_name}, headers=headers
        )

        if get_endpoint_response.status_code == 200:
            logger.info(f"Serving endpoint {model_name} already exists, updating it instead of creating a new one.")
            # The serving endpoint already exists, we will update it instead of creating a new one.
            data = {
                "served_entities": [
                    {
                        "name": model_name,
                        "entity_name": model,
                        "entity_version": model_version,
                        "min_provisioned_throughput": min_provisioned_throughput,
                        "max_provisioned_throughput": max_provisioned_throughput,
                    }
                ]
            }

            response = requests.put(
                url=f"{databricks_host}/api/2.0/serving-endpoints/{model_name}/config",
                json=data,
                headers=headers,
            )
        else:
            logger.info(f"Creating serving endpoint {model_name} on Databricks model serving!")
            # Send the POST request to create the serving endpoint.
            data = {
                "name": model_name,
                "config": {
                    "served_entities": [
                        {
                            "name": model_name,
                            "entity_name": model,
                            "entity_version": model_version,
                            "min_provisioned_throughput": min_provisioned_throughput,
                            "max_provisioned_throughput": max_provisioned_throughput,
                        }
                    ]
                },
            }

            response = requests.post(url=f"{databricks_host}/api/2.0/serving-endpoints", json=data, headers=headers)

        if response.status_code == 200:
            logger.info(
                f"Successfully started creating/updating serving endpoint {model_name} on Databricks model serving!"
            )
        else:
            raise ValueError(f"Failed to create serving endpoint: {response.json()}.")

        logger.info(
            f"Waiting for serving endpoint {model_name} to be ready, this might take a few minutes... You can check "
            f"the status of the endpoint at {databricks_host}/ml/endpoints/{model_name}"
        )
        from openai import OpenAI

        client = OpenAI(
            api_key=databricks_token,
            base_url=f"{databricks_host}/serving-endpoints",
        )
        # Wait for the deployment to be ready.
        num_retries = deploy_timeout // 60
        for _ in range(num_retries):
            try:
                if data_format == TrainDataFormat.CHAT:
                    client.chat.completions.create(
                        messages=[{"role": "user", "content": "hi"}], model=model_name, max_tokens=1
                    )
                elif data_format == TrainDataFormat.COMPLETION:
                    client.completions.create(prompt="hi", model=model_name, max_tokens=1)
                logger.info(f"Databricks model serving endpoint {model_name} is ready!")
                return
            except Exception:
                time.sleep(60)

        raise ValueError(
            f"Failed to create serving endpoint {model_name} on Databricks model serving platform within "
            f"{deploy_timeout} seconds."
        )

    @staticmethod
    def finetune(
        job: TrainingJobDatabricks,
        model: str,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | str | None = "chat",
        train_kwargs: dict[str, Any] | None = None,
    ) -> str:
        if isinstance(train_data_format, str):
            if train_data_format == "chat":
                train_data_format = TrainDataFormat.CHAT
            elif train_data_format == "completion":
                train_data_format = TrainDataFormat.COMPLETION
            else:
                raise ValueError(
                    f"String `train_data_format` must be one of 'chat' or 'completion', but received: {train_data_format}."
                )

        if "train_data_path" not in train_kwargs:
            raise ValueError("The `train_data_path` must be provided to finetune on Databricks.")
        # Add the file name to the directory path.
        train_kwargs["train_data_path"] = DatabricksProvider.upload_data(
            train_data, train_kwargs["train_data_path"], train_data_format
        )

        try:
            from databricks.model_training import foundation_model as fm
        except ImportError:
            raise ImportError(
                "To use Databricks finetuning, please install the databricks_genai package via "
                "`pip install databricks_genai`."
            )

        if "register_to" not in train_kwargs:
            raise ValueError("The `register_to` must be provided to finetune on Databricks.")

        # Allow users to override the host and token. This is useful on Databricks hosted runtime.
        databricks_host = train_kwargs.pop("databricks_host", None)
        databricks_token = train_kwargs.pop("databricks_token", None)

        skip_deploy = train_kwargs.pop("skip_deploy", False)
        deploy_timeout = train_kwargs.pop("deploy_timeout", 900)

        logger.info("Starting finetuning on Databricks... this might take a few minutes to finish.")
        finetuning_run = fm.create(
            model=model,
            **train_kwargs,
        )

        job.run = finetuning_run

        # Wait for the finetuning run to be ready.
        while True:
            job.run = fm.get(job.run)
            if job.run.status.display_name == "Completed":
                logger.info("Finetuning run completed successfully!")
                break
            elif job.run.status.display_name == "Failed":
                raise ValueError(
                    f"Finetuning run failed with status: {job.run.status.display_name}. Please check the Databricks "
                    f"workspace for more details. Finetuning job's metadata: {job.run}."
                )
            else:
                time.sleep(60)

        if skip_deploy:
            return None

        job.launch_started = True
        model_to_deploy = train_kwargs.get("register_to")
        job.endpoint_name = model_to_deploy.replace(".", "_")
        DatabricksProvider.deploy_finetuned_model(
            model_to_deploy, train_data_format, databricks_host, databricks_token, deploy_timeout
        )
        job.launch_completed = True
        # The finetuned model name should be in the format: "databricks/<endpoint_name>".
        return f"databricks/{job.endpoint_name}"

    @staticmethod
    def upload_data(train_data: list[dict[str, Any]], databricks_unity_catalog_path: str, data_format: TrainDataFormat):
        logger.info("Uploading finetuning data to Databricks Unity Catalog...")
        file_path = _save_data_to_local_file(train_data, data_format)

        w = _get_workspace_client()
        _create_directory_in_databricks_unity_catalog(w, databricks_unity_catalog_path)

        try:
            with open(file_path, "rb") as f:
                target_path = os.path.join(databricks_unity_catalog_path, os.path.basename(file_path))
                w.files.upload(target_path, f, overwrite=True)
            logger.info("Successfully uploaded finetuning data to Databricks Unity Catalog!")
            return target_path
        except Exception as e:
            raise ValueError(f"Failed to upload finetuning data to Databricks Unity Catalog: {e}")


def _get_workspace_client() -> "WorkspaceClient":
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError:
        raise ImportError(
            "To use Databricks finetuning, please install the databricks-sdk package via "
            "`pip install databricks-sdk`."
        )
    return WorkspaceClient()


def _create_directory_in_databricks_unity_catalog(w: "WorkspaceClient", databricks_unity_catalog_path: str):
    pattern = r"^/Volumes/(?P<catalog>[^/]+)/(?P<schema>[^/]+)/(?P<volume>[^/]+)(/[^/]+)+$"
    match = re.match(pattern, databricks_unity_catalog_path)
    if not match:
        raise ValueError(
            f"Databricks Unity Catalog path must be in the format '/Volumes/<catalog>/<schema>/<volume>/...', but "
            f"received: {databricks_unity_catalog_path}."
        )

    catalog = match.group("catalog")
    schema = match.group("schema")
    volume = match.group("volume")

    try:
        volume_path = f"{catalog}.{schema}.{volume}"
        w.volumes.read(volume_path)
    except Exception:
        raise ValueError(
            f"Databricks Unity Catalog volume does not exist: {volume_path}, please create it on the Databricks "
            "workspace."
        )

    try:
        w.files.get_directory_metadata(databricks_unity_catalog_path)
        logger.info(f"Directory {databricks_unity_catalog_path} already exists, skip creating it.")
    except Exception:
        # Create the directory if it doesn't exist, we don't raise an error because this is a common case.
        logger.info(f"Creating directory {databricks_unity_catalog_path} in Databricks Unity Catalog...")
        w.files.create_directory(databricks_unity_catalog_path)
        logger.info(f"Successfully created directory {databricks_unity_catalog_path} in Databricks Unity Catalog!")


def _save_data_to_local_file(train_data: list[dict[str, Any]], data_format: TrainDataFormat):
    import uuid

    file_name = f"finetuning_{uuid.uuid4()}.jsonl"

    finetune_dir = get_finetune_directory()
    file_path = os.path.join(finetune_dir, file_name)
    file_path = os.path.abspath(file_path)
    with open(file_path, "w") as f:
        for item in train_data:
            if data_format == TrainDataFormat.CHAT:
                _validate_chat_data(item)
            elif data_format == TrainDataFormat.COMPLETION:
                _validate_completion_data(item)

            f.write(ujson.dumps(item) + "\n")
    return file_path


def _validate_chat_data(data: dict[str, Any]):
    if "messages" not in data:
        raise ValueError(
            "Each finetuning data must be a dict with a 'messages' key when `task=CHAT_COMPLETION`, but "
            f"received: {data}"
        )

    if not isinstance(data["messages"], list):
        raise ValueError(
            "The value of the 'messages' key in each finetuning data must be a list of dicts with keys 'role' and "
            f"'content' when `task=CHAT_COMPLETION`, but received: {data['messages']}"
        )

    for message in data["messages"]:
        if "role" not in message:
            raise ValueError(f"Each message in the 'messages' list must contain a 'role' key, but received: {message}.")
        if "content" not in message:
            raise ValueError(
                f"Each message in the 'messages' list must contain a 'content' key, but received: {message}."
            )


def _validate_completion_data(data: dict[str, Any]):
    if "prompt" not in data:
        raise ValueError(
            "Each finetuning data must be a dict with a 'prompt' key when `task=INSTRUCTION_FINETUNE`, but "
            f"received: {data}"
        )
    if "response" not in data and "completion" not in data:
        raise ValueError(
            "Each finetuning data must be a dict with a 'response' or 'completion' key when "
            f"`task=INSTRUCTION_FINETUNE`, but received: {data}"
        )
```

## File: `clients/embedding.py`
```
from typing import Any, Callable

import litellm
import numpy as np

from dspy.clients.cache import request_cache


class Embedder:
    """DSPy embedding class.

    The class for computing embeddings for text inputs. This class provides a unified interface for both:

    1. Hosted embedding models (e.g. OpenAI's text-embedding-3-small) via litellm integration
    2. Custom embedding functions that you provide

    For hosted models, simply pass the model name as a string (e.g., "openai/text-embedding-3-small"). The class will use
    litellm to handle the API calls and caching.

    For custom embedding models, pass a callable function that:
    - Takes a list of strings as input.
    - Returns embeddings as either:
        - A 2D numpy array of float32 values
        - A 2D list of float32 values
    - Each row should represent one embedding vector

    Args:
        model: The embedding model to use. This can be either a string (representing the name of the hosted embedding
            model, must be an embedding model supported by litellm) or a callable that represents a custom embedding
            model.
        batch_size (int, optional): The default batch size for processing inputs in batches. Defaults to 200.
        caching (bool, optional): Whether to cache the embedding response when using a hosted model. Defaults to True.
        **kwargs: Additional default keyword arguments to pass to the embedding model.

    Examples:
        Example 1: Using a hosted model.

        ```python
        import dspy

        embedder = dspy.Embedder("openai/text-embedding-3-small", batch_size=100)
        embeddings = embedder(["hello", "world"])

        assert embeddings.shape == (2, 1536)
        ```

        Example 2: Using any local embedding model, e.g. from https://huggingface.co/models?library=sentence-transformers.

        ```python
        # pip install sentence_transformers
        import dspy
        from sentence_transformers import SentenceTransformer

        # Load an extremely efficient local model for retrieval
        model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")

        embedder = dspy.Embedder(model.encode)
        embeddings = embedder(["hello", "world"], batch_size=1)

        assert embeddings.shape == (2, 1024)
        ```

        Example 3: Using a custom function.

        ```python
        import dspy
        import numpy as np

        def my_embedder(texts):
            return np.random.rand(len(texts), 10)

        embedder = dspy.Embedder(my_embedder)
        embeddings = embedder(["hello", "world"], batch_size=1)

        assert embeddings.shape == (2, 10)
        ```
    """

    def __init__(self, model: str | Callable, batch_size: int = 200, caching: bool = True, **kwargs: dict[str, Any]):
        self.model = model
        self.batch_size = batch_size
        self.caching = caching
        self.default_kwargs = kwargs

    def _preprocess(self, inputs, batch_size=None, caching=None, **kwargs):
        if isinstance(inputs, str):
            is_single_input = True
            inputs = [inputs]
        else:
            is_single_input = False

        if not all(isinstance(inp, str) for inp in inputs):
            raise ValueError("All inputs must be strings.")

        batch_size = batch_size or self.batch_size
        caching = caching or self.caching
        merged_kwargs = self.default_kwargs.copy()
        merged_kwargs.update(kwargs)

        input_batches = []
        for i in range(0, len(inputs), batch_size):
            input_batches.append(inputs[i : i + batch_size])

        return input_batches, caching, merged_kwargs, is_single_input

    def _postprocess(self, embeddings_list, is_single_input):
        embeddings = np.array(embeddings_list, dtype=np.float32)
        if is_single_input:
            return embeddings[0]
        else:
            return np.array(embeddings, dtype=np.float32)

    def __call__(self, inputs: str | list[str], batch_size: int | None = None, caching: bool | None = None, **kwargs: dict[str, Any]) -> np.ndarray:
        """Compute embeddings for the given inputs.

        Args:
            inputs: The inputs to compute embeddings for, can be a single string or a list of strings.
            batch_size (int, optional): The batch size for processing inputs. If None, defaults to the batch_size set
                during initialization.
            caching (bool, optional): Whether to cache the embedding response when using a hosted model. If None,
                defaults to the caching setting from initialization.
            kwargs: Additional keyword arguments to pass to the embedding model. These will override the default
                kwargs provided during initialization.

        Returns:
            numpy.ndarray: If the input is a single string, returns a 1D numpy array representing the embedding.
            If the input is a list of strings, returns a 2D numpy array of embeddings, one embedding per row.
        """
        input_batches, caching, kwargs, is_single_input = self._preprocess(inputs, batch_size, caching, **kwargs)

        compute_embeddings = _cached_compute_embeddings if caching else _compute_embeddings

        embeddings_list = []

        for batch in input_batches:
            embeddings_list.extend(compute_embeddings(self.model, batch, caching=caching, **kwargs))
        return self._postprocess(embeddings_list, is_single_input)

    async def acall(self, inputs, batch_size=None, caching=None, **kwargs):
        input_batches, caching, kwargs, is_single_input = self._preprocess(inputs, batch_size, caching, **kwargs)

        embeddings_list = []
        acompute_embeddings = _cached_acompute_embeddings if caching else _acompute_embeddings

        for batch in input_batches:
            embeddings_list.extend(await acompute_embeddings(self.model, batch, caching=caching, **kwargs))
        return self._postprocess(embeddings_list, is_single_input)


def _compute_embeddings(model, batch_inputs, caching=False, **kwargs):
    if isinstance(model, str):
        caching = caching and litellm.cache is not None
        embedding_response = litellm.embedding(model=model, input=batch_inputs, caching=caching, **kwargs)
        return [data["embedding"] for data in embedding_response.data]
    elif callable(model):
        return model(batch_inputs, **kwargs)
    else:
        raise ValueError(f"`model` in `dspy.Embedder` must be a string or a callable, but got {type(model)}.")


@request_cache(ignored_args_for_cache_key=["api_key", "api_base", "base_url"])
def _cached_compute_embeddings(model, batch_inputs, caching=True, **kwargs):
    return _compute_embeddings(model, batch_inputs, caching=caching, **kwargs)


async def _acompute_embeddings(model, batch_inputs, caching=False, **kwargs):
    if isinstance(model, str):
        caching = caching and litellm.cache is not None
        embedding_response = await litellm.aembedding(model=model, input=batch_inputs, caching=caching, **kwargs)
        return [data["embedding"] for data in embedding_response.data]
    elif callable(model):
        return model(batch_inputs, **kwargs)
    else:
        raise ValueError(f"`model` in `dspy.Embedder` must be a string or a callable, but got {type(model)}.")


@request_cache(ignored_args_for_cache_key=["api_key", "api_base", "base_url"])
async def _cached_acompute_embeddings(model, batch_inputs, caching=True, **kwargs):
    return await _acompute_embeddings(model, batch_inputs, caching=caching, **kwargs)
```

## File: `clients/lm_local_arbor.py`
```
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypedDict

import openai
import requests

import dspy
from dspy.clients.provider import Provider, ReinforceJob, TrainingJob
from dspy.clients.utils_finetune import GRPOGroup, TrainDataFormat, TrainingStatus, save_data

if TYPE_CHECKING:
    from dspy.clients.lm import LM


class GRPOTrainKwargs(TypedDict):
    num_generations: int


class ArborTrainingJob(TrainingJob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.provider_file_id = None
        self.provider_job_id = None

    def cancel(self):
        if ArborProvider.does_job_exist(self.provider_job_id):
            status = self.status()
            if ArborProvider.is_terminal_training_status(status):
                err_msg = "Jobs that are complete cannot be canceled."
                err_msg += f" Job with ID {self.provider_job_id} is done."
                raise Exception(err_msg)
            openai.fine_tuning.jobs.cancel(self.provider_job_id)
            self.provider_job_id = None

        if self.provider_file_id is not None:
            if ArborProvider.does_file_exist(self.provider_file_id):
                openai.files.delete(self.provider_file_id)
            self.provider_file_id = None

        super().cancel()

    def status(self) -> TrainingStatus:
        status = ArborProvider.get_training_status(self.provider_job_id)
        return status


class ArborReinforceJob(ReinforceJob):
    DEFAULT_TRAIN_KWARGS = {  # noqa: RUF012
        "temperature": 0.9,
        "beta": 0.04,
        "num_iterations": 1,
        "per_device_train_batch_size": 8,
        "learning_rate": 1e-6,
        "gradient_accumulation_steps": 1,
        # This is false by default in TRL, but I think it makes sense to be true for us
        "gradient_checkpointing": True,
        "lr_scheduler_type": "constant_with_warmup",
        "max_prompt_length": None,
        "max_completion_length": None,
        "gradient_checkpointing_kwargs": None,
        "bf16": False,
        "scale_rewards": True,
        "max_grad_norm": 1.0,
        "report_to": "none",
        "log_completions": True,
        "logging_steps": 100,
        # By default, none is the model's max context length
        "max_context_length": None,
        "lora": False,
    }

    def __init__(self, lm: "LM", train_kwargs: GRPOTrainKwargs):
        # The teleprompter must ensure that this is set
        if "num_generations" not in train_kwargs:
            raise ValueError("num_generations must be set in the training kwargs")

        self.lm = lm
        self.train_kwargs = train_kwargs
        self.provider_job_id = None
        self.checkpoints = {}
        self.last_checkpoint = None

    def initialize(self):
        # TODO(GRPO Team): Set provider job ID
        num_generations = self.train_kwargs.get("num_generations")
        temperature = self.train_kwargs.get("temperature", self.DEFAULT_TRAIN_KWARGS["temperature"])
        beta = self.train_kwargs.get("beta", self.DEFAULT_TRAIN_KWARGS["beta"])
        num_iterations = self.train_kwargs.get("num_iterations", self.DEFAULT_TRAIN_KWARGS["num_iterations"])
        per_device_train_batch_size = self.train_kwargs.get(
            "per_device_train_batch_size", self.DEFAULT_TRAIN_KWARGS["per_device_train_batch_size"]
        )
        learning_rate = self.train_kwargs.get("learning_rate", self.DEFAULT_TRAIN_KWARGS["learning_rate"])
        gradient_accumulation_steps = self.train_kwargs.get(
            "gradient_accumulation_steps", self.DEFAULT_TRAIN_KWARGS["gradient_accumulation_steps"]
        )
        gradient_checkpointing = self.train_kwargs.get(
            "gradient_checkpointing", self.DEFAULT_TRAIN_KWARGS["gradient_checkpointing"]
        )
        lr_scheduler_type = self.train_kwargs.get("lr_scheduler_type", self.DEFAULT_TRAIN_KWARGS["lr_scheduler_type"])
        max_prompt_length = self.train_kwargs.get("max_prompt_length", self.DEFAULT_TRAIN_KWARGS["max_prompt_length"])
        max_completion_length = self.train_kwargs.get(
            "max_completion_length", self.DEFAULT_TRAIN_KWARGS["max_completion_length"]
        )
        bf16 = self.train_kwargs.get("bf16", self.DEFAULT_TRAIN_KWARGS["bf16"])
        scale_rewards = self.train_kwargs.get("scale_rewards", self.DEFAULT_TRAIN_KWARGS["scale_rewards"])
        gradient_checkpointing_kwargs = self.train_kwargs.get(
            "gradient_checkpointing_kwargs", self.DEFAULT_TRAIN_KWARGS["gradient_checkpointing_kwargs"]
        )
        max_grad_norm = self.train_kwargs.get("max_grad_norm", self.DEFAULT_TRAIN_KWARGS["max_grad_norm"])
        report_to = self.train_kwargs.get("report_to", self.DEFAULT_TRAIN_KWARGS["report_to"])
        log_completions = self.train_kwargs.get("log_completions", self.DEFAULT_TRAIN_KWARGS["log_completions"])
        logging_steps = self.train_kwargs.get("logging_steps", self.DEFAULT_TRAIN_KWARGS["logging_steps"])
        max_context_length = self.train_kwargs.get(
            "max_context_length", self.DEFAULT_TRAIN_KWARGS["max_context_length"]
        )
        lora = self.train_kwargs.get("lora", self.DEFAULT_TRAIN_KWARGS["lora"])
        api_base = self.lm.kwargs["api_base"]

        suffix = "dspy"
        finetune_model = ArborProvider._remove_provider_prefix(self.lm.model)
        data = {
            "model": finetune_model,
            "suffix": suffix,
            "num_generations": num_generations,
            "temperature": temperature,
            "beta": beta,
            "num_iterations": num_iterations,
            "per_device_train_batch_size": per_device_train_batch_size,
            "learning_rate": learning_rate,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_checkpointing": gradient_checkpointing,
            "lr_scheduler_type": lr_scheduler_type,
            "max_prompt_length": max_prompt_length,
            "max_completion_length": max_completion_length,
            "bf16": bf16,
            "scale_rewards": scale_rewards,
            "gradient_checkpointing_kwargs": gradient_checkpointing_kwargs,
            "max_grad_norm": max_grad_norm,
            "report_to": report_to,
            "log_completions": log_completions,
            "logging_steps": logging_steps,
            "max_context_length": max_context_length,
            "lora": lora,
        }
        url = f"{api_base}fine_tuning/grpo/initialize"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url=url, headers=headers, json=data)
        assert response.status_code == 200, f"Failed to initialize GRPO: {response}"
        self.lm.model = ArborProvider._add_provider_prefix(finetune_model)
        # self.provider_job_id = response.json().get("job_id")  # TODO: To be updated

    def _run_grpo_step_one_group(
        self, train_group: GRPOGroup, train_data_format: TrainDataFormat | str | None = None
    ):
        # TODO: Check that the data follows the intended format
        api_base = self.lm.kwargs["api_base"]
        # api_key = self.lm.kwargs["api_key"]

        finetune_model = ArborProvider._remove_provider_prefix(self.lm.model)
        data = {"model": finetune_model, "batch": train_group}
        url = f"{api_base}fine_tuning/grpo/step"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=data)
        assert response.status_code == 200, f"Failed to run a GRPO step: {response.text}"
        response = response.json()
        assert "current_model" in response, f"Response does not contain the next model ID to be used: {response}"
        current_model = response["current_model"]
        self.lm.model = ArborProvider._add_provider_prefix(current_model)

    def step(self, train_data: list[GRPOGroup], train_data_format: TrainDataFormat | str | None):
        # Note: TrainDataFormat specifies the format for the inner most dict.
        # Because we run GRPO at the group level, train_data will be a list of
        # groups, where each group is a list of GRPOChatData. Our teleprompters
        # ensure that we pass the right data format.
        # We can consider making this distinction clearer, e.g., by having two
        # different step methods or changing our smallets data format to be the
        # GRPO group.
        # TODO: Support step on the server side
        assert (
            train_data_format == TrainDataFormat.GRPO_CHAT
        ), f"GRPO only supports the GRPO_CHAT data format. Got {train_data_format} instead."
        for group in train_data:
            self._run_grpo_step_one_group(group, train_data_format)

    def save_checkpoint(self, checkpoint_name: str, score: float | None = None):
        api_base = self.lm.kwargs["api_base"]
        url = f"{api_base}fine_tuning/grpo/checkpoint"
        headers = {"Content-Type": "application/json"}
        body = {"checkpoint_name": checkpoint_name}
        response = requests.post(url, headers=headers, json=body)
        assert response.status_code == 200, f"Failed to save checkpoint: {response.text}"
        response = response.json()

        last_checkpoint = response["last_checkpoint"]
        checkpoints = response["checkpoints"]
        checkpoint_model_path = checkpoints[last_checkpoint]
        self.checkpoints[last_checkpoint] = {
            "model_path": checkpoint_model_path,
            "score": score,
        }
        self.last_checkpoint = last_checkpoint

    def terminate(self):
        api_base = self.lm.kwargs["api_base"]

        url = f"{api_base}fine_tuning/grpo/terminate"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers)
        assert response.status_code == 200, f"Failed to terminate GRPO: {response.text}"

        response = response.json()
        current_model = response["current_model"]
        self.lm.model = ArborProvider._add_provider_prefix(current_model)

    def cancel(self):
        if ArborProvider.does_job_exist(self.provider_job_id):
            status = self.status()
            if ArborProvider.is_terminal_training_status(status):
                err_msg = "Jobs that are complete cannot be canceled."
                err_msg += f" Job with ID {self.provider_job_id} is done."
                raise Exception(err_msg)
            openai.fine_tuning.jobs.cancel(self.provider_job_id)
            self.provider_job_id = None

    def status(self) -> TrainingStatus:
        status = ArborProvider.get_training_status(self.provider_job_id)
        return status


class ArborProvider(Provider):
    def __init__(self):
        super().__init__()
        self.finetunable = True
        self.reinforceable = True
        self.TrainingJob = ArborTrainingJob
        self.ReinforceJob = ArborReinforceJob

    @staticmethod
    def launch(lm: "LM", launch_kwargs: dict[str, Any] | None = None):
        model = ArborProvider._remove_provider_prefix(lm.model)

        api_base = lm.kwargs["api_base"]

        launch_kwargs = launch_kwargs or lm.launch_kwargs

        # Make request to launch endpoint
        response = requests.post(f"{api_base}chat/launch", json={"model": model, "launch_kwargs": launch_kwargs})

        if response.status_code != 200:
            raise Exception(f"Failed to launch model. Status code: {response.status_code}, Response: {response.text}")

        print(f"Inference server for model {model} launched successfully")

    @staticmethod
    def kill(lm: "LM", launch_kwargs: dict[str, Any] | None = None):
        api_base = lm.kwargs["api_base"]

        response = requests.post(
            f"{api_base}chat/kill",
        )

        if response.status_code != 200:
            raise Exception(f"Failed to kill model. Status code: {response.status_code}, Response: {response.text}")

        print("Inference killed successfully")

    @staticmethod
    def _remove_provider_prefix(model: str) -> str:
        if model.startswith("openai/"):
            model = model[7:]
        if model.startswith("arbor:"):
            model = model[6:]
        return model

    @staticmethod
    def _add_provider_prefix(model: str) -> str:
        if not model.startswith("openai/arbor:"):
            model = "openai/arbor:" + model
        return model

    @staticmethod
    def _get_arbor_base_api():
        # TODO: We will delete this method once we start passing the LM object
        # to finetune.
        import dspy.settings as settings

        if not hasattr(settings, "arbor_api_base"):
            raise ValueError(
                "Arbor API base not set. Please set the `dspy.settings.arbor_api_base` to the URL for the Arbor server (e.g. 'http://localhost:8000/v1/')."
            )
        return dspy.settings.arbor_api_base

    @staticmethod
    def finetune(
        job: ArborTrainingJob,
        model: str,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> str:
        # TODO: We want to re-factor finetune so that it takes in an LM.
        # Until then, we use the following to get the api information. The
        # following is a dummy call to ensure that dspy.settings.arbor_base_api
        # is set.
        ArborProvider._get_arbor_base_api()

        model = ArborProvider._remove_provider_prefix(model)

        print("[Arbor Provider] Validating the data format")
        ArborProvider.validate_data_format(train_data_format)

        print("[Arbor Provider] Saving the data to a file")
        data_path = save_data(train_data)
        print(f"[Arbor Provider] Data saved to {data_path}")

        print("[Arbor Provider] Uploading the data to the provider")
        provider_file_id = ArborProvider.upload_data(data_path)
        job.provider_file_id = provider_file_id

        print("[Arbor Provider] Starting remote training")
        provider_job_id = ArborProvider._start_remote_training(
            train_file_id=job.provider_file_id,
            model=model,
            train_kwargs=train_kwargs,
        )
        job.provider_job_id = provider_job_id
        print(f"[Arbor Provider] Job started with the Arbor Job ID {provider_job_id}")

        print("[Arbor Provider] Waiting for training to complete")
        ArborProvider.wait_for_job(job, train_kwargs)

        print("[Arbor Provider] Attempting to retrieve the trained model")
        model = ArborProvider.get_trained_model(job)
        print(f"[Arbor Provider] Model retrieved: {model}")

        return ArborProvider._add_provider_prefix(model)

    @staticmethod
    def does_job_exist(job_id: str, training_kwargs: dict[str, Any]) -> bool:
        try:
            original_base_url = openai.base_url
            openai.base_url = ArborProvider._get_arbor_base_api()
            openai.fine_tuning.jobs.retrieve(job_id)
            openai.base_url = original_base_url
            return True
        except Exception:
            return False

    @staticmethod
    def does_file_exist(file_id: str, training_kwargs: dict[str, Any]) -> bool:
        try:
            original_base_url = openai.base_url
            openai.base_url = ArborProvider._get_arbor_base_api()
            openai.files.retrieve(file_id)
            openai.base_url = original_base_url
            return True
        except Exception:
            return False

    @staticmethod
    def is_terminal_training_status(status: TrainingStatus) -> bool:
        return status in [
            TrainingStatus.succeeded,
            TrainingStatus.failed,
            TrainingStatus.cancelled,
        ]

    @staticmethod
    def get_training_status(job_id: str, training_kwargs: dict[str, Any]) -> TrainingStatus:
        provider_status_to_training_status = {
            "validating_files": TrainingStatus.pending,
            "queued": TrainingStatus.pending,
            "running": TrainingStatus.running,
            "succeeded": TrainingStatus.succeeded,
            "failed": TrainingStatus.failed,
            "cancelled": TrainingStatus.cancelled,
            "pending": TrainingStatus.pending,
            "pending_pause": TrainingStatus.pending,
            "pending_resume": TrainingStatus.pending,
            "paused": TrainingStatus.pending,
            "pending_cancel": TrainingStatus.pending,
        }

        if job_id is None:
            print("There is no active job.")
            return TrainingStatus.not_started

        err_msg = f"Job with ID {job_id} does not exist."
        assert ArborProvider.does_job_exist(job_id, training_kwargs), err_msg

        original_base_url = openai.base_url
        openai.base_url = ArborProvider._get_arbor_base_api()
        provider_job = openai.fine_tuning.jobs.retrieve(job_id)
        openai.base_url = original_base_url

        provider_status = provider_job.status
        status = provider_status_to_training_status[provider_status]

        return status

    @staticmethod
    def validate_data_format(data_format: TrainDataFormat):
        supported_data_formats = [
            TrainDataFormat.CHAT,
            TrainDataFormat.COMPLETION,
            TrainDataFormat.GRPO_CHAT,
        ]

        if data_format not in supported_data_formats:
            err_msg = f"Arbor does not support the data format {data_format}."
            raise ValueError(err_msg)

    @staticmethod
    def upload_data(data_path: str, training_kwargs: dict[str, Any]) -> str:
        original_base_url = openai.base_url
        openai.base_url = ArborProvider._get_arbor_base_api()
        provider_file = openai.files.create(
            file=open(data_path, "rb"),
            purpose="fine-tune",
        )
        openai.base_url = original_base_url

        return provider_file.id

    @staticmethod
    def _start_remote_training(train_file_id: str, model: str, train_kwargs: dict[str, Any]) -> str:
        train_kwargs = train_kwargs or {}
        original_base_url = openai.base_url
        openai.base_url = ArborProvider._get_arbor_base_api()
        provider_job = openai.fine_tuning.jobs.create(
            model=model,
            training_file=train_file_id,
            hyperparameters=train_kwargs,
        )
        openai.base_url = original_base_url
        return provider_job.id

    @staticmethod
    def wait_for_job(
        job: TrainingJob,
        training_kwargs: dict[str, Any],
        poll_frequency: int = 20,
    ):
        done = False
        cur_event_id = None
        reported_estimated_time = False
        while not done:
            # Report estimated time if not already reported
            if not reported_estimated_time:
                original_base_url = openai.base_url
                openai.base_url = ArborProvider._get_arbor_base_api()
                remote_job = openai.fine_tuning.jobs.retrieve(job.provider_job_id)
                openai.base_url = original_base_url

                timestamp = remote_job.estimated_finish
                if timestamp:
                    estimated_finish_dt = datetime.fromtimestamp(timestamp)
                    delta_dt = estimated_finish_dt - datetime.now()
                    print(f"[Arbor Provider] The Arbor estimated time remaining is: {delta_dt}")
                    reported_estimated_time = True

            # Get new events
            original_base_url = openai.base_url
            openai.base_url = ArborProvider._get_arbor_base_api()
            page = openai.fine_tuning.jobs.list_events(fine_tuning_job_id=job.provider_job_id, limit=1)
            openai.base_url = original_base_url

            new_event = page.data[0] if page.data else None
            if new_event and new_event.id != cur_event_id:
                dt = datetime.fromtimestamp(new_event.created_at)
                print(f"[Arbor Provider] {dt} {new_event.message}")
                cur_event_id = new_event.id

            # Sleep and update the flag
            time.sleep(poll_frequency)
            done = ArborProvider.is_terminal_training_status(job.status())

    @staticmethod
    def get_trained_model(job, training_kwargs: dict[str, Any]):
        status = job.status()
        if status != TrainingStatus.succeeded:
            err_msg = f"Job status is {status}."
            err_msg += f" Must be {TrainingStatus.succeeded} to retrieve model."
            raise Exception(err_msg)

        original_base_url = openai.base_url
        openai.base_url = ArborProvider._get_arbor_base_api()
        provider_job = openai.fine_tuning.jobs.retrieve(job.provider_job_id)
        openai.base_url = original_base_url

        finetuned_model = provider_job.fine_tuned_model
        return finetuned_model
```

## File: `clients/lm_local.py`
```
import datetime
import logging
import random
import socket
import string
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any

import requests

from dspy.clients.provider import Provider, TrainingJob
from dspy.clients.utils_finetune import TrainDataFormat, save_data

if TYPE_CHECKING:
    from dspy.clients.lm import LM

logger = logging.getLogger(__name__)


class LocalProvider(Provider):
    def __init__(self):
        super().__init__()
        self.finetunable = True
        self.TrainingJob = TrainingJob

    @staticmethod
    def launch(lm: "LM", launch_kwargs: dict[str, Any] | None = None):
        try:
            import sglang  # noqa: F401
        except ImportError:
            raise ImportError(
                "For local model launching, please install sglang."
                "Navigate to https://docs.sglang.ai/start/install.html for the latest installation instructions!"
            )

        if hasattr(lm, "process"):
            logger.info("Server is already launched.")
            return

        launch_kwargs = launch_kwargs or lm.launch_kwargs

        import os

        model = lm.model
        if model.startswith("openai/"):
            model = model[7:]
        if model.startswith("local:"):
            model = model[6:]
        if model.startswith("huggingface/"):
            model = model[len("huggingface/") :]

        logger.info(f"Grabbing a free port to launch an SGLang server for model {model}")
        logger.info(f"We see that CUDA_VISIBLE_DEVICES is {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
        port = get_free_port()
        timeout = launch_kwargs.get("timeout", 1800)
        command = f"python -m sglang.launch_server --model-path {model} --port {port} --host 0.0.0.0"

        # We will manually stream & capture logs.
        process = subprocess.Popen(
            command.replace("\\\n", " ").replace("\\", " ").split(),
            text=True,
            stdout=subprocess.PIPE,  # We'll read from pipe
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
        )

        # A threading.Event to control printing after the server is ready.
        # This will store *all* lines (both before and after readiness).
        logger.info(f"SGLang server process started with PID {process.pid}.")
        stop_printing_event = threading.Event()
        logs_buffer = []

        def _tail_process(proc, buffer, stop_event):
            while True:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    # Process ended and no new line
                    break
                if line:
                    buffer.append(line)
                    # Print only if stop_event is not set
                    if not stop_event.is_set():
                        print(line, end="")

        # Start a background thread to read from the process continuously
        thread = threading.Thread(
            target=_tail_process,
            args=(process, logs_buffer, stop_printing_event),
            daemon=True,
        )
        thread.start()

        # Wait until the server is ready (or times out)
        base_url = f"http://localhost:{port}"
        try:
            wait_for_server(base_url, timeout=timeout)
        except TimeoutError:
            # If the server doesn't come up, we might want to kill it:
            process.kill()
            raise

        # Once server is ready, we tell the thread to stop printing further lines.
        stop_printing_event.set()

        # A convenience getter so the caller can see all logs so far (and future).
        def get_logs() -> str:
            # Join them all into a single string, or you might return a list
            return "".join(logs_buffer)

        # Let the user know server is up
        logger.info(f"Server ready on random port {port}! Logs are available via lm.get_logs() method on returned lm.")

        lm.kwargs["api_base"] = f"http://localhost:{port}/v1"
        lm.kwargs["api_key"] = "local"
        lm.get_logs = get_logs
        lm.process = process
        lm.thread = thread

    @staticmethod
    def kill(lm: "LM", launch_kwargs: dict[str, Any] | None = None):
        from sglang.utils import terminate_process

        if not hasattr(lm, "process"):
            logger.info("No running server to kill.")
            return
        # Ideally, the following happens atomically
        terminate_process(lm.process)
        lm.thread.join()
        del lm.process
        del lm.thread
        del lm.get_logs
        logger.info("Server killed.")

    @staticmethod
    def finetune(
        job: TrainingJob,
        model: str,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> str:
        if model.startswith("openai/"):
            model = model[7:]
        if model.startswith("local:"):
            model = model[6:]

        if train_data_format != TrainDataFormat.CHAT:
            raise ValueError("Only chat models are supported for local finetuning.")

        data_path = save_data(train_data)
        logger.info(f"Train data saved to {data_path}")
        output_dir = create_output_dir(model, data_path)

        default_train_kwargs = {
            "device": None,
            "use_peft": False,
            "num_train_epochs": 5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-5,
            "max_seq_length": None,
            "packing": True,
            "bf16": True,
            "output_dir": output_dir,
        }
        train_kwargs = {**default_train_kwargs, **(train_kwargs or {})}
        output_dir = train_kwargs["output_dir"]  # user might have changed the output_dir

        logger.info(f"Starting local training, will save to {output_dir}")
        train_sft_locally(
            model_name=model,
            train_data=train_data,
            train_kwargs=train_kwargs,
        )

        logger.info("Training complete")
        return f"openai/local:{output_dir}"


def create_output_dir(model_name, data_path):
    model_str = model_name.replace("/", "-")
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rnd_str = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    model_identifier = f"{rnd_str}_{model_str}_{time_str}"
    output_dir = data_path.replace(".jsonl", "_" + model_identifier)
    return output_dir


def train_sft_locally(model_name, train_data, train_kwargs):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer, setup_chat_format
    except ImportError:
        raise ImportError(
            "For local finetuning, please install torch, transformers, and trl "
            "by running `pip install -U torch transformers accelerate trl peft`"
        )

    device = train_kwargs.get("device", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

    # Set up the chat format; generally only for non-chat model variants, hence the try-except.
    try:
        model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)
    except Exception:
        pass

    if tokenizer.pad_token_id is None:
        logger.info("Adding pad token to tokenizer")
        tokenizer.add_special_tokens({"pad_token": "[!#PAD#!]"})

    logger.info("Creating dataset")
    if "max_seq_length" not in train_kwargs:
        train_kwargs["max_seq_length"] = 4096
        logger.info(
            f"The 'train_kwargs' parameter didn't include a 'max_seq_length', defaulting to {train_kwargs['max_seq_length']}"
        )

    from datasets import Dataset

    hf_dataset = Dataset.from_list(train_data)

    def tokenize_function(example):
        return encode_sft_example(example, tokenizer, train_kwargs["max_seq_length"])  # noqa: F821

    tokenized_dataset = hf_dataset.map(tokenize_function, batched=False)
    tokenized_dataset.set_format(type="torch")
    tokenized_dataset = tokenized_dataset.filter(lambda example: (example["labels"] != -100).any())

    use_peft = train_kwargs.get("use_peft", False)
    peft_config = None

    if use_peft:
        from peft import LoraConfig

        rank_dimension = 32
        lora_alpha = 64
        lora_dropout = 0.05

        peft_config = LoraConfig(
            r=rank_dimension,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    sft_config = SFTConfig(
        output_dir=train_kwargs["output_dir"],
        num_train_epochs=train_kwargs["num_train_epochs"],
        per_device_train_batch_size=train_kwargs["per_device_train_batch_size"],
        gradient_accumulation_steps=train_kwargs["gradient_accumulation_steps"],
        learning_rate=train_kwargs["learning_rate"],
        max_grad_norm=2.0,  # note that the current SFTConfig default is 1.0
        logging_steps=20,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        save_steps=10_000,
        bf16=train_kwargs["bf16"],
        max_seq_length=train_kwargs["max_seq_length"],
        packing=train_kwargs["packing"],
        dataset_kwargs={  # We need to pass dataset_kwargs because we are processing the dataset ourselves
            "add_special_tokens": False,  # Special tokens handled by template
            "append_concat_token": False,  # No additional separator needed
        },
    )

    logger.info("Starting training")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=tokenized_dataset,
        peft_config=peft_config,
    )

    # Train!
    trainer.train()

    # Save the model!
    trainer.save_model()

    merge = True
    if use_peft and merge:
        from peft import AutoPeftModelForCausalLM

        # Load PEFT model on CPU
        model_ = AutoPeftModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=sft_config.output_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        merged_model = model_.merge_and_unload()
        merged_model.save_pretrained(sft_config.output_dir, safe_serialization=True, max_shard_size="5GB")

    # Clean up!
    import gc

    del model
    del tokenizer
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return sft_config.output_dir


def get_free_port() -> int:
    """
    Return a free TCP port on localhost.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def wait_for_server(base_url: str, timeout: int | None = None) -> None:
    """
    Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server (e.g. http://localhost:1234)
        timeout: Maximum time to wait in seconds. None means wait forever.
    """
    start_time = time.time()
    while True:
        try:
            response = requests.get(
                f"{base_url}/v1/models",
                headers={"Authorization": "Bearer None"},
            )
            if response.status_code == 200:
                # A small extra sleep to ensure server is fully up.
                time.sleep(5)
                break

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Server did not become ready within timeout period")
        except requests.exceptions.RequestException:
            # Server not up yet, wait and retry
            time.sleep(1)


def encode_sft_example(example, tokenizer, max_seq_length):
    """
    This function encodes a single example into a format that can be used for sft training.
    Here, we assume each example has a 'messages' field. Each message in it is a dict with 'role' and 'content' fields.
    We use the `apply_chat_template` function from the tokenizer to tokenize the messages and prepare the input and label tensors.

    Code obtained from the allenai/open-instruct repository: https://github.com/allenai/open-instruct/blob/4365dea3d1a6111e8b2712af06b22a4512a0df88/open_instruct/finetune.py
    """
    import torch

    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_length,
        add_generation_prompt=False,
    )
    labels = input_ids.clone()
    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            # we calculate the start index of this non-assistant message
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer.apply_chat_template(
                    conversation=messages[:message_idx],  # here marks the end of the previous messages
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # next, we calculate the end index of this non-assistant message
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # for intermediate messages that follow with an assistant message, we need to
                # set `add_generation_prompt=True` to avoid the assistant generation prefix being included in the loss
                # (e.g., `<|assistant|>`)
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=True,
                ).shape[1]
            else:
                # for the last message or the message that doesn't follow with an assistant message,
                # we don't need to add the assistant generation prefix
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # set the label to -100 for the non-assistant part
            labels[:, message_start_idx:message_end_idx] = -100
            if max_seq_length and message_end_idx >= max_seq_length:
                break
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }
```

## File: `clients/lm.py`
```
import logging
import os
import re
import threading
from typing import Any, Literal, cast

import litellm
from anyio.streams.memory import MemoryObjectSendStream
from asyncer import syncify

import dspy
from dspy.clients.cache import request_cache
from dspy.clients.openai import OpenAIProvider
from dspy.clients.provider import Provider, ReinforceJob, TrainingJob
from dspy.clients.utils_finetune import TrainDataFormat
from dspy.dsp.utils.settings import settings
from dspy.utils.callback import BaseCallback

from .base_lm import BaseLM

logger = logging.getLogger(__name__)


class LM(BaseLM):
    """
    A language model supporting chat or text completion requests for use with DSPy modules.
    """

    def __init__(
        self,
        model: str,
        model_type: Literal["chat", "text"] = "chat",
        temperature: float = 0.0,
        max_tokens: int = 4000,
        cache: bool = True,
        cache_in_memory: bool = True,
        callbacks: list[BaseCallback] | None = None,
        num_retries: int = 3,
        provider: Provider | None = None,
        finetuning_model: str | None = None,
        launch_kwargs: dict[str, Any] | None = None,
        train_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Create a new language model instance for use with DSPy modules and programs.

        Args:
            model: The model to use. This should be a string of the form ``"llm_provider/llm_name"``
                   supported by LiteLLM. For example, ``"openai/gpt-4o"``.
            model_type: The type of the model, either ``"chat"`` or ``"text"``.
            temperature: The sampling temperature to use when generating responses.
            max_tokens: The maximum number of tokens to generate per response.
            cache: Whether to cache the model responses for reuse to improve performance
                   and reduce costs.
            cache_in_memory (deprecated): To enable additional caching with LRU in memory.
            callbacks: A list of callback functions to run before and after each request.
            num_retries: The number of times to retry a request if it fails transiently due to
                         network error, rate limiting, etc. Requests are retried with exponential
                         backoff.
            provider: The provider to use. If not specified, the provider will be inferred from the model.
            finetuning_model: The model to finetune. In some providers, the models available for finetuning is different
                from the models available for inference.
        """
        # Remember to update LM.copy() if you modify the constructor!
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.cache_in_memory = cache_in_memory
        self.provider = provider or self.infer_provider()
        self.callbacks = callbacks or []
        self.history = []
        self.num_retries = num_retries
        self.finetuning_model = finetuning_model
        self.launch_kwargs = launch_kwargs or {}
        self.train_kwargs = train_kwargs or {}

        # Handle model-specific configuration for different model families
        model_family = model.split("/")[-1].lower() if "/" in model else model.lower()

        # Match pattern: o[1,3,4] at the start, optionally followed by -mini and anything else
        model_pattern = re.match(r"^o([134])(?:-mini)?", model_family)

        if model_pattern:
            # Handle OpenAI reasoning models (o1, o3)
            assert (
                max_tokens >= 20_000 and temperature == 1.0
            ), "OpenAI's reasoning models require passing temperature=1.0 and max_tokens >= 20_000 to `dspy.LM(...)`"
            self.kwargs = dict(temperature=temperature, max_completion_tokens=max_tokens, **kwargs)
        else:
            self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)

    def _get_cached_completion_fn(self, completion_fn, cache, enable_memory_cache):
        ignored_args_for_cache_key = ["api_key", "api_base", "base_url"]
        if cache and enable_memory_cache:
            completion_fn = request_cache(
                cache_arg_name="request",
                ignored_args_for_cache_key=ignored_args_for_cache_key,
            )(completion_fn)
        elif cache:
            completion_fn = request_cache(
                cache_arg_name="request",
                ignored_args_for_cache_key=ignored_args_for_cache_key,
                enable_memory_cache=False,
            )(completion_fn)
        else:
            completion_fn = completion_fn

        if not cache or litellm.cache is None:
            litellm_cache_args = {"no-cache": True, "no-store": True}
        else:
            litellm_cache_args = {"no-cache": False, "no-store": False}

        return completion_fn, litellm_cache_args

    def forward(self, prompt=None, messages=None, **kwargs):
        # Build the request.
        cache = kwargs.pop("cache", self.cache)
        enable_memory_cache = kwargs.pop("cache_in_memory", self.cache_in_memory)

        messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {**self.kwargs, **kwargs}

        completion = litellm_completion if self.model_type == "chat" else litellm_text_completion
        completion, litellm_cache_args = self._get_cached_completion_fn(completion, cache, enable_memory_cache)

        results = completion(
            request=dict(model=self.model, messages=messages, **kwargs),
            num_retries=self.num_retries,
            cache=litellm_cache_args,
        )

        if any(c.finish_reason == "length" for c in results["choices"]):
            logger.warning(
                f"LM response was truncated due to exceeding max_tokens={self.kwargs['max_tokens']}. "
                "You can inspect the latest LM interactions with `dspy.inspect_history()`. "
                "To avoid truncation, consider passing a larger max_tokens when setting up dspy.LM. "
                f"You may also consider increasing the temperature (currently {self.kwargs['temperature']}) "
                " if the reason for truncation is repetition."
            )

        if not getattr(results, "cache_hit", False) and dspy.settings.usage_tracker and hasattr(results, "usage"):
            settings.usage_tracker.add_usage(self.model, dict(results.usage))
        return results

    async def aforward(self, prompt=None, messages=None, **kwargs):
        # Build the request.
        cache = kwargs.pop("cache", self.cache)
        enable_memory_cache = kwargs.pop("cache_in_memory", self.cache_in_memory)

        messages = messages or [{"role": "user", "content": prompt}]
        kwargs = {**self.kwargs, **kwargs}

        completion = alitellm_completion if self.model_type == "chat" else alitellm_text_completion
        completion, litellm_cache_args = self._get_cached_completion_fn(completion, cache, enable_memory_cache)

        results = await completion(
            request=dict(model=self.model, messages=messages, **kwargs),
            num_retries=self.num_retries,
            cache=litellm_cache_args,
        )

        if any(c.finish_reason == "length" for c in results["choices"]):
            logger.warning(
                f"LM response was truncated due to exceeding max_tokens={self.kwargs['max_tokens']}. "
                "You can inspect the latest LM interactions with `dspy.inspect_history()`. "
                "To avoid truncation, consider passing a larger max_tokens when setting up dspy.LM. "
                f"You may also consider increasing the temperature (currently {self.kwargs['temperature']}) "
                " if the reason for truncation is repetition."
            )

        if not getattr(results, "cache_hit", False) and dspy.settings.usage_tracker and hasattr(results, "usage"):
            settings.usage_tracker.add_usage(self.model, dict(results.usage))
        return results

    def launch(self, launch_kwargs: dict[str, Any] | None = None):
        self.provider.launch(self, launch_kwargs)

    def kill(self, launch_kwargs: dict[str, Any] | None = None):
        self.provider.kill(self, launch_kwargs)

    def finetune(
        self,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> TrainingJob:
        from dspy import settings as settings

        err = f"Provider {self.provider} does not support fine-tuning."
        assert self.provider.finetunable, err

        def thread_function_wrapper():
            return self._run_finetune_job(job)

        thread = threading.Thread(target=thread_function_wrapper)
        train_kwargs = train_kwargs or self.train_kwargs
        model_to_finetune = self.finetuning_model or self.model
        job = self.provider.TrainingJob(
            thread=thread,
            model=model_to_finetune,
            train_data=train_data,
            train_data_format=train_data_format,
            train_kwargs=train_kwargs,
        )
        thread.start()

        return job

    def reinforce(self, train_kwargs) -> ReinforceJob:
        # TODO(GRPO Team): Should we return an initialized job here?
        from dspy import settings as settings

        err = f"Provider {self.provider} does not implement the reinforcement learning interface."
        assert self.provider.reinforceable, err

        job = self.provider.ReinforceJob(lm=self, train_kwargs=train_kwargs)
        job.initialize()
        return job

    def _run_finetune_job(self, job: TrainingJob):
        # TODO(enhance): We should listen for keyboard interrupts somewhere.
        # Requires TrainingJob.cancel() to be implemented for each provider.
        try:
            model = self.provider.finetune(
                job=job,
                model=job.model,
                train_data=job.train_data,
                train_data_format=job.train_data_format,
                train_kwargs=job.train_kwargs,
            )
            lm = self.copy(model=model)
            job.set_result(lm)
        except Exception as err:
            logger.error(err)
            job.set_result(err)

    def infer_provider(self) -> Provider:
        if OpenAIProvider.is_provider_model(self.model):
            return OpenAIProvider()
        return Provider()

    def dump_state(self):
        state_keys = [
            "model",
            "model_type",
            "cache",
            "cache_in_memory",
            "num_retries",
            "finetuning_model",
            "launch_kwargs",
            "train_kwargs",
        ]
        return {key: getattr(self, key) for key in state_keys} | self.kwargs


def _get_stream_completion_fn(
    request: dict[str, Any],
    cache_kwargs: dict[str, Any],
    sync=True,
):
    stream = dspy.settings.send_stream
    caller_predict = dspy.settings.caller_predict

    if stream is None:
        return None

    # The stream is already opened, and will be closed by the caller.
    stream = cast(MemoryObjectSendStream, stream)
    caller_predict_id = id(caller_predict) if caller_predict else None

    if dspy.settings.track_usage:
        request["stream_options"] = {"include_usage": True}

    async def stream_completion(request: dict[str, Any], cache_kwargs: dict[str, Any]):
        response = await litellm.acompletion(
            cache=cache_kwargs,
            stream=True,
            **request,
        )
        chunks = []
        async for chunk in response:
            if caller_predict_id:
                # Add the predict id to the chunk so that the stream listener can identify which predict produces it.
                chunk.predict_id = caller_predict_id
            chunks.append(chunk)
            await stream.send(chunk)
        return litellm.stream_chunk_builder(chunks)

    def sync_stream_completion():
        syncified_stream_completion = syncify(stream_completion)
        return syncified_stream_completion(request, cache_kwargs)

    async def async_stream_completion():
        return await stream_completion(request, cache_kwargs)

    if sync:
        return sync_stream_completion
    else:
        return async_stream_completion


def litellm_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    stream_completion = _get_stream_completion_fn(request, cache, sync=True)
    if stream_completion is None:
        return litellm.completion(
            cache=cache,
            num_retries=num_retries,
            retry_strategy="exponential_backoff_retry",
            **request,
        )

    return stream_completion()


def litellm_text_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    # Extract the provider and model from the model string.
    # TODO: Not all the models are in the format of "provider/model"
    model = request.pop("model").split("/", 1)
    provider, model = model[0] if len(model) > 1 else "openai", model[-1]

    # Use the API key and base from the request, or from the environment.
    api_key = request.pop("api_key", None) or os.getenv(f"{provider}_API_KEY")
    api_base = request.pop("api_base", None) or os.getenv(f"{provider}_API_BASE")

    # Build the prompt from the messages.
    prompt = "\n\n".join([x["content"] for x in request.pop("messages")] + ["BEGIN RESPONSE:"])

    return litellm.text_completion(
        cache=cache,
        model=f"text-completion-openai/{model}",
        api_key=api_key,
        api_base=api_base,
        prompt=prompt,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        **request,
    )


async def alitellm_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    stream_completion = _get_stream_completion_fn(request, cache, sync=False)
    if stream_completion is None:
        return await litellm.acompletion(
            cache=cache,
            num_retries=num_retries,
            retry_strategy="exponential_backoff_retry",
            **request,
        )

    return await stream_completion()


async def alitellm_text_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    model = request.pop("model").split("/", 1)
    provider, model = model[0] if len(model) > 1 else "openai", model[-1]

    # Use the API key and base from the request, or from the environment.
    api_key = request.pop("api_key", None) or os.getenv(f"{provider}_API_KEY")
    api_base = request.pop("api_base", None) or os.getenv(f"{provider}_API_BASE")

    # Build the prompt from the messages.
    prompt = "\n\n".join([x["content"] for x in request.pop("messages")] + ["BEGIN RESPONSE:"])

    return await litellm.atext_completion(
        cache=cache,
        model=f"text-completion-openai/{model}",
        api_key=api_key,
        api_base=api_base,
        prompt=prompt,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        **request,
    )
```

## File: `clients/openai.py`
```
import re
import time
from datetime import datetime
from typing import Any

import openai

from dspy.clients.provider import Provider, TrainingJob
from dspy.clients.utils_finetune import TrainDataFormat, TrainingStatus, save_data

_OPENAI_MODELS = [
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "tts-1",
    "tts-1-1106",
    "chatgpt-4o-latest",
    "dall-e-2",
    "whisper-1",
    "gpt-3.5-turbo-instruct",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "babbage-002",
    "davinci-002",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "dall-e-3",
    "gpt-4o-mini",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
    "o1-preview",
    "gpt-4o-audio-preview-2024-10-01",
    "o1-mini-2024-09-12",
    "gpt-4o-audio-preview",
    "tts-1-hd",
    "tts-1-hd-1106",
    "o1-preview-2024-09-12",
    "o1-mini",
    "gpt-4-1106-preview",
    "text-embedding-ada-002",
    "gpt-3.5-turbo-16k",
    "text-embedding-3-small",
    "text-embedding-3-large",
    "gpt-4o-realtime-preview-2024-10-01",
    "gpt-4o-realtime-preview",
    "gpt-3.5-turbo-1106",
    "gpt-4-0613",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4",
    "gpt-3.5-turbo-instruct-0914",
]


class TrainingJobOpenAI(TrainingJob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.provider_file_id = None
        self.provider_job_id = None

    def cancel(self):
        # Cancel the provider job
        if OpenAIProvider.does_job_exist(self.provider_job_id):
            status = self.status()
            if OpenAIProvider.is_terminal_training_status(status):
                err_msg = "Jobs that are complete cannot be canceled."
                err_msg += f" Job with ID {self.provider_job_id} is done."
                raise Exception(err_msg)
            openai.fine_tuning.jobs.cancel(self.provider_job_id)
            self.provider_job_id = None

        # Delete the provider file
        if self.provider_file_id is not None:
            if OpenAIProvider.does_file_exist(self.provider_file_id):
                openai.files.delete(self.provider_file_id)
            self.provider_file_id = None

        # Call the super's cancel method after the custom cancellation logic
        super().cancel()

    def status(self) -> TrainingStatus:
        status = OpenAIProvider.get_training_status(self.provider_job_id)
        return status


class OpenAIProvider(Provider):
    def __init__(self):
        super().__init__()
        self.finetunable = True
        self.TrainingJob = TrainingJobOpenAI

    @staticmethod
    def is_provider_model(model: str) -> bool:
        model = OpenAIProvider._remove_provider_prefix(model)

        # Check if the model is a base OpenAI model
        # TODO(enhance) The following list can be replaced with
        # openai.models.list(), but doing so might require a key. Is there a
        # way to get the list of models without a key?
        if model in _OPENAI_MODELS:
            return True

        # Check if the model is a fine-tuned OpneAI model. Fine-tuned OpenAI
        # models have the prefix "ft:<BASE_MODEL_NAME>:", followed by a string
        # specifying the fine-tuned model. The following RegEx pattern is used
        # to match the base model name.
        # TODO(enhance): This part can be updated to match the actual fine-tuned
        # model names by making a call to the OpenAI API to be more exact, but
        # this might require an API key with the right permissions.
        match = re.match(r"ft:([^:]+):", model)
        if match and match.group(1) in _OPENAI_MODELS:
            return True

        return False

    @staticmethod
    def _remove_provider_prefix(model: str) -> str:
        provider_prefix = "openai/"
        return model.replace(provider_prefix, "")

    @staticmethod
    def finetune(
        job: TrainingJobOpenAI,
        model: str,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> str:
        model = OpenAIProvider._remove_provider_prefix(model)

        print("[OpenAI Provider] Validating the data format")
        OpenAIProvider.validate_data_format(train_data_format)

        print("[OpenAI Provider] Saving the data to a file")
        data_path = save_data(train_data)
        print(f"[OpenAI Provider] Data saved to {data_path}")

        print("[OpenAI Provider] Uploading the data to the provider")
        provider_file_id = OpenAIProvider.upload_data(data_path)
        job.provider_file_id = provider_file_id

        print("[OpenAI Provider] Starting remote training")
        provider_job_id = OpenAIProvider._start_remote_training(
            train_file_id=job.provider_file_id,
            model=model,
            train_kwargs=train_kwargs,
        )
        job.provider_job_id = provider_job_id
        print(f"[OpenAI Provider] Job started with the OpenAI Job ID {provider_job_id}")

        print("[OpenAI Provider] Waiting for training to complete")
        # TODO(feature): Could we stream OAI logs?
        OpenAIProvider.wait_for_job(job)

        print("[OpenAI Provider] Attempting to retrieve the trained model")
        model = OpenAIProvider.get_trained_model(job)
        print(f"[OpenAI Provider] Model retrieved: {model}")

        return model

    @staticmethod
    def does_job_exist(job_id: str) -> bool:
        try:
            # TODO(nit): This call may fail for other reasons. We should check
            # the error message to ensure that the job does not exist.
            openai.fine_tuning.jobs.retrieve(job_id)
            return True
        except Exception:
            return False

    @staticmethod
    def does_file_exist(file_id: str) -> bool:
        try:
            # TODO(nit): This call may fail for other reasons. We should check
            # the error message to ensure that the file does not exist.
            openai.files.retrieve(file_id)
            return True
        except Exception:
            return False

    @staticmethod
    def is_terminal_training_status(status: TrainingStatus) -> bool:
        return status in [
            TrainingStatus.succeeded,
            TrainingStatus.failed,
            TrainingStatus.cancelled,
        ]

    @staticmethod
    def get_training_status(job_id: str) -> TrainingStatus:
        provider_status_to_training_status = {
            "validating_files": TrainingStatus.pending,
            "queued": TrainingStatus.pending,
            "running": TrainingStatus.running,
            "succeeded": TrainingStatus.succeeded,
            "failed": TrainingStatus.failed,
            "cancelled": TrainingStatus.cancelled,
        }

        # Check if there is an active job
        if job_id is None:
            print("There is no active job.")
            return TrainingStatus.not_started

        err_msg = f"Job with ID {job_id} does not exist."
        assert OpenAIProvider.does_job_exist(job_id), err_msg

        # Retrieve the provider's job and report the status
        provider_job = openai.fine_tuning.jobs.retrieve(job_id)
        provider_status = provider_job.status
        status = provider_status_to_training_status[provider_status]

        return status

    @staticmethod
    def validate_data_format(data_format: TrainDataFormat):
        supported_data_formats = [
            TrainDataFormat.CHAT,
            TrainDataFormat.COMPLETION,
        ]
        if data_format not in supported_data_formats:
            err_msg = f"OpenAI does not support the data format {data_format}."
            raise ValueError(err_msg)

    @staticmethod
    def upload_data(data_path: str) -> str:
        # Upload the data to the provider
        provider_file = openai.files.create(
            file=open(data_path, "rb"),
            purpose="fine-tune",
        )
        return provider_file.id

    @staticmethod
    def _start_remote_training(train_file_id: str, model: str, train_kwargs: dict[str, Any] | None = None) -> str:
        train_kwargs = train_kwargs or {}
        provider_job = openai.fine_tuning.jobs.create(
            model=model,
            training_file=train_file_id,
            hyperparameters=train_kwargs,
        )
        return provider_job.id

    @staticmethod
    def wait_for_job(
        job: TrainingJobOpenAI,
        poll_frequency: int = 20,
    ):
        # Poll for the job until it is done
        done = False
        cur_event_id = None
        reported_estimated_time = False
        while not done:
            # Report estimated time if not already reported
            if not reported_estimated_time:
                remote_job = openai.fine_tuning.jobs.retrieve(job.provider_job_id)
                timestamp = remote_job.estimated_finish
                if timestamp:
                    estimated_finish_dt = datetime.fromtimestamp(timestamp)
                    delta_dt = estimated_finish_dt - datetime.now()
                    print(f"[OpenAI Provider] The OpenAI estimated time remaining is: {delta_dt}")
                    reported_estimated_time = True

            # Get new events
            page = openai.fine_tuning.jobs.list_events(fine_tuning_job_id=job.provider_job_id, limit=1)
            new_event = page.data[0] if page.data else None
            if new_event and new_event.id != cur_event_id:
                dt = datetime.fromtimestamp(new_event.created_at)
                print(f"[OpenAI Provider] {dt} {new_event.message}")
                cur_event_id = new_event.id

            # Sleep and update the flag
            time.sleep(poll_frequency)
            done = OpenAIProvider.is_terminal_training_status(job.status())

    @staticmethod
    def get_trained_model(job):
        status = job.status()
        if status != TrainingStatus.succeeded:
            err_msg = f"Job status is {status}."
            err_msg += f" Must be {TrainingStatus.succeeded} to retrieve model."
            raise Exception(err_msg)

        provider_job = openai.fine_tuning.jobs.retrieve(job.provider_job_id)
        finetuned_model = provider_job.fine_tuned_model
        return finetuned_model
```

## File: `clients/provider.py`
```
from abc import abstractmethod
from concurrent.futures import Future
from threading import Thread
from typing import TYPE_CHECKING, Any

from dspy.clients.utils_finetune import TrainDataFormat

if TYPE_CHECKING:
    from dspy.clients.lm import LM


class TrainingJob(Future):
    def __init__(
        self,
        thread: Thread | None = None,
        model: str | None = None,
        train_data: list[dict[str, Any]] | None = None,
        train_data_format: TrainDataFormat | None = None,
        train_kwargs: dict[str, Any] | None = None,
    ):
        self.thread = thread
        self.model = model
        self.train_data = train_data
        self.train_data_format = train_data_format
        self.train_kwargs = train_kwargs or {}
        super().__init__()

    # Subclasses should override the cancel method to cancel the job; then call
    # the super's cancel method so that the future can be cancelled.
    def cancel(self):
        super().cancel()

    @abstractmethod
    def status(self):
        raise NotImplementedError


class ReinforceJob:
    def __init__(self, lm: "LM", train_kwargs: dict[str, Any] | None = None):
        self.lm = lm
        self.train_kwargs = train_kwargs or {}
        self.checkpoints = {}
        self.last_checkpoint = None

    @abstractmethod
    def initialize(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, train_data: list[dict[str, Any]], train_data_format: TrainDataFormat | str | None = None):
        raise NotImplementedError

    @abstractmethod
    def terminate(self):
        raise NotImplementedError

    @abstractmethod
    def update_model(self):
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, checkpoint_name: str):
        raise NotImplementedError

    def cancel(self):
        raise NotImplementedError

    def status(self):
        raise NotImplementedError


class Provider:
    def __init__(self):
        self.finetunable = False
        self.reinforceable = False
        self.TrainingJob = TrainingJob
        self.ReinforceJob = ReinforceJob

    @staticmethod
    def is_provider_model(model: str) -> bool:
        # Subclasses should actually check whether a model is supported if they
        # want to have the model provider auto-discovered.
        return False

    @staticmethod
    def launch(lm: "LM", launch_kwargs: dict[str, Any] | None = None):
        # Note that "launch" and "kill" methods might be called even if there
        # is a launched LM or no launched LM to kill. These methods should be
        # resillient to such cases.
        pass

    @staticmethod
    def kill(lm: "LM", launch_kwargs: dict[str, Any] | None = None):
        # We assume that LM.launch_kwargs dictionary will contain the necessary
        # information for a provider to launch and/or kill an LM. This is the
        # reeason why the argument here is named launch_kwargs and not
        # kill_kwargs.
        pass

    @staticmethod
    def finetune(
        job: TrainingJob,
        model: str,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | str | None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError
```

## File: `clients/utils_finetune.py`
```
import os
from enum import Enum
from typing import Any, Literal, TypedDict

import ujson

import dspy
from dspy.adapters.base import Adapter
from dspy.utils.caching import DSPY_CACHEDIR


class TrainingStatus(str, Enum):
    not_started = "not_started"
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class TrainDataFormat(str, Enum):
    CHAT = "chat"
    COMPLETION = "completion"
    GRPO_CHAT = "grpo_chat"


class Message(TypedDict):
    role: Literal["user"] | Literal["assistant"] | Literal["system"]
    content: str


class MessageAssistant(TypedDict):
    role: Literal["assistant"]
    content: str


class GRPOChatData(TypedDict):
    messages: list[Message]
    completion: MessageAssistant
    reward: float


GRPOGroup = list[GRPOChatData]


def infer_data_format(adapter: Adapter) -> str:
    if isinstance(adapter, dspy.ChatAdapter):
        return TrainDataFormat.CHAT
    raise ValueError(f"Could not infer the data format for: {adapter}")


def get_finetune_directory() -> str:
    default_finetunedir = os.path.join(DSPY_CACHEDIR, "finetune")
    finetune_dir = os.environ.get("DSPY_FINETUNEDIR") or default_finetunedir
    finetune_dir = os.path.abspath(finetune_dir)
    os.makedirs(finetune_dir, exist_ok=True)
    return finetune_dir


def write_lines(file_path, data):
    with open(file_path, "w") as f:
        for item in data:
            f.write(ujson.dumps(item) + "\n")


def save_data(
    data: list[dict[str, Any]],
) -> str:
    from datasets.fingerprint import Hasher

    # Assign a unique name to the file based on the data hash
    hash = Hasher.hash(data)
    file_name = f"{hash}.jsonl"

    finetune_dir = get_finetune_directory()
    file_path = os.path.join(finetune_dir, file_name)
    file_path = os.path.abspath(file_path)
    with open(file_path, "w") as f:
        for item in data:
            f.write(ujson.dumps(item) + "\n")
    return file_path


def validate_data_format(
    data: list[dict[str, Any]],
    data_format: TrainDataFormat,
):
    find_err_funcs = {
        TrainDataFormat.CHAT: find_data_error_chat,
        TrainDataFormat.COMPLETION: find_data_errors_completion,
    }
    err = f"Data format {data_format} is not supported."
    assert data_format in find_err_funcs, err
    find_err_func = find_err_funcs[data_format]

    if not isinstance(data, list):
        err = f"Data is not a list. Found data type: {type(data)}"
        raise ValueError(err)

    data_dict_errors = []
    for ind, data_dict in enumerate(data):
        err = f"Not a dictionary -- found data type: {type(data_dict)}"
        if isinstance(data_dict, dict):
            err = find_err_func(data_dict)
        if err:
            err_dict = {"index": ind, "error": err}
            data_dict_errors.append(err_dict)

    if data_dict_errors:
        finetune_dir = get_finetune_directory()
        log_path = os.path.join(finetune_dir, "data_format_errors.log")
        log_path = os.path.abspath(log_path)
        write_lines(log_path, data_dict_errors)

        err = f"Data format errors found.  For more details, see the log file: {log_path}"
        raise ValueError(err)


def find_data_errors_completion(data_dict: dict[str, str]) -> str | None:
    keys = ["prompt", "completion"]

    assert isinstance(data_dict, dict)
    expected_keys = sorted(keys)
    found_keys = sorted(data_dict.keys())
    if set(expected_keys) != set(found_keys):
        return f"Expected Keys: {expected_keys}; Found Keys: {found_keys}"

    for key in keys:
        if not isinstance(data_dict[key], str):
            return f"Expected `{key}` to be of type `str`. Found: {type(data_dict[key])}"


# Following functions are modified from the OpenAI cookbook:
# https://cookbook.openai.com/examples/chat_finetuning_data_prep
def find_data_error_chat(messages: dict[str, Any]) -> str | None:
    assert isinstance(messages, dict)

    expected_keys = ["messages"]
    found_keys = sorted(messages.keys())
    if set(expected_keys) != set(found_keys):
        return f"Expected Keys: {expected_keys}; Found Keys: {found_keys}"

    if not isinstance(messages["messages"], list):
        return f"The value of the `messages` key should be a list instance. Found: {type(messages['messages'])}"

    for ind, message in enumerate(messages["messages"]):
        err = find_data_error_chat_message(message)
        if err:
            return f"Error in message at index {ind}: {err}"


def find_data_error_chat_message(message: dict[str, Any]) -> str | None:
    assert isinstance(message, dict)

    message_keys = sorted(["role", "content"])
    found_keys = sorted(message.keys())
    if set(message_keys) != set(found_keys):
        return f"Expected Keys: {message_keys}; Found Keys: {found_keys}"

    expected_roles = sorted(["assistant", "system", "user"])
    found_role = message["role"]
    if found_role not in expected_roles:
        return f"Expected Roles: {expected_roles}; Found Role: {found_role}"

    if not isinstance(message["content"], str):
        return f"Expected Content Type: `str`; Found Content Type: {type(message['content'])}"
```
