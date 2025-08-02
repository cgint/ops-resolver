# Directory Structure
_Includes files where the actual content might be omitted. This way the LLM can still use the file structure to understand the project._
```
.
├── __init__.py
├── __metadata__.py
├── adapters
│   ├── __init__.py
│   ├── base.py
│   ├── chat_adapter.py
│   ├── json_adapter.py
│   ├── two_step_adapter.py
│   ├── types
│   │   ├── __init__.py
│   │   ├── audio.py
│   │   ├── base_type.py
│   │   ├── code.py
│   │   ├── history.py
│   │   ├── image.py
│   │   └── tool.py
│   ├── utils.py
│   └── xml_adapter.py
├── clients
│   ├── __init__.py
│   ├── base_lm.py
│   ├── cache.py
│   ├── databricks.py
│   ├── embedding.py
│   ├── lm_local_arbor.py
│   ├── lm_local.py
│   ├── lm.py
│   ├── openai.py
│   ├── provider.py
│   └── utils_finetune.py
├── datasets
│   ├── __init__.py
│   ├── alfworld
│   │   ├── __init__.py
│   │   └── alfworld.py
│   ├── colors.py
│   ├── dataloader.py
│   ├── dataset.py
│   ├── gsm8k.py
│   ├── hotpotqa.py
│   └── math.py
├── dsp
│   ├── __init__.py
│   ├── colbertv2.py
│   └── utils
│       ├── __init__.py
│       ├── dpr.py
│       ├── settings.py
│       └── utils.py
├── evaluate
│   ├── __init__.py
│   ├── auto_evaluation.py
│   ├── evaluate.py
│   └── metrics.py
├── predict
│   ├── __init__.py
│   ├── aggregation.py
│   ├── avatar
│   │   ├── __init__.py
│   │   ├── avatar.py
│   │   ├── models.py
│   │   └── signatures.py
│   ├── best_of_n.py
│   ├── chain_of_thought.py
│   ├── code_act.py
│   ├── knn.py
│   ├── multi_chain_comparison.py
│   ├── parallel.py
│   ├── parameter.py
│   ├── predict.py
│   ├── program_of_thought.py
│   ├── react.py
│   ├── refine.py
│   └── retry.py
├── primitives
│   ├── __init__.py
│   ├── base_module.py
│   ├── example.py
│   ├── module.py
│   ├── prediction.py
│   ├── python_interpreter.py
│   └── runner.js
├── propose
│   ├── __init__.py
│   ├── dataset_summary_generator.py
│   ├── grounded_proposer.py
│   ├── propose_base.py
│   └── utils.py
├── retrievers
│   ├── __init__.py
│   ├── databricks_rm.py
│   ├── embeddings.py
│   ├── retrieve.py
│   └── weaviate_rm.py
├── signatures
│   ├── __init__.py
│   ├── field.py
│   ├── signature.py
│   └── utils.py
├── streaming
│   ├── __init__.py
│   ├── messages.py
│   ├── streamify.py
│   └── streaming_listener.py
├── teleprompt
│   ├── __init__.py
│   ├── avatar_optimizer.py
│   ├── bettertogether.py
│   ├── bootstrap_finetune.py
│   ├── bootstrap.py
│   ├── copro_optimizer.py
│   ├── ensemble.py
│   ├── grpo.py
│   ├── infer_rules.py
│   ├── knn_fewshot.py
│   ├── mipro_optimizer_v2.py
│   ├── random_search.py
│   ├── signature_opt.py
│   ├── simba_utils.py
│   ├── simba.py
│   ├── teleprompt_optuna.py
│   ├── teleprompt.py
│   ├── utils.py
│   └── vanilla.py
└── utils
    ├── __init__.py
    ├── asyncify.py
    ├── caching.py
    ├── callback.py
    ├── dummies.py
    ├── exceptions.py
    ├── inspect_history.py
    ├── langchain_tool.py
    ├── logging_utils.py
    ├── mcp.py
    ├── parallelizer.py
    ├── saving.py
    ├── syncify.py
    ├── unbatchify.py
    └── usage_tracker.py
```

# File Contents

## File: `__init__.py`
```
from dspy.predict import *
from dspy.primitives import *
from dspy.retrievers import *
from dspy.signatures import *
from dspy.teleprompt import *

from dspy.evaluate import Evaluate  # isort: skip
from dspy.clients import *  # isort: skip
from dspy.adapters import Adapter, ChatAdapter, JSONAdapter, XMLAdapter, TwoStepAdapter, Image, Audio, History, Type, Tool, ToolCalls, Code  # isort: skip
from dspy.utils.logging_utils import configure_dspy_loggers, disable_logging, enable_logging
from dspy.utils.asyncify import asyncify
from dspy.utils.syncify import syncify
from dspy.utils.saving import load
from dspy.streaming.streamify import streamify
from dspy.utils.usage_tracker import track_usage

from dspy.dsp.utils.settings import settings
from dspy.dsp.colbertv2 import ColBERTv2
from dspy.clients import DSPY_CACHE
from dspy.__metadata__ import __name__, __version__, __description__, __url__, __author__, __author_email__

configure_dspy_loggers(__name__)

# Singleton definitions and aliasing
configure = settings.configure
context = settings.context

BootstrapRS = BootstrapFewShotWithRandomSearch

cache = DSPY_CACHE
```

## File: `__metadata__.py`
```
#replace_package_name_marker
__name__="dspy"
#replace_package_version_marker
__version__="3.0.0b3"
__description__="DSPy"
__url__="https://github.com/stanfordnlp/dspy"
__author__="Omar Khattab"
__author_email__="okhattab@stanford.edu"```

## File: `adapters/__init__.py`
```
from dspy.adapters.base import Adapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.two_step_adapter import TwoStepAdapter
from dspy.adapters.types import Audio, Code, History, Image, Tool, ToolCalls, Type
from dspy.adapters.xml_adapter import XMLAdapter

__all__ = [
    "Adapter",
    "ChatAdapter",
    "Type",
    "History",
    "Image",
    "Audio",
    "Code",
    "JSONAdapter",
    "XMLAdapter",
    "TwoStepAdapter",
    "Tool",
    "ToolCalls",
]
```

## File: `adapters/base.py`
```
import logging
from typing import TYPE_CHECKING, Any, get_origin

import json_repair
import litellm

from dspy.adapters.types import History
from dspy.adapters.types.base_type import split_message_content_for_custom_types
from dspy.adapters.types.tool import Tool, ToolCalls
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback, with_callbacks

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dspy.clients.lm import LM


class Adapter:
    def __init__(self, callbacks: list[BaseCallback] | None = None, use_native_function_calling: bool = False):
        self.callbacks = callbacks or []
        self.use_native_function_calling = use_native_function_calling

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # Decorate format() and parse() method with with_callbacks
        cls.format = with_callbacks(cls.format)
        cls.parse = with_callbacks(cls.parse)

    def _call_preprocess(
        self,
        lm: "LM",
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        if self.use_native_function_calling:
            tool_call_input_field_name = self._get_tool_call_input_field_name(signature)
            tool_call_output_field_name = self._get_tool_call_output_field_name(signature)

            if tool_call_output_field_name and tool_call_input_field_name is None:
                raise ValueError(
                    f"You provided an output field {tool_call_output_field_name} to receive the tool calls information, "
                    "but did not provide any tools as the input. Please provide a list of tools as the input by adding an "
                    "input field with type `list[dspy.Tool]`."
                )

            if tool_call_output_field_name and litellm.supports_function_calling(model=lm.model):
                tools = inputs[tool_call_input_field_name]
                tools = tools if isinstance(tools, list) else [tools]

                litellm_tools = []
                for tool in tools:
                    litellm_tools.append(tool.format_as_litellm_function_call())

                lm_kwargs["tools"] = litellm_tools

                signature_for_native_function_calling = signature.delete(tool_call_output_field_name)
                signature_for_native_function_calling = signature_for_native_function_calling.delete(
                    tool_call_input_field_name
                )

                return signature_for_native_function_calling

        return signature

    def _call_postprocess(
        self,
        processed_signature: type[Signature],
        original_signature: type[Signature],
        outputs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        values = []

        tool_call_output_field_name = self._get_tool_call_output_field_name(original_signature)

        for output in outputs:
            output_logprobs = None
            tool_calls = None
            text = output

            if isinstance(output, dict):
                text = output["text"]
                output_logprobs = output.get("logprobs")
                tool_calls = output.get("tool_calls")

            if text:
                value = self.parse(processed_signature, text)
                for field_name in original_signature.output_fields.keys():
                    if field_name not in value:
                        # We need to set the field not present in the processed signature to None for consistency.
                        value[field_name] = None
            else:
                value = {}
                for field_name in original_signature.output_fields.keys():
                    value[field_name] = None

            if tool_calls and tool_call_output_field_name:
                tool_calls = [
                    {
                        "name": v["function"]["name"],
                        "args": json_repair.loads(v["function"]["arguments"]),
                    }
                    for v in tool_calls
                ]
                value[tool_call_output_field_name] = ToolCalls.from_dict_list(tool_calls)

            if output_logprobs:
                value["logprobs"] = output_logprobs

            values.append(value)

        return values

    def __call__(
        self,
        lm: "LM",
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        processed_signature = self._call_preprocess(lm, lm_kwargs, signature, inputs)
        inputs = self.format(processed_signature, demos, inputs)

        outputs = lm(messages=inputs, **lm_kwargs)
        return self._call_postprocess(processed_signature, signature, outputs)

    async def acall(
        self,
        lm: "LM",
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        processed_signature = self._call_preprocess(lm, lm_kwargs, signature, inputs)
        inputs = self.format(processed_signature, demos, inputs)

        outputs = await lm.acall(messages=inputs, **lm_kwargs)
        return self._call_postprocess(processed_signature, signature, outputs)

    def format(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Format the input messages for the LM call.

        This method converts the DSPy structured input along with few-shot examples and conversation history into
        multiturn messages as expected by the LM. For custom adapters, this method can be overridden to customize
        the formatting of the input messages.

        In general we recommend the messages to have the following structure:
        ```
        [
            {"role": "system", "content": system_message},
            # Begin few-shot examples
            {"role": "user", "content": few_shot_example_1_input},
            {"role": "assistant", "content": few_shot_example_1_output},
            {"role": "user", "content": few_shot_example_2_input},
            {"role": "assistant", "content": few_shot_example_2_output},
            ...
            # End few-shot examples
            # Begin conversation history
            {"role": "user", "content": conversation_history_1_input},
            {"role": "assistant", "content": conversation_history_1_output},
            {"role": "user", "content": conversation_history_2_input},
            {"role": "assistant", "content": conversation_history_2_output},
            ...
            # End conversation history
            {"role": "user", "content": current_input},
        ]

        And system message should contain the field description, field structure, and task description.
        ```


        Args:
            signature: The DSPy signature for which to format the input messages.
            demos: A list of few-shot examples.
            inputs: The input arguments to the DSPy module.

        Returns:
            A list of multiturn messages as expected by the LM.
        """
        inputs_copy = dict(inputs)

        # If the signature and inputs have conversation history, we need to format the conversation history and
        # remove the history field from the signature.
        history_field_name = self._get_history_field_name(signature)
        if history_field_name:
            # In order to format the conversation history, we need to remove the history field from the signature.
            signature_without_history = signature.delete(history_field_name)
            conversation_history = self.format_conversation_history(
                signature_without_history,
                history_field_name,
                inputs_copy,
            )

        messages = []
        system_message = (
            f"{self.format_field_description(signature)}\n"
            f"{self.format_field_structure(signature)}\n"
            f"{self.format_task_description(signature)}"
        )
        messages.append({"role": "system", "content": system_message})
        messages.extend(self.format_demos(signature, demos))
        if history_field_name:
            # Conversation history and current input
            content = self.format_user_message_content(signature_without_history, inputs_copy, main_request=True)
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": content})
        else:
            # Only current input
            content = self.format_user_message_content(signature, inputs_copy, main_request=True)
            messages.append({"role": "user", "content": content})

        messages = split_message_content_for_custom_types(messages)
        return messages

    def format_field_description(self, signature: type[Signature]) -> str:
        """Format the field description for the system message.

        This method formats the field description for the system message. It should return a string that contains
        the field description for the input fields and the output fields.

        Args:
            signature: The DSPy signature for which to format the field description.

        Returns:
            A string that contains the field description for the input fields and the output fields.
        """
        raise NotImplementedError

    def format_field_structure(self, signature: type[Signature]) -> str:
        """Format the field structure for the system message.

        This method formats the field structure for the system message. It should return a string that dictates the
        format the input fields should be provided to the LM, and the format the output fields will be in the response.
        Refer to the ChatAdapter and JsonAdapter for an example.

        Args:
            signature: The DSPy signature for which to format the field structure.
        """
        raise NotImplementedError

    def format_task_description(self, signature: type[Signature]) -> str:
        """Format the task description for the system message.

        This method formats the task description for the system message. In most cases this is just a thin wrapper
        over `signature.instructions`.

        Args:
            signature: The DSPy signature of the DSpy module.

        Returns:
            A string that describes the task.
        """
        raise NotImplementedError

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        """Format the user message content.

        This method formats the user message content, which can be used in formatting few-shot examples, conversation
        history, and the current input.

        Args:
            signature: The DSPy signature for which to format the user message content.
            inputs: The input arguments to the DSPy module.
            prefix: A prefix to the user message content.
            suffix: A suffix to the user message content.

        Returns:
            A string that contains the user message content.
        """
        raise NotImplementedError

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message: str | None = None,
    ) -> str:
        """Format the assistant message content.

        This method formats the assistant message content, which can be used in formatting few-shot examples,
        conversation history.

        Args:
            signature: The DSPy signature for which to format the assistant message content.
            outputs: The output fields to be formatted.
            missing_field_message: A message to be used when a field is missing.

        Returns:
            A string that contains the assistant message content.
        """
        raise NotImplementedError

    def format_demos(self, signature: type[Signature], demos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format the few-shot examples.

        This method formats the few-shot examples as multiturn messages.

        Args:
            signature: The DSPy signature for which to format the few-shot examples.
            demos: A list of few-shot examples, each element is a dictionary with keys of the input and output fields of
                the signature.

        Returns:
            A list of multiturn messages.
        """
        complete_demos = []
        incomplete_demos = []

        for demo in demos:
            # Check if all fields are present and not None
            is_complete = all(k in demo and demo[k] is not None for k in signature.fields)

            # Check if demo has at least one input and one output field
            has_input = any(k in demo for k in signature.input_fields)
            has_output = any(k in demo for k in signature.output_fields)

            if is_complete:
                complete_demos.append(demo)
            elif has_input and has_output:
                # We only keep incomplete demos that have at least one input and one output field
                incomplete_demos.append(demo)

        messages = []

        incomplete_demo_prefix = "This is an example of the task, though some input or output fields are not supplied."
        for demo in incomplete_demos:
            messages.append(
                {
                    "role": "user",
                    "content": self.format_user_message_content(signature, demo, prefix=incomplete_demo_prefix),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": self.format_assistant_message_content(
                        signature, demo, missing_field_message="Not supplied for this particular example. "
                    ),
                }
            )

        for demo in complete_demos:
            messages.append({"role": "user", "content": self.format_user_message_content(signature, demo)})
            messages.append(
                {
                    "role": "assistant",
                    "content": self.format_assistant_message_content(
                        signature, demo, missing_field_message="Not supplied for this conversation history message. "
                    ),
                }
            )

        return messages

    def _get_history_field_name(self, signature: type[Signature]) -> bool:
        for name, field in signature.input_fields.items():
            if field.annotation == History:
                return name
        return None

    def _get_tool_call_input_field_name(self, signature: type[Signature]) -> bool:
        for name, field in signature.input_fields.items():
            # Look for annotation `list[dspy.Tool]` or `dspy.Tool`
            origin = get_origin(field.annotation)
            if origin is list and field.annotation.__args__[0] == Tool:
                return name
            if field.annotation == Tool:
                return name
        return None

    def _get_tool_call_output_field_name(self, signature: type[Signature]) -> bool:
        for name, field in signature.output_fields.items():
            if field.annotation == ToolCalls:
                return name
        return None

    def format_conversation_history(
        self,
        signature: type[Signature],
        history_field_name: str,
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Format the conversation history.

        This method formats the conversation history and the current input as multiturn messages.

        Args:
            signature: The DSPy signature for which to format the conversation history.
            history_field_name: The name of the history field in the signature.
            inputs: The input arguments to the DSPy module.

        Returns:
            A list of multiturn messages.
        """
        conversation_history = inputs[history_field_name].messages if history_field_name in inputs else None

        if conversation_history is None:
            return []

        messages = []
        for message in conversation_history:
            messages.append(
                {
                    "role": "user",
                    "content": self.format_user_message_content(signature, message),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": self.format_assistant_message_content(signature, message),
                }
            )

        # Remove the history field from the inputs
        del inputs[history_field_name]

        return messages

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        """Parse the LM output into a dictionary of the output fields.

        This method parses the LM output into a dictionary of the output fields.

        Args:
            signature: The DSPy signature for which to parse the LM output.
            completion: The LM output to be parsed.

        Returns:
            A dictionary of the output fields.
        """
        raise NotImplementedError
```

## File: `adapters/chat_adapter.py`
```
import re
import textwrap
from typing import Any, NamedTuple

from litellm import ContextWindowExceededError
from pydantic.fields import FieldInfo

from dspy.adapters.base import Adapter
from dspy.adapters.utils import (
    format_field_value,
    get_annotation_name,
    get_field_description_string,
    parse_value,
    translate_field_type,
)
from dspy.clients.lm import LM
from dspy.signatures.signature import Signature
from dspy.utils.exceptions import AdapterParseError

field_header_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")


class FieldInfoWithName(NamedTuple):
    name: str
    info: FieldInfo


class ChatAdapter(Adapter):
    def __call__(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)
        except Exception as e:
            # fallback to JSONAdapter
            from dspy.adapters.json_adapter import JSONAdapter

            if isinstance(e, ContextWindowExceededError) or isinstance(self, JSONAdapter):
                # On context window exceeded error or already using JSONAdapter, we don't want to retry with a different
                # adapter.
                raise e
            return JSONAdapter()(lm, lm_kwargs, signature, demos, inputs)

    async def acall(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            return await super().acall(lm, lm_kwargs, signature, demos, inputs)
        except Exception as e:
            # fallback to JSONAdapter
            from dspy.adapters.json_adapter import JSONAdapter

            if isinstance(e, ContextWindowExceededError) or isinstance(self, JSONAdapter):
                # On context window exceeded error or already using JSONAdapter, we don't want to retry with a different
                # adapter.
                raise e
            return await JSONAdapter().acall(lm, lm_kwargs, signature, demos, inputs)

    def format_field_description(self, signature: type[Signature]) -> str:
        return (
            f"Your input fields are:\n{get_field_description_string(signature.input_fields)}\n"
            f"Your output fields are:\n{get_field_description_string(signature.output_fields)}"
        )

    def format_field_structure(self, signature: type[Signature]) -> str:
        """
        `ChatAdapter` requires input and output fields to be in their own sections, with section header using markers
        `[[ ## field_name ## ]]`. An arbitrary field `completed` ([[ ## completed ## ]]) is added to the end of the
        output fields section to indicate the end of the output fields.
        """
        parts = []
        parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

        def format_signature_fields_for_instructions(fields: dict[str, FieldInfo]):
            return self.format_field_with_value(
                fields_with_values={
                    FieldInfoWithName(name=field_name, info=field_info): translate_field_type(field_name, field_info)
                    for field_name, field_info in fields.items()
                },
            )

        parts.append(format_signature_fields_for_instructions(signature.input_fields))
        parts.append(format_signature_fields_for_instructions(signature.output_fields))
        parts.append("[[ ## completed ## ]]\n")
        return "\n\n".join(parts).strip()

    def format_task_description(self, signature: type[Signature]) -> str:
        instructions = textwrap.dedent(signature.instructions)
        objective = ("\n" + " " * 8).join([""] + instructions.splitlines())
        return f"In adhering to this structure, your objective is: {objective}"

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        messages = [prefix]
        for k, v in signature.input_fields.items():
            if k in inputs:
                value = inputs.get(k)
                formatted_field_value = format_field_value(field_info=v, value=value)
                messages.append(f"[[ ## {k} ## ]]\n{formatted_field_value}")

        if main_request:
            output_requirements = self.user_message_output_requirements(signature)
            if output_requirements is not None:
                messages.append(output_requirements)

        messages.append(suffix)
        return "\n\n".join(messages).strip()

    def user_message_output_requirements(self, signature: type[Signature]) -> str:
        """Returns a simplified format reminder for the language model.

        In chat-based interactions, language models may lose track of the required output format
        as the conversation context grows longer. This method generates a concise reminder of
        the expected output structure that can be included in user messages.

        Args:
            signature (Type[Signature]): The DSPy signature defining the expected input/output fields.

        Returns:
            str: A simplified description of the required output format.

        Note:
            This is a more lightweight version of `format_field_structure` specifically designed
            for inline reminders within chat messages.
        """

        def type_info(v):
            if v.annotation is not str:
                return f" (must be formatted as a valid Python {get_annotation_name(v.annotation)})"
            else:
                return ""

        message = "Respond with the corresponding output fields, starting with the field "
        message += ", then ".join(f"`[[ ## {f} ## ]]`{type_info(v)}" for f, v in signature.output_fields.items())
        message += ", and then ending with the marker for `[[ ## completed ## ]]`."
        return message

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message=None,
    ) -> str:
        assistant_message_content = self.format_field_with_value(
            {
                FieldInfoWithName(name=k, info=v): outputs.get(k, missing_field_message)
                for k, v in signature.output_fields.items()
            },
        )
        assistant_message_content += "\n\n[[ ## completed ## ]]\n"
        return assistant_message_content

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        sections = [(None, [])]

        for line in completion.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                # If the header pattern is found, split the rest of the line as content
                header = match.group(1)
                remaining_content = line[match.end() :].strip()
                sections.append((header, [remaining_content] if remaining_content else []))
            else:
                sections[-1][1].append(line)

        sections = [(k, "\n".join(v).strip()) for k, v in sections]

        fields = {}
        for k, v in sections:
            if (k not in fields) and (k in signature.output_fields):
                try:
                    fields[k] = parse_value(v, signature.output_fields[k].annotation)
                except Exception as e:
                    raise AdapterParseError(
                        adapter_name="ChatAdapter",
                        signature=signature,
                        lm_response=completion,
                        message=f"Failed to parse field {k} with value {v} from the LM response. Error message: {e}",
                    )
        if fields.keys() != signature.output_fields.keys():
            raise AdapterParseError(
                adapter_name="ChatAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=fields,
            )

        return fields

    def format_field_with_value(self, fields_with_values: dict[FieldInfoWithName, Any]) -> str:
        """
        Formats the values of the specified fields according to the field's DSPy type (input or output),
        annotation (e.g. str, int, etc.), and the type of the value itself. Joins the formatted values
        into a single string, which is is a multiline string if there are multiple fields.

        Args:
            fields_with_values: A dictionary mapping information about a field to its corresponding
                value.

        Returns:
            The joined formatted values of the fields, represented as a string
        """
        output = []
        for field, field_value in fields_with_values.items():
            formatted_field_value = format_field_value(field_info=field.info, value=field_value)
            output.append(f"[[ ## {field.name} ## ]]\n{formatted_field_value}")

        return "\n\n".join(output).strip()

    def format_finetune_data(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> dict[str, list[Any]]:
        """
        Format the call data into finetuning data according to the OpenAI API specifications.

        For the chat adapter, this means formatting the data as a list of messages, where each message is a dictionary
        with a "role" and "content" key. The role can be "system", "user", or "assistant". Then, the messages are
        wrapped in a dictionary with a "messages" key.
        """
        system_user_messages = self.format(  # returns a list of dicts with the keys "role" and "content"
            signature=signature, demos=demos, inputs=inputs
        )
        assistant_message_content = self.format_assistant_message_content(  # returns a string, without the role
            signature=signature, outputs=outputs
        )
        assistant_message = {"role": "assistant", "content": assistant_message_content}
        messages = system_user_messages + [assistant_message]
        return {"messages": messages}
```

## File: `adapters/json_adapter.py`
```
import json
import logging
from typing import Any, get_origin

import json_repair
import litellm
import pydantic
import regex
from pydantic.fields import FieldInfo

from dspy.adapters.chat_adapter import ChatAdapter, FieldInfoWithName
from dspy.adapters.types.tool import ToolCalls
from dspy.adapters.utils import (
    format_field_value,
    get_annotation_name,
    parse_value,
    serialize_for_json,
    translate_field_type,
)
from dspy.clients.lm import LM
from dspy.signatures.signature import Signature, SignatureMeta
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)


def _has_open_ended_mapping(signature: SignatureMeta) -> bool:
    """
    Check whether any output field in the signature has an open-ended mapping type,
    such as dict[str, Any]. Structured Outputs require explicit properties, so such fields
    are incompatible.
    """
    for field in signature.output_fields.values():
        annotation = field.annotation
        if get_origin(annotation) is dict:
            return True
    return False


class JSONAdapter(ChatAdapter):
    def __init__(self, callbacks: list[BaseCallback] | None = None, use_native_function_calling: bool = True):
        # JSONAdapter uses native function calling by default.
        super().__init__(callbacks=callbacks, use_native_function_calling=use_native_function_calling)

    def _json_adapter_call_common(self, lm, lm_kwargs, signature, demos, inputs, call_fn):
        """Common call logic to be used for both sync and async calls."""
        provider = lm.model.split("/", 1)[0] or "openai"
        params = litellm.get_supported_openai_params(model=lm.model, custom_llm_provider=provider)

        if not params or "response_format" not in params:
            return call_fn(lm, lm_kwargs, signature, demos, inputs)

        has_tool_calls = any(field.annotation == ToolCalls for field in signature.output_fields.values())
        if _has_open_ended_mapping(signature) or (not self.use_native_function_calling and has_tool_calls):
            # We found that structured output mode doesn't work well with dspy.ToolCalls as output field.
            # So we fall back to json mode if native function calling is disabled and ToolCalls is present.
            lm_kwargs["response_format"] = {"type": "json_object"}
            return call_fn(lm, lm_kwargs, signature, demos, inputs)

    def __call__(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        result = self._json_adapter_call_common(lm, lm_kwargs, signature, demos, inputs, super().__call__)
        if result:
            return result

        try:
            structured_output_model = _get_structured_outputs_response_format(
                signature, self.use_native_function_calling
            )
            lm_kwargs["response_format"] = structured_output_model
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)
        except Exception:
            logger.warning("Failed to use structured output format, falling back to JSON mode.")
            lm_kwargs["response_format"] = {"type": "json_object"}
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)

    async def acall(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        result = self._json_adapter_call_common(lm, lm_kwargs, signature, demos, inputs, super().acall)
        if result:
            return await result

        try:
            structured_output_model = _get_structured_outputs_response_format(signature)
            lm_kwargs["response_format"] = structured_output_model
            return await super().acall(lm, lm_kwargs, signature, demos, inputs)
        except Exception:
            logger.warning("Failed to use structured output format, falling back to JSON mode.")
            lm_kwargs["response_format"] = {"type": "json_object"}
            return await super().acall(lm, lm_kwargs, signature, demos, inputs)

    def format_field_structure(self, signature: type[Signature]) -> str:
        parts = []
        parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

        def format_signature_fields_for_instructions(fields: dict[str, FieldInfo], role: str):
            return self.format_field_with_value(
                fields_with_values={
                    FieldInfoWithName(name=field_name, info=field_info): translate_field_type(field_name, field_info)
                    for field_name, field_info in fields.items()
                },
                role=role,
            )

        parts.append("Inputs will have the following structure:")
        parts.append(format_signature_fields_for_instructions(signature.input_fields, role="user"))
        parts.append("Outputs will be a JSON object with the following fields.")
        parts.append(format_signature_fields_for_instructions(signature.output_fields, role="assistant"))
        return "\n\n".join(parts).strip()

    def user_message_output_requirements(self, signature: type[Signature]) -> str:
        def type_info(v):
            return (
                f" (must be formatted as a valid Python {get_annotation_name(v.annotation)})"
                if v.annotation is not str
                else ""
            )

        message = "Respond with a JSON object in the following order of fields: "
        message += ", then ".join(f"`{f}`{type_info(v)}" for f, v in signature.output_fields.items())
        message += "."
        return message

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message=None,
    ) -> str:
        fields_with_values = {
            FieldInfoWithName(name=k, info=v): outputs.get(k, missing_field_message)
            for k, v in signature.output_fields.items()
        }
        return self.format_field_with_value(fields_with_values, role="assistant")

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        pattern = r"\{(?:[^{}]|(?R))*\}"
        match = regex.search(pattern, completion, regex.DOTALL)
        if match:
            completion = match.group(0)
        fields = json_repair.loads(completion)

        if not isinstance(fields, dict):
            raise AdapterParseError(
                adapter_name="JSONAdapter",
                signature=signature,
                lm_response=completion,
                message="LM response cannot be serialized to a JSON object.",
            )

        fields = {k: v for k, v in fields.items() if k in signature.output_fields}

        # Attempt to cast each value to type signature.output_fields[k].annotation.
        for k, v in fields.items():
            if k in signature.output_fields:
                fields[k] = parse_value(v, signature.output_fields[k].annotation)

        if fields.keys() != signature.output_fields.keys():
            raise AdapterParseError(
                adapter_name="JSONAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=fields,
            )

        return fields

    def format_field_with_value(self, fields_with_values: dict[FieldInfoWithName, Any], role: str = "user") -> str:
        """
        Formats the values of the specified fields according to the field's DSPy type (input or output),
        annotation (e.g. str, int, etc.), and the type of the value itself. Joins the formatted values
        into a single string, which is a multiline string if there are multiple fields.

        Args:
            fields_with_values: A dictionary mapping information about a field to its corresponding value.
        Returns:
            The joined formatted values of the fields, represented as a string.
        """
        if role == "user":
            output = []
            for field, field_value in fields_with_values.items():
                formatted_field_value = format_field_value(field_info=field.info, value=field_value)
                output.append(f"[[ ## {field.name} ## ]]\n{formatted_field_value}")
            return "\n\n".join(output).strip()
        else:
            d = fields_with_values.items()
            d = {k.name: v for k, v in d}
            return json.dumps(serialize_for_json(d), indent=2)

    def format_finetune_data(
        self, signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any], outputs: dict[str, Any]
    ) -> dict[str, list[Any]]:
        # TODO: implement format_finetune_data method in JSONAdapter
        raise NotImplementedError


def _get_structured_outputs_response_format(
    signature: SignatureMeta,
    use_native_function_calling: bool = True,
) -> type[pydantic.BaseModel]:
    """
    Builds a Pydantic model from a DSPy signature's output_fields and ensures the generated JSON schema
    is compatible with OpenAI Structured Outputs (all objects have a "required" key listing every property,
    and additionalProperties is always false).

    IMPORTANT: If any field's annotation is an open-ended mapping (e.g. dict[str, Any]), then a structured
    schema cannot be generated since all properties must be explicitly declared. In that case, an exception
    is raised so that the caller can fall back to using a plain "json_object" response_format.
    """
    # Although we've already performed an early check, we keep this here as a final guard.
    for name, field in signature.output_fields.items():
        annotation = field.annotation
        if get_origin(annotation) is dict:
            raise ValueError(
                f"Field '{name}' has an open-ended mapping type which is not supported by Structured Outputs."
            )

    fields = {}
    for name, field in signature.output_fields.items():
        annotation = field.annotation
        if use_native_function_calling and annotation == ToolCalls:
            # Skip ToolCalls field if native function calling is enabled.
            continue
        default = field.default if hasattr(field, "default") else ...
        fields[name] = (annotation, default)

    # Build the model with extra fields forbidden.
    pydantic_model = pydantic.create_model(
        "DSPyProgramOutputs",
        __config__=pydantic.ConfigDict(extra="forbid"),
        **fields,
    )

    # Generate the initial schema.
    schema = pydantic_model.model_json_schema()

    # Remove any DSPy-specific metadata.
    for prop in schema.get("properties", {}).values():
        prop.pop("json_schema_extra", None)

    def enforce_required(schema_part: dict):
        """
        Recursively ensure that:
            - for any object schema, a "required" key is added with all property names (or [] if no properties)
            - additionalProperties is set to False regardless of the previous value.
            - the same enforcement is run for nested arrays and definitions.
        """
        if schema_part.get("type") == "object":
            props = schema_part.get("properties")
            if props is not None:
                # For objects with explicitly declared properties:
                schema_part["required"] = list(props.keys())
                schema_part["additionalProperties"] = False
                for sub_schema in props.values():
                    if isinstance(sub_schema, dict):
                        enforce_required(sub_schema)
            else:
                # For objects with no properties (should not happen normally but a fallback).
                schema_part["properties"] = {}
                schema_part["required"] = []
                schema_part["additionalProperties"] = False
        if schema_part.get("type") == "array" and isinstance(schema_part.get("items"), dict):
            enforce_required(schema_part["items"])
        # Also enforce in any nested definitions / $defs.
        for key in ("$defs", "definitions"):
            if key in schema_part:
                for def_schema in schema_part[key].values():
                    enforce_required(def_schema)

    enforce_required(schema)

    # Override the model's JSON schema generation to return our precomputed schema.
    pydantic_model.model_json_schema = lambda *args, **kwargs: schema

    return pydantic_model
```

## File: `adapters/two_step_adapter.py`
```
from typing import Any

import json_repair

from dspy.adapters.base import Adapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types import ToolCalls
from dspy.adapters.utils import get_field_description_string
from dspy.clients import LM
from dspy.signatures.field import InputField
from dspy.signatures.signature import Signature, make_signature

"""
NOTE/TODO/FIXME:

The main issue below is that the second step's signature is entirely created on the fly and is invoked with a chat
adapter explicitly constructed with no demonstrations. This means that it cannot "learn" or get optimized.
"""


class TwoStepAdapter(Adapter):
    """
    A two-stage adapter that:
        1. Uses a simpler, more natural prompt for the main LM
        2. Uses a smaller LM with chat adapter to extract structured data from the response of main LM
    This adapter uses a common __call__ logic defined in base Adapter class.
    This class is particularly useful when interacting with reasoning models as the main LM since reasoning models
    are known to struggle with structured outputs.

    Example:
    ```
    import dspy
    lm = dspy.LM(model="openai/o3-mini", max_tokens=10000, temperature = 1.0)
    adapter = dspy.TwoStepAdapter(dspy.LM("openai/gpt-4o-mini"))
    dspy.configure(lm=lm, adapter=adapter)
    program = dspy.ChainOfThought("question->answer")
    result = program("What is the capital of France?")
    print(result)
    ```
    """

    def __init__(self, extraction_model: LM, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(extraction_model, LM):
            raise ValueError("extraction_model must be an instance of LM")
        self.extraction_model = extraction_model

    def format(
        self, signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Format a prompt for the first stage with the main LM.
        This no specific structure is required for the main LM, we customize the format method
        instead of format_field_description or format_field_structure.

        Args:
            signature: The signature of the original task
            demos: A list of demo examples
            inputs: The current input

        Returns:
            A list of messages to be passed to the main LM.
        """
        messages = []

        # Create a task description for the main LM
        task_description = self.format_task_description(signature)
        messages.append({"role": "system", "content": task_description})

        messages.extend(self.format_demos(signature, demos))

        # Format the current input
        messages.append({"role": "user", "content": self.format_user_message_content(signature, inputs)})

        return messages

    def parse(self, signature: Signature, completion: str) -> dict[str, Any]:
        """
        Use a smaller LM (extraction_model) with chat adapter to extract structured data
        from the raw completion text of the main LM.

        Args:
            signature: The signature of the original task
            completion: The completion from the main LM

        Returns:
            A dictionary containing the extracted structured data.
        """
        # The signature is supposed to be "text -> {original output fields}"
        extractor_signature = self._create_extractor_signature(signature)

        try:
            # Call the smaller LM to extract structured data from the raw completion text with ChatAdapter
            parsed_result = ChatAdapter()(
                lm=self.extraction_model,
                lm_kwargs={},
                signature=extractor_signature,
                demos=[],
                inputs={"text": completion},
            )
            return parsed_result[0]

        except Exception as e:
            raise ValueError(f"Failed to parse response from the original completion: {completion}") from e

    async def acall(
        self,
        lm: "LM",
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        inputs = self.format(signature, demos, inputs)

        outputs = await lm.acall(messages=inputs, **lm_kwargs)
        # The signature is supposed to be "text -> {original output fields}"
        extractor_signature = self._create_extractor_signature(signature)

        values = []

        tool_call_output_field_name = self._get_tool_call_output_field_name(signature)
        for output in outputs:
            output_logprobs = None
            tool_calls = None
            text = output

            if isinstance(output, dict):
                text = output["text"]
                output_logprobs = output.get("logprobs")
                tool_calls = output.get("tool_calls")

            try:
                # Call the smaller LM to extract structured data from the raw completion text with ChatAdapter
                value = await ChatAdapter().acall(
                    lm=self.extraction_model,
                    lm_kwargs={},
                    signature=extractor_signature,
                    demos=[],
                    inputs={"text": text},
                )
                value = value[0]

            except Exception as e:
                raise ValueError(f"Failed to parse response from the original completion: {output}") from e

            if tool_calls and tool_call_output_field_name:
                tool_calls = [
                    {
                        "name": v["function"]["name"],
                        "args": json_repair.loads(v["function"]["arguments"]),
                    }
                    for v in tool_calls
                ]
                value[tool_call_output_field_name] = ToolCalls.from_dict_list(tool_calls)

            if output_logprobs is not None:
                value["logprobs"] = output_logprobs

            values.append(value)
        return values

    def format_task_description(self, signature: Signature) -> str:
        """Create a description of the task based on the signature"""
        parts = []

        parts.append("You are a helpful assistant that can solve tasks based on user input.")
        parts.append("As input, you will be provided with:\n" + get_field_description_string(signature.input_fields))
        parts.append("Your outputs must contain:\n" + get_field_description_string(signature.output_fields))
        parts.append("You should lay out your outputs in detail so that your answer can be understood by another agent")

        if signature.instructions:
            parts.append(f"Specific instructions: {signature.instructions}")

        return "\n".join(parts)

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
    ) -> str:
        parts = [prefix]

        for name in signature.input_fields.keys():
            if name in inputs:
                parts.append(f"{name}: {inputs.get(name, '')}")

        parts.append(suffix)
        return "\n\n".join(parts).strip()

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message: str | None = None,
    ) -> str:
        parts = []

        for name in signature.output_fields.keys():
            if name in outputs:
                parts.append(f"{name}: {outputs.get(name, missing_field_message)}")

        return "\n\n".join(parts).strip()

    def _create_extractor_signature(
        self,
        original_signature: type[Signature],
    ) -> type[Signature]:
        """Create a new signature containing a new 'text' input field and all output fields.

        Args:
            original_signature: The original signature to extract output fields from

        Returns:
            A new Signature type with a text input field and all output fields
        """
        # Create new fields dict with 'text' input field and all output fields
        new_fields = {
            "text": (str, InputField()),
            **{name: (field.annotation, field) for name, field in original_signature.output_fields.items()},
        }

        outputs_str = ", ".join([f"`{field}`" for field in original_signature.output_fields])
        instructions = f"The input is a text that should contain all the necessary information to produce the fields {outputs_str}. \
            Your job is to extract the fields from the text verbatim. Extract precisely the appropriate value (content) for each field."

        return make_signature(new_fields, instructions)
```

## File: `adapters/types/__init__.py`
```
from dspy.adapters.types.audio import Audio
from dspy.adapters.types.base_type import Type
from dspy.adapters.types.code import Code
from dspy.adapters.types.history import History
from dspy.adapters.types.image import Image
from dspy.adapters.types.tool import Tool, ToolCalls

__all__ = ["History", "Image", "Audio", "Type", "Tool", "ToolCalls", "Code"]
```

## File: `adapters/types/audio.py`
```
import base64
import io
import mimetypes
import os
from typing import Any, Union

import pydantic
import requests

from dspy.adapters.types.base_type import Type

try:
    import soundfile as sf

    SF_AVAILABLE = True
except ImportError:
    SF_AVAILABLE = False


class Audio(Type):
    data: str
    audio_format: str

    model_config = {
        "frozen": True,
        "extra": "forbid",
    }

    def format(self) -> list[dict[str, Any]]:
        try:
            data = self.data
        except Exception as e:
            raise ValueError(f"Failed to format audio for DSPy: {e}")
        return [{
            "type": "input_audio",
            "input_audio": {
                "data": data,
                "format": self.audio_format
            }
        }]


    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, values: Any) -> Any:
        """
        Validate input for Audio, expecting 'data' and 'audio_format' keys in dictionary.
        """
        if isinstance(values, cls):
            return {"data": values.data, "audio_format": values.audio_format}
        return encode_audio(values)

    @classmethod
    def from_url(cls, url: str) -> "Audio":
        """
        Download an audio file from URL and encode it as base64.
        """
        response = requests.get(url)
        response.raise_for_status()
        mime_type = response.headers.get("Content-Type", "audio/wav")
        if not mime_type.startswith("audio/"):
            raise ValueError(f"Unsupported MIME type for audio: {mime_type}")
        audio_format = mime_type.split("/")[1]
        encoded_data = base64.b64encode(response.content).decode("utf-8")
        return cls(data=encoded_data, audio_format=audio_format)

    @classmethod
    def from_file(cls, file_path: str) -> "Audio":
        """
        Read local audio file and encode it as base64.
        """
        if not os.path.isfile(file_path):
            raise ValueError(f"File not found: {file_path}")

        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type or not mime_type.startswith("audio/"):
            raise ValueError(f"Unsupported MIME type for audio: {mime_type}")

        with open(file_path, "rb") as file:
            file_data = file.read()

        audio_format = mime_type.split("/")[1]
        encoded_data = base64.b64encode(file_data).decode("utf-8")
        return cls(data=encoded_data, audio_format=audio_format)

    @classmethod
    def from_array(
        cls, array: Any, sampling_rate: int, format: str = "wav"
    ) -> "Audio":
        """
        Process numpy-like array and encode it as base64. Uses sampling rate and audio format for encoding.
        """
        if not SF_AVAILABLE:
            raise ImportError("soundfile is required to process audio arrays.")

        byte_buffer = io.BytesIO()
        sf.write(
            byte_buffer,
            array,
            sampling_rate,
            format=format.upper(),
            subtype="PCM_16",
        )
        encoded_data = base64.b64encode(byte_buffer.getvalue()).decode("utf-8")
        return cls(data=encoded_data, audio_format=format)

    def __str__(self) -> str:
        return self.serialize_model()

    def __repr__(self) -> str:
        length = len(self.data)
        return f"Audio(data=<AUDIO_BASE_64_ENCODED({length})>, audio_format='{self.audio_format}')"

def encode_audio(audio: Union[str, bytes, dict, "Audio", Any], sampling_rate: int = 16000, format: str = "wav") -> dict:
    """
    Encode audio to a dict with 'data' and 'audio_format'.
    
    Accepts: local file path, URL, data URI, dict, Audio instance, numpy array, or bytes (with known format).
    """
    if isinstance(audio, dict) and "data" in audio and "audio_format" in audio:
        return audio
    elif isinstance(audio, Audio):
        return {"data": audio.data, "audio_format": audio.audio_format}
    elif isinstance(audio, str) and audio.startswith("data:audio/"):
        try:
            header, b64data = audio.split(",", 1)
            mime = header.split(";")[0].split(":")[1]
            audio_format = mime.split("/")[1]
            return {"data": b64data, "audio_format": audio_format}
        except Exception as e:
            raise ValueError(f"Malformed audio data URI: {e}")
    elif isinstance(audio, str) and os.path.isfile(audio):
        a = Audio.from_file(audio)
        return {"data": a.data, "audio_format": a.audio_format}
    elif isinstance(audio, str) and audio.startswith("http"):
        a = Audio.from_url(audio)
        return {"data": a.data, "audio_format": a.audio_format}
    elif SF_AVAILABLE and hasattr(audio, "shape"):
        a = Audio.from_array(audio, sampling_rate=sampling_rate, format=format)
        return {"data": a.data, "audio_format": a.audio_format}
    elif isinstance(audio, bytes):
        encoded = base64.b64encode(audio).decode("utf-8")
        return {"data": encoded, "audio_format": format}
    else:
        raise ValueError(f"Unsupported type for encode_audio: {type(audio)}")
```

## File: `adapters/types/base_type.py`
```
import json
import re
from typing import Any, get_args, get_origin

import json_repair
import pydantic

CUSTOM_TYPE_START_IDENTIFIER = "<<CUSTOM-TYPE-START-IDENTIFIER>>"
CUSTOM_TYPE_END_IDENTIFIER = "<<CUSTOM-TYPE-END-IDENTIFIER>>"


class Type(pydantic.BaseModel):
    """Base class to support creating custom types for DSPy signatures.

    This is the parent class of DSPy custom types, e.g, dspy.Image. Subclasses must implement the `format` method to
    return a list of dictionaries (same as the Array of content parts in the OpenAI API user message's content field).

    Example:

        ```python
        class Image(Type):
            url: str

            def format(self) -> list[dict[str, Any]]:
                return [{"type": "image_url", "image_url": {"url": self.url}}]
        ```
    """

    def format(self) -> list[dict[str, Any]] | str:
        raise NotImplementedError

    @classmethod
    def description(cls) -> str:
        """Description of the custom type"""
        return ""

    @classmethod
    def extract_custom_type_from_annotation(cls, annotation):
        """Extract all custom types from the annotation.

        This is used to extract all custom types from the annotation of a field, while the annotation can
        have arbitrary level of nesting. For example, we detect `Tool` is in `list[dict[str, Tool]]`.
        """
        # Direct match. Nested type like `list[dict[str, Event]]` passes `isinstance(annotation, type)` in python 3.10
        # while fails in python 3.11. To accomodate users using python 3.10, we need to capture the error and ignore it.
        try:
            if isinstance(annotation, type) and issubclass(annotation, cls):
                return [annotation]
        except TypeError:
            pass

        origin = get_origin(annotation)
        if origin is None:
            return []

        result = []
        # Recurse into all type args
        for arg in get_args(annotation):
            result.extend(cls.extract_custom_type_from_annotation(arg))

        return result

    @pydantic.model_serializer()
    def serialize_model(self):
        formatted = self.format()
        if isinstance(formatted, list):
            return f"{CUSTOM_TYPE_START_IDENTIFIER}{self.format()}{CUSTOM_TYPE_END_IDENTIFIER}"
        return formatted


def split_message_content_for_custom_types(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split user message content into a list of content blocks.

    This method splits each user message's content in the `messages` list to be a list of content block, so that
    the custom types like `dspy.Image` can be properly formatted for better quality. For example, the split content
    may look like below if the user message has a `dspy.Image` object:

    ```
    [
        {"type": "text", "text": "{text_before_image}"},
        {"type": "image_url", "image_url": {"url": "{image_url}"}},
        {"type": "text", "text": "{text_after_image}"},
    ]
    ```

    This is implemented by finding the `<<CUSTOM-TYPE-START-IDENTIFIER>>` and `<<CUSTOM-TYPE-END-IDENTIFIER>>`
    in the user message content and splitting the content around them. The `<<CUSTOM-TYPE-START-IDENTIFIER>>`
    and `<<CUSTOM-TYPE-END-IDENTIFIER>>` are the reserved identifiers for the custom types as in `dspy.Type`.

    Args:
        messages: a list of messages sent to the LM. The format is the same as [OpenAI API's messages
            format](https://platform.openai.com/docs/guides/chat-completions/response-format).

    Returns:
        A list of messages with the content split into a list of content blocks around custom types content.
    """
    for message in messages:
        if message["role"] != "user":
            # Custom type messages are only in user messages
            continue

        pattern = rf"{CUSTOM_TYPE_START_IDENTIFIER}(.*?){CUSTOM_TYPE_END_IDENTIFIER}"
        result = []
        last_end = 0
        # DSPy adapter always formats user input into a string content before custom type splitting
        content: str = message["content"]

        for match in re.finditer(pattern, content, re.DOTALL):
            start, end = match.span()

            # Add text before the current block
            if start > last_end:
                result.append({"type": "text", "text": content[last_end:start]})

            # Parse the JSON inside the block
            custom_type_content = match.group(1).strip()
            try:
                parsed = json_repair.loads(custom_type_content)
                for custom_type_content in parsed:
                    result.append(custom_type_content)
            except json.JSONDecodeError:
                # fallback to raw string if it's not valid JSON
                parsed = {"type": "text", "text": custom_type_content}
                result.append(parsed)

            last_end = end

        if last_end == 0:
            # No custom type found, return the original message
            continue

        # Add any remaining text after the last match
        if last_end < len(content):
            result.append({"type": "text", "text": content[last_end:]})

        message["content"] = result

    return messages
```

## File: `adapters/types/code.py`
```
import re
from typing import Any, ClassVar

import pydantic
from pydantic import create_model

from dspy.adapters.types.base_type import Type


class Code(Type):
    """Code type in DSPy.

    This type is useful for code generation and code analysis.

    Example 1: dspy.Code as output type in code generation:

    ```python
    import dspy

    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


    class CodeGeneration(dspy.Signature):
        '''Generate python code to answer the question.'''

        question: str = dspy.InputField(description="The question to answer")
        code: dspy.Code["java"] = dspy.OutputField(description="The code to execute")


    predict = dspy.Predict(CodeGeneration)

    result = predict(question="Given an array, find if any of the two numbers sum up to 10")
    print(result.code)
    ```

    Example 2: dspy.Code as input type in code analysis:

    ```python
    import dspy
    import inspect

    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

    class CodeAnalysis(dspy.Signature):
        '''Analyze the time complexity of the function.'''

        code: dspy.Code["python"] = dspy.InputField(description="The function to analyze")
        result: str = dspy.OutputField(description="The time complexity of the function")


    predict = dspy.Predict(CodeAnalysis)


    def sleepsort(x):
        import time

        for i in x:
            time.sleep(i)
            print(i)

    result = predict(code=inspect.getsource(sleepsort))
    print(result.result)
    ```
    """

    code: str

    language: ClassVar[str] = "python"

    def format(self):
        return f"{self.code}"

    @pydantic.model_serializer()
    def serialize_model(self):
        """Override to bypass the <<CUSTOM-TYPE-START-IDENTIFIER>> and <<CUSTOM-TYPE-END-IDENTIFIER>> tags."""
        return self.format()

    @classmethod
    def description(cls) -> str:
        return (
            "Code represented in a string, specified in the `code` field. If this is an output field, the code "
            "field should follow the markdown code block format, e.g. \n```python\n{code}\n``` or \n```cpp\n{code}\n```"
            f"\nProgramming language: {cls.language}"
        )

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any):
        if isinstance(data, cls):
            return data

        if isinstance(data, str):
            return {"code": _filter_code(data)}

        if isinstance(data, dict):
            if "code" not in data:
                raise ValueError("`code` field is required for `dspy.Code`")
            if not isinstance(data["code"], str):
                raise ValueError(f"`code` field must be a string, but received type: {type(data['code'])}")
            return {"code": _filter_code(data["code"])}

        raise ValueError(f"Received invalid value for `dspy.Code`: {data}")


def _filter_code(code: str) -> str:
    """Extract code from markdown code blocks, stripping any language identifier."""
    # Case 1: format like:
    # ```python
    # {code_block}
    # ```
    regex_pattern = r"```(?:[^\n]*)\n(.*?)```"
    match = re.search(regex_pattern, code, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Case 2: ```<code>``` (no language, single-line)
    regex_pattern_simple = r"```(.*?)```"
    match = re.search(regex_pattern_simple, code, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback case
    return code


# Patch __class_getitem__ directly on the class to support dspy.Code["python"] syntax
def _code_class_getitem(cls, language):
    code_with_language_cls = create_model(f"{cls.__name__}_{language}", __base__=cls)
    code_with_language_cls.language = language
    return code_with_language_cls


Code.__class_getitem__ = classmethod(_code_class_getitem)
```

## File: `adapters/types/history.py`
```
from typing import Any

import pydantic


class History(pydantic.BaseModel):
    """Class representing the conversation history.

    The conversation history is a list of messages, each message entity should have keys from the associated signature.
    For example, if you have the following signature:

    ```
    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        answer: str = dspy.OutputField()
    ```

    Then the history should be a list of dictionaries with keys "question" and "answer".

    Example:
        ```
        import dspy

        dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            history: dspy.History = dspy.InputField()
            answer: str = dspy.OutputField()

        history = dspy.History(
            messages=[
                {"question": "What is the capital of France?", "answer": "Paris"},
                {"question": "What is the capital of Germany?", "answer": "Berlin"},
            ]
        )

        predict = dspy.Predict(MySignature)
        outputs = predict(question="What is the capital of France?", history=history)
        ```

    Example of capturing the conversation history:
        ```
        import dspy

        dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

        class MySignature(dspy.Signature):
            question: str = dspy.InputField()
            history: dspy.History = dspy.InputField()
            answer: str = dspy.OutputField()

        predict = dspy.Predict(MySignature)
        outputs = predict(question="What is the capital of France?")
        history = dspy.History(messages=[{"question": "What is the capital of France?", **outputs}])
        outputs_with_history = predict(question="Are you sure?", history=history)
        ```
    """

    messages: list[dict[str, Any]]

    model_config = {
        "frozen": True,
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "extra": "forbid",
    }
```

## File: `adapters/types/image.py`
```
import base64
import io
import mimetypes
import os
from typing import Any, Union
from urllib.parse import urlparse

import pydantic
import requests

from dspy.adapters.types.base_type import Type

try:
    from PIL import Image as PILImage

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class Image(Type):
    url: str

    model_config = {
        "frozen": True,
        "str_strip_whitespace": True,
        "validate_assignment": True,
        "extra": "forbid",
    }

    def format(self) -> list[dict[str, Any]] | str:
        try:
            image_url = encode_image(self.url)
        except Exception as e:
            raise ValueError(f"Failed to format image for DSPy: {e}")
        return [{"type": "image_url", "image_url": {"url": image_url}}]

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, values):
        # Allow the model to accept either a URL string or a dictionary with a single 'url' key
        if isinstance(values, str):
            # if a string, assume it's the URL directly and wrap it in a dict
            return {"url": values}
        elif isinstance(values, dict) and set(values.keys()) == {"url"}:
            # if it's a dict, ensure it has only the 'url' key
            return values
        elif isinstance(values, cls):
            return values.model_dump()
        else:
            raise TypeError("Expected a string URL or a dictionary with a key 'url'.")

    # If all my inits just call encode_image, should that be in this class
    @classmethod
    def from_url(cls, url: str, download: bool = False):
        return cls(url=encode_image(url, download))

    @classmethod
    def from_file(cls, file_path: str):
        return cls(url=encode_image(file_path))

    @classmethod
    def from_PIL(cls, pil_image):  # noqa: N802
        return cls(url=encode_image(pil_image))

    def __str__(self):
        return self.serialize_model()

    def __repr__(self):
        if "base64" in self.url:
            len_base64 = len(self.url.split("base64,")[1])
            image_type = self.url.split(";")[0].split("/")[-1]
            return f"Image(url=data:image/{image_type};base64,<IMAGE_BASE_64_ENCODED({len_base64!s})>)"
        return f"Image(url='{self.url}')"


def is_url(string: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urlparse(string)
        return all([result.scheme in ("http", "https", "gs"), result.netloc])
    except ValueError:
        return False


def encode_image(image: Union[str, bytes, "PILImage.Image", dict], download_images: bool = False) -> str:
    """
    Encode an image or file to a base64 data URI.

    Args:
        image: The image or file to encode. Can be a PIL Image, file path, URL, or data URI.
        download_images: Whether to download images from URLs.

    Returns:
        str: The data URI of the file or the URL if download_images is False.

    Raises:
        ValueError: If the file type is not supported.
    """
    if isinstance(image, dict) and "url" in image:
        # NOTE: Not doing other validation for now
        return image["url"]
    elif isinstance(image, str):
        if image.startswith("data:"):
            # Already a data URI
            return image
        elif os.path.isfile(image):
            # File path
            return _encode_image_from_file(image)
        elif is_url(image):
            # URL
            if download_images:
                return _encode_image_from_url(image)
            else:
                # Return the URL as is
                return image
        else:
            # Unsupported string format
            print(f"Unsupported file string: {image}")
            raise ValueError(f"Unsupported file string: {image}")
    elif PIL_AVAILABLE and isinstance(image, PILImage.Image):
        # PIL Image
        return _encode_pil_image(image)
    elif isinstance(image, bytes):
        # Raw bytes
        if not PIL_AVAILABLE:
            raise ImportError("Pillow is required to process image bytes.")
        img = PILImage.open(io.BytesIO(image))
        return _encode_pil_image(img)
    elif isinstance(image, Image):
        return image.url
    else:
        print(f"Unsupported image type: {type(image)}")
        raise ValueError(f"Unsupported image type: {type(image)}")


def _encode_image_from_file(file_path: str) -> str:
    """Encode a file from a file path to a base64 data URI."""
    with open(file_path, "rb") as file:
        file_data = file.read()

    # Use mimetypes to guess directly from the file path
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for file: {file_path}")

    encoded_data = base64.b64encode(file_data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_data}"


def _encode_image_from_url(image_url: str) -> str:
    """Encode a file from a URL to a base64 data URI."""
    response = requests.get(image_url)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")

    # Use the content type from the response headers if available
    if content_type:
        mime_type = content_type
    else:
        # Try to guess MIME type from URL
        mime_type, _ = mimetypes.guess_type(image_url)
        if mime_type is None:
            raise ValueError(f"Could not determine MIME type for URL: {image_url}")

    encoded_data = base64.b64encode(response.content).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_data}"


def _encode_pil_image(image: "PILImage") -> str:
    """Encode a PIL Image object to a base64 data URI."""
    buffered = io.BytesIO()
    file_format = image.format or "PNG"
    image.save(buffered, format=file_format)

    # Get the correct MIME type using the image format
    file_extension = file_format.lower()
    mime_type, _ = mimetypes.guess_type(f"file.{file_extension}")
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for image format: {file_format}")

    encoded_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_data}"


def _get_file_extension(path_or_url: str) -> str:
    """Extract the file extension from a file path or URL."""
    extension = os.path.splitext(urlparse(path_or_url).path)[1].lstrip(".").lower()
    return extension or "png"  # Default to 'png' if no extension found


def is_image(obj) -> bool:
    """Check if the object is an image or a valid media file reference."""
    if PIL_AVAILABLE and isinstance(obj, PILImage.Image):
        return True
    if isinstance(obj, str):
        if obj.startswith("data:"):
            return True
        elif os.path.isfile(obj):
            return True
        elif is_url(obj):
            return True
    return False
```

## File: `adapters/types/tool.py`
```
import asyncio
import inspect
from typing import TYPE_CHECKING, Any, Callable, Type, get_origin, get_type_hints

from jsonschema import ValidationError, validate
from pydantic import BaseModel, TypeAdapter, create_model

from dspy.adapters.types.base_type import Type
from dspy.dsp.utils.settings import settings
from dspy.utils.callback import with_callbacks

if TYPE_CHECKING:
    import mcp
    from langchain.tools import BaseTool

_TYPE_MAPPING = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list, "object": dict}


class Tool(Type):
    """Tool class.

    This class is used to simplify the creation of tools for tool calling (function calling) in LLMs. Only supports
    functions for now.
    """

    func: Callable
    name: str | None = None
    desc: str | None = None
    args: dict[str, Any] | None = None
    arg_types: dict[str, Any] | None = None
    arg_desc: dict[str, str] | None = None
    has_kwargs: bool = False

    def __init__(
        self,
        func: Callable,
        name: str | None = None,
        desc: str | None = None,
        args: dict[str, Any] | None = None,
        arg_types: dict[str, Any] | None = None,
        arg_desc: dict[str, str] | None = None,
    ):
        """Initialize the Tool class.

        Users can choose to specify the `name`, `desc`, `args`, and `arg_types`, or let the `dspy.Tool`
        automatically infer the values from the function. For values that are specified by the user, automatic inference
        will not be performed on them.

        Args:
            func (Callable): The actual function that is being wrapped by the tool.
            name (Optional[str], optional): The name of the tool. Defaults to None.
            desc (Optional[str], optional): The description of the tool. Defaults to None.
            args (Optional[dict[str, Any]], optional): The args and their schema of the tool, represented as a
                dictionary from arg name to arg's json schema. Defaults to None.
            arg_types (Optional[dict[str, Any]], optional): The argument types of the tool, represented as a dictionary
                from arg name to the type of the argument. Defaults to None.
            arg_desc (Optional[dict[str, str]], optional): Descriptions for each arg, represented as a
                dictionary from arg name to description string. Defaults to None.

        Example:

        ```python
        def foo(x: int, y: str = "hello"):
            return str(x) + y

        tool = Tool(foo)
        print(tool.args)
        # Expected output: {'x': {'type': 'integer'}, 'y': {'type': 'string', 'default': 'hello'}}
        ```
        """
        super().__init__(func=func, name=name, desc=desc, args=args, arg_types=arg_types, arg_desc=arg_desc)
        self._parse_function(func, arg_desc)

    def _parse_function(self, func: Callable, arg_desc: dict[str, str] | None = None):
        """Helper method that parses a function to extract the name, description, and args.

        This is a helper function that automatically infers the name, description, and args of the tool from the
        provided function. In order to make the inference work, the function must have valid type hints.
        """
        annotations_func = func if inspect.isfunction(func) or inspect.ismethod(func) else func.__call__
        name = getattr(func, "__name__", type(func).__name__)
        desc = getattr(func, "__doc__", None) or getattr(annotations_func, "__doc__", "")
        args = {}
        arg_types = {}

        # Use inspect.signature to get all arg names
        sig = inspect.signature(annotations_func)
        # Get available type hints
        available_hints = get_type_hints(annotations_func)
        # Build a dictionary of arg name -> type (defaulting to Any when missing)
        hints = {param_name: available_hints.get(param_name, Any) for param_name in sig.parameters.keys()}
        default_values = {param_name: sig.parameters[param_name].default for param_name in sig.parameters.keys()}

        # Process each argument's type to generate its JSON schema.
        for k, v in hints.items():
            arg_types[k] = v
            if k == "return":
                continue
            # Check if the type (or its origin) is a subclass of Pydantic's BaseModel
            origin = get_origin(v) or v
            if isinstance(origin, type) and issubclass(origin, BaseModel):
                # Get json schema, and replace $ref with the actual schema
                v_json_schema = _resolve_json_schema_reference(v.model_json_schema())
                args[k] = v_json_schema
            else:
                args[k] = _resolve_json_schema_reference(TypeAdapter(v).json_schema())
            if default_values[k] is not inspect.Parameter.empty:
                args[k]["default"] = default_values[k]
            if arg_desc and k in arg_desc:
                args[k]["description"] = arg_desc[k]

        self.name = self.name or name
        self.desc = self.desc or desc
        self.args = self.args or args
        self.arg_types = self.arg_types or arg_types
        self.has_kwargs = any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values())

    def _validate_and_parse_args(self, **kwargs):
        # Validate the args value comply to the json schema.
        for k, v in kwargs.items():
            if k not in self.args:
                if self.has_kwargs:
                    continue
                else:
                    raise ValueError(f"Arg {k} is not in the tool's args.")
            try:
                instance = v.model_dump() if hasattr(v, "model_dump") else v
                type_str = self.args[k].get("type")
                if type_str is not None and type_str != "Any":
                    validate(instance=instance, schema=self.args[k])
            except ValidationError as e:
                raise ValueError(f"Arg {k} is invalid: {e.message}")

        # Parse the args to the correct type.
        parsed_kwargs = {}
        for k, v in kwargs.items():
            if k in self.arg_types and self.arg_types[k] != Any:
                # Create a pydantic model wrapper with a dummy field `value` to parse the arg to the correct type.
                # This is specifically useful for handling nested Pydantic models like `list[list[MyPydanticModel]]`
                pydantic_wrapper = create_model("Wrapper", value=(self.arg_types[k], ...))
                parsed = pydantic_wrapper.model_validate({"value": v})
                parsed_kwargs[k] = parsed.value
            else:
                parsed_kwargs[k] = v
        return parsed_kwargs

    def format(self):
        return str(self)

    def format_as_litellm_function_call(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.desc,
                "parameters": {
                    "type": "object",
                    "properties": self.args,
                    "required": list(self.args.keys()),
                },
            },
        }

    def _run_async_in_sync(self, coroutine):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)

        return loop.run_until_complete(coroutine)

    @with_callbacks
    def __call__(self, **kwargs):
        parsed_kwargs = self._validate_and_parse_args(**kwargs)
        result = self.func(**parsed_kwargs)
        if asyncio.iscoroutine(result):
            if settings.allow_tool_async_sync_conversion:
                return self._run_async_in_sync(result)
            else:
                raise ValueError(
                    "You are calling `__call__` on an async tool, please use `acall` instead or set "
                    "`allow_async=True` to run the async tool in sync mode."
                )
        return result

    @with_callbacks
    async def acall(self, **kwargs):
        parsed_kwargs = self._validate_and_parse_args(**kwargs)
        result = self.func(**parsed_kwargs)
        if asyncio.iscoroutine(result):
            return await result
        else:
            # We should allow calling a sync tool in the async path.
            return result

    @classmethod
    def from_mcp_tool(cls, session: "mcp.client.session.ClientSession", tool: "mcp.types.Tool") -> "Tool":
        """
        Build a DSPy tool from an MCP tool and a ClientSession.

        Args:
            session: The MCP session to use.
            tool: The MCP tool to convert.

        Returns:
            A Tool object.
        """
        from dspy.utils.mcp import convert_mcp_tool

        return convert_mcp_tool(session, tool)

    @classmethod
    def from_langchain(cls, tool: "BaseTool") -> "Tool":
        """
        Build a DSPy tool from a LangChain tool.

        Args:
            tool: The LangChain tool to convert.

        Returns:
            A Tool object.

        Example:

        ```python
        import asyncio
        import dspy
        from langchain.tools import tool as lc_tool

        @lc_tool
        def add(x: int, y: int):
            "Add two numbers together."
            return x + y

        dspy_tool = dspy.Tool.from_langchain(add)

        async def run_tool():
            return await dspy_tool.acall(x=1, y=2)

        print(asyncio.run(run_tool()))
        # 3
        ```
        """
        from dspy.utils.langchain_tool import convert_langchain_tool

        return convert_langchain_tool(tool)

    def __repr__(self):
        return f"Tool(name={self.name}, desc={self.desc}, args={self.args})"

    def __str__(self):
        desc = f", whose description is <desc>{self.desc}</desc>.".replace("\n", "  ") if self.desc else "."
        arg_desc = f"It takes arguments {self.args}."
        return f"{self.name}{desc} {arg_desc}"


class ToolCalls(Type):
    class ToolCall(BaseModel):
        name: str
        args: dict[str, Any]

    tool_calls: list[ToolCall]

    @classmethod
    def from_dict_list(cls, tool_calls_dicts: list[dict[str, Any]]) -> "ToolCalls":
        """Convert a list of dictionaries to a ToolCalls instance.

        Args:
            dict_list: A list of dictionaries, where each dictionary should have 'name' and 'args' keys.

        Returns:
            A ToolCalls instance.

        Example:

            ```python
            tool_calls_dict = [
                {"name": "search", "args": {"query": "hello"}},
                {"name": "translate", "args": {"text": "world"}}
            ]
            tool_calls = ToolCalls.from_dict_list(tool_calls_dict)
            ```
        """
        tool_calls = [cls.ToolCall(**item) for item in tool_calls_dicts]
        return cls(tool_calls=tool_calls)

    @classmethod
    def description(cls) -> str:
        return (
            "Tool calls information, including the name of the tools and the arguments to be passed to it. "
            "Arguments must be provided in JSON format."
        )

    def format(self) -> list[dict[str, Any]]:
        # The tool_call field is compatible with OpenAI's tool calls schema.
        return [
            {
                "type": "tool_calls",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.args,
                        },
                    }
                    for tool_call in self.tool_calls
                ],
            }
        ]


def _resolve_json_schema_reference(schema: dict) -> dict:
    """Recursively resolve json model schema, expanding all references."""

    # If there are no definitions to resolve, return the main schema
    if "$defs" not in schema and "definitions" not in schema:
        return schema

    def resolve_refs(obj: Any) -> Any:
        if not isinstance(obj, (dict, list)):
            return obj
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_path = obj["$ref"].split("/")[-1]
                return resolve_refs(schema["$defs"][ref_path])
            return {k: resolve_refs(v) for k, v in obj.items()}

        # Must be a list
        return [resolve_refs(item) for item in obj]

    # Resolve all references in the main schema
    resolved_schema = resolve_refs(schema)
    # Remove the $defs key as it's no longer needed
    resolved_schema.pop("$defs", None)
    return resolved_schema


def convert_input_schema_to_tool_args(
    schema: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Type], dict[str, str]]:
    """Convert an input json schema to tool arguments compatible with DSPy Tool.

    Args:
        schema: An input json schema describing the tool's input parameters

    Returns:
        A tuple of (args, arg_types, arg_desc) for DSPy Tool definition.
    """
    args, arg_types, arg_desc = {}, {}, {}
    properties = schema.get("properties", None)
    if properties is None:
        return args, arg_types, arg_desc

    required = schema.get("required", [])

    defs = schema.get("$defs", {})

    for name, prop in properties.items():
        if len(defs) > 0:
            prop = _resolve_json_schema_reference({"$defs": defs, **prop})
        args[name] = prop
        arg_types[name] = _TYPE_MAPPING.get(prop.get("type"), Any)
        arg_desc[name] = prop.get("description", "No description provided.")
        if name in required:
            arg_desc[name] += " (Required)"

    return args, arg_types, arg_desc
```

## File: `adapters/utils.py`
```
import ast
import enum
import inspect
import json
from collections.abc import Mapping
from typing import Any, Literal, Union, get_args, get_origin

import json_repair
import pydantic
from pydantic import TypeAdapter
from pydantic.fields import FieldInfo

from dspy.adapters.types.base_type import Type
from dspy.signatures.utils import get_dspy_field_type


def serialize_for_json(value: Any) -> Any:
    """
    Formats the specified value so that it can be serialized as a JSON string.

    Args:
        value: The value to format as a JSON string.
    Returns:
        The formatted value, which is serializable as a JSON string.
    """
    # Attempt to format the value as a JSON-compatible object using pydantic, falling back to
    # a string representation of the value if that fails (e.g. if the value contains an object
    # that pydantic doesn't recognize or can't serialize)
    try:
        return TypeAdapter(type(value)).dump_python(value, mode="json")
    except Exception:
        return str(value)


def format_field_value(field_info: FieldInfo, value: Any, assume_text=True) -> str | dict:
    """
    Formats the value of the specified field according to the field's DSPy type (input or output),
    annotation (e.g. str, int, etc.), and the type of the value itself.

    Args:
      field_info: Information about the field, including its DSPy field type and annotation.
      value: The value of the field.
    Returns:
      The formatted value of the field, represented as a string.
    """
    string_value = None
    if isinstance(value, list) and field_info.annotation is str:
        # If the field has no special type requirements, format it as a nice numbered list for the LM.
        string_value = _format_input_list_field_value(value)
    else:
        jsonable_value = serialize_for_json(value)
        if isinstance(jsonable_value, dict) or isinstance(jsonable_value, list):
            string_value = json.dumps(jsonable_value, ensure_ascii=False)
        else:
            # If the value is not a Python representation of a JSON object or Array
            # (e.g. the value is a JSON string), just use the string representation of the value
            # to avoid double-quoting the JSON string (which would hurt accuracy for certain
            # tasks, e.g. tasks that rely on computing string length)
            string_value = str(jsonable_value)

    if assume_text:
        return string_value
    else:
        return {"type": "text", "text": string_value}


def _get_json_schema(field_type):
    def move_type_to_front(d):
        # Move the 'type' key to the front of the dictionary, recursively, for LLM readability/adherence.
        if isinstance(d, Mapping):
            return {
                k: move_type_to_front(v) for k, v in sorted(d.items(), key=lambda item: (item[0] != "type", item[0]))
            }
        elif isinstance(d, list):
            return [move_type_to_front(item) for item in d]
        return d

    schema = pydantic.TypeAdapter(field_type).json_schema()
    schema = move_type_to_front(schema)
    return schema


def translate_field_type(field_name, field_info):
    field_type = field_info.annotation

    if get_dspy_field_type(field_info) == "input" or field_type is str:
        desc = ""
    elif field_type is bool:
        desc = "must be True or False"
    elif field_type in (int, float):
        desc = f"must be a single {field_type.__name__} value"
    elif inspect.isclass(field_type) and issubclass(field_type, enum.Enum):
        enum_vals = "; ".join(str(member.value) for member in field_type)
        desc = f"must be one of: {enum_vals}"
    elif hasattr(field_type, "__origin__") and field_type.__origin__ is Literal:
        desc = (
            # Strongly encourage the LM to avoid choosing values that don't appear in the
            # literal or returning a value of the form 'Literal[<selected_value>]'
            f"must exactly match (no extra characters) one of: {'; '.join([str(x) for x in field_type.__args__])}"
        )
    else:
        desc = f"must adhere to the JSON schema: {json.dumps(_get_json_schema(field_type), ensure_ascii=False)}"

    desc = (" " * 8) + f"# note: the value you produce {desc}" if desc else ""
    return f"{{{field_name}}}{desc}"


def find_enum_member(enum, identifier):
    """
    Finds the enum member corresponding to the specified identifier, which may be the
    enum member's name or value.

    Args:
        enum: The enum to search for the member.
        identifier: If the enum is explicitly-valued, this is the value of the enum member to find.
                    If the enum is auto-valued, this is the name of the enum member to find.
    Returns:
        The enum member corresponding to the specified identifier.
    """
    # Check if the identifier is a valid enum member value *before* checking if it's a valid enum
    # member name, since the identifier will be a value for explicitly-valued enums. This handles
    # the (rare) case where an enum member value is the same as another enum member's name in
    # an explicitly-valued enum
    for member in enum:
        if member.value == identifier:
            return member

    # If the identifier is not a valid enum member value, check if it's a valid enum member name,
    # since the identifier will be a member name for auto-valued enums
    if identifier in enum.__members__:
        return enum[identifier]

    raise ValueError(f"{identifier} is not a valid name or value for the enum {enum.__name__}")


def parse_value(value, annotation):
    if annotation is str:
        return str(value)

    if isinstance(annotation, enum.EnumMeta):
        return find_enum_member(annotation, value)

    origin = get_origin(annotation)

    if origin is Literal:
        allowed = get_args(annotation)
        if value in allowed:
            return value

        if isinstance(value, str):
            v = value.strip()
            if v.startswith(("Literal[", "str[")) and v.endswith("]"):
                v = v[v.find("[") + 1 : -1]
            if len(v) > 1 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]

            if v in allowed:
                return v

        raise ValueError(f"{value!r} is not one of {allowed!r}")

    if not isinstance(value, str):
        return TypeAdapter(annotation).validate_python(value)

    candidate = json_repair.loads(value)  # json_repair.loads returns "" on failure.
    if candidate == "" and value != "":
        try:
            candidate = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            candidate = value

    try:
        return TypeAdapter(annotation).validate_python(candidate)
    except pydantic.ValidationError as e:
        if issubclass(annotation, Type):
            try:
                # For dspy.Type, try parsing from the original value in case it has a custom parser
                return TypeAdapter(annotation).validate_python(value)
            except Exception:
                raise e

        if origin is Union and type(None) in get_args(annotation) and str in get_args(annotation):
            return str(candidate)
        raise


def get_annotation_name(annotation):
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is None:
        if hasattr(annotation, "__name__"):
            return annotation.__name__
        else:
            return str(annotation)

    if origin is Literal:
        args_str = ", ".join(
            _quoted_string_for_literal_type_annotation(a) if isinstance(a, str) else get_annotation_name(a)
            for a in args
        )
        return f"{get_annotation_name(origin)}[{args_str}]"
    else:
        args_str = ", ".join(get_annotation_name(a) for a in args)
        return f"{get_annotation_name(origin)}[{args_str}]"


def get_field_description_string(fields: dict) -> str:
    field_descriptions = []
    for idx, (k, v) in enumerate(fields.items()):
        field_message = f"{idx + 1}. `{k}`"
        field_message += f" ({get_annotation_name(v.annotation)})"
        desc = v.json_schema_extra["desc"] if v.json_schema_extra["desc"] != f"${{{k}}}" else ""

        custom_types = Type.extract_custom_type_from_annotation(v.annotation)
        for custom_type in custom_types:
            if len(custom_type.description()) > 0:
                desc += f"\n    Type description of {get_annotation_name(custom_type)}: {custom_type.description()}"

        field_message += f": {desc}"
        field_message += (
            f"\nConstraints: {v.json_schema_extra['constraints']}" if v.json_schema_extra.get("constraints") else ""
        )
        field_descriptions.append(field_message)
    return "\n".join(field_descriptions).strip()


def _format_input_list_field_value(value: list[Any]) -> str:
    """
    Formats the value of an input field of type list[Any].

    Args:
      value: The value of the list-type input field.
    Returns:
      A string representation of the input field's list value.
    """
    if len(value) == 0:
        return "N/A"
    if len(value) == 1:
        return _format_blob(value[0])

    return "\n".join([f"[{idx + 1}] {_format_blob(txt)}" for idx, txt in enumerate(value)])


def _format_blob(blob: str) -> str:
    """
    Formats the specified text blobs so that an LM can parse it correctly within a list
    of multiple text blobs.

    Args:
        blob: The text blob to format.
    Returns:
        The formatted text blob.
    """
    if "\n" not in blob and "«" not in blob and "»" not in blob:
        return f"«{blob}»"

    modified_blob = blob.replace("\n", "\n    ")
    return f"«««\n    {modified_blob}\n»»»"


def _quoted_string_for_literal_type_annotation(s: str) -> str:
    """
    Return the specified string quoted for inclusion in a literal type annotation.
    """
    has_single = "'" in s
    has_double = '"' in s

    if has_single and not has_double:
        # Only single quotes => enclose in double quotes
        return f'"{s}"'
    elif has_double and not has_single:
        # Only double quotes => enclose in single quotes
        return f"'{s}'"
    elif has_single and has_double:
        # Both => enclose in single quotes; escape each single quote with \'
        escaped = s.replace("'", "\\'")
        return f"'{escaped}'"
    else:
        # Neither => enclose in single quotes
        return f"'{s}'"
```

## File: `adapters/xml_adapter.py`
```
import re
from typing import Any

from dspy.adapters.chat_adapter import ChatAdapter, FieldInfoWithName
from dspy.adapters.utils import format_field_value
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback


class XMLAdapter(ChatAdapter):
    def __init__(self, callbacks: list[BaseCallback] | None = None):
        super().__init__(callbacks)
        self.field_pattern = re.compile(r"<(?P<name>\w+)>((?P<content>.*?))</\1>", re.DOTALL)

    def format_field_with_value(self, fields_with_values: dict[FieldInfoWithName, Any]) -> str:
        output = []
        for field, field_value in fields_with_values.items():
            formatted = format_field_value(field_info=field.info, value=field_value)
            output.append(f"<{field.name}>\n{formatted}\n</{field.name}>")
        return "\n\n".join(output).strip()

    def user_message_output_requirements(self, signature: type[Signature]) -> str:
        message = "Respond with the corresponding output fields wrapped in XML tags"
        message += ", then ".join(f"`<{f}>`" for f in signature.output_fields)
        message += ", and then end with the `<completed>` tag."
        return message

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        fields = {}
        for match in self.field_pattern.finditer(completion):
            name = match.group("name")
            content = match.group("content").strip()
            if name in signature.output_fields and name not in fields:
                fields[name] = content
        # Cast values using base class parse_value helper
        for k, v in fields.items():
            fields[k] = self._parse_field_value(signature.output_fields[k], v, completion, signature)
        if fields.keys() != signature.output_fields.keys():
            from dspy.utils.exceptions import AdapterParseError

            raise AdapterParseError(
                adapter_name="XMLAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=fields,
            )
        return fields

    def _parse_field_value(self, field_info, raw, completion, signature):
        from dspy.adapters.utils import parse_value

        try:
            return parse_value(raw, field_info.annotation)
        except Exception as e:
            from dspy.utils.exceptions import AdapterParseError

            raise AdapterParseError(
                adapter_name="XMLAdapter",
                signature=signature,
                lm_response=completion,
                message=f"Failed to parse field {field_info} with value {raw}: {e}",
            )
```

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

## File: `datasets/__init__.py`
```
from dspy.datasets.alfworld import AlfWorld
from dspy.datasets.colors import Colors
from dspy.datasets.dataloader import DataLoader
from dspy.datasets.dataset import Dataset
from dspy.datasets.hotpotqa import HotPotQA
from dspy.datasets.math import MATH

__all__ = [
    "Colors",
    "DataLoader",
    "Dataset",
    "HotPotQA",
    "MATH",
]
```

## File: `datasets/alfworld/__init__.py`
```
from dspy.datasets.alfworld.alfworld import AlfWorld
```

## File: `datasets/alfworld/alfworld.py`
```
import os
import queue
import random


def env_worker(inq, outq):
    """
    Worker process: creates a single AlfredTWEnv instance,
    handles 'init' (with task idx) and 'step' (with action).
    """

    try:
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import alfworld.agents.environment as environment
        import yaml
    except ImportError:
        raise ImportError(
            "alfworld is not installed. "
            "Please install it via `pip install alfworld==0.3.5` then run `alfworld-download`."
        )

    buf = io.StringIO()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "base_config.yml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    with redirect_stdout(buf), redirect_stderr(buf):
        base_env = environment.AlfredTWEnv(config, train_eval="train")

    env = None
    while True:
        cmd, data = inq.get()
        if cmd == "init":
            env = base_env.init_env(batch_size=1)
            env.skip(data)
            task_def, info = env.reset()
            outq.put((task_def[0], info))
        elif cmd == "step":
            obs, rew, done, info = env.step([data])
            outq.put((obs, rew, done, info))
        elif cmd == "close":
            outq.put("CLOSED")
            break
        else:
            outq.put("UNKNOWN_CMD")


class EnvPool:
    """
    Pool of processes, each with a unique env_worker.
    Acquire a worker using a context manager for safe usage:
        with pool.session() as sess:
            sess.init(5)              # init with idx=5
            obs, rew, done, info = sess.step("go north")
            ...
    """

    def __init__(self, size=2):
        self.size = size
        self.workers = []
        self.available = queue.Queue()

        try:
            import multiprocess as mp
        except ImportError:
            raise ImportError("multiprocess is not installed. " "Please install it via `pip install multiprocess`.")

        # Must call set_start_method('spawn') here, before creating any processes
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            # If it's already set, ignore
            pass

        ctx = mp.get_context("spawn")
        for i in range(size):
            inq = ctx.Queue()
            outq = ctx.Queue()
            p = ctx.Process(target=env_worker, args=(inq, outq), daemon=True)
            p.start()
            self.workers.append((inq, outq, p))
            self.available.put(i)

    def _acquire(self):
        wid = self.available.get()
        return wid, self.workers[wid][0], self.workers[wid][1]

    def _release(self, wid):
        self.available.put(wid)

    def close_all(self):
        """Close all processes in the pool."""
        while not self.available.empty():
            wid = self.available.get()
            inq, outq, proc = self.workers[wid]
            inq.put(("close", None))
            outq.get()  # Wait 'CLOSED'
            inq.close()
            outq.close()
            proc.join()

    def session(self):
        """Context manager that acquires/releases a single worker."""
        return _EnvSession(self)


class _EnvSession:
    """
    A context manager that acquires a worker from the pool,
    provides .init(idx) and .step(action), then releases the worker.
    """

    def __init__(self, pool: EnvPool):
        self.pool = pool
        self.wid = None
        self.inq = None
        self.outq = None

    def __enter__(self):
        self.wid, self.inq, self.outq = self.pool._acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool._release(self.wid)

    def init(self, idx):
        self.inq.put(("init", idx))
        return self.outq.get()  # (task_def, info)

    def step(self, action):
        self.inq.put(("step", action))
        return self.outq.get()  # (obs, rew, done, info)


class AlfWorld:
    def __init__(self, max_threads=20):
        self.POOL = EnvPool(size=max_threads)

        import dspy

        dataset = [dspy.Example(idx=idx).with_inputs("idx") for idx in range(3500)]
        random.Random(0).shuffle(dataset)

        trainset, devset = dataset[:3000], dataset[-500:]
        assert len(trainset) + len(devset) <= len(dataset)

        self.trainset = trainset
        self.devset = devset

    def __del__(self):
        self.POOL.close_all()
```

## File: `datasets/colors.py`
```
import random

from dspy.datasets.dataset import Dataset

### A bunch of colors, originally from matplotlib
all_colors = [
    "alice blue",
    "dodger blue",
    "light sky blue",
    "deep sky blue",
    "sky blue",
    "steel blue",
    "light steel blue",
    "medium blue",
    "navy blue",
    "blue",
    "royal blue",
    "cadet blue",
    "cornflower blue",
    "medium slate blue",
    "slate blue",
    "dark slate blue",
    "powder blue",
    "turquoise",
    "dark turquoise",
    "medium turquoise",
    "pale turquoise",
    "light sea green",
    "medium sea green",
    "sea green",
    "forest green",
    "green yellow",
    "lime green",
    "dark green",
    "green",
    "lime",
    "chartreuse",
    "lawn green",
    "yellow green",
    "olive green",
    "dark olive green",
    "medium spring green",
    "spring green",
    "medium aquamarine",
    "aquamarine",
    "aqua",
    "cyan",
    "dark cyan",
    "teal",
    "medium orchid",
    "dark orchid",
    "orchid",
    "blue violet",
    "violet",
    "dark violet",
    "plum",
    "thistle",
    "magenta",
    "fuchsia",
    "dark magenta",
    "medium purple",
    "purple",
    "rebecca purple",
    "dark red",
    "fire brick",
    "indian red",
    "light coral",
    "dark salmon",
    "light salmon",
    "salmon",
    "red",
    "crimson",
    "tomato",
    "coral",
    "orange red",
    "dark orange",
    "orange",
    "yellow",
    "gold",
    "light goldenrod yellow",
    "pale goldenrod",
    "goldenrod",
    "dark goldenrod",
    "beige",
    "moccasin",
    "blanched almond",
    "navajo white",
    "antique white",
    "bisque",
    "burlywood",
    "dark khaki",
    "khaki",
    "tan",
    "wheat",
    "snow",
    "floral white",
    "old lace",
    "ivory",
    "linen",
    "seashell",
    "honeydew",
    "mint cream",
    "azure",
    "lavender",
    "ghost white",
    "white smoke",
    "gainsboro",
    "light gray",
    "silver",
    "dark gray",
    "gray",
    "dim gray",
    "slate gray",
    "light slate gray",
    "dark slate gray",
    "black",
    "medium violet red",
    "pale violet red",
    "deep pink",
    "hot pink",
    "light pink",
    "pink",
    "peach puff",
    "rosy brown",
    "saddle brown",
    "sandy brown",
    "chocolate",
    "peru",
    "sienna",
    "brown",
    "maroon",
    "white",
    "misty rose",
    "lavender blush",
    "papaya whip",
    "lemon chiffon",
    "light yellow",
    "corn silk",
    "pale green",
    "light green",
    "olive drab",
    "olive",
    "dark sea green",
]


class Colors(Dataset):
    def __init__(self, sort_by_suffix=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.sort_by_suffix = sort_by_suffix
        colors = self.sorted_by_suffix(all_colors)

        train_size = int(
            len(colors) * 0.6
        )  # chosen to ensure that similar colors aren't repeated between train and dev
        train_colors, dev_colors = colors[:train_size], colors[train_size:]

        self._train = [{"color": color} for color in train_colors]
        self._dev = [{"color": color} for color in dev_colors]

        random.Random(0).shuffle(self._train)
        random.Random(0).shuffle(self._dev)

    def sorted_by_suffix(self, colors):
        if not self.sort_by_suffix:
            return colors

        if isinstance(colors[0], str):
            sorted_colors = sorted(colors, key=lambda x: x[::-1])
        else:
            sorted_colors = sorted(colors, key=lambda x: x["color"][::-1])

        return sorted_colors
```

## File: `datasets/dataloader.py`
```
import random
from collections.abc import Mapping
from typing import TYPE_CHECKING

import dspy
from dspy.datasets.dataset import Dataset

if TYPE_CHECKING:
    import pandas as pd


class DataLoader(Dataset):
    def __init__(self):
        pass

    def from_huggingface(
        self,
        dataset_name: str,
        *args,
        input_keys: tuple[str] = (),
        fields: tuple[str] | None = None,
        **kwargs,
    ) -> Mapping[str, list[dspy.Example]] | list[dspy.Example]:
        if fields and not isinstance(fields, tuple):
            raise ValueError("Invalid fields provided. Please provide a tuple of fields.")

        if not isinstance(input_keys, tuple):
            raise ValueError("Invalid input keys provided. Please provide a tuple of input keys.")

        from datasets import load_dataset

        dataset = load_dataset(dataset_name, *args, **kwargs)

        if isinstance(dataset, list) and isinstance(kwargs["split"], list):
            dataset = {split_name: dataset[idx] for idx, split_name in enumerate(kwargs["split"])}

        try:
            returned_split = {}
            for split_name in dataset.keys():
                if fields:
                    returned_split[split_name] = [
                        dspy.Example({field: row[field] for field in fields}).with_inputs(*input_keys)
                        for row in dataset[split_name]
                    ]
                else:
                    returned_split[split_name] = [
                        dspy.Example({field: row[field] for field in row.keys()}).with_inputs(*input_keys)
                        for row in dataset[split_name]
                    ]

            return returned_split
        except AttributeError:
            if fields:
                return [
                    dspy.Example({field: row[field] for field in fields}).with_inputs(*input_keys) for row in dataset
                ]
            else:
                return [
                    dspy.Example({field: row[field] for field in row.keys()}).with_inputs(*input_keys)
                    for row in dataset
                ]

    def from_csv(
        self,
        file_path: str,
        fields: list[str] | None = None,
        input_keys: tuple[str] = (),
    ) -> list[dspy.Example]:
        from datasets import load_dataset

        dataset = load_dataset("csv", data_files=file_path)["train"]

        if not fields:
            fields = list(dataset.features)

        return [dspy.Example({field: row[field] for field in fields}).with_inputs(*input_keys) for row in dataset]

    def from_pandas(
        self,
        df: "pd.DataFrame",
        fields: list[str] | None = None,
        input_keys: tuple[str] = (),
    ) -> list[dspy.Example]:
        if fields is None:
            fields = list(df.columns)

        return [
            dspy.Example({field: row[field] for field in fields}).with_inputs(*input_keys) for _, row in df.iterrows()
        ]

    def from_json(
        self,
        file_path: str,
        fields: list[str] | None = None,
        input_keys: tuple[str] = (),
    ) -> list[dspy.Example]:
        from datasets import load_dataset

        dataset = load_dataset("json", data_files=file_path)["train"]

        if not fields:
            fields = list(dataset.features)

        return [dspy.Example({field: row[field] for field in fields}).with_inputs(*input_keys) for row in dataset]

    def from_parquet(
        self,
        file_path: str,
        fields: list[str] | None = None,
        input_keys: tuple[str] = (),
    ) -> list[dspy.Example]:
        from datasets import load_dataset

        dataset = load_dataset("parquet", data_files=file_path)["train"]

        if not fields:
            fields = list(dataset.features)

        return [dspy.Example({field: row[field] for field in fields}).with_inputs(*input_keys) for row in dataset]

    def from_rm(self, num_samples: int, fields: list[str], input_keys: list[str]) -> list[dspy.Example]:
        try:
            rm = dspy.settings.rm
            try:
                return [
                    dspy.Example({field: row[field] for field in fields}).with_inputs(*input_keys)
                    for row in rm.get_objects(num_samples=num_samples, fields=fields)
                ]
            except AttributeError:
                raise ValueError(
                    "Retrieval module does not support `get_objects`. Please use a different retrieval module."
                )
        except AttributeError:
            raise ValueError(
                "Retrieval module not found. Please set a retrieval module using `dspy.settings.configure`."
            )

    def sample(
        self,
        dataset: list[dspy.Example],
        n: int,
        *args,
        **kwargs,
    ) -> list[dspy.Example]:
        if not isinstance(dataset, list):
            raise ValueError(
                f"Invalid dataset provided of type {type(dataset)}. Please provide a list of `dspy.Example`s."
            )

        return random.sample(dataset, n, *args, **kwargs)

    def train_test_split(
        self,
        dataset: list[dspy.Example],
        train_size: int | float = 0.75,
        test_size: int | float | None = None,
        random_state: int | None = None,
    ) -> Mapping[str, list[dspy.Example]]:
        if random_state is not None:
            random.seed(random_state)

        dataset_shuffled = dataset.copy()
        random.shuffle(dataset_shuffled)

        if train_size is not None and isinstance(train_size, float) and (0 < train_size < 1):
            train_end = int(len(dataset_shuffled) * train_size)
        elif train_size is not None and isinstance(train_size, int):
            train_end = train_size
        else:
            raise ValueError(
                "Invalid `train_size`. Please provide a float between 0 and 1 to represent the proportion of the "
                "dataset to include in the train split or an int to represent the absolute number of samples to "
                f"include in the train split. Received `train_size`: {train_size}."
            )

        if test_size is not None:
            if isinstance(test_size, float) and (0 < test_size < 1):
                test_end = int(len(dataset_shuffled) * test_size)
            elif isinstance(test_size, int):
                test_end = test_size
            else:
                raise ValueError(
                    "Invalid `test_size`. Please provide a float between 0 and 1 to represent the proportion of the "
                    "dataset to include in the test split or an int to represent the absolute number of samples to "
                    f"include in the test split. Received `test_size`: {test_size}."
                )
            if train_end + test_end > len(dataset_shuffled):
                raise ValueError(
                    "`train_size` + `test_size` cannot exceed the total number of samples. Received "
                    f"`train_size`: {train_end}, `test_size`: {test_end}, and `dataset_size`: {len(dataset_shuffled)}."
                )
        else:
            test_end = len(dataset_shuffled) - train_end

        train_dataset = dataset_shuffled[:train_end]
        test_dataset = dataset_shuffled[train_end : train_end + test_end]

        return {"train": train_dataset, "test": test_dataset}
```

## File: `datasets/dataset.py`
```
import random
import uuid

from dspy import Example
from dspy.dsp.utils import dotdict


class Dataset:
    def __init__(self, train_seed=0, train_size=None, eval_seed=0, dev_size=None, test_size=None, input_keys=None):
        self.train_size = train_size
        self.train_seed = train_seed
        self.dev_size = dev_size
        self.dev_seed = eval_seed
        self.test_size = test_size
        self.test_seed = eval_seed
        self.input_keys = input_keys or []

        self.do_shuffle = True

        self.name = self.__class__.__name__

    def reset_seeds(self, train_seed=None, train_size=None, eval_seed=None, dev_size=None, test_size=None):
        self.train_size = train_size or self.train_size
        self.train_seed = train_seed or self.train_seed
        self.dev_size = dev_size or self.dev_size
        self.dev_seed = eval_seed or self.dev_seed
        self.test_size = test_size or self.test_size
        self.test_seed = eval_seed or self.test_seed

        if hasattr(self, "_train_"):
            del self._train_

        if hasattr(self, "_dev_"):
            del self._dev_

        if hasattr(self, "_test_"):
            del self._test_

    @property
    def train(self):
        if not hasattr(self, "_train_"):
            self._train_ = self._shuffle_and_sample("train", self._train, self.train_size, self.train_seed)

        return self._train_

    @property
    def dev(self):
        if not hasattr(self, "_dev_"):
            self._dev_ = self._shuffle_and_sample("dev", self._dev, self.dev_size, self.dev_seed)

        return self._dev_

    @property
    def test(self):
        if not hasattr(self, "_test_"):
            self._test_ = self._shuffle_and_sample("test", self._test, self.test_size, self.test_seed)

        return self._test_

    def _shuffle_and_sample(self, split, data, size, seed=0):
        data = list(data)

        # Shuffle the data irrespective of the requested size.
        base_rng = random.Random(seed)

        if self.do_shuffle:
            base_rng.shuffle(data)

        data = data[:size]
        output = []

        for example in data:
            example_obj = Example(**example, dspy_uuid=str(uuid.uuid4()), dspy_split=split)
            if self.input_keys:
                example_obj = example_obj.with_inputs(*self.input_keys)
            output.append(example_obj)
        # TODO: NOTE: Ideally we use these uuids for dedup internally, for demos and internal train/val splits.
        # Now, some tasks (like convQA and Colors) have overlapping examples. Here, we should allow the user to give us
        # a uuid field that would respect this in some way. This means that we need a more refined concept that
        # uuid (each example is unique) and more like a group_uuid.

        return output

    @classmethod
    def prepare_by_seed(
        cls,
        train_seeds=None,
        train_size=16,
        dev_size=1000,
        divide_eval_per_seed=True,
        eval_seed=2023,
        **kwargs,
    ):
        train_seeds = train_seeds or [1, 2, 3, 4, 5]
        data_args = dotdict(train_size=train_size, eval_seed=eval_seed, dev_size=dev_size, test_size=0, **kwargs)
        dataset = cls(**data_args)

        eval_set = dataset.dev
        eval_sets, train_sets = [], []

        examples_per_seed = dev_size // len(train_seeds) if divide_eval_per_seed else dev_size
        eval_offset = 0

        for train_seed in train_seeds:
            data_args.train_seed = train_seed
            dataset.reset_seeds(**data_args)

            eval_sets.append(eval_set[eval_offset : eval_offset + examples_per_seed])
            train_sets.append(dataset.train)

            assert len(eval_sets[-1]) == examples_per_seed, len(eval_sets[-1])
            assert len(train_sets[-1]) == train_size, len(train_sets[-1])

            if divide_eval_per_seed:
                eval_offset += examples_per_seed

        return dotdict(train_sets=train_sets, eval_sets=eval_sets)
```

## File: `datasets/gsm8k.py`
```
import random

import tqdm


class GSM8K:
    def __init__(self):
        self.do_shuffle = False

        from datasets import load_dataset

        dataset = load_dataset("gsm8k", "main")

        hf_official_train = dataset["train"]
        hf_official_test = dataset["test"]
        official_train = []
        official_test = []

        for example in tqdm.tqdm(hf_official_train):
            question = example["question"]

            answer = example["answer"].strip().split()
            assert answer[-2] == "####"

            gold_reasoning = " ".join(answer[:-2])
            answer = str(int(answer[-1].replace(",", "")))

            official_train.append({"question": question, "gold_reasoning": gold_reasoning, "answer": answer})

        for example in tqdm.tqdm(hf_official_test):
            question = example["question"]

            answer = example["answer"].strip().split()
            assert answer[-2] == "####"

            gold_reasoning = " ".join(answer[:-2])
            answer = str(int(answer[-1].replace(",", "")))

            official_test.append({"question": question, "gold_reasoning": gold_reasoning, "answer": answer})

        rng = random.Random(0)
        rng.shuffle(official_train)

        rng = random.Random(0)
        rng.shuffle(official_test)

        trainset = official_train[:200]
        devset = official_train[200:500]
        testset = official_test[:]

        import dspy

        trainset = [dspy.Example(**x).with_inputs("question") for x in trainset]
        devset = [dspy.Example(**x).with_inputs("question") for x in devset]
        testset = [dspy.Example(**x).with_inputs("question") for x in testset]

        self.train = trainset
        self.dev = devset
        self.test = testset


def parse_integer_answer(answer, only_first_line=True):
    try:
        if only_first_line:
            answer = answer.strip().split("\n")[0]

        # find the last token that has a number in it
        answer = [token for token in answer.split() if any(c.isdigit() for c in token)][-1]
        answer = answer.split(".")[0]
        answer = "".join([c for c in answer if c.isdigit()])
        answer = int(answer)

    except (ValueError, IndexError):
        answer = 0

    return answer


def gsm8k_metric(gold, pred, trace=None):
    return int(parse_integer_answer(str(gold.answer))) == int(parse_integer_answer(str(pred.answer)))
```

## File: `datasets/hotpotqa.py`
```
import random

from dspy.datasets.dataset import Dataset


class HotPotQA(Dataset):
    def __init__(
        self,
        *args,
        only_hard_examples=True,
        keep_details="dev_titles",
        unofficial_dev=True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert only_hard_examples, (
            "Care must be taken when adding support for easy examples."
            "Dev must be all hard to match official dev, but training can be flexible."
        )

        from datasets import load_dataset

        hf_official_train = load_dataset("hotpot_qa", "fullwiki", split="train", trust_remote_code=True)
        hf_official_dev = load_dataset("hotpot_qa", "fullwiki", split="validation", trust_remote_code=True)

        official_train = []
        for raw_example in hf_official_train:
            if raw_example["level"] == "hard":
                if keep_details is True:
                    keys = ["id", "question", "answer", "type", "supporting_facts", "context"]
                elif keep_details == "dev_titles":
                    keys = ["question", "answer", "supporting_facts"]
                else:
                    keys = ["question", "answer"]

                example = {k: raw_example[k] for k in keys}

                if "supporting_facts" in example:
                    example["gold_titles"] = set(example["supporting_facts"]["title"])
                    del example["supporting_facts"]

                official_train.append(example)

        rng = random.Random(0)
        rng.shuffle(official_train)

        self._train = official_train[: len(official_train) * 75 // 100]

        if unofficial_dev:
            self._dev = official_train[len(official_train) * 75 // 100 :]
        else:
            self._dev = None

        for example in self._train:
            if keep_details == "dev_titles":
                del example["gold_titles"]

        test = []
        for raw_example in hf_official_dev:
            assert raw_example["level"] == "hard"
            example = {k: raw_example[k] for k in ["id", "question", "answer", "type", "supporting_facts"]}
            if "supporting_facts" in example:
                example["gold_titles"] = set(example["supporting_facts"]["title"])
                del example["supporting_facts"]
            test.append(example)

        self._test = test


if __name__ == "__main__":
    from dspy.dsp.utils import dotdict

    data_args = dotdict(train_seed=1, train_size=16, eval_seed=2023, dev_size=200 * 5, test_size=0)
    dataset = HotPotQA(**data_args)

    print(dataset)
    print(dataset.train[0].question)
    print(dataset.train[15].question)

    print(len(dataset.train), len(dataset.dev), len(dataset.test))

    print(dataset.dev[0].question)
    print(dataset.dev[340].question)
    print(dataset.dev[937].question)

"""
What was the population of the city where Woodward Avenue ends in 2010?
Where did the star , who is also an executive producer, of the Mick begin her carrer?
16 1000 0
Both London and German have seen attacks during war, there was one specific type of attack that Germany called the blitz, what did London call a similar attack?
Pre-Madonna was a collection of demos by the singer who was a leading presence during the emergence of what network?
Alan Mills composed the classic folk song that tells the story of what?
"""
```

## File: `datasets/math.py`
```
import random
import re


class MATH:
    def __init__(self, subset):
        from datasets import load_dataset

        import dspy

        ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", subset)

        # NOTE: Defaults to sub-splitting MATH's 'test' split into train/dev/test, presuming that current
        # LMs are trained on MATH's train. Makes no difference for gpt-4o-mini, but might for other models.

        dataset = [
            dspy.Example(
                question=example["problem"], reasoning=example["solution"], answer=extract_answer(example["solution"])
            ).with_inputs("question")
            for example in ds["test"]
        ]

        size = min(350, len(dataset) // 3)
        random.Random(0).shuffle(dataset)
        self.train, self.dev, self.test = dataset[:size], dataset[size : 2 * size], dataset[2 * size :]

    def metric(self, example, pred, trace=None):
        try:
            import math_equivalence
        except ImportError:
            raise ImportError("MATH's metric requires `pip install git+https://github.com/hendrycks/math.git`")

        return math_equivalence.is_equiv(example.answer, pred.answer)


def extract_answer(s):
    start = s.find("\\boxed{")
    if start == -1:
        return None

    idx = start + len("\\boxed{")
    brace_level = 1

    answer = ""
    while idx < len(s) and brace_level > 0:
        c = s[idx]
        if c == "{":
            brace_level += 1
        elif c == "}":
            brace_level -= 1
            if brace_level == 0:
                break
        answer += c
        idx += 1

    answer = re.sub(r"\\text\{[^}]*\}", "", answer)
    answer = re.sub(r"\\!", "", answer)
    return answer.strip()


"""
NOTE: MATH's official math_equivalence.is_equiv does not seem to have perfect recall.
Consider its behavior on reference values like `left[\frac{1}{2}, \frac{4}{3}\right]`.
"""
```

## File: `dsp/__init__.py`
```
Content omitted due to reason: BINARY
```

## File: `dsp/colbertv2.py`
```
from typing import Any

import requests

from dspy.clients.cache import request_cache
from dspy.dsp.utils import dotdict

# TODO: Ideally, this takes the name of the index and looks up its port.


class ColBERTv2:
    """Wrapper for the ColBERTv2 Retrieval."""

    def __init__(
        self,
        url: str = "http://0.0.0.0",
        port: str | int | None = None,
        post_requests: bool = False,
    ):
        self.post_requests = post_requests
        self.url = f"{url}:{port}" if port else url

    def __call__(
        self,
        query: str,
        k: int = 10,
        simplify: bool = False,
    ) -> list[str] | list[dotdict]:
        if self.post_requests:
            topk: list[dict[str, Any]] = colbertv2_post_request(self.url, query, k)
        else:
            topk: list[dict[str, Any]] = colbertv2_get_request(self.url, query, k)

        if simplify:
            return [psg["long_text"] for psg in topk]

        return [dotdict(psg) for psg in topk]


@request_cache()
def colbertv2_get_request_v2(url: str, query: str, k: int):
    assert k <= 100, "Only k <= 100 is supported for the hosted ColBERTv2 server at the moment."

    payload = {"query": query, "k": k}
    res = requests.get(url, params=payload, timeout=10)

    topk = res.json()["topk"][:k]
    topk = [{**d, "long_text": d["text"]} for d in topk]
    return topk[:k]


@request_cache()
def colbertv2_get_request_v2_wrapped(*args, **kwargs):
    return colbertv2_get_request_v2(*args, **kwargs)


colbertv2_get_request = colbertv2_get_request_v2_wrapped


@request_cache()
def colbertv2_post_request_v2(url: str, query: str, k: int):
    headers = {"Content-Type": "application/json; charset=utf-8"}
    payload = {"query": query, "k": k}
    res = requests.post(url, json=payload, headers=headers, timeout=10)

    return res.json()["topk"][:k]


@request_cache()
def colbertv2_post_request_v2_wrapped(*args, **kwargs):
    return colbertv2_post_request_v2(*args, **kwargs)


colbertv2_post_request = colbertv2_post_request_v2_wrapped


class ColBERTv2RetrieverLocal:
    def __init__(self, passages: list[str], colbert_config=None, load_only: bool = False):
        """Colbertv2 retriever module

        Args:
            passages (list[str]): list of passages
            colbert_config (ColBERTConfig, optional): colbert config for building and searching. Defaults to None.
            load_only (bool, optional): whether to load the index or build and then load. Defaults to False.
        """
        assert (
            colbert_config is not None
        ), "Please pass a valid colbert_config, which you can import from colbert.infra.config import ColBERTConfig and modify it"
        self.colbert_config = colbert_config

        assert (
            self.colbert_config.checkpoint is not None
        ), "Please pass a valid checkpoint like colbert-ir/colbertv2.0, which you can modify in the ColBERTConfig with attribute name checkpoint"
        self.passages = passages

        assert (
            self.colbert_config.index_name is not None
        ), "Please pass a valid index_name, which you can modify in the ColBERTConfig with attribute name index_name"
        self.passages = passages

        if not load_only:
            print(
                f"Building the index for experiment {self.colbert_config.experiment} with index name "
                f"{self.colbert_config.index_name}"
            )
            self.build_index()

        print(
            f"Loading the index for experiment {self.colbert_config.experiment} with index name "
            f"{self.colbert_config.index_name}"
        )
        self.searcher = self.get_index()

    def build_index(self):
        try:
            import colbert  # noqa: F401
        except ImportError:
            print(
                "Colbert not found. Please check your installation or install the module using pip install "
                "colbert-ai[faiss-gpu,torch]."
            )

        from colbert import Indexer
        from colbert.infra import Run, RunConfig

        with Run().context(RunConfig(nranks=self.colbert_config.nranks, experiment=self.colbert_config.experiment)):
            indexer = Indexer(checkpoint=self.colbert_config.checkpoint, config=self.colbert_config)
            indexer.index(name=self.colbert_config.index_name, collection=self.passages, overwrite=True)

    def get_index(self):
        try:
            import colbert  # noqa: F401
        except ImportError:
            print(
                "Colbert not found. Please check your installation or install the module using pip install "
                "colbert-ai[faiss-gpu,torch]."
            )

        from colbert import Searcher
        from colbert.infra import Run, RunConfig

        with Run().context(RunConfig(experiment=self.colbert_config.experiment)):
            searcher = Searcher(index=self.colbert_config.index_name, collection=self.passages)
        return searcher

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, query: str, k: int = 7, **kwargs):
        import torch

        if kwargs.get("filtered_pids"):
            filtered_pids = kwargs.get("filtered_pids")
            assert isinstance(filtered_pids, list) and all(isinstance(pid, int) for pid in filtered_pids), "The filtered pids should be a list of integers"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            results = self.searcher.search(
                query,
                # Number of passages to receive
                k=k,
                # Passing the filter function of relevant
                filter_fn=lambda pids: torch.tensor(
                    [pid for pid in pids if pid in filtered_pids], dtype=torch.int32
                ).to(device),
            )
        else:
            searcher_results = self.searcher.search(query, k=k)
        results = []
        for pid, rank, score in zip(*searcher_results, strict=False):  # noqa: B007
            results.append(dotdict({"long_text": self.searcher.collection[pid], "score": score, "pid": pid}))
        return results


class ColBERTv2RerankerLocal:
    def __init__(self, colbert_config=None, checkpoint: str = "bert-base-uncased"):
        try:
            import colbert  # noqa: F401
        except ImportError:
            print(
                "Colbert not found. Please check your installation or install the module using pip install "
                "colbert-ai[faiss-gpu,torch]."
            )
        """_summary_

        Args:
            colbert_config (ColBERTConfig, optional): Colbert config. Defaults to None.
            checkpoint_name (str, optional): checkpoint for embeddings. Defaults to 'bert-base-uncased'.
        """
        self.colbert_config = colbert_config
        self.checkpoint = checkpoint
        self.colbert_config.checkpoint = checkpoint

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, query: str, passages: list[str] | None = None):
        assert len(passages) > 0, "Passages should not be empty"

        import numpy as np
        from colbert.modeling.colbert import ColBERT
        from colbert.modeling.tokenization.doc_tokenization import DocTokenizer
        from colbert.modeling.tokenization.query_tokenization import QueryTokenizer

        passages = passages or []
        self.colbert_config.nway = len(passages)
        query_tokenizer = QueryTokenizer(self.colbert_config, verbose=1)
        doc_tokenizer = DocTokenizer(self.colbert_config)
        query_ids, query_masks = query_tokenizer.tensorize([query])
        doc_ids, doc_masks = doc_tokenizer.tensorize(passages)

        col = ColBERT(self.checkpoint, self.colbert_config)
        q = col.query(query_ids, query_masks)
        doc_ids, doc_masks = col.doc(doc_ids, doc_masks, keep_dims="return_mask")
        q_duplicated = q.repeat_interleave(len(passages), dim=0).contiguous()
        tensor_scores = col.score(q_duplicated, doc_ids, doc_masks)
        passage_score_arr = np.array([score.cpu().detach().numpy().tolist() for score in tensor_scores])
        return passage_score_arr
```

## File: `dsp/utils/__init__.py`
```
from dspy.dsp.utils.dpr import *
from dspy.dsp.utils.settings import *
from dspy.dsp.utils.utils import *
```

## File: `dsp/utils/dpr.py`
```
"""
Source: DPR Implementation from Facebook Research
https://github.com/facebookresearch/DPR/tree/master/dpr
Original license: https://github.com/facebookresearch/DPR/blob/main/LICENSE
"""

import copy
import logging
import unicodedata

import regex

logger = logging.getLogger(__name__)


class Tokens:
    """A class to represent a list of tokenized text."""

    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i:j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return "".join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token

        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if "pos" not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if "lemma" not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if "ner" not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.

        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [
            (s, e + 1)
            for s in range(len(words))
            for e in range(s, min(s + n, len(words)))
            if not _skip(words[s : e + 1])
        ]

        # Concatenate into strings
        if as_strings:
            ngrams = ["{}".format(" ".join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get("non_ent", "O")
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while idx < len(entities) and entities[idx] == ner_tag:
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


class Tokenizer:
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """

    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()


class SimpleTokenizer(Tokenizer):
    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
    NON_WS = r"[^\p{Z}\p{C}]"

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            "(%s)|(%s)" % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE,
        )
        if len(kwargs.get("annotators", {})) > 0:
            logger.warning(
                "%s only tokenizes! Skipping annotators: %s",
                type(self).__name__,
                kwargs.get("annotators"),
            )
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = list(self._regexp.finditer(text))
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append(
                (
                    token,
                    text[start_ws:end_ws],
                    span,
                )
            )
        return Tokens(data, self.annotators)


def has_answer(tokenized_answers, text):
    text = DPR_normalize(text)

    for single_answer in tokenized_answers:
        for i in range(0, len(text) - len(single_answer) + 1):
            if single_answer == text[i : i + len(single_answer)]:
                return True

    return False


def locate_answers(tokenized_answers, text):
    """
    Returns each occurrence of an answer as (offset, endpos) in terms of *characters*.
    """
    tokenized_text = DPR_tokenize(text)
    occurrences = []

    text_words, text_word_positions = tokenized_text.words(uncased=True), tokenized_text.offsets()
    answers_words = [ans.words(uncased=True) for ans in tokenized_answers]

    for single_answer in answers_words:
        for i in range(0, len(text_words) - len(single_answer) + 1):
            if single_answer == text_words[i : i + len(single_answer)]:
                (offset, _), (_, endpos) = text_word_positions[i], text_word_positions[i + len(single_answer) - 1]
                occurrences.append((offset, endpos))

    return occurrences


STokenizer = SimpleTokenizer()


def DPR_tokenize(text):  # noqa: N802
    return STokenizer.tokenize(unicodedata.normalize("NFD", text))


def DPR_normalize(text):  # noqa: N802
    return DPR_tokenize(text).words(uncased=True)


# Source: https://github.com/shmsw25/qa-hard-em/blob/master/prepro_util.py
def strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)
```

## File: `dsp/utils/settings.py`
```
import asyncio
import contextvars
import copy
import threading
from contextlib import contextmanager

from dspy.dsp.utils.utils import dotdict

DEFAULT_CONFIG = dotdict(
    lm=None,
    adapter=None,
    rm=None,
    branch_idx=0,
    trace=[],
    callbacks=[],
    async_max_workers=8,
    send_stream=None,
    disable_history=False,
    track_usage=False,
    usage_tracker=None,
    caller_predict=None,
    caller_modules=None,
    stream_listeners=[],
    provide_traceback=False,  # Whether to include traceback information in error logs.
    num_threads=8,  # Number of threads to use for parallel processing.
    max_errors=10,  # Maximum errors before halting operations.
    # If true, async tools can be called in sync mode by getting converted to sync.
    allow_tool_async_sync_conversion=False,
)

# Global base configuration and owner tracking
main_thread_config = copy.deepcopy(DEFAULT_CONFIG)
config_owner_thread_id = None
config_owner_async_task = None

# Global lock for settings configuration
global_lock = threading.Lock()

thread_local_overrides = contextvars.ContextVar("context_overrides", default=dotdict())


class Settings:
    """
    A singleton class for DSPy configuration settings.
    Thread-safe global configuration.
    - 'configure' can be called by only one 'owner' thread (the first thread that calls it).
    - Other threads see the configured global values from 'main_thread_config'.
    - 'context' sets thread-local overrides. These overrides propagate to threads spawned
      inside that context block, when (and only when!) using a ParallelExecutor that copies overrides.

      1. Only one unique thread (which can be any thread!) can call dspy.configure.
      2. It affects a global state, visible to all. As a result, user threads work, but they shouldn't be
         mixed with concurrent changes to dspy.configure from the "main" thread.
         (TODO: In the future, add warnings: if there are near-in-time user-thread reads followed by .configure calls.)
      3. Any thread can use dspy.context. It propagates to child threads created with DSPy primitives: Parallel, asyncify, etc.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def lock(self):
        return global_lock

    def __getattr__(self, name):
        overrides = thread_local_overrides.get()
        if name in overrides:
            return overrides[name]
        elif name in main_thread_config:
            return main_thread_config[name]
        else:
            raise AttributeError(f"'Settings' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in ("_instance",):
            super().__setattr__(name, value)
        else:
            self.configure(**{name: value})

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __contains__(self, key):
        overrides = thread_local_overrides.get()
        return key in overrides or key in main_thread_config

    def get(self, key, default=None):
        try:
            return self[key]
        except AttributeError:
            return default

    def copy(self):
        overrides = thread_local_overrides.get()
        return dotdict({**main_thread_config, **overrides})

    @property
    def config(self):
        return self.copy()

    def _ensure_configure_allowed(self):
        global main_thread_config, config_owner_thread_id, config_owner_async_task
        current_thread_id = threading.get_ident()

        if config_owner_thread_id is None:
            # First `configure` call assigns the owner thread id.
            config_owner_thread_id = current_thread_id

        if config_owner_thread_id != current_thread_id:
            # Disallow a second `configure` calls from other threads.
            raise RuntimeError("dspy.settings can only be changed by the thread that initially configured it.")

        # Async task doesn't allow a second `configure` call, must use dspy.context(...) instead.
        is_async_task = False
        try:
            if asyncio.current_task() is not None:
                is_async_task = True
        except RuntimeError:
            # This exception (e.g., "no current task") means we are not in an async loop/task,
            # or asyncio module itself is not fully functional in this specific sub-thread context.
            is_async_task = False

        if not is_async_task:
            return

        if config_owner_async_task is None:
            # First `configure` call assigns the owner async task.
            config_owner_async_task = asyncio.current_task()
            return

        # We are in an async task. Now check for IPython and allow calling `configure` from IPython.
        in_ipython = False
        try:
            from IPython import get_ipython

            # get_ipython is a global injected by IPython environments.
            # We check its existence and type to be more robust.
            in_ipython = get_ipython() is not None
        except Exception:
            # If `IPython` is not installed or `get_ipython` failed, we are not in an IPython environment.
            in_ipython = False

        if not in_ipython and config_owner_async_task != asyncio.current_task():
            raise RuntimeError(
                "dspy.settings.configure(...) can only be called from the same async task that called it first. Please "
                "use `dspy.context(...)` in other async tasks instead."
            )

    def configure(self, **kwargs):
        # If no exception is raised, the `configure` call is allowed.
        self._ensure_configure_allowed()

        # Update global config
        for k, v in kwargs.items():
            main_thread_config[k] = v

    @contextmanager
    def context(self, **kwargs):
        """
        Context manager for temporary configuration changes at the thread level.
        Does not affect global configuration. Changes only apply to the current thread.
        If threads are spawned inside this block using ParallelExecutor, they will inherit these overrides.
        """

        original_overrides = thread_local_overrides.get().copy()
        new_overrides = dotdict({**main_thread_config, **original_overrides, **kwargs})
        token = thread_local_overrides.set(new_overrides)

        try:
            yield
        finally:
            thread_local_overrides.reset(token)

    def __repr__(self):
        overrides = thread_local_overrides.get()
        combined_config = {**main_thread_config, **overrides}
        return repr(combined_config)


settings = Settings()
```

## File: `dsp/utils/utils.py`
```
import copy
import datetime
import itertools
import os
from collections import defaultdict

import tqdm


def print_message(*s, condition=True, pad=False, sep=None):
    s = " ".join([str(x) for x in s])
    msg = "[{}] {}".format(datetime.datetime.now().strftime("%b %d, %H:%M:%S"), s)

    if condition:
        msg = msg if not pad else f"\n{msg}\n"
        print(msg, flush=True, sep=sep)

    return msg


def timestamp(daydir=False):
    format_str = f"%Y-%m{'/' if daydir else '-'}%d{'/' if daydir else '_'}%H.%M.%S"
    result = datetime.datetime.now().strftime(format_str)
    return result


def file_tqdm(file):
    print(f"#> Reading {file.name}")

    with tqdm.tqdm(
        total=os.path.getsize(file.name) / 1024.0 / 1024.0,
        unit="MiB",
    ) as pbar:
        for line in file:
            yield line
            pbar.update(len(line) / 1024.0 / 1024.0)

        pbar.close()


def create_directory(path):
    if os.path.exists(path):
        print("\n")
        print_message("#> Note: Output directory", path, "already exists\n\n")
    else:
        print("\n")
        print_message("#> Creating directory", path, "\n\n")
        os.makedirs(path)


def deduplicate(seq: list[str]) -> list[str]:
    """
    Source: https://stackoverflow.com/a/480227/1493011
    """

    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def batch(group, bsize, provide_offset=False):
    offset = 0
    while offset < len(group):
        batch_data = group[offset : offset + bsize]
        yield ((offset, batch_data) if provide_offset else batch_data)
        offset += len(batch_data)
    return


class dotdict(dict):  # noqa: N801
    def __getattr__(self, key):
        if key.startswith("__") and key.endswith("__"):
            return super().__getattr__(key)
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key.startswith("__") and key.endswith("__"):
            super().__setattr__(key, value)
        else:
            self[key] = value

    def __delattr__(self, key):
        if key.startswith("__") and key.endswith("__"):
            super().__delattr__(key)
        else:
            del self[key]

    def __deepcopy__(self, memo):
        # Use the default dict copying method to avoid infinite recursion.
        return dotdict(copy.deepcopy(dict(self), memo))


class dotdict_lax(dict):  # noqa: N801
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def flatten(data_list):
    result = []
    for child_list in data_list:
        result += child_list

    return result


def zipstar(data_list, lazy=False):
    """
    A much faster A, B, C = zip(*[(a, b, c), (a, b, c), ...])
    May return lists or tuples.
    """

    if len(data_list) == 0:
        return data_list

    width = len(data_list[0])

    if width < 100:
        return [[elem[idx] for elem in data_list] for idx in range(width)]

    zipped_data = zip(*data_list, strict=False)

    return zipped_data if lazy else list(zipped_data)


def zip_first(list1, list2):
    length = len(list1) if type(list1) in [tuple, list] else None

    zipped_data = list(zip(list1, list2, strict=False))

    assert length in [None, len(zipped_data)], "zip_first() failure: length differs!"

    return zipped_data


def int_or_float(val):
    if "." in val:
        return float(val)

    return int(val)


def groupby_first_item(lst):
    groups = defaultdict(list)

    for first, *rest in lst:
        rest = rest[0] if len(rest) == 1 else rest
        groups[first].append(rest)

    return groups


def process_grouped_by_first_item(lst):
    """
    Requires items in list to already be grouped by first item.
    """

    groups = defaultdict(list)

    started = False
    last_group = None

    for first, *rest in lst:
        rest = rest[0] if len(rest) == 1 else rest

        if started and first != last_group:
            yield (last_group, groups[last_group])
            assert first not in groups, f"{first} seen earlier --- violates precondition."

        groups[first].append(rest)

        last_group = first
        started = True

    return groups


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks
        Example: grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
        Source: https://docs.python.org/3/library/itertools.html#itertools-recipes
    """

    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def lengths2offsets(lengths):
    offset = 0

    for length in lengths:
        yield (offset, offset + length)
        offset += length

    return


# see https://stackoverflow.com/a/45187287
class NullContextManager:
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource

    def __enter__(self):
        return self.dummy_resource

    def __exit__(self, *args):
        pass


def load_batch_backgrounds(args, qids):
    if args.qid2backgrounds is None:
        return None

    qbackgrounds = []

    for qid in qids:
        back = args.qid2backgrounds[qid]

        if len(back) and isinstance(back[0], int):
            x = [args.collection[pid] for pid in back]
        else:
            x = [args.collectionX.get(pid, "") for pid in back]

        x = " [SEP] ".join(x)
        qbackgrounds.append(x)

    return qbackgrounds
```

## File: `evaluate/__init__.py`
```
from dspy.evaluate.auto_evaluation import CompleteAndGrounded, SemanticF1
from dspy.evaluate.evaluate import Evaluate
from dspy.evaluate.metrics import EM, answer_exact_match, answer_passage_match, normalize_text

__all__ = [
    "EM",
    "normalize_text",
    "answer_exact_match",
    "answer_passage_match",
    "Evaluate",
    "SemanticF1",
    "CompleteAndGrounded",
]
```

## File: `evaluate/auto_evaluation.py`
```
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.primitives import Module
from dspy.signatures import InputField, OutputField, Signature


class SemanticRecallPrecision(Signature):
    """
    Compare a system's response to the ground truth to compute its recall and precision.
    If asked to reason, enumerate key ideas in each response, and whether they are present in the other response.
    """

    question: str = InputField()
    ground_truth: str = InputField()
    system_response: str = InputField()
    recall: float = OutputField(desc="fraction (out of 1.0) of ground truth covered by the system response")
    precision: float = OutputField(desc="fraction (out of 1.0) of system response covered by the ground truth")


class DecompositionalSemanticRecallPrecision(Signature):
    """
    Compare a system's response to the ground truth to compute recall and precision of key ideas.
    You will first enumerate key ideas in each response, discuss their overlap, and then report recall and precision.
    """

    question: str = InputField()
    ground_truth: str = InputField()
    system_response: str = InputField()
    ground_truth_key_ideas: str = OutputField(desc="enumeration of key ideas in the ground truth")
    system_response_key_ideas: str = OutputField(desc="enumeration of key ideas in the system response")
    discussion: str = OutputField(desc="discussion of the overlap between ground truth and system response")
    recall: float = OutputField(desc="fraction (out of 1.0) of ground truth covered by the system response")
    precision: float = OutputField(desc="fraction (out of 1.0) of system response covered by the ground truth")


def f1_score(precision, recall):
    precision, recall = max(0.0, min(1.0, precision)), max(0.0, min(1.0, recall))
    return 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)


class SemanticF1(Module):
    def __init__(self, threshold=0.66, decompositional=False):
        self.threshold = threshold

        if decompositional:
            self.module = ChainOfThought(DecompositionalSemanticRecallPrecision)
        else:
            self.module = ChainOfThought(SemanticRecallPrecision)

    def forward(self, example, pred, trace=None):
        scores = self.module(question=example.question, ground_truth=example.response, system_response=pred.response)
        score = f1_score(scores.precision, scores.recall)

        return score if trace is None else score >= self.threshold



###########


class AnswerCompleteness(Signature):
    """
    Estimate the completeness of a system's responses, against the ground truth.
    You will first enumerate key ideas in each response, discuss their overlap, and then report completeness.
    """

    question: str = InputField()
    ground_truth: str = InputField()
    system_response: str = InputField()
    ground_truth_key_ideas: str = OutputField(desc="enumeration of key ideas in the ground truth")
    system_response_key_ideas: str = OutputField(desc="enumeration of key ideas in the system response")
    discussion: str = OutputField(desc="discussion of the overlap between ground truth and system response")
    completeness: float = OutputField(desc="fraction (out of 1.0) of ground truth covered by the system response")



class AnswerGroundedness(Signature):
    """
    Estimate the groundedness of a system's responses, against real retrieved documents written by people.
    You will first enumerate whatever non-trivial or check-worthy claims are made in the system response, and then
    discuss the extent to which some or all of them can be deduced from the retrieved context and basic commonsense.
    """

    question: str = InputField()
    retrieved_context: str = InputField()
    system_response: str = InputField()
    system_response_claims: str = OutputField(desc="enumeration of non-trivial or check-worthy claims in the system response")
    discussion: str = OutputField(desc="discussion of how supported the claims are by the retrieved context")
    groundedness: float = OutputField(desc="fraction (out of 1.0) of system response supported by the retrieved context")


class CompleteAndGrounded(Module):
    def __init__(self, threshold=0.66):
        self.threshold = threshold
        self.completeness_module = ChainOfThought(AnswerCompleteness)
        self.groundedness_module = ChainOfThought(AnswerGroundedness)

    def forward(self, example, pred, trace=None):
        completeness = self.completeness_module(question=example.question, ground_truth=example.response, system_response=pred.response)
        groundedness = self.groundedness_module(question=example.question, retrieved_context=pred.context, system_response=pred.response)
        score = f1_score(groundedness.groundedness, completeness.completeness)

        return score if trace is None else score >= self.threshold
```

## File: `evaluate/evaluate.py`
```
import importlib
import logging
import types
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import pandas as pd

import tqdm

import dspy
from dspy.primitives.prediction import Prediction
from dspy.utils.callback import with_callbacks
from dspy.utils.parallelizer import ParallelExecutor

try:
    from IPython.display import HTML
    from IPython.display import display as display

except ImportError:

    def display(obj: Any):
        """
        Display the specified Python object in the console.

        :param obj: The Python object to display.
        """
        print(obj)

    def HTML(x: str) -> str:  # noqa: N802
        """
        Obtain the HTML representation of the specified string.
        """
        # NB: This method exists purely for code compatibility with the IPython HTML() function in
        # environments where IPython is not available. In such environments where IPython is not
        # available, this method will simply return the input string.
        return x


# TODO: Counting failures and having a max_failure count. When that is exceeded (also just at the end),
# we print the number of failures, the first N examples that failed, and the first N exceptions raised.

logger = logging.getLogger(__name__)


class EvaluationResult(Prediction):
    """
    A class that represents the result of an evaluation.
    It is a subclass of `dspy.Prediction` that contains the following fields

    - score: An float value (e.g., 67.30) representing the overall performance
    - results: a list of (example, prediction, score) tuples for each example in devset
    """
    def __init__(self, score: float, results: list[tuple["dspy.Example", "dspy.Example", Any]]):
        super().__init__(score=score, results=results)

    def __repr__(self):
        return f"EvaluationResult(score={self.score}, results=<list of {len(self.results)} results>)"


class Evaluate:
    """DSPy Evaluate class.

    This class is used to evaluate the performance of a DSPy program. Users need to provide a evaluation dataset and
    a metric function in order to use this class. This class supports parallel evaluation on the provided dataset.
    """

    def __init__(
        self,
        *,
        devset: list["dspy.Example"],
        metric: Callable | None = None,
        num_threads: int | None = None,
        display_progress: bool = False,
        display_table: bool | int = False,
        max_errors: int | None = None,
        provide_traceback: bool | None = None,
        failure_score: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            devset (list[dspy.Example]): the evaluation dataset.
            metric (Callable): The metric function to use for evaluation.
            num_threads (Optional[int]): The number of threads to use for parallel evaluation.
            display_progress (bool): Whether to display progress during evaluation.
            display_table (Union[bool, int]): Whether to display the evaluation results in a table.
                If a number is passed, the evaluation results will be truncated to that number before displayed.
            max_errors (Optional[int]): The maximum number of errors to allow before
                stopping evaluation. If ``None``, inherits from ``dspy.settings.max_errors``.
            provide_traceback (Optional[bool]): Whether to provide traceback information during evaluation.
            failure_score (float): The default score to use if evaluation fails due to an exception.
        """
        self.devset = devset
        self.metric = metric
        self.num_threads = num_threads
        self.display_progress = display_progress
        self.display_table = display_table
        self.max_errors = max_errors
        self.provide_traceback = provide_traceback
        self.failure_score = failure_score

    @with_callbacks
    def __call__(
        self,
        program: "dspy.Module",
        metric: Callable | None = None,
        devset: list["dspy.Example"] | None = None,
        num_threads: int | None = None,
        display_progress: bool | None = None,
        display_table: bool | int | None = None,
        callback_metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """
        Args:
            program (dspy.Module): The DSPy program to evaluate.
            metric (Callable): The metric function to use for evaluation. if not provided, use `self.metric`.
            devset (list[dspy.Example]): the evaluation dataset. if not provided, use `self.devset`.
            num_threads (Optional[int]): The number of threads to use for parallel evaluation. if not provided, use
                `self.num_threads`.
            display_progress (bool): Whether to display progress during evaluation. if not provided, use
                `self.display_progress`.
            display_table (Union[bool, int]): Whether to display the evaluation results in a table. if not provided, use
                `self.display_table`. If a number is passed, the evaluation results will be truncated to that number before displayed.
            callback_metadata (dict): Metadata to be used for evaluate callback handlers.

        Returns:
            The evaluation results are returned as a dspy.EvaluationResult object containing the following attributes:
            
            - score: A float percentage score (e.g., 67.30) representing overall performance
            
            - results: a list of (example, prediction, score) tuples for each example in devset
        """
        metric = metric if metric is not None else self.metric
        devset = devset if devset is not None else self.devset
        num_threads = num_threads if num_threads is not None else self.num_threads
        display_progress = display_progress if display_progress is not None else self.display_progress
        display_table = display_table if display_table is not None else self.display_table

        if callback_metadata:
            logger.debug(f"Evaluate is called with callback metadata: {callback_metadata}")

        tqdm.tqdm._instances.clear()

        executor = ParallelExecutor(
            num_threads=num_threads,
            disable_progress_bar=not display_progress,
            max_errors=(
                self.max_errors
                if self.max_errors is not None
                else dspy.settings.max_errors
            ),
            provide_traceback=self.provide_traceback,
            compare_results=True,
        )

        def process_item(example):
            prediction = program(**example.inputs())
            score = metric(example, prediction)

            # Increment assert and suggest failures to program's attributes
            if hasattr(program, "_assert_failures"):
                program._assert_failures += dspy.settings.get("assert_failures")
            if hasattr(program, "_suggest_failures"):
                program._suggest_failures += dspy.settings.get("suggest_failures")

            return prediction, score

        results = executor.execute(process_item, devset)
        assert len(devset) == len(results)

        results = [((dspy.Prediction(), self.failure_score) if r is None else r) for r in results]
        results = [(example, prediction, score) for example, (prediction, score) in zip(devset, results, strict=False)]
        ncorrect, ntotal = sum(score for *_, score in results), len(devset)

        logger.info(f"Average Metric: {ncorrect} / {ntotal} ({round(100 * ncorrect / ntotal, 1)}%)")

        if display_table:
            if importlib.util.find_spec("pandas") is not None:
                # Rename the 'correct' column to the name of the metric object
                metric_name = metric.__name__ if isinstance(metric, types.FunctionType) else metric.__class__.__name__
                # Construct a pandas DataFrame from the results
                result_df = self._construct_result_table(results, metric_name)

                self._display_result_table(result_df, display_table, metric_name)
            else:
                logger.warning("Skipping table display since `pandas` is not installed.")

        return EvaluationResult(
            score=round(100 * ncorrect / ntotal, 2),
            results=results,
        )


    def _construct_result_table(
        self, results: list[tuple["dspy.Example", "dspy.Example", Any]], metric_name: str
    ) -> "pd.DataFrame":
        """
        Construct a pandas DataFrame from the specified result list.
        Let's not try to change the name of this method as it may be patched by external tracing tools.

        Args:
            results: The list of results to construct the result DataFrame from.
            metric_name: The name of the metric used for evaluation.

        Returns:
            The constructed pandas DataFrame.
        """
        import pandas as pd

        data = [
            (
                merge_dicts(example, prediction) | {"correct": score}
                if prediction_is_dictlike(prediction)
                else dict(example) | {"prediction": prediction, "correct": score}
            )
            for example, prediction, score in results
        ]

        # Truncate every cell in the DataFrame (DataFrame.applymap was renamed to DataFrame.map in Pandas 2.1.0)
        result_df = pd.DataFrame(data)
        result_df = result_df.map(truncate_cell) if hasattr(result_df, "map") else result_df.applymap(truncate_cell)

        return result_df.rename(columns={"correct": metric_name})

    def _display_result_table(self, result_df: "pd.DataFrame", display_table: bool | int, metric_name: str):
        """
        Display the specified result DataFrame in a table format.

        Args:
            result_df: The result DataFrame to display.
            display_table: Whether to display the evaluation results in a table.
                If a number is passed, the evaluation results will be truncated to that number before displayed.
            metric_name: The name of the metric used for evaluation.
        """
        if isinstance(display_table, bool):
            df_to_display = result_df.copy()
            truncated_rows = 0
        else:
            df_to_display = result_df.head(display_table).copy()
            truncated_rows = len(result_df) - display_table

        df_to_display = stylize_metric_name(df_to_display, metric_name)

        display_dataframe(df_to_display)

        if truncated_rows > 0:
            # Simplified message about the truncated rows
            message = f"""
            <div style='
                text-align: center;
                font-size: 16px;
                font-weight: bold;
                color: #555;
                margin: 10px 0;'>
                ... {truncated_rows} more rows not displayed ...
            </div>
            """
            display(HTML(message))


def prediction_is_dictlike(prediction):
    # Downstream logic for displaying dictionary-like predictions depends solely on the predictions
    # having a method called `items()` for iterating through key/value pairs
    return hasattr(prediction, "items") and callable(prediction.items)


def merge_dicts(d1, d2) -> dict:
    merged = {}
    for k, v in d1.items():
        if k in d2:
            merged[f"example_{k}"] = v
        else:
            merged[k] = v

    for k, v in d2.items():
        if k in d1:
            merged[f"pred_{k}"] = v
        else:
            merged[k] = v

    return merged


def truncate_cell(content) -> str:
    """Truncate content of a cell to 25 words."""
    words = str(content).split()
    if len(words) > 25:
        return " ".join(words[:25]) + "..."
    return content


def stylize_metric_name(df: "pd.DataFrame", metric_name: str) -> "pd.DataFrame":
    """
    Stylize the cell contents of a pandas DataFrame corresponding to the specified metric name.

    :param df: The pandas DataFrame for which to stylize cell contents.
    :param metric_name: The name of the metric for which to stylize DataFrame cell contents.
    """
    df[metric_name] = df[metric_name].apply(
        lambda x: f"✔️ [{x:.3f}]" if x and isinstance(x, float) else f"✔️ [{x}]" if x else ""
    )
    return df


def display_dataframe(df: "pd.DataFrame"):
    """
    Display the specified Pandas DataFrame in the console.

    :param df: The Pandas DataFrame to display.
    """
    import pandas as pd

    if is_in_ipython_notebook_environment():
        display(configure_dataframe_for_ipython_notebook_display(df))
    else:
        # Pretty print the DataFrame to the console
        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None
        ):  # more options can be specified also
            print(df)


def configure_dataframe_for_ipython_notebook_display(df: "pd.DataFrame") -> "pd.DataFrame":
    """Set various pandas display options for DataFrame in an IPython notebook environment."""
    import pandas as pd

    pd.options.display.max_colwidth = 70
    return df


def is_in_ipython_notebook_environment():
    """
    Check if the current environment is an IPython notebook environment.

    :return: True if the current environment is an IPython notebook environment, False otherwise.
    """
    try:
        from IPython import get_ipython

        # This is a best-effort check to see if we are in an IPython notebook environment
        return "IPKernelApp" in getattr(get_ipython(), "config", {})
    except ImportError:
        return False


# FIXME: TODO: The merge_dicts stuff above is way too quick and dirty.
# TODO: the display_table can't handle False but can handle 0!
# Not sure how it works with True exactly, probably fails too.
```

## File: `evaluate/metrics.py`
```
# TODO: This should move internally. Same for passage_match. dspy.metrics.answer_exact_match, dspy.metrics.answer_passage_match

import re
import string
import unicodedata
from collections import Counter

from dspy.dsp.utils.utils import print_message


def EM(prediction, answers_list):  # noqa: N802
    assert isinstance(answers_list, list)

    return max(em_score(prediction, ans) for ans in answers_list)


def F1(prediction, answers_list):  # noqa: N802
    assert isinstance(answers_list, list)

    return max(f1_score(prediction, ans) for ans in answers_list)


def HotPotF1(prediction, answers_list):  # noqa: N802
    assert isinstance(answers_list, list)

    return max(hotpot_f1_score(prediction, ans) for ans in answers_list)


def normalize_text(s):
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_score(prediction, ground_truth):
    return normalize_text(prediction) == normalize_text(ground_truth)


# See: https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
# See: https://rajpurkar.github.io/SQuAD-explorer/ under Evaluation Script
# See: QReCC's


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if len(prediction_tokens) == len(ground_truth_tokens) == 0:
        # Unlike most tasks, QReCC and SQuAD-2.0 assign 1.0 in this edge case. We don't for uniformity.
        print_message("\n#> F1 Metric: Rare edge case of len(prediction_tokens) == len(ground_truth_tokens) == 0.\n")

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def hotpot_f1_score(prediction, ground_truth):
    normalized_prediction = normalize_text(prediction)
    normalized_ground_truth = normalize_text(ground_truth)

    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return 0
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return 0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def precision_score(prediction, ground_truth):
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if len(prediction_tokens) == len(ground_truth_tokens) == 0:
        # Unlike most tasks, QReCC and SQuAD-2.0 assign 1.0 in this edge case. We don't for uniformity.
        print_message("\n#> Precision Metric: Rare edge case of len(prediction_tokens) == len(ground_truth_tokens) == 0.\n")

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_tokens)

    return precision


def _passage_match(passages: list[str], answers: list[str]) -> bool:
    """Returns True if any of the passages contains the answer."""
    from dspy.dsp.utils import DPR_normalize, has_answer

    def passage_has_answers(passage: str, answers: list[str]) -> bool:
        """Returns True if the passage contains the answer."""
        return has_answer(
            tokenized_answers=[DPR_normalize(normalize_text(ans)) for ans in answers],
            text=normalize_text(passage),
        )

    return any(passage_has_answers(psg, answers) for psg in passages)


def _answer_match(prediction, answers, frac=1.0):
    """Returns True if the prediction matches any of the answers."""

    if frac >= 1.0:
        return EM(prediction, answers)

    return F1(prediction, answers) >= frac


def answer_exact_match(example, pred, trace=None, frac=1.0):
    if isinstance(example.answer, str):
        return _answer_match(pred.answer, [example.answer], frac=frac)
    elif isinstance(example.answer, list):
        return _answer_match(pred.answer, example.answer, frac=frac)

    raise ValueError(f"Invalid answer type: {type(example.answer)}")


def answer_passage_match(example, pred, trace=None):
    if isinstance(example.answer, str):
        return _passage_match(pred.context, [example.answer])
    elif isinstance(example.answer, list):
        return _passage_match(pred.context, example.answer)

    raise ValueError(f"Invalid answer type: {type(example.answer)}")
```

## File: `predict/__init__.py`
```
from dspy.predict.aggregation import majority
from dspy.predict.best_of_n import BestOfN
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.code_act import CodeAct
from dspy.predict.knn import KNN
from dspy.predict.multi_chain_comparison import MultiChainComparison
from dspy.predict.parallel import Parallel
from dspy.predict.predict import Predict
from dspy.predict.program_of_thought import ProgramOfThought
from dspy.predict.react import ReAct, Tool
from dspy.predict.refine import Refine

__all__ = [
    "majority",
    "BestOfN",
    "ChainOfThought",
    "CodeAct",
    "KNN",
    "MultiChainComparison",
    "Predict",
    "ProgramOfThought",
    "ReAct",
    "Refine",
    "Tool",
    "Parallel",
]
```

## File: `predict/aggregation.py`
```
from dspy.evaluate import normalize_text
from dspy.primitives.prediction import Completions, Prediction


def default_normalize(s):
    return normalize_text(s) or None


def majority(prediction_or_completions, normalize=default_normalize, field=None):
    """
    Returns the most common completion for the target field (or the last field) in the signature.
    When normalize returns None, that completion is ignored.
    In case of a tie, earlier completion are prioritized.
    """

    assert any(isinstance(prediction_or_completions, t) for t in [Prediction, Completions, list])
    type(prediction_or_completions)

    # Get the completions
    if isinstance(prediction_or_completions, Prediction):
        completions = prediction_or_completions.completions
    else:
        completions = prediction_or_completions

    try:
        signature = completions.signature
    except Exception:
        signature = None

    if not field:
        if signature:
            field = list(signature.output_fields.keys())[-1]
        else:
            field = list(completions[0].keys())[-1]

    # Normalize
    normalize = normalize if normalize else lambda x: x
    normalized_values = [normalize(completion[field]) for completion in completions]
    normalized_values_ = [x for x in normalized_values if x is not None]

    # Count
    value_counts = {}
    for value in normalized_values_ or normalized_values:
        value_counts[value] = value_counts.get(value, 0) + 1

    majority_value = max(value_counts, key=value_counts.get)

    # Return the first completion with the majority value in the field
    for completion in completions:
        if normalize(completion[field]) == majority_value:
            break

    # if input_type == Prediction:
    return Prediction.from_completions([completion], signature=signature)
```

## File: `predict/avatar/__init__.py`
```
from dspy.predict.avatar.avatar import *
from dspy.predict.avatar.models import *
from dspy.predict.avatar.signatures import *
```

## File: `predict/avatar/avatar.py`
```
from copy import deepcopy

from pydantic.fields import FieldInfo

import dspy
from dspy.predict.avatar.models import Action, ActionOutput, Tool
from dspy.predict.avatar.signatures import Actor
from dspy.signatures.signature import ensure_signature


def get_number_with_suffix(number: int) -> str:
    if number == 1:
        return "1st"
    elif number == 2:
        return "2nd"
    elif number == 3:
        return "3rd"
    else:
        return f"{number}th"


class Avatar(dspy.Module):
    def __init__(
        self,
        signature,
        tools,
        max_iters=3,
        verbose=False,
    ):
        self.signature = ensure_signature(signature)
        self.input_fields = self.signature.input_fields
        self.output_fields = self.signature.output_fields

        self.finish_tool = Tool(
            tool=None,
            name="Finish",
            desc="returns the final output and finishes the task",
        )

        self.tools = tools + [self.finish_tool]
        self.actor_signature = Actor

        for field in list(self.input_fields.keys())[::-1]:
            self.actor_signature = self.actor_signature.append(
                field,
                self._get_field(self.input_fields[field]),
                type_=self.input_fields[field].annotation,
            )

        self.verbose = verbose
        self.max_iters = max_iters
        self.actor = dspy.TypedPredictor(self.actor_signature)

        self.actor_clone = deepcopy(self.actor)

    def _get_field(self, field_info: FieldInfo):
        if field_info.json_schema_extra["__dspy_field_type"] == "input":
            return dspy.InputField(
                prefix=field_info.json_schema_extra["prefix"],
                desc=field_info.json_schema_extra["desc"],
                format=field_info.json_schema_extra["format"] if "format" in field_info.json_schema_extra else None,
            )
        elif field_info.json_schema_extra["__dspy_field_type"] == "output":
            return dspy.OutputField(
                prefix=field_info.json_schema_extra["prefix"],
                desc=field_info.json_schema_extra["desc"],
                format=field_info.json_schema_extra["format"] if "format" in field_info.json_schema_extra else None,
            )
        else:
            raise ValueError(f"Unknown field type: {field_info.json_schema_extra['__dspy_field_type']}")

    def _update_signature(self, idx: int, omit_action: bool = False):
        self.actor.signature = self.actor.signature.with_updated_fields(
            f"action_{idx}", Action, __dspy_field_type="input"
        )

        self.actor.signature = self.actor.signature.append(
            f"result_{idx}",
            dspy.InputField(
                prefix=f"Result {idx}:",
                desc=f"{get_number_with_suffix(idx)} result",
                type_=str,
            ),
        )

        if omit_action:
            for field in list(self.output_fields.keys()):
                self.actor.signature = self.actor.signature.append(
                    field,
                    self._get_field(self.output_fields[field]),
                    type_=self.output_fields[field].annotation,
                )
        else:
            self.actor.signature = self.actor.signature.append(
                f"action_{idx+1}",
                dspy.OutputField(
                    prefix=f"Action {idx+1}:",
                    desc=f"{get_number_with_suffix(idx+1)} action to taken",
                ),
            )
            self.actor.signature = self.actor.signature.with_updated_fields(
                f"action_{idx+1}",
                Action,
            )

    def _call_tool(self, tool_name: str, tool_input_query: str) -> str:
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.tool.run(tool_input_query)

    def forward(self, **kwargs):
        if self.verbose:
            print("Starting the task...")

        args = {
            "goal": self.signature.__doc__,
            "tools": [tool.name for tool in self.tools],
        }

        for key in self.input_fields.keys():
            if key in kwargs:
                args[key] = kwargs[key]

        idx = 1
        tool_name = None
        action_results: list[ActionOutput] = []
        max_iters = None if "max_iters" not in kwargs else kwargs["max_iters"]

        while tool_name != "Finish" and (max_iters > 0 if max_iters else True):
            actor_output = self.actor(**args)
            action = getattr(actor_output, f"action_{idx}")

            tool_name = action.tool_name
            tool_input_query = action.tool_input_query

            if self.verbose:
                print(f"Action {idx}: {tool_name} ({tool_input_query})")

            if tool_name != "Finish":
                tool_output = self._call_tool(tool_name, tool_input_query)
                action_results.append(
                    ActionOutput(tool_name=tool_name, tool_input_query=tool_input_query, tool_output=tool_output)
                )

                self._update_signature(idx)

                args[f"action_{idx}"] = action
                args[f"result_{idx}"] = tool_output
            else:
                self._update_signature(idx, omit_action=True)

                args[f"action_{idx}"] = action
                args[f"result_{idx}"] = "Gathered all information needed to finish the task."
                break

            idx += 1

            if max_iters:
                max_iters -= 1

        final_answer = self.actor(**args)
        self.actor = deepcopy(self.actor_clone)

        return dspy.Prediction(
            **{key: getattr(final_answer, key) for key in self.output_fields.keys()},
            actions=action_results,
        )
```

## File: `predict/avatar/models.py`
```
from typing import Any

from pydantic import BaseModel, Field


class Tool(BaseModel):
    tool: Any
    name: str
    desc: str | None
    input_type: str | None = None

    def __str__(self) -> str:
        return f"{self.name}{f'(valid_input: {self.input_type})' if self.input_type else ''}: {self.desc}"

    def __repr__(self) -> str:
        return self.__str__()


class Action(BaseModel):
    tool_name: Any = Field(..., description="Name of the tool to use.")
    tool_input_query: Any = Field(..., description="Query to pass as input to the tool.")


class ActionOutput(BaseModel):
    tool_name: str
    tool_input_query: str
    tool_output: str
```

## File: `predict/avatar/signatures.py`
```
import dspy
from dspy.predict.avatar.models import Action


class Actor(dspy.Signature):
    """You will be given `Tools` which will be a list of tools to use to accomplish the `Goal`. Given the user query, your task is to decide which tool to use and what input values to provide.

    You will output action needed to accomplish the `Goal`. `Action` should have a tool to use and the input query to pass to the tool.

    Note: You can opt to use no tools and provide the final answer directly. You can also one tool multiple times with different input queries if applicable."""

    goal: str = dspy.InputField(
        prefix="Goal:",
        desc="Task to be accomplished.",
    )
    tools: list[str] = dspy.InputField(
        prefix="Tools:",
        desc="list of tools to use",
    )
    action_1: Action = dspy.OutputField(
        prefix="Action 1:",
        desc="1st action to take.",
    )
```

## File: `predict/best_of_n.py`
```
from typing import Callable

import dspy
from dspy.predict.predict import Module, Prediction


class BestOfN(Module):
    def __init__(
        self,
        module: Module,
        N: int,  # noqa: N803
        reward_fn: Callable[[dict, Prediction], float],
        threshold: float,
        fail_count: int | None = None,
    ):
        """
        Runs a module up to `N` times with different temperatures and returns the best prediction
        out of `N` attempts or the first prediction that passes the `threshold`.

        Args:
            module (Module): The module to run.
            N (int): The number of times to run the module.
            reward_fn (Callable[[dict, Prediction], float]): The reward function which takes in the args passed to the module, the resulting prediction, and returns a scalar reward.
            threshold (float): The threshold for the reward function.
            fail_count (Optional[int], optional): The number of times the module can fail before raising an error. Defaults to N if not provided.

        Example:
            ```python
            import dspy

            dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

            # Define a QA module with chain of thought
            qa = dspy.ChainOfThought("question -> answer")

            # Define a reward function that checks for one-word answers
            def one_word_answer(args, pred):
                return 1.0 if len(pred.answer.split()) == 1 else 0.0

            # Create a refined module that tries up to 3 times
            best_of_3 = dspy.BestOfN(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0)

            # Use the refined module
            result = best_of_3(question="What is the capital of Belgium?").answer
            # Returns: Brussels
            ```
        """
        self.module = module
        self.reward_fn = lambda *args: reward_fn(*args)  # to prevent this from becoming a parameter
        self.threshold = threshold
        self.N = N
        self.fail_count = fail_count or N  # default to N if fail_count is not provided

    def forward(self, **kwargs):
        lm = self.module.get_lm() or dspy.settings.lm
        temps = [lm.kwargs["temperature"]] + [0.5 + i * (0.5 / self.N) for i in range(self.N)]
        temps = list(dict.fromkeys(temps))[: self.N]
        best_pred, best_trace, best_reward = None, None, -float("inf")

        for idx, t in enumerate(temps):
            lm_ = lm.copy(temperature=t)
            mod = self.module.deepcopy()
            mod.set_lm(lm_)

            try:
                with dspy.context(trace=[]):
                    pred = mod(**kwargs)
                    trace = dspy.settings.trace.copy()

                    # NOTE: Not including the trace of reward_fn.
                    reward = self.reward_fn(kwargs, pred)

                if reward > best_reward:
                    best_reward, best_pred, best_trace = reward, pred, trace

                if reward >= self.threshold:
                    break

            except Exception as e:
                print(f"BestOfN: Attempt {idx + 1} failed with temperature {t}: {e}")
                if idx > self.fail_count:
                    raise e
                self.fail_count -= 1

        if best_trace:
            dspy.settings.trace.extend(best_trace)
        return best_pred
```

## File: `predict/chain_of_thought.py`
```
from typing import Any

from pydantic.fields import FieldInfo

import dspy
from dspy.primitives.module import Module
from dspy.signatures.signature import Signature, ensure_signature


class ChainOfThought(Module):
    def __init__(
        self,
        signature: str | type[Signature],
        rationale_field: FieldInfo | None = None,
        rationale_field_type: type = str,
        **config: dict[str, Any],
    ):
        """
        A module that reasons step by step in order to predict the output of a task.

        Args:
            signature (Type[dspy.Signature]): The signature of the module.
            rationale_field (Optional[Union[dspy.OutputField, pydantic.fields.FieldInfo]]): The field that will contain the reasoning.
            rationale_field_type (Type): The type of the rationale field.
            **config: The configuration for the module.
        """
        super().__init__()
        signature = ensure_signature(signature)
        prefix = "Reasoning: Let's think step by step in order to"
        desc = "${reasoning}"
        rationale_field_type = rationale_field.annotation if rationale_field else rationale_field_type
        rationale_field = rationale_field if rationale_field else dspy.OutputField(prefix=prefix, desc=desc)
        extended_signature = signature.prepend(name="reasoning", field=rationale_field, type_=rationale_field_type)
        self.predict = dspy.Predict(extended_signature, **config)

    def forward(self, **kwargs):
        return self.predict(**kwargs)

    async def aforward(self, **kwargs):
        return await self.predict.acall(**kwargs)
```

## File: `predict/code_act.py`
```
import inspect
import logging
from typing import Callable

import dspy
from dspy.adapters.types.tool import Tool
from dspy.predict.program_of_thought import ProgramOfThought
from dspy.predict.react import ReAct
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.signatures.signature import Signature, ensure_signature

logger = logging.getLogger(__name__)

class CodeAct(ReAct, ProgramOfThought):
    """
    CodeAct is a module that utilizes the Code Interpreter and predefined tools to solve the problem.
    """

    def __init__(self, signature: str | type[Signature], tools: list[Callable], max_iters: int = 5, interpreter: PythonInterpreter | None = None):
        """
        Initializes the CodeAct class with the specified model, temperature, and max tokens.

        Args:
            signature (Union[str, Type[Signature]]): The signature of the module.
            tools (list[Callable]): The tool callables to be used. CodeAct only accepts functions and not callable objects.
            max_iters (int): The maximum number of iterations to generate the answer.
            interpreter: PythonInterpreter instance to use. If None, a new one is instantiated.
        Example:
            ```python
            from dspy.predict import CodeAct
            def factorial(n):
                if n == 1:
                    return 1
                return n * factorial(n-1)

            act = CodeAct("n->factorial", tools=[factorial])
            act(n=5) # 120
            ```
        """
        self.signature = ensure_signature(signature)
        self.max_iters = max_iters
        self.history = []

        tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        if any(
            not inspect.isfunction(tool.func) for tool in tools
        ):
            raise ValueError("CodeAct only accepts functions and not callable objects.")
        tools = {tool.name: tool for tool in tools}

        instructions = self._build_instructions(self.signature, tools)

        codeact_signature = (
            dspy.Signature({**self.signature.input_fields}, "\n".join(instructions))
            .append("trajectory", dspy.InputField(), type_=str)
            .append("generated_code", dspy.OutputField(desc="Python code that when executed, produces output relevant to answering the question"), type_=str)
            .append("finished", dspy.OutputField(desc="a boolean flag to determine if the process is done"), type_=bool)
        )

        extract_signature = dspy.Signature(
            {**self.signature.input_fields, **self.signature.output_fields},
            self.signature.instructions,
        ).append("trajectory", dspy.InputField(), type_=str)

        self.tools: dict[str, Tool] = tools
        self.codeact = dspy.Predict(codeact_signature)
        self.extractor = dspy.ChainOfThought(extract_signature)
        # It will raises exception when dspy cannot find available deno instance by now.
        self.interpreter = interpreter or PythonInterpreter()

    def _build_instructions(self, signature, tools):
        instructions = [f"{signature.instructions}\n"] if signature.instructions else []
        inputs = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
        outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])

        instructions.append(
            f"You are an intelligent agent. For each episode, you will receive the fields {inputs} as input.\n"
            f"Your goal is to generate executable Python code that collects any necessary information for producing {outputs}.\n"
            "For each iteration, you will generate a code snippet that either solves the task or progresses towards the solution.\n"
            "Ensure any output you wish to extract from the code is printed to the console. The code should be enclosed in a fenced code block.\n"
            f"When all information for producing the outputs ({outputs}) are available to be extracted, mark `finished=True` besides the final Python code.\n"
            "You have access to the Python Standard Library and the following functions:"
        )

        for idx, tool in enumerate(tools.values()):
            instructions.append(f"({idx + 1}) {tool}")

        return instructions

    def forward(self, **kwargs):
        # Define the tool funcitons in the interpreter
        for tool in self.tools.values():
            self.interpreter(inspect.getsource(tool.func))

        trajectory = {}
        max_iters = kwargs.pop("max_iters", self.max_iters)
        for idx in range(max_iters):
            code_data = self.codeact(trajectory=trajectory, **kwargs)
            output = None
            code, error = self._parse_code(code_data)

            if error:
                trajectory[f"observation_{idx}"] = f"Failed to parse the generated code: {error}"
                continue

            trajectory[f"generated_code_{idx}"] = code
            output, error = self._execute_code(code)

            if not error:
                trajectory[f"code_output_{idx}"] = output
            else:
                trajectory[f"observation_{idx}"] = f"Failed to execute the generated code: {error}"

            if code_data.finished:
                break

        extract = self._call_with_potential_trajectory_truncation(self.extractor, trajectory, **kwargs)
        self.interpreter.shutdown()
        return dspy.Prediction(trajectory=trajectory, **extract)
```

## File: `predict/knn.py`
```
import numpy as np

from dspy.clients import Embedder
from dspy.primitives import Example


class KNN:
    def __init__(self, k: int, trainset: list[Example], vectorizer: Embedder):
        """
        A k-nearest neighbors retriever that finds similar examples from a training set.

        Args:
            k: Number of nearest neighbors to retrieve
            trainset: List of training examples to search through
            vectorizer: The `Embedder` to use for vectorization

        Example:
            ```python
            import dspy
            from sentence_transformers import SentenceTransformer

            # Create a training dataset with examples
            trainset = [
                dspy.Example(input="hello", output="world"),
                # ... more examples ...
            ]

            # Initialize KNN with a sentence transformer model
            knn = KNN(
                k=3,
                trainset=trainset,
                vectorizer=dspy.Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode)
            )

            # Find similar examples
            similar_examples = knn(input="hello")
            ```
        """
        self.k = k
        self.trainset = trainset
        self.embedding = vectorizer
        trainset_casted_to_vectorize = [
            " | ".join([f"{key}: {value}" for key, value in example.items() if key in example._input_keys])
            for example in self.trainset
        ]
        self.trainset_vectors = self.embedding(trainset_casted_to_vectorize).astype(np.float32)

    def __call__(self, **kwargs) -> list:
        input_example_vector = self.embedding([" | ".join([f"{key}: {val}" for key, val in kwargs.items()])])
        scores = np.dot(self.trainset_vectors, input_example_vector.T).squeeze()
        nearest_samples_idxs = scores.argsort()[-self.k :][::-1]
        return [self.trainset[cur_idx] for cur_idx in nearest_samples_idxs]
```

## File: `predict/multi_chain_comparison.py`
```
from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.signatures import InputField, OutputField
from dspy.signatures.signature import ensure_signature


class MultiChainComparison(Module):
    def __init__(self, signature, M=3, temperature=0.7, **config):  # noqa: N803
        super().__init__()

        self.M = M
        signature = ensure_signature(signature)

        *_, self.last_key = signature.output_fields.keys()

        for idx in range(M):
            signature = signature.append(
                f"reasoning_attempt_{idx+1}",
                InputField(
                    prefix=f"Student Attempt #{idx+1}:",
                    desc="${reasoning attempt}",
                ),
            )

        signature = signature.prepend(
            "rationale",
            OutputField(
                prefix="Accurate Reasoning: Thank you everyone. Let's now holistically",
                desc="${corrected reasoning}",
            ),
        )

        self.predict = Predict(signature, temperature=temperature, **config)

    def forward(self, completions, **kwargs):
        attempts = []

        for c in completions:
            rationale = c.get("rationale", c.get("reasoning")).strip().split("\n")[0].strip()
            answer = str(c[self.last_key]).strip().split("\n")[0].strip()
            attempts.append(
                f"«I'm trying to {rationale} I'm not sure but my prediction is {answer}»",
            )

        assert (
            len(attempts) == self.M
        ), f"The number of attempts ({len(attempts)}) doesn't match the expected number M ({self.M}). Please set the correct value for M when initializing MultiChainComparison."

        kwargs = {
            **{f"reasoning_attempt_{idx+1}": attempt for idx, attempt in enumerate(attempts)},
            **kwargs,
        }
        return self.predict(**kwargs)
```

## File: `predict/parallel.py`
```
import threading
from typing import Any

from dspy.dsp.utils.settings import settings
from dspy.primitives.example import Example
from dspy.utils.parallelizer import ParallelExecutor


class Parallel:
    def __init__(
        self,
        num_threads: int | None = None,
        max_errors: int | None = None,
        access_examples: bool = True,
        return_failed_examples: bool = False,
        provide_traceback: bool | None = None,
        disable_progress_bar: bool = False,
    ):
        super().__init__()
        self.num_threads = num_threads or settings.num_threads
        self.max_errors = settings.max_errors if max_errors is None else max_errors
        self.access_examples = access_examples
        self.return_failed_examples = return_failed_examples
        self.provide_traceback = provide_traceback
        self.disable_progress_bar = disable_progress_bar

        self.error_count = 0
        self.error_lock = threading.Lock()
        self.cancel_jobs = threading.Event()
        self.failed_examples = []
        self.exceptions = []

    def forward(self, exec_pairs: list[tuple[Any, Example]], num_threads: int | None = None) -> list[Any]:
        num_threads = num_threads if num_threads is not None else self.num_threads

        executor = ParallelExecutor(
            num_threads=num_threads,
            max_errors=self.max_errors,
            provide_traceback=self.provide_traceback,
            disable_progress_bar=self.disable_progress_bar,
        )

        def process_pair(pair):
            result = None
            module, example = pair

            if isinstance(example, Example):
                if self.access_examples:
                    result = module(**example.inputs())
                else:
                    result = module(example)
            elif isinstance(example, dict):
                result = module(**example)
            elif isinstance(example, list) and module.__class__.__name__ == "Parallel":
                result = module(example)
            elif isinstance(example, tuple):
                result = module(*example)
            else:
                raise ValueError(
                    f"Invalid example type: {type(example)}, only supported types are Example, dict, list and tuple"
                )
            return result

        # Execute the processing function over the execution pairs
        results = executor.execute(process_pair, exec_pairs)

        if self.return_failed_examples:
            return results, self.failed_examples, self.exceptions
        else:
            return results

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)
```

## File: `predict/parameter.py`
```
class Parameter:
    pass
```

## File: `predict/predict.py`
```
import logging
import random

from pydantic import BaseModel

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.clients.base_lm import BaseLM
from dspy.clients.lm import LM
from dspy.dsp.utils.settings import settings
from dspy.predict.parameter import Parameter
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.signatures.signature import Signature, ensure_signature
from dspy.utils.callback import BaseCallback

logger = logging.getLogger(__name__)


class Predict(Module, Parameter):
    def __init__(self, signature: str | type[Signature], callbacks: list[BaseCallback] | None = None, **config):
        super().__init__(callbacks=callbacks)
        self.stage = random.randbytes(8).hex()
        self.signature = ensure_signature(signature)
        self.config = config
        self.reset()

    def reset(self):
        self.lm = None
        self.traces = []
        self.train = []
        self.demos = []

    def dump_state(self):
        state_keys = ["traces", "train"]
        state = {k: getattr(self, k) for k in state_keys}

        state["demos"] = []
        for demo in self.demos:
            demo = demo.copy()

            for field in demo:
                # FIXME: Saving BaseModels as strings in examples doesn't matter because you never re-access as an object
                demo[field] = serialize_object(demo[field])

            state["demos"].append(demo)

        state["signature"] = self.signature.dump_state()
        state["lm"] = self.lm.dump_state() if self.lm else None
        return state

    def load_state(self, state: dict) -> "Predict":
        """Load the saved state of a `Predict` object.

        Args:
            state: The saved state of a `Predict` object.

        Returns:
            Self to allow method chaining.
        """
        excluded_keys = ["signature", "extended_signature", "lm"]
        for name, value in state.items():
            # `excluded_keys` are fields that go through special handling.
            if name not in excluded_keys:
                setattr(self, name, value)

        self.signature = self.signature.load_state(state["signature"])
        self.lm = LM(**state["lm"]) if state["lm"] else None

        if "extended_signature" in state:  # legacy, up to and including 2.5, for CoT.
            raise NotImplementedError("Loading extended_signature is no longer supported in DSPy 2.6+")

        return self

    def _get_positional_args_error_message(self):
        input_fields = list(self.signature.input_fields.keys())
        return (
            "Positional arguments are not allowed when calling `dspy.Predict`, must use keyword arguments "
            f"that match your signature input fields: '{', '.join(input_fields)}'. For example: "
            f"`predict({input_fields[0]}=input_value, ...)`."
        )

    def __call__(self, *args, **kwargs):
        if args:
            raise ValueError(self._get_positional_args_error_message())

        return super().__call__(**kwargs)

    async def acall(self, *args, **kwargs):
        if args:
            raise ValueError(self._get_positional_args_error_message())

        return await super().acall(**kwargs)

    def _forward_preprocess(self, **kwargs):
        # Extract the three privileged keyword arguments.
        assert "new_signature" not in kwargs, "new_signature is no longer a valid keyword argument."
        signature = ensure_signature(kwargs.pop("signature", self.signature))
        demos = kwargs.pop("demos", self.demos)
        config = dict(**self.config, **kwargs.pop("config", {}))

        # Get the right LM to use.
        lm = kwargs.pop("lm", self.lm) or settings.lm

        if lm is None:
            raise ValueError(
                "No LM is loaded. Please configure the LM using `dspy.configure(lm=dspy.LM(...))`. e.g, "
                "`dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))`"
            )

        if isinstance(lm, str):
            # Many users mistakenly use `dspy.configure(lm="openai/gpt-4o-mini")` instead of
            # `dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))`, so we are providing a specific error message.
            raise ValueError(
                f"LM must be an instance of `dspy.BaseLM`, not a string. Instead of using a string like "
                f"'dspy.configure(lm=\"{lm}\")', please configure the LM like 'dspy.configure(lm=dspy.LM(\"{lm}\"))'"
            )
        elif not isinstance(lm, BaseLM):
            raise ValueError(f"LM must be an instance of `dspy.BaseLM`, not {type(lm)}. Received `lm={lm}`.")

        # If temperature is unset or <=0.15, and n > 1, set temperature to 0.7 to keep randomness.
        temperature = config.get("temperature") or lm.kwargs.get("temperature")
        num_generations = config.get("n") or lm.kwargs.get("n") or lm.kwargs.get("num_generations") or 1

        if (temperature is None or temperature <= 0.15) and num_generations > 1:
            config["temperature"] = 0.7

        if "prediction" in kwargs:
            if (
                isinstance(kwargs["prediction"], dict)
                and kwargs["prediction"].get("type") == "content"
                and "content" in kwargs["prediction"]
            ):
                # If the `prediction` is the standard predicted outputs format
                # (https://platform.openai.com/docs/guides/predicted-outputs), we remvoe it from input kwargs and add it
                # to the lm kwargs.
                config["prediction"] = kwargs.pop("prediction")

        if not all(k in kwargs for k in signature.input_fields):
            present = [k for k in signature.input_fields if k in kwargs]
            missing = [k for k in signature.input_fields if k not in kwargs]
            logger.warning(
                "Not all input fields were provided to module. Present: %s. Missing: %s.",
                present,
                missing,
            )
        return lm, config, signature, demos, kwargs

    def _forward_postprocess(self, completions, signature, **kwargs):
        pred = Prediction.from_completions(completions, signature=signature)
        if kwargs.pop("_trace", True) and settings.trace is not None:
            trace = settings.trace
            trace.append((self, {**kwargs}, pred))
        return pred

    def _should_stream(self):
        stream_listeners = settings.stream_listeners or []
        should_stream = settings.send_stream is not None
        if should_stream and len(stream_listeners) > 0:
            should_stream = any(stream_listener.predict == self for stream_listener in stream_listeners)

        return should_stream

    def forward(self, **kwargs):
        lm, config, signature, demos, kwargs = self._forward_preprocess(**kwargs)

        adapter = settings.adapter or ChatAdapter()

        if self._should_stream():
            with settings.context(caller_predict=self):
                completions = adapter(lm, lm_kwargs=config, signature=signature, demos=demos, inputs=kwargs)
        else:
            with settings.context(send_stream=None):
                completions = adapter(lm, lm_kwargs=config, signature=signature, demos=demos, inputs=kwargs)

        return self._forward_postprocess(completions, signature, **kwargs)

    async def aforward(self, **kwargs):
        lm, config, signature, demos, kwargs = self._forward_preprocess(**kwargs)

        adapter = settings.adapter or ChatAdapter()
        if self._should_stream():
            with settings.context(caller_predict=self):
                completions = await adapter.acall(lm, lm_kwargs=config, signature=signature, demos=demos, inputs=kwargs)
        else:
            with settings.context(send_stream=None):
                completions = await adapter.acall(lm, lm_kwargs=config, signature=signature, demos=demos, inputs=kwargs)

        return self._forward_postprocess(completions, signature, **kwargs)

    def update_config(self, **kwargs):
        self.config = {**self.config, **kwargs}

    def get_config(self):
        return self.config

    def __repr__(self):
        return f"{self.__class__.__name__}({self.signature})"


def serialize_object(obj):
    """
    Recursively serialize a given object into a JSON-compatible format.
    Supports Pydantic models, lists, dicts, and primitive types.
    """
    if isinstance(obj, BaseModel):
        # Use model_dump to convert the model into a JSON-serializable dict
        return obj.model_dump()
    elif isinstance(obj, list):
        return [serialize_object(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(serialize_object(item) for item in obj)
    elif isinstance(obj, dict):
        return {key: serialize_object(value) for key, value in obj.items()}
    else:
        return obj


# # TODO: FIXME: Hmm, I guess expected behavior is that contexts can
# affect execution. Well, we need to determine whether context dominates, __init__ demoninates, or forward dominates.
# Generally, unless overwritten, we'd see n=None, temperature=None.
# That will eventually mean we have to learn them.
```

## File: `predict/program_of_thought.py`
```
import json
import logging
import re

import dspy
from dspy.primitives.module import Module
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.signatures.signature import Signature, ensure_signature

logger = logging.getLogger(__name__)


class ProgramOfThought(Module):
    """
    A DSPy module that runs Python programs to solve a problem.
    This module requires deno to be installed. Please install deno following https://docs.deno.com/runtime/getting_started/installation/

    Example:
    ```
    import dspy

    lm = dspy.LM('openai/gpt-4o-mini')
    dspy.configure(lm=lm)
    pot = dspy.ProgramOfThought("question -> answer")
    pot(question="what is 1+1?")
    ```
    """

    def __init__(self, signature: str | type[Signature], max_iters: int = 3, interpreter: PythonInterpreter | None = None):
        """
        Args:
            signature: The signature of the module.
            max_iters: The maximum number of iterations to retry code generation and execution.
            interpreter: PythonInterpreter instance to use. If None, a new one is instantiated.
        """
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters

        self.input_fields = signature.input_fields
        self.output_fields = signature.output_fields

        self.code_generate = dspy.ChainOfThought(
            dspy.Signature(
                self._generate_signature("generate").fields,
                self._generate_instruction("generate"),
            ),
        )
        self.code_regenerate = dspy.ChainOfThought(
            dspy.Signature(
                self._generate_signature("regenerate").fields,
                self._generate_instruction("regenerate"),
            ),
        )
        self.generate_answer = dspy.ChainOfThought(
            dspy.Signature(
                self._generate_signature("answer").fields,
                self._generate_instruction("answer"),
            ),
        )
        # It will raises exception when dspy cannot find available deno instance by now.
        self.interpreter = interpreter or PythonInterpreter()

    def _generate_signature(self, mode):
        signature_dict = dict(self.input_fields)
        fields_for_mode = {
            "generate": {
                "generated_code": dspy.OutputField(
                    prefix="Code:",
                    desc="python code that answers the question",
                    format=str,
                ),
            },
            "regenerate": {
                "previous_code": dspy.InputField(
                    prefix="Previous Code:",
                    desc="previously-generated python code that errored",
                    format=str,
                ),
                "error": dspy.InputField(
                    prefix="Error:",
                    desc="error message from previously-generated python code",
                ),
                "generated_code": dspy.OutputField(
                    prefix="Code:",
                    desc="python code that answers the question",
                    format=str,
                ),
            },
            "answer": {
                "final_generated_code": dspy.InputField(
                    prefix="Code:",
                    desc="python code that answers the question",
                    format=str,
                ),
                "code_output": dspy.InputField(
                    prefix="Code Output:",
                    desc="output of previously-generated python code",
                ),
            }
            | self.signature.output_fields,
        }
        signature_dict.update(fields_for_mode[mode])
        return dspy.Signature(signature_dict)

    def _generate_instruction(self, mode):
        mode_inputs = ", ".join(
            [f"`{field_name}`" for field_name in self._generate_signature(mode).input_fields],
        )
        mode_outputs = ", ".join(
            [f"`{field_name}`" for field_name in self._generate_signature(mode).output_fields],
        )
        final_outputs = ", ".join(
            [f"`{field_name}`" for field_name in self.output_fields],
        )
        if mode == "generate":
            instr = [
                f"You will be given {mode_inputs} and you will respond with {mode_outputs}.",
                f"Generating executable Python code that programmatically computes the correct {mode_outputs}.",
                "After you're done with the computation and think you have the answer, make sure to provide your answer by calling the preloaded function `final_answer()`.",
                f'You should structure your answer in a dict object, like {{"field_a": answer_a, ...}}, evaluates to the correct value mapping for {final_outputs}.',
            ]
        elif mode == "regenerate":
            instr = [
                f"You are given {mode_inputs} due to an error in previous code.",
                "Your task is to correct the error and provide the new `generated_code`.",
            ]
        else:  # mode == 'answer'
            instr = [
                f"Given the final code {mode_inputs}, provide the final {mode_outputs}.",
            ]

        return "\n".join(instr)

    def _parse_code(self, code_data):
        code = code_data.get("generated_code", "").split("---", 1)[0].split("\n\n\n", 1)[0]
        code_match = re.search(r"```python[ \n](.*?)[ \n]```?", code, re.DOTALL)
        code_block = (code_match.group(1) if code_match else code).replace("\\n", "\n")
        if not code_block:
            return code, "Error: Empty code after parsing."
        if "\n" not in code_block and code_block.count("=") > 1:
            return code, "Error: Code format is not correct."
        lines = code_block.split("\n")
        last_line_match = re.match(r"^(\w+)\s*=", lines[-1].strip())
        if last_line_match and len(lines) > 1:
            code_block += "\n" + last_line_match.group(1)
        else:
            code_block = re.sub(
                r"([a-zA-Z_]\w* *=.*?)(?=[a-zA-Z_]\w* *=)",
                r"\1\n",
                code_block,
            )
            code_block = re.sub(
                r"([a-zA-Z_]\w* *=.*?)([a-zA-Z_]\w*)$",
                r"\1\n\2",
                code_block,
            )
        return code_block, None

    def _execute_code(self, code):
        """
        Execute the code using PythonInterpreter and return the output or error.
        """
        if not code:
            return None, "Error: Empty code before execution."

        try:
            # Since it's more complex structure now, just blindly use json to represents all.
            output = json.dumps(self.interpreter.execute(code))
            return output, None
        except Exception as e:
            return None, str(e)

    def forward(self, **kwargs):
        input_kwargs = {field_name: kwargs[field_name] for field_name in self.input_fields}
        code_data = self.code_generate(**input_kwargs)
        output = None
        code, error = self._parse_code(code_data)
        if not error:
            output, error = self._execute_code(code)
        hop = 1
        # Retying code generation and execution until no error or reach max_iters
        while error is not None:
            logger.error(f"Error in code execution: {error}")
            if hop == self.max_iters:
                self.interpreter.shutdown()
                raise RuntimeError(f"Max hops reached. Failed to run ProgramOfThought: {error}")
            input_kwargs.update({"previous_code": code, "error": error})
            code_data = self.code_regenerate(**input_kwargs)
            code, error = self._parse_code(code_data)
            if not error:
                output, error = self._execute_code(code)
            hop += 1
        input_kwargs.update({"final_generated_code": code, "code_output": output})
        answer_gen_result = self.generate_answer(**input_kwargs)
        self.interpreter.shutdown()
        return answer_gen_result
```

## File: `predict/react.py`
```
import logging
from typing import TYPE_CHECKING, Any, Callable, Literal

from litellm import ContextWindowExceededError

import dspy
from dspy.adapters.types.tool import Tool
from dspy.primitives.module import Module
from dspy.signatures.signature import ensure_signature

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature


class ReAct(Module):
    def __init__(self, signature: type["Signature"], tools: list[Callable], max_iters: int = 10):
        """
        ReAct stands for "Reasoning and Acting," a popular paradigm for building tool-using agents.
        In this approach, the language model is iteratively provided with a list of tools and has
        to reason about the current situation. The model decides whether to call a tool to gather more
        information or to finish the task based on its reasoning process. The DSPy version of ReAct is
        generalized to work over any signature, thanks to signature polymorphism.

        Args:
            signature: The signature of the module, which defines the input and output of the react module.
            tools (list[Callable]): A list of functions, callable objects, or `dspy.Tool` instances.
            max_iters (Optional[int]): The maximum number of iterations to run. Defaults to 10.

        Example:

        ```python
        def get_weather(city: str) -> str:
            return f"The weather in {city} is sunny."

        react = dspy.ReAct(signature="question->answer", tools=[get_weather])
        pred = react(question="What is the weather in Tokyo?")
        ```
        """
        super().__init__()
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters

        tools = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        tools = {tool.name: tool for tool in tools}

        inputs = ", ".join([f"`{k}`" for k in signature.input_fields.keys()])
        outputs = ", ".join([f"`{k}`" for k in signature.output_fields.keys()])
        instr = [f"{signature.instructions}\n"] if signature.instructions else []

        instr.extend(
            [
                f"You are an Agent. In each episode, you will be given the fields {inputs} as input. And you can see your past trajectory so far.",
                f"Your goal is to use one or more of the supplied tools to collect any necessary information for producing {outputs}.\n",
                "To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.",
                "After each tool call, you receive a resulting observation, which gets appended to your trajectory.\n",
                "When writing next_thought, you may reason about the current situation and plan for future steps.",
                "When selecting the next_tool_name and its next_tool_args, the tool must be one of:\n",
            ]
        )

        tools["finish"] = Tool(
            func=lambda: "Completed.",
            name="finish",
            desc=f"Marks the task as complete. That is, signals that all information for producing the outputs, i.e. {outputs}, are now available to be extracted.",
            args={},
        )

        for idx, tool in enumerate(tools.values()):
            instr.append(f"({idx + 1}) {tool}")
        instr.append("When providing `next_tool_args`, the value inside the field must be in JSON format")

        react_signature = (
            dspy.Signature({**signature.input_fields}, "\n".join(instr))
            .append("trajectory", dspy.InputField(), type_=str)
            .append("next_thought", dspy.OutputField(), type_=str)
            .append("next_tool_name", dspy.OutputField(), type_=Literal[tuple(tools.keys())])
            .append("next_tool_args", dspy.OutputField(), type_=dict[str, Any])
        )

        fallback_signature = dspy.Signature(
            {**signature.input_fields, **signature.output_fields},
            signature.instructions,
        ).append("trajectory", dspy.InputField(), type_=str)

        self.tools = tools
        self.react = dspy.Predict(react_signature)
        self.extract = dspy.ChainOfThought(fallback_signature)

    def _format_trajectory(self, trajectory: dict[str, Any]):
        adapter = dspy.settings.adapter or dspy.ChatAdapter()
        trajectory_signature = dspy.Signature(f"{', '.join(trajectory.keys())} -> x")
        return adapter.format_user_message_content(trajectory_signature, trajectory)

    def forward(self, **input_args):
        trajectory = {}
        max_iters = input_args.pop("max_iters", self.max_iters)
        for idx in range(max_iters):
            try:
                pred = self._call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
            except ValueError as err:
                logger.warning(f"Ending the trajectory: Agent failed to select a valid tool: {_fmt_exc(err)}")
                break

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            try:
                trajectory[f"observation_{idx}"] = self.tools[pred.next_tool_name](**pred.next_tool_args)
            except Exception as err:
                trajectory[f"observation_{idx}"] = f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"

            if pred.next_tool_name == "finish":
                break

        extract = self._call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
        return dspy.Prediction(trajectory=trajectory, **extract)

    async def aforward(self, **input_args):
        trajectory = {}
        max_iters = input_args.pop("max_iters", self.max_iters)
        for idx in range(max_iters):
            try:
                pred = await self._async_call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
            except ValueError as err:
                logger.warning(f"Ending the trajectory: Agent failed to select a valid tool: {_fmt_exc(err)}")
                break

            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            try:
                trajectory[f"observation_{idx}"] = await self.tools[pred.next_tool_name].acall(**pred.next_tool_args)
            except Exception as err:
                trajectory[f"observation_{idx}"] = f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"

            if pred.next_tool_name == "finish":
                break

        extract = await self._async_call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
        return dspy.Prediction(trajectory=trajectory, **extract)

    def _call_with_potential_trajectory_truncation(self, module, trajectory, **input_args):
        for _ in range(3):
            try:
                return module(
                    **input_args,
                    trajectory=self._format_trajectory(trajectory),
                )
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                trajectory = self.truncate_trajectory(trajectory)

    async def _async_call_with_potential_trajectory_truncation(self, module, trajectory, **input_args):
        for _ in range(3):
            try:
                return await module.acall(
                    **input_args,
                    trajectory=self._format_trajectory(trajectory),
                )
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                trajectory = self.truncate_trajectory(trajectory)

    def truncate_trajectory(self, trajectory):
        """Truncates the trajectory so that it fits in the context window.

        Users can override this method to implement their own truncation logic.
        """
        keys = list(trajectory.keys())
        if len(keys) < 4:
            # Every tool call has 4 keys: thought, tool_name, tool_args, and observation.
            raise ValueError(
                "The trajectory is too long so your prompt exceeded the context window, but the trajectory cannot be "
                "truncated because it only has one tool call."
            )

        for key in keys[:4]:
            trajectory.pop(key)

        return trajectory


def _fmt_exc(err: BaseException, *, limit: int = 5) -> str:
    """
    Return a one-string traceback summary.
    * `limit` - how many stack frames to keep (from the innermost outwards).
    """

    import traceback

    return "\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__, limit=limit)).strip()


"""
Thoughts and Planned Improvements for dspy.ReAct.

TOPIC 01: How Trajectories are Formatted, or rather when they are formatted.

Right now, both sub-modules are invoked with a `trajectory` argument, which is a string formatted in `forward`. Though
the formatter uses a general adapter.format_fields, the tracing of DSPy only sees the string, not the formatting logic.

What this means is that, in demonstrations, even if the user adjusts the adapter for a fixed program, the demos' format
will not update accordingly, but the inference-time trajectories will.

One way to fix this is to support `format=fn` in the dspy.InputField() for "trajectory" in the signatures. But this
means that care must be taken that the adapter is accessed at `forward` runtime, not signature definition time.

Another potential fix is to more natively support a "variadic" input field, where the input is a list of dictionaries,
or a big dictionary, and have each adatper format it accordingly.

Trajectories also affect meta-programming modules that view the trace later. It's inefficient O(n^2) to view the
trace of every module repeating the prefix.


TOPIC 03: Simplifying ReAct's __init__ by moving modular logic to the Tool class.
    * Handling exceptions and error messages.
    * More cleanly defining the "finish" tool, perhaps as a runtime-defined function?


TOPIC 04: Default behavior when the trajectory gets too long.


TOPIC 05: Adding more structure around how the instruction is formatted.
    * Concretely, it's now a string, so an optimizer can and does rewrite it freely.
    * An alternative would be to add more structure, such that a certain template is fixed but values are variable?


TOPIC 06: Idiomatically allowing tools that maintain state across iterations, but not across different `forward` calls.
    * So the tool would be newly initialized at the start of each `forward` call, but maintain state across iterations.
    * This is pretty useful for allowing the agent to keep notes or count certain things, etc.
"""
```

## File: `predict/refine.py`
```
import inspect
import textwrap
from typing import Callable

import ujson

import dspy
from dspy.adapters.utils import get_field_description_string
from dspy.predict.predict import Prediction
from dspy.signatures import InputField, OutputField, Signature

from .predict import Module


class OfferFeedback(Signature):
    """
    In the discussion, assign blame to each module that contributed to the final reward being below the threshold, if
    any. Then, prescribe concrete advice of how the module should act on its future input when we retry the process, if
    it were to receive the same or similar inputs. If a module is not to blame, the advice should be N/A.
    The module will not see its own history, so it needs to rely on entirely concrete and actionable advice from you
    to avoid the same mistake on the same or similar inputs.
    """

    program_code: str = InputField(desc="The code of the program that we are analyzing")
    modules_defn: str = InputField(desc="The definition of each module in the program, including its I/O")
    program_inputs: str = InputField(desc="The inputs to the program that we are analyzing")
    program_trajectory: str = InputField(desc="The trajectory of the program's execution, showing each module's I/O")
    program_outputs: str = InputField(desc="The outputs of the program that we are analyzing")
    reward_code: str = InputField(desc="The code of the reward function that we are analyzing")
    target_threshold: float = InputField(desc="The target threshold for the reward function")
    reward_value: float = InputField(desc="The reward value assigned to the program's outputs")
    module_names: list[str] = InputField(desc="The names of the modules in the program, for which we seek advice")
    discussion: str = OutputField(desc="Discussing blame of where each module went wrong, if it did")
    advice: dict[str, str] = OutputField(
        desc="For each module, describe very concretely, in this order: the specific scenarios in which it has made "
        "mistakes in the past and what each mistake was, followed by what it should do differently in that kind of"
        "scenario in the future. If the module is not to blame, write N/A."
    )


class Refine(Module):
    def __init__(
        self,
        module: Module,
        N: int,  # noqa: N803
        reward_fn: Callable[[dict, Prediction], float],
        threshold: float,
        fail_count: int | None = None,
    ):
        """
        Refines a module by running it up to N times with different temperatures and returns the best prediction.

        This module runs the provided module multiple times with varying temperature settings and selects
        either the first prediction that exceeds the specified threshold or the one with the highest reward.
        If no prediction meets the threshold, it automatically generates feedback to improve future predictions.


        Args:
            module (Module): The module to refine.
            N (int): The number of times to run the module. must
            reward_fn (Callable): The reward function.
            threshold (float): The threshold for the reward function.
            fail_count (Optional[int], optional): The number of times the module can fail before raising an error

        Example:
            ```python
            import dspy

            dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

            # Define a QA module with chain of thought
            qa = dspy.ChainOfThought("question -> answer")

            # Define a reward function that checks for one-word answers
            def one_word_answer(args, pred):
                return 1.0 if len(pred.answer.split()) == 1 else 0.0

            # Create a refined module that tries up to 3 times
            best_of_3 = dspy.Refine(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0)

            # Use the refined module
            result = best_of_3(question="What is the capital of Belgium?").answer
            # Returns: Brussels
            ```
        """
        self.module = module
        self.reward_fn = lambda *args: reward_fn(*args)  # to prevent this from becoming a parameter
        self.threshold = threshold
        self.N = N
        self.fail_count = fail_count or N  # default to N if fail_count is not provided
        self.module_code = inspect.getsource(module.__class__)
        try:
            self.reward_fn_code = inspect.getsource(reward_fn)
        except TypeError:
            self.reward_fn_code = inspect.getsource(reward_fn.__class__)

    def forward(self, **kwargs):
        lm = self.module.get_lm() or dspy.settings.lm
        temps = [lm.kwargs["temperature"]] + [0.5 + i * (0.5 / self.N) for i in range(self.N)]
        temps = list(dict.fromkeys(temps))[: self.N]
        best_pred, best_trace, best_reward = None, None, -float("inf")
        advice = None
        adapter = dspy.settings.adapter or dspy.ChatAdapter()

        for idx, t in enumerate(temps):
            lm_ = lm.copy(temperature=t)
            mod = self.module.deepcopy()
            mod.set_lm(lm_)

            predictor2name = {predictor: name for name, predictor in mod.named_predictors()}
            signature2name = {predictor.signature: name for name, predictor in mod.named_predictors()}
            module_names = [name for name, _ in mod.named_predictors()]

            try:
                with dspy.context(trace=[]):
                    if not advice:
                        outputs = mod(**kwargs)
                    else:

                        class WrapperAdapter(adapter.__class__):
                            def __call__(self, lm, lm_kwargs, signature, demos, inputs):
                                inputs["hint_"] = advice.get(signature2name[signature], "N/A")  # noqa: B023
                                signature = signature.append(
                                    "hint_", InputField(desc="A hint to the module from an earlier run")
                                )
                                return adapter(lm, lm_kwargs, signature, demos, inputs)

                        with dspy.context(adapter=WrapperAdapter()):
                            outputs = mod(**kwargs)

                    trace = dspy.settings.trace.copy()

                    # TODO: Remove the hint from the trace, if it's there.

                    # NOTE: Not including the trace of reward_fn.
                    reward = self.reward_fn(kwargs, outputs)

                if reward > best_reward:
                    best_reward, best_pred, best_trace = reward, outputs, trace

                if self.threshold is not None and reward >= self.threshold:
                    break

                if idx == self.N - 1:
                    break

                modules = {"program_code": self.module_code, "modules_defn": inspect_modules(mod)}
                trajectory = [{"module_name": predictor2name[p], "inputs": i, "outputs": dict(o)} for p, i, o in trace]
                trajectory = {
                    "program_inputs": kwargs,
                    "program_trajectory": trajectory,
                    "program_outputs": dict(outputs),
                }
                reward = {
                    "reward_code": self.reward_fn_code,
                    "target_threshold": self.threshold,
                    "reward_value": reward,
                }

                advise_kwargs = dict(**modules, **trajectory, **reward, module_names=module_names)
                # advise_kwargs = {k: ujson.dumps(recursive_mask(v), indent=2) for k, v in advise_kwargs.items()}
                # only dumps if it's a list or dict
                advise_kwargs = {
                    k: v if isinstance(v, str) else ujson.dumps(recursive_mask(v), indent=2)
                    for k, v in advise_kwargs.items()
                }
                advice = dspy.Predict(OfferFeedback)(**advise_kwargs).advice
                # print(f"Advice for each module: {advice}")

            except Exception as e:
                print(f"Refine: Attempt failed with temperature {t}: {e}")
                if idx > self.fail_count:
                    raise e
                self.fail_count -= 1
        if best_trace:
            dspy.settings.trace.extend(best_trace)
        return best_pred


def inspect_modules(program):
    separator = "-" * 80
    output = [separator]

    for _, (name, predictor) in enumerate(program.named_predictors()):
        signature = predictor.signature
        instructions = textwrap.dedent(signature.instructions)
        instructions = ("\n" + "\t" * 2).join([""] + instructions.splitlines())

        output.append(f"Module {name}")
        output.append("\n\tInput Fields:")
        output.append(("\n" + "\t" * 2).join([""] + get_field_description_string(signature.input_fields).splitlines()))
        output.append("\tOutput Fields:")
        output.append(("\n" + "\t" * 2).join([""] + get_field_description_string(signature.output_fields).splitlines()))
        output.append(f"\tOriginal Instructions: {instructions}")
        output.append(separator)

    return "\n".join([o.strip("\n") for o in output])


def recursive_mask(o):
    # If the object is already serializable, return it.
    try:
        ujson.dumps(o)
        return o
    except TypeError:
        pass

    # If it's a dictionary, apply recursively to its values.
    if isinstance(o, dict):
        return {k: recursive_mask(v) for k, v in o.items()}
    # If it's a list, apply recursively.
    elif isinstance(o, list):
        return [recursive_mask(v) for v in o]
    # If it's a tuple, apply recursively.
    elif isinstance(o, tuple):
        return tuple(recursive_mask(v) for v in o)
    # Otherwise, replace it with a placeholder string (or use repr(o)).
    else:
        return f"<non-serializable: {type(o).__name__}>"
```

## File: `predict/retry.py`
```
# import copy

# import dspy

# from .predict import Predict


# class Retry(Predict):
#     def __init__(self, module):
#         super().__init__(module.signature)
#         self.module = module
#         self.original_signature = module.signature
#         self.original_forward = module.forward
#         self.new_signature = self._create_new_signature(self.original_signature)

#     def _create_new_signature(self, signature):
#         # Add "Past" input fields for each output field
#         for key, value in signature.output_fields.items():
#             actual_prefix = value.json_schema_extra["prefix"].split(":")[0] + ":"
#             signature = signature.append(f"past_{key}", dspy.InputField(
#                 prefix="Previous " + actual_prefix,
#                 desc=f"past {actual_prefix[:-1]} with errors",
#                 format=value.json_schema_extra.get("format"),
#             ))

#         signature = signature.append("feedback", dspy.InputField(
#             prefix="Instructions:",
#             desc="Some instructions you must satisfy",
#             format=str,
#         ))

#         return signature

#     def forward(self, *, past_outputs, **kwargs):
#         # Take into account the possible new signature, as in TypedPredictor
#         new_signature = kwargs.pop("new_signature", None)
#         if new_signature:
#             self.original_signature = new_signature
#             self.new_signature = self._create_new_signature(self.original_signature)

#         # Convert the dict past_outputs={"answer": ...} to kwargs
#         # {past_answer=..., ...}
#         for key, value in past_outputs.items():
#             past_key = f"past_{key}"
#             if past_key in self.new_signature.input_fields:
#                 kwargs[past_key] = value
#         # Tell the wrapped module to use the new signature.
#         # Note: This only works if the wrapped module is a Predict or ChainOfThought.
#         kwargs["new_signature"] = self.new_signature
#         return self.original_forward(**kwargs)

#     def __call__(self, **kwargs):
#         copy.deepcopy(kwargs)
#         kwargs["_trace"] = False
#         kwargs.setdefault("demos", self.demos if self.demos is not None else [])

#         # perform backtracking
#         if dspy.settings.backtrack_to == self:
#             for key, value in dspy.settings.backtrack_to_args.items():
#                 kwargs.setdefault(key, value)
#             pred = self.forward(**kwargs)
#         else:
#             pred = self.module(**kwargs)

#         # now pop multiple reserved keys
#         # NOTE(shangyin) past_outputs seems not useful to include in demos,
#         # therefore dropped
#         for key in ["_trace", "demos", "signature", "new_signature", "config", "lm", "past_outputs"]:
#             kwargs.pop(key, None)

#         if dspy.settings.trace is not None:
#             trace = dspy.settings.trace
#             trace.append((self, {**kwargs}, pred))
#         return pred
```

## File: `primitives/__init__.py`
```
from dspy.primitives.base_module import BaseModule
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.primitives.prediction import Completions, Prediction
from dspy.primitives.python_interpreter import PythonInterpreter

__all__ = [
    "Example",
    "BaseModule",
    "Prediction",
    "Completions",
    "Module",
    "PythonInterpreter",
]
```

## File: `primitives/base_module.py`
```
import copy
import logging
from collections import deque
from collections.abc import Generator
from pathlib import Path

import cloudpickle
import ujson

from dspy.utils.saving import get_dependency_versions

# NOTE: Note: It's important (temporary decision) to maintain named_parameters that's different in behavior from
# named_sub_modules for the time being.


logger = logging.getLogger(__name__)


class BaseModule:
    def __init__(self):
        pass

    def named_parameters(self):
        """
        Unlike PyTorch, handles (non-recursive) lists of parameters too.
        """

        import dspy
        from dspy.predict.parameter import Parameter

        visited = set()
        named_parameters = []

        def add_parameter(param_name, param_value):
            if isinstance(param_value, Parameter):
                if id(param_value) not in visited:
                    visited.add(id(param_value))
                    named_parameters.append((param_name, param_value))

            elif isinstance(param_value, dspy.Module):
                # When a sub-module is pre-compiled, keep it frozen.
                if not getattr(param_value, "_compiled", False):
                    for sub_name, param in param_value.named_parameters():
                        add_parameter(f"{param_name}.{sub_name}", param)

        if isinstance(self, Parameter):
            add_parameter("self", self)

        for name, value in self.__dict__.items():
            if isinstance(value, Parameter):
                add_parameter(name, value)

            elif isinstance(value, dspy.Module):
                # When a sub-module is pre-compiled, keep it frozen.
                if not getattr(value, "_compiled", False):
                    for sub_name, param in value.named_parameters():
                        add_parameter(f"{name}.{sub_name}", param)

            elif isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    add_parameter(f"{name}[{idx}]", item)

            elif isinstance(value, dict):
                for key, item in value.items():
                    add_parameter(f"{name}['{key}']", item)

        return named_parameters

    def named_sub_modules(self, type_=None, skip_compiled=False) -> Generator[tuple[str, "BaseModule"], None, None]:
        """Find all sub-modules in the module, as well as their names.

        Say `self.children[4]['key'].sub_module` is a sub-module. Then the name will be
        `children[4]['key'].sub_module`. But if the sub-module is accessible at different
        paths, only one of the paths will be returned.
        """
        if type_ is None:
            type_ = BaseModule

        queue = deque([("self", self)])
        seen = {id(self)}

        def add_to_queue(name, item):
            if id(item) not in seen:
                seen.add(id(item))
                queue.append((name, item))

        while queue:
            name, item = queue.popleft()

            if isinstance(item, type_):
                yield name, item

            if isinstance(item, BaseModule):
                if skip_compiled and getattr(item, "_compiled", False):
                    continue
                for sub_name, sub_item in item.__dict__.items():
                    add_to_queue(f"{name}.{sub_name}", sub_item)

            elif isinstance(item, (list, tuple)):
                for i, sub_item in enumerate(item):
                    add_to_queue(f"{name}[{i}]", sub_item)

            elif isinstance(item, dict):
                for key, sub_item in item.items():
                    add_to_queue(f"{name}[{key}]", sub_item)

    def parameters(self):
        return [param for _, param in self.named_parameters()]

    def deepcopy(self):
        """Deep copy the module.

        This is a tweak to the default python deepcopy that only deep copies `self.parameters()`, and for other
        attributes, we just do the shallow copy.
        """
        try:
            # If the instance itself is copyable, we can just deep copy it.
            # Otherwise we will have to create a new instance and copy over the attributes one by one.
            return copy.deepcopy(self)
        except Exception:
            pass

        # Create an empty instance.
        new_instance = self.__class__.__new__(self.__class__)
        # Set attribuetes of the copied instance.
        for attr, value in self.__dict__.items():
            if isinstance(value, BaseModule):
                setattr(new_instance, attr, value.deepcopy())
            else:
                try:
                    # Try to deep copy the attribute
                    setattr(new_instance, attr, copy.deepcopy(value))
                except Exception:
                    logging.warning(
                        f"Failed to deep copy attribute '{attr}' of {self.__class__.__name__}, "
                        "falling back to shallow copy or reference copy."
                    )
                    try:
                        # Fallback to shallow copy if deep copy fails
                        setattr(new_instance, attr, copy.copy(value))
                    except Exception:
                        # If even the shallow copy fails, we just copy over the reference.
                        setattr(new_instance, attr, value)

        return new_instance

    def reset_copy(self):
        """Deep copy the module and reset all parameters."""
        new_instance = self.deepcopy()

        for param in new_instance.parameters():
            param.reset()

        return new_instance

    def dump_state(self):
        return {name: param.dump_state() for name, param in self.named_parameters()}

    def load_state(self, state):
        for name, param in self.named_parameters():
            param.load_state(state[name])

    def save(self, path, save_program=False, modules_to_serialize=None):
        """Save the module.

        Save the module to a directory or a file. There are two modes:
        - `save_program=False`: Save only the state of the module to a json or pickle file, based on the value of
            the file extension.
        - `save_program=True`: Save the whole module to a directory via cloudpickle, which contains both the state and
            architecture of the model.

        If `save_program=True` and `modules_to_serialize` are provided, it will register those modules for serialization 
        with cloudpickle's `register_pickle_by_value`. This causes cloudpickle to serialize the module by value rather 
        than by reference, ensuring the module is fully preserved along with the saved program. This is useful 
        when you have custom modules that need to be serialized alongside your program. If None, then no modules 
        will be registered for serialization.

        We also save the dependency versions, so that the loaded model can check if there is a version mismatch on
        critical dependencies or DSPy version.

        Args:
            path (str): Path to the saved state file, which should be a .json or .pkl file when `save_program=False`,
                and a directory when `save_program=True`.
            save_program (bool): If True, save the whole module to a directory via cloudpickle, otherwise only save
                the state.
            modules_to_serialize (list): A list of modules to serialize with cloudpickle's `register_pickle_by_value`.
                If None, then no modules will be registered for serialization.

        """
        metadata = {}
        metadata["dependency_versions"] = get_dependency_versions()
        path = Path(path)

        if save_program:
            if path.suffix:
                raise ValueError(
                    f"`path` must point to a directory without a suffix when `save_program=True`, but received: {path}"
                )
            if path.exists() and not path.is_dir():
                raise NotADirectoryError(f"The path '{path}' exists but is not a directory.")

            if not path.exists():
                # Create the directory (and any parent directories)
                path.mkdir(parents=True)

            try:
                modules_to_serialize = modules_to_serialize or []
                for module in modules_to_serialize:
                    cloudpickle.register_pickle_by_value(module)

                with open(path / "program.pkl", "wb") as f:
                    cloudpickle.dump(self, f)
            except Exception as e:
                raise RuntimeError(
                    f"Saving failed with error: {e}. Please remove the non-picklable attributes from your DSPy program, "
                    "or consider using state-only saving by setting `save_program=False`."
                )
            with open(path / "metadata.json", "w", encoding="utf-8") as f:
                ujson.dump(metadata, f, indent=2, ensure_ascii=False)

            return

        state = self.dump_state()
        state["metadata"] = metadata
        if path.suffix == ".json":
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(ujson.dumps(state, indent=2 , ensure_ascii=False))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to save state to {path} with error: {e}. Your DSPy program may contain non "
                    "json-serializable objects, please consider saving the state in .pkl by using `path` ending "
                    "with `.pkl`, or saving the whole program by setting `save_program=True`."
                )
        elif path.suffix == ".pkl":
            with open(path, "wb") as f:
                cloudpickle.dump(state, f)
        else:
            raise ValueError(f"`path` must end with `.json` or `.pkl` when `save_program=False`, but received: {path}")

    def load(self, path):
        """Load the saved module. You may also want to check out dspy.load, if you want to
        load an entire program, not just the state for an existing program.

        Args:
            path (str): Path to the saved state file, which should be a .json or a .pkl file
        """
        path = Path(path)

        if path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                state = ujson.loads(f.read())
        elif path.suffix == ".pkl":
            with open(path, "rb") as f:
                state = cloudpickle.load(f)
        else:
            raise ValueError(f"`path` must end with `.json` or `.pkl`, but received: {path}")

        dependency_versions = get_dependency_versions()
        saved_dependency_versions = state["metadata"]["dependency_versions"]
        for key, saved_version in saved_dependency_versions.items():
            if dependency_versions[key] != saved_version:
                logger.warning(
                    f"There is a mismatch of {key} version between saved model and current environment. "
                    f"You saved with `{key}=={saved_version}`, but now you have "
                    f"`{key}=={dependency_versions[key]}`. This might cause errors or performance downgrade "
                    "on the loaded model, please consider loading the model in the same environment as the "
                    "saving environment."
                )
        self.load_state(state)
```

## File: `primitives/example.py`
```
class Example:
    def __init__(self, base=None, **kwargs):
        # Internal storage and other attributes
        self._store = {}
        self._demos = []
        self._input_keys = None

        # Initialize from a base Example if provided
        if base and isinstance(base, type(self)):
            self._store = base._store.copy()

        # Initialize from a dict if provided
        elif base and isinstance(base, dict):
            self._store = base.copy()

        # Update with provided kwargs
        self._store.update(kwargs)

    def __getattr__(self, key):
        if key.startswith("__") and key.endswith("__"):
            raise AttributeError
        if key in self._store:
            return self._store[key]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key.startswith("_") or key in dir(self.__class__):
            super().__setattr__(key, value)
        else:
            self._store[key] = value

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        self._store[key] = value

    def __delitem__(self, key):
        del self._store[key]

    def __contains__(self, key):
        return key in self._store

    def __len__(self):
        return len([k for k in self._store if not k.startswith("dspy_")])

    def __repr__(self):
        # return f"Example({self._store})" + f" (input_keys={self._input_keys}, demos={self._demos})"
        d = {k: v for k, v in self._store.items() if not k.startswith("dspy_")}
        return f"Example({d})" + f" (input_keys={self._input_keys})"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return isinstance(other, Example) and self._store == other._store

    def __hash__(self):
        return hash(tuple(self._store.items()))

    def keys(self, include_dspy=False):
        return [k for k in self._store.keys() if not k.startswith("dspy_") or include_dspy]

    def values(self, include_dspy=False):
        return [v for k, v in self._store.items() if not k.startswith("dspy_") or include_dspy]

    def items(self, include_dspy=False):
        return [(k, v) for k, v in self._store.items() if not k.startswith("dspy_") or include_dspy]

    def get(self, key, default=None):
        return self._store.get(key, default)

    def with_inputs(self, *keys):
        copied = self.copy()
        copied._input_keys = set(keys)
        return copied

    def inputs(self):
        if self._input_keys is None:
            raise ValueError("Inputs have not been set for this example. Use `example.with_inputs()` to set them.")

        # return items that are in input_keys
        d = {key: self._store[key] for key in self._store if key in self._input_keys}
        # return type(self)(d)
        new_instance = type(self)(base=d)
        new_instance._input_keys = self._input_keys  # Preserve input_keys in new instance
        return new_instance

    def labels(self):
        # return items that are NOT in input_keys
        input_keys = self.inputs().keys()
        d = {key: self._store[key] for key in self._store if key not in input_keys}
        return type(self)(d)

    def __iter__(self):
        return iter(dict(self._store))

    def copy(self, **kwargs):
        return type(self)(base=self, **kwargs)

    def without(self, *keys):
        copied = self.copy()
        for key in keys:
            del copied[key]
        return copied

    def toDict(self):  # noqa: N802
        return self._store.copy()
```

## File: `primitives/module.py`
```
import inspect
import logging

import magicattr

from dspy.dsp.utils.settings import settings, thread_local_overrides
from dspy.predict.parallel import Parallel
from dspy.primitives.base_module import BaseModule
from dspy.primitives.example import Example
from dspy.utils.callback import with_callbacks
from dspy.utils.inspect_history import pretty_print_history
from dspy.utils.usage_tracker import track_usage

logger = logging.getLogger(__name__)


class ProgramMeta(type):
    """Metaclass ensuring every ``dspy.Module`` instance is properly initialised."""

    def __call__(cls, *args, **kwargs):
        # Create the instance without invoking ``__init__`` so we can inject
        # the base initialization beforehand.
        obj = cls.__new__(cls, *args, **kwargs)
        if isinstance(obj, cls):
            # ``_base_init`` sets attributes that should exist on all modules
            # even when a subclass forgets to call ``super().__init__``.
            Module._base_init(obj)
            cls.__init__(obj, *args, **kwargs)

            # Guarantee existence of critical attributes if ``__init__`` didn't
            # create them.
            if not hasattr(obj, "callbacks"):
                obj.callbacks = []
            if not hasattr(obj, "history"):
                obj.history = []
        return obj


class Module(BaseModule, metaclass=ProgramMeta):
    def _base_init(self):
        self._compiled = False
        self.callbacks = []
        self.history = []

    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []
        self._compiled = False
        # LM calling history of the module.
        self.history = []

    @with_callbacks
    def __call__(self, *args, **kwargs):
        caller_modules = settings.caller_modules or []
        caller_modules = list(caller_modules)
        caller_modules.append(self)

        with settings.context(caller_modules=caller_modules):
            if settings.track_usage and thread_local_overrides.get().get("usage_tracker") is None:
                with track_usage() as usage_tracker:
                    output = self.forward(*args, **kwargs)
                output.set_lm_usage(usage_tracker.get_total_tokens())
                return output

            return self.forward(*args, **kwargs)

    @with_callbacks
    async def acall(self, *args, **kwargs):
        caller_modules = settings.caller_modules or []
        caller_modules = list(caller_modules)
        caller_modules.append(self)

        with settings.context(caller_modules=caller_modules):
            if settings.track_usage and thread_local_overrides.get().get("usage_tracker") is None:
                with track_usage() as usage_tracker:
                    output = await self.aforward(*args, **kwargs)
                    output.set_lm_usage(usage_tracker.get_total_tokens())
                    return output

            return await self.aforward(*args, **kwargs)

    def named_predictors(self):
        from dspy.predict.predict import Predict

        return [(name, param) for name, param in self.named_parameters() if isinstance(param, Predict)]

    def predictors(self):
        return [param for _, param in self.named_predictors()]

    def set_lm(self, lm):
        for _, param in self.named_predictors():
            param.lm = lm

    def get_lm(self):
        all_used_lms = [param.lm for _, param in self.named_predictors()]

        if len(set(all_used_lms)) == 1:
            return all_used_lms[0]

        raise ValueError("Multiple LMs are being used in the module. There's no unique LM to return.")

    def __repr__(self):
        s = []

        for name, param in self.named_predictors():
            s.append(f"{name} = {param}")

        return "\n".join(s)

    def map_named_predictors(self, func):
        """Applies a function to all named predictors."""
        for name, predictor in self.named_predictors():
            set_attribute_by_name(self, name, func(predictor))
        return self

    def inspect_history(self, n: int = 1):
        return pretty_print_history(self.history, n)

    def batch(
        self,
        examples: list[Example],
        num_threads: int | None = None,
        max_errors: int | None = None,
        return_failed_examples: bool = False,
        provide_traceback: bool | None = None,
        disable_progress_bar: bool = False,
    ) -> list[Example] | tuple[list[Example], list[Example], list[Exception]]:
        """
        Processes a list of dspy.Example instances in parallel using the Parallel module.

        Args:
            examples: List of dspy.Example instances to process.
            num_threads: Number of threads to use for parallel processing.
            max_errors: Maximum number of errors allowed before stopping execution.
                If ``None``, inherits from ``dspy.settings.max_errors``.
            return_failed_examples: Whether to return failed examples and exceptions.
            provide_traceback: Whether to include traceback information in error logs.
            disable_progress_bar: Whether to display the progress bar.

        Returns:
            List of results, and optionally failed examples and exceptions.
        """
        # Create a list of execution pairs (self, example)
        exec_pairs = [(self, example.inputs()) for example in examples]

        # Create an instance of Parallel
        parallel_executor = Parallel(
            num_threads=num_threads,
            max_errors=max_errors,
            return_failed_examples=return_failed_examples,
            provide_traceback=provide_traceback,
            disable_progress_bar=disable_progress_bar,
        )

        # Execute the forward method of Parallel
        if return_failed_examples:
            results, failed_examples, exceptions = parallel_executor.forward(exec_pairs)
            return results, failed_examples, exceptions
        else:
            results = parallel_executor.forward(exec_pairs)
            return results

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)

        if name == "forward" and callable(attr):
            # Check if forward is called through __call__ or directly
            stack = inspect.stack()
            forward_called_directly = len(stack) <= 1 or stack[1].function != "__call__"

            if forward_called_directly:
                logger.warning(
                    f"Calling {self.__class__.__name__}.forward() directly is discouraged. "
                    f"Please use {self.__class__.__name__}() instead."
                )

        return attr


def set_attribute_by_name(obj, name, value):
    magicattr.set(obj, name, value)
```

## File: `primitives/prediction.py`
```
from dspy.primitives.example import Example


class Prediction(Example):
    """A prediction object that contains the output of a DSPy module.
    
    Prediction inherits from Example.
    
    To allow feedback-augmented scores, Prediction supports comparison operations
    (<, >, <=, >=) for Predictions with a `score` field. The comparison operations
    compare the 'score' values as floats. For equality comparison, Predictions are equal
    if their underlying data stores are equal (inherited from Example).
    
    Arithmetic operations (+, /, etc.) are also supported for Predictions with a 'score'
    field, operating on the score value.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        del self._demos
        del self._input_keys

        self._completions = None
        self._lm_usage = None

    def get_lm_usage(self):
        return self._lm_usage

    def set_lm_usage(self, value):
        self._lm_usage = value

    @classmethod
    def from_completions(cls, list_or_dict, signature=None):
        obj = cls()
        obj._completions = Completions(list_or_dict, signature=signature)
        obj._store = {k: v[0] for k, v in obj._completions.items()}

        return obj

    def __repr__(self):
        store_repr = ",\n    ".join(f"{k}={v!r}" for k, v in self._store.items())

        if self._completions is None or len(self._completions) == 1:
            return f"Prediction(\n    {store_repr}\n)"

        num_completions = len(self._completions)
        return f"Prediction(\n    {store_repr},\n    completions=Completions(...)\n) ({num_completions-1} completions omitted)"

    def __str__(self):
        return self.__repr__()

    def __float__(self):
        if "score" not in self._store:
            raise ValueError("Prediction object does not have a 'score' field to convert to float.")
        return float(self._store["score"])

    def __add__(self, other):
        if isinstance(other, (float, int)):
            return self.__float__() + other
        elif isinstance(other, Prediction):
            return self.__float__() + float(other)
        raise TypeError(f"Unsupported type for addition: {type(other)}")

    def __radd__(self, other):
        if isinstance(other, (float, int)):
            return other + self.__float__()
        elif isinstance(other, Prediction):
            return float(other) + self.__float__()
        raise TypeError(f"Unsupported type for addition: {type(other)}")

    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            return self.__float__() / other
        elif isinstance(other, Prediction):
            return self.__float__() / float(other)
        raise TypeError(f"Unsupported type for division: {type(other)}")

    def __rtruediv__(self, other):
        if isinstance(other, (float, int)):
            return other / self.__float__()
        elif isinstance(other, Prediction):
            return float(other) / self.__float__()
        raise TypeError(f"Unsupported type for division: {type(other)}")

    def __lt__(self, other):
        if isinstance(other, (float, int)):
            return self.__float__() < other
        elif isinstance(other, Prediction):
            return self.__float__() < float(other)
        raise TypeError(f"Unsupported type for comparison: {type(other)}")

    def __le__(self, other):
        if isinstance(other, (float, int)):
            return self.__float__() <= other
        elif isinstance(other, Prediction):
            return self.__float__() <= float(other)
        raise TypeError(f"Unsupported type for comparison: {type(other)}")

    def __gt__(self, other):
        if isinstance(other, (float, int)):
            return self.__float__() > other
        elif isinstance(other, Prediction):
            return self.__float__() > float(other)
        raise TypeError(f"Unsupported type for comparison: {type(other)}")

    def __ge__(self, other):
        if isinstance(other, (float, int)):
            return self.__float__() >= other
        elif isinstance(other, Prediction):
            return self.__float__() >= float(other)
        raise TypeError(f"Unsupported type for comparison: {type(other)}")

    @property
    def completions(self):
        return self._completions


class Completions:
    def __init__(self, list_or_dict, signature=None):
        self.signature = signature

        if isinstance(list_or_dict, list):
            kwargs = {}
            for arg in list_or_dict:
                for k, v in arg.items():
                    kwargs.setdefault(k, []).append(v)
        else:
            kwargs = list_or_dict

        assert all(isinstance(v, list) for v in kwargs.values()), "All values must be lists"

        if kwargs:
            length = len(next(iter(kwargs.values())))
            assert all(len(v) == length for v in kwargs.values()), "All lists must have the same length"

        self._completions = kwargs

    def items(self):
        return self._completions.items()

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError("Index out of range")

            return Prediction(**{k: v[key] for k, v in self._completions.items()})

        return self._completions[key]

    def __getattr__(self, name):
        if name == "_completions":
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        if name in self._completions:
            return self._completions[name]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __len__(self):
        # Return the length of the list for one of the keys
        # It assumes all lists have the same length
        return len(next(iter(self._completions.values())))

    def __contains__(self, key):
        return key in self._completions

    def __repr__(self):
        items_repr = ",\n    ".join(f"{k}={v!r}" for k, v in self._completions.items())
        return f"Completions(\n    {items_repr}\n)"

    def __str__(self):
        # return str(self._completions)
        return self.__repr__()
```

## File: `primitives/python_interpreter.py`
```
import json
import os
import subprocess
from os import PathLike
from types import TracebackType
from typing import Any


class InterpreterError(RuntimeError):
    pass


class PythonInterpreter:
    r"""
    PythonInterpreter that runs code in a sandboxed environment using Deno and Pyodide.

    Prerequisites:
    - Deno (https://docs.deno.com/runtime/getting_started/installation/).

    Example Usage:
    ```python
    code_string = "print('Hello'); 1 + 2"
    with PythonInterpreter() as interp:
        output = interp(code_string) # If final statement is non-None, prints the numeric result, else prints captured output
    ```
    """

    def __init__(
        self,
        deno_command: list[str] | None = None,
        enable_read_paths: list[PathLike | str] | None = None,
        enable_write_paths: list[PathLike | str] | None = None,
        enable_env_vars: list[str] | None = None,
        enable_network_access: list[str] | None = None,
        sync_files: bool = True,
    ) -> None:
        """
        Args:
            deno_command: command list to launch Deno.
            enable_read_paths: Files or directories to allow reading from in the sandbox.
            enable_write_paths: Files or directories to allow writing to in the sandbox.
            enable_env_vars: Environment variable names to allow in the sandbox.
            enable_network_access: Domains or IPs to allow network access in the sandbox.
            sync_files: If set, syncs changes within the sandbox back to original files after execution.
        """
        if isinstance(deno_command, dict):
            deno_command = None  # no-op, just a guard in case someone passes a dict

        self.enable_read_paths = enable_read_paths or []
        self.enable_write_paths = enable_write_paths or []
        self.enable_env_vars = enable_env_vars or []
        self.enable_network_access = enable_network_access or []
        self.sync_files = sync_files
        # TODO later on add enable_run (--allow-run) by proxying subprocess.run through Deno.run() to fix 'emscripten does not support processes' error

        if deno_command:
            self.deno_command = list(deno_command)
        else:
            args = ["deno", "run", "--allow-read"]
            self._env_arg  = ""
            if self.enable_env_vars:
                user_vars = [str(v).strip() for v in self.enable_env_vars]
                args.append("--allow-env=" + ",".join(user_vars))
                self._env_arg = ",".join(user_vars)
            if self.enable_network_access:
                args.append(f"--allow-net={','.join(str(x) for x in self.enable_network_access)}")
            if self.enable_write_paths:
                args.append(f"--allow-write={','.join(str(x) for x in self.enable_write_paths)}")

            args.append(self._get_runner_path())

            # For runner.js to load in env vars
            if self._env_arg:
                args.append(self._env_arg)
            self.deno_command = args

        self.deno_process = None
        self._mounted_files = False

    def _get_runner_path(self) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "runner.js")

    def _mount_files(self):
        if self._mounted_files:
            return
        paths_to_mount = []
        if self.enable_read_paths:
            paths_to_mount.extend(self.enable_read_paths)
        if self.enable_write_paths:
            paths_to_mount.extend(self.enable_write_paths)
        if not paths_to_mount:
            return
        for path in paths_to_mount:
            if not path:
                continue
            if not os.path.exists(path):
                if self.enable_write_paths and path in self.enable_write_paths:
                    open(path, "a").close()
                else:
                    raise FileNotFoundError(f"Cannot mount non-existent file: {path}")
            virtual_path = f"/sandbox/{os.path.basename(path)}"
            mount_msg = json.dumps({"mount_file": str(path), "virtual_path": virtual_path})
            self.deno_process.stdin.write(mount_msg + "\n")
            self.deno_process.stdin.flush()
        self._mounted_files = True

    def _sync_files(self):
        if not self.enable_write_paths or not self.sync_files:
            return
        for path in self.enable_write_paths:
            virtual_path = f"/sandbox/{os.path.basename(path)}"
            sync_msg = json.dumps({
                "sync_file": virtual_path,
                "host_file": str(path)
            })
            self.deno_process.stdin.write(sync_msg + "\n")
            self.deno_process.stdin.flush()


    def _ensure_deno_process(self) -> None:
        if self.deno_process is None or self.deno_process.poll() is not None:
            try:
                self.deno_process = subprocess.Popen(
                    self.deno_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="UTF-8",
                    env=os.environ.copy()
                )
            except FileNotFoundError as e:
                install_instructions = (
                    "Deno executable not found. Please install Deno to proceed.\n"
                    "Installation instructions:\n"
                    "> curl -fsSL https://deno.land/install.sh | sh\n"
                    "*or*, on macOS with Homebrew:\n"
                    "> brew install deno\n"
                    "For additional configurations: https://docs.deno.com/runtime/getting_started/installation/"
                )
                raise InterpreterError(install_instructions) from e

    def _inject_variables(self, code: str, variables: dict[str, Any]) -> str:
        # Insert Python assignments for each variable at the top of the code
        injected_lines = []
        for key, value in variables.items():
            if not key.isidentifier():
                raise InterpreterError(f"Invalid variable name: '{key}'")
            python_value = self._serialize_value(value)
            injected_lines.append(f"{key} = {python_value}")
        injected_code = "\n".join(injected_lines) + "\n" + code
        return injected_code

    def _serialize_value(self, value: Any) -> str:
        # Basic safe serialization
        if isinstance(value, str):
            return repr(value)
        elif isinstance(value, (int, float, bool)):
            return str(value)
        elif value is None:
            return "None"
        elif isinstance(value, list) or isinstance(value, dict):
            return json.dumps(value)
        else:
            raise InterpreterError(f"Unsupported value type: {type(value).__name__}")

    def execute(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        variables = variables or {}
        code = self._inject_variables(code, variables)
        self._ensure_deno_process()
        self._mount_files()

        # Send the code as JSON
        input_data = json.dumps({"code": code})
        try:
            self.deno_process.stdin.write(input_data + "\n")
            self.deno_process.stdin.flush()
        except BrokenPipeError:
            # If the process died, restart and try again once
            self._ensure_deno_process()
            self.deno_process.stdin.write(input_data + "\n")
            self.deno_process.stdin.flush()

        # Read one JSON line from stdout
        output_line = self.deno_process.stdout.readline().strip()
        if not output_line:
            # Possibly the subprocess died or gave no output
            err_output = self.deno_process.stderr.read()
            raise InterpreterError(f"No output from Deno subprocess. Stderr: {err_output}")

        # Parse that line as JSON
        try:
            result = json.loads(output_line)
        except json.JSONDecodeError:
            # If not valid JSON, just return raw text
            result = {"output": output_line}

        # If we have an error, determine if it's a SyntaxError or other error using error.errorType.
        if "error" in result:
            error_msg = result["error"]
            error_type = result.get("errorType", "Sandbox Error")
            if error_type == "FinalAnswer":
                # The `FinalAnswer` trick to receive output from the sandbox interpreter,
                # just simply replace the output with the arguments.
                result["output"] = result.get("errorArgs", None)
            elif error_type == "SyntaxError":
                raise SyntaxError(f"Invalid Python syntax. message: {error_msg}")
            else:
                raise InterpreterError(f"{error_type}: {result.get('errorArgs') or error_msg}")

        # If there's no error or got `FinalAnswer`, return the "output" field
        self._sync_files()
        return result.get("output", None)

    def __enter__(self):
        return self

    # All exception fields are ignored and the runtime will automatically re-raise the exception
    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ):
        self.shutdown()

    def __call__(
        self,
        code: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        return self.execute(code, variables)

    def shutdown(self) -> None:
        if self.deno_process and self.deno_process.poll() is None:
            shutdown_message = json.dumps({"shutdown": True}) + "\n"
            self.deno_process.stdin.write(shutdown_message)
            self.deno_process.stdin.flush()
            self.deno_process.stdin.close()
            self.deno_process.wait()
            self.deno_process = None
```

## File: `primitives/runner.js`
```
// Adapted from "Simon Willison’s TILs" (https://til.simonwillison.net/deno/pyodide-sandbox)

import pyodideModule from "npm:pyodide/pyodide.js";
import { readLines } from "https://deno.land/std@0.186.0/io/mod.ts";

const pyodide = await pyodideModule.loadPyodide();

try {
  const env_vars = (Deno.args[0] ?? "").split(",").filter(Boolean);
  for (const key of env_vars) {
    const val = Deno.env.get(key);
    if (val !== undefined) {
      pyodide.runPython(`
import os
os.environ[${JSON.stringify(key)}] = ${JSON.stringify(val)}
      `);
    }
  }
} catch (e) {
  console.error("Error setting environment variables in Pyodide:", e);
}

for await (const line of readLines(Deno.stdin)) {
  let input;
  try {
    input = JSON.parse(line);
  } catch (error) {
    console.log(JSON.stringify({
      error: "Invalid JSON input: " + error.message,
      errorType: "ValueError"
    }));
    continue;
  }

  if (input.mount_file) {
      const hostPath = input.mount_file;
      const virtualPath = input.virtual_path || hostPath;
      try {
          const contents = await Deno.readFile(hostPath);
          const dirs = virtualPath.split('/').slice(1, -1);
          let cur = '';
          for (const d of dirs) {
              cur += '/' + d;
              try {
                  pyodide.FS.mkdir(cur);
              } catch (e) {
                  if (!(e && e.message && e.message.includes('File exists'))) {
                      console.log("[DEBUG] Error creating directory in Pyodide file system:", cur, "|", e.message);
                  }
              }
          }
          pyodide.FS.writeFile(virtualPath, contents);
      } catch (e) {
          console.log(JSON.stringify({error: "Failed to mount file: " + e.message}));
      }
      continue;      
  }

  if (input.sync_file) {
      const virtualPath = input.sync_file;
      const hostPath = input.host_file || virtualPath;
      try {
          const contents = pyodide.FS.readFile(virtualPath);
          await Deno.writeFile(hostPath, contents);
      } catch (e) {
          console.log("[DEBUG] Failed to sync file:", hostPath, "|", e.message);
      }
      continue;
  }


  // Expecting an object like { "code": "...", ... }
  if (typeof input !== 'object' || input === null) {
    console.log(JSON.stringify({
      error: "Input is not a JSON object",
      errorType: "ValueError"
    }));
    continue;
  }

  // Check for shutdown
  if (input.shutdown) {
    break;
  }

  const code = input.code || "";

  // Wrap execution in a try/catch so we can handle syntax errors, etc.
  try {
    await pyodide.loadPackagesFromImports(code);
    // 1. Temporarily override stdout/stderr so we can capture prints.
    pyodide.runPython(`
import sys
import io

# Keep references to the old stdout/stderr so we can restore them later
old_stdout = sys.stdout
old_stderr = sys.stderr

# New "file-like" buffers
buf_stdout = io.StringIO()
buf_stderr = io.StringIO()

sys.stdout = buf_stdout
sys.stderr = buf_stderr
    `);

    // 2. Setup proper exception arguments extractor and FinalAnswer bridge
    // The idea is borrowed from `smolagents` that uses the exception to simulate non-local exit
    pyodide.runPython(`
import json

def last_exception_args():
    return json.dumps(sys.last_exc.args) if sys.last_exc else None 

class FinalAnswer(Exception):
    pass

def final_answer(*args):
    raise FinalAnswer(*args)
      `);

    // 3. Run the user's code asynchronously
    const result = await pyodide.runPythonAsync(code);

    // 4. Retrieve captured stdout/stderr
    const capturedStdout = pyodide.runPython("buf_stdout.getvalue()");
    const capturedStderr = pyodide.runPython("buf_stderr.getvalue()");

    // 5. Restore original stdout/stderr
    pyodide.runPython(`
sys.stdout = old_stdout
sys.stderr = old_stderr
    `);

    // 6. Build our output object according to the rules:
    //    - If result is None (or Python "None" => JS null), output all prints
    //    - Else output the result only
    // Note: `None` in Python becomes `null` in JS.
    let output;
    if (result === null || result === undefined) {
      // The final statement was None or no return => deliver printed output
      // If you want to combine capturedStderr as well, you can append it
      // But here we'll just do stdout for clarity
      output = capturedStdout;
      // If there's something in stderr, you might want to include that or log it
      // output += capturedStderr;
    } else {
      // If the code returned a real value, just return that
      try {
        output = result.toJs();
      } catch (e) {
        output = result;
      }
    }

    console.log(JSON.stringify({ output }));
  } catch (error) {
    // We have an error => check if it's a SyntaxError or something else
    // The Python error class name is stored in error.type: https://pyodide.org/en/stable/usage/api/js-api.html#pyodide.ffi.PythonError
    const errorType = error.type || "Error";
    // error.message is mostly blank.
    const errorMessage = (error.message || "").trim();
    // The arguments of the exception are stored in sys.last_exc.args,
    // which is always helpful but pyodide don't extract them for us.
    // Use a bridge function to get them.
    let errorArgs = [];
    if (errorType !== "SyntaxError") {
      // Only python exceptions have args.
      const last_exception_args = pyodide.globals.get("last_exception_args");
      // Regarding https://pyodide.org/en/stable/usage/type-conversions.html#type-translations-errors,
      // we do a addtional `json.dumps` and `JSON.parse` on the values, to avoid the possible memory leak.
      errorArgs = JSON.parse(last_exception_args()) || [];
    }


    console.log(JSON.stringify({
      error: errorMessage,
      errorArgs: errorArgs,
      errorType: errorType
    }));
  }
}```

## File: `propose/__init__.py`
```
from dspy.propose.grounded_proposer import GroundedProposer

__all__ = [
    "GroundedProposer",
]
```

## File: `propose/dataset_summary_generator.py`
```
import re

import dspy
from dspy.propose.utils import strip_prefix


class ObservationSummarizer(dspy.Signature):
    ("""Given a series of observations I have made about my dataset, please summarize them into a brief 2-3 sentence summary which highlights only the most important details.""")
    observations = dspy.InputField(desc="Observations I have made about my dataset")
    summary = dspy.OutputField(desc="Two to Three sentence summary of only the most significant highlights of my observations")

class DatasetDescriptor(dspy.Signature):
    ("""Given several examples from a dataset please write observations about trends that hold for most or all of the samples. """
    """Some areas you may consider in your observations: topics, content, syntax, conciceness, etc. """
    """It will be useful to make an educated guess as to the nature of the task this dataset will enable. Don't be afraid to be creative""")

    examples = dspy.InputField(desc="Sample data points from the dataset")
    observations = dspy.OutputField(desc="Somethings that holds true for most or all of the data you observed")

class DatasetDescriptorWithPriorObservations(dspy.Signature):
    ("""Given several examples from a dataset please write observations about trends that hold for most or all of the samples. """
    """I will also provide you with a few observations I have already made.  Please add your own observations or if you feel the observations are comprehensive say 'COMPLETE' """
    """Some areas you may consider in your observations: topics, content, syntax, conciceness, etc. """
    """It will be useful to make an educated guess as to the nature of the task this dataset will enable. Don't be afraid to be creative""")

    examples = dspy.InputField(desc="Sample data points from the dataset")
    prior_observations = dspy.InputField(desc="Some prior observations I made about the data")
    observations = dspy.OutputField(desc="Somethings that holds true for most or all of the data you observed or COMPLETE if you have nothing to add")

def order_input_keys_in_string(unordered_repr):
    # Regex pattern to match the input keys structure
    pattern = r"input_keys=\{([^\}]+)\}"

    # Function to reorder keys
    def reorder_keys(match):
        # Extracting the keys from the match
        keys_str = match.group(1)
        # Splitting the keys, stripping extra spaces, and sorting them
        keys = sorted(key.strip() for key in keys_str.split(","))
        # Formatting the sorted keys back into the expected structure
        return f"input_keys={{{', '.join(keys)}}}"

    # Using re.sub to find all matches of the pattern and replace them using the reorder_keys function
    ordered_repr = re.sub(pattern, reorder_keys, unordered_repr)

    return ordered_repr

def create_dataset_summary(trainset, view_data_batch_size, prompt_model, log_file=None, verbose=False):
    if verbose:
        print("\nBootstrapping dataset summary (this will be used to generate instructions)...")
    upper_lim = min(len(trainset), view_data_batch_size)
    prompt_model = prompt_model if prompt_model else dspy.settings.lm
    with dspy.settings.context(lm=prompt_model):
        observation = dspy.Predict(DatasetDescriptor, n=1, temperature=1.0)(examples=order_input_keys_in_string(trainset[0:upper_lim].__repr__()))
    observations = observation["observations"]

    if log_file:
        log_file.write("PRODUCING DATASET SUMMARY\n")

    skips = 0
    try:
        max_calls = 10
        calls = 0
        for b in range(view_data_batch_size, len(trainset), view_data_batch_size):
            calls+=1
            if calls >= max_calls:
                break
            if verbose:
                print(f"b: {b}")
            upper_lim = min(len(trainset), b+view_data_batch_size)
            with dspy.settings.context(lm=prompt_model):
                output = dspy.Predict(DatasetDescriptorWithPriorObservations, n=1, temperature=1.0)(prior_observations=observations, examples=order_input_keys_in_string(trainset[b:upper_lim].__repr__()))
            if len(output["observations"]) >= 8 and output["observations"][:8].upper() == "COMPLETE":
                skips += 1
                if skips >= 5:
                    break
                continue
            observations += output["observations"]

            if log_file:
                log_file.write(f"observations {observations}\n")
    except Exception as e:
        if verbose:
            print(f"e {e}. using observations from past round for a summary.")

    if prompt_model:
        with dspy.settings.context(lm=prompt_model):
            summary = dspy.Predict(ObservationSummarizer, n=1, temperature=1.0)(observations=observations)
    else:
        summary = dspy.Predict(ObservationSummarizer, n=1, temperature=1.0)(observations=observations)
    if verbose:
        print(f"summary: {summary}")
    if log_file:
        log_file.write(f"summary: {summary}\n")

    if verbose:
        print(f"\nGenerated summary: {strip_prefix(summary.summary)}\n")

    return strip_prefix(summary.summary)
```

## File: `propose/grounded_proposer.py`
```
import random

import dspy
from dspy.propose.dataset_summary_generator import create_dataset_summary
from dspy.propose.propose_base import Proposer
from dspy.propose.utils import (
    create_example_string,
    create_predictor_level_history_string,
    get_dspy_source_code,
    strip_prefix,
)
from dspy.teleprompt.utils import get_prompt_model, get_signature

# Hardcoded variables (TODO: update)
MAX_INSTRUCT_IN_HISTORY = 5  # 10

TIPS = {
        "none": "",
        "creative": "Don't be afraid to be creative when creating the new instruction!",
        "simple": "Keep the instruction clear and concise.",
        "description": "Make sure your instruction is very informative and descriptive.",
        "high_stakes": "The instruction should include a high stakes scenario in which the LM must solve the task!",
        "persona": 'Include a persona that is relevant to the task in the instruction (ie. "You are a ...")',
    }

### SIGNATURES USED TO HELP WITH INSTRUCTION GENERATION ###

class DescribeProgram(dspy.Signature):
    (
        """Below is some pseudo-code for a pipeline that solves tasks with calls to language models. Please describe what type of task this program appears to be designed to solve, and how it appears to work."""
    )
    program_code = dspy.InputField(
        format=str,
        desc="Pseudocode for a language model program designed to solve a particular task.",
        prefix="PROGRAM CODE:",
    )
    program_example = dspy.InputField(
        format=str,
        desc="An example of the program in use.",
        prefix="EXAMPLE OF PROGRAM IN USE:",
    )
    program_description = dspy.OutputField(
        desc="Describe what task the program is designed to solve, and how it goes about solving this task.",
        prefix="SUMMARY OF PROGRAM ABOVE:",
    )


class DescribeModule(dspy.Signature):
    (
        """Below is some pseudo-code for a pipeline that solves tasks with calls to language models. Please describe the purpose of one of the specified module in this pipeline."""
    )
    program_code = dspy.InputField(
        format=str,
        desc="Pseudocode for a language model program designed to solve a particular task.",
        prefix="PROGRAM CODE:",
    )
    program_example = dspy.InputField(
        format=str,
        desc="An example of the program in use.",
        prefix="EXAMPLE OF PROGRAM IN USE:",
    )
    program_description = dspy.InputField(
        desc="Summary of the task the program is designed to solve, and how it goes about solving it.",
        prefix="SUMMARY OF PROGRAM ABOVE:",
    )
    module = dspy.InputField(
        desc="The module in the program that we want to describe.", prefix="MODULE:",
    )
    module_description = dspy.OutputField(
        desc="Description of the module's role in the broader program.",
        prefix="MODULE DESCRIPTION:",
    )


def generate_instruction_class(
    use_dataset_summary=True,
    program_aware=True,
    use_task_demos=True,
    use_instruct_history=True,
    use_tip=True,
):
    class GenerateSingleModuleInstruction(dspy.Signature):
        (
            """Use the information below to learn about a task that we are trying to solve using calls to an LM, then generate a new instruction that will be used to prompt a Language Model to better solve the task."""
        )
        if use_dataset_summary:
            dataset_description = dspy.InputField(
                desc="A description of the dataset that we are using.",
                prefix="DATASET SUMMARY:",
            )
        if program_aware:
            program_code = dspy.InputField(
                format=str,
                desc="Language model program designed to solve a particular task.",
                prefix="PROGRAM CODE:",
            )
            program_description = dspy.InputField(
                desc="Summary of the task the program is designed to solve, and how it goes about solving it.",
                prefix="PROGRAM DESCRIPTION:",
            )
            module = dspy.InputField(
                desc="The module to create an instruction for.", prefix="MODULE:",
            )
            module_description = dspy.InputField(
                desc="Description of the module to create an instruction for.", prefix="MODULE DESCRIPTION:",
            )
        task_demos = dspy.InputField(
            format=str,
            desc="Example inputs/outputs of our module.",
            prefix="TASK DEMO(S):",
        )
        if use_instruct_history:
            previous_instructions = dspy.InputField(
                format=str,
                desc="Previous instructions we've attempted, along with their associated scores.",
                prefix="PREVIOUS INSTRUCTIONS:",
            )
        basic_instruction = dspy.InputField(
            format=str, desc="Basic instruction.", prefix="BASIC INSTRUCTION:",
        )
        if use_tip:
            tip = dspy.InputField(
                format=str,
                desc="A suggestion for how to go about generating the new instruction.",
                prefix="TIP:",
            )
        proposed_instruction = dspy.OutputField(
            desc="Propose an instruction that will be used to prompt a Language Model to perform this task.",
            prefix="PROPOSED INSTRUCTION:",
        )

    return dspy.Predict(GenerateSingleModuleInstruction)

### CLASS RESPONSIBLE FOR GENERATING A NEW INSTRUCTION, USING THE HELPER SIGNATURES ABOVE ###

class GenerateModuleInstruction(dspy.Module):
    def __init__(
        self,
        program_code_string=None,
        use_dataset_summary=True,
        program_aware=False,
        use_task_demos=True,
        use_instruct_history=True,
        use_tip=True,
        verbose=False,
    ):
        super().__init__()
        self.use_dataset_summary = use_dataset_summary
        self.program_aware = program_aware
        self.use_task_demos = use_task_demos
        self.use_instruct_history = use_instruct_history
        self.use_tip = use_tip
        self.verbose = verbose

        self.program_code_string = program_code_string
        self.describe_program = dspy.Predict(DescribeProgram)
        self.describe_module = dspy.Predict(DescribeModule)
        self.generate_module_instruction = generate_instruction_class(
            use_dataset_summary=use_dataset_summary,
            program_aware=program_aware,
            use_task_demos=use_task_demos,
            use_instruct_history=use_instruct_history,
            use_tip=use_tip,
        )

    def forward(
        self,
        demo_candidates,
        pred_i,
        demo_set_i,
        program,
        previous_instructions,
        data_summary,
        num_demos_in_context=3,
        tip=None,
    ):
        def gather_examples_from_sets(candidate_sets, max_examples):
            """Helper function to gather up to augmented examples from given sets."""
            count = 0
            for candidate_set in candidate_sets:
                for example in candidate_set:
                    if "augmented" in example.keys():
                        fields_to_use = get_signature(program.predictors()[pred_i]).fields
                        yield create_example_string(fields_to_use, example)
                        count += 1
                        if count >= max_examples:
                            return

        # Construct full program demo or single module demo depending on settings
        basic_instruction = get_signature(program.predictors()[pred_i]).instructions
        task_demos = ""

        if self.use_task_demos:
            # Combine current and adjacent sets
            adjacent_sets = (
                [demo_candidates[pred_i][demo_set_i]] +
                demo_candidates[pred_i][demo_set_i + 1:] +
                demo_candidates[pred_i][:demo_set_i]
            )

            # Gather examples up to the required count
            example_strings = gather_examples_from_sets(adjacent_sets, num_demos_in_context)
            task_demos = "\n\n".join(example_strings) + "\n\n"

        # Default to no demos provided if no examples were gathered, or if we're using the first demo set
        if not task_demos.strip() or demo_set_i == 0:
            task_demos = "No task demos provided."

        # Summarize the program
        program_description = "Not available"
        module_code = "Not provided"
        module_description = "Not provided"
        if self.program_aware:
            try:
                program_description = strip_prefix(
                    self.describe_program(
                        program_code=self.program_code_string, program_example=task_demos,
                    ).program_description,
                )
                if self.verbose:
                    print(f"PROGRAM DESCRIPTION: {program_description}")

                inputs = []
                outputs = []
                for field_name, field in get_signature(program.predictors()[pred_i]).fields.items():
                    # Access the '__dspy_field_type' from the extra metadata
                    dspy_field_type = field.json_schema_extra.get("__dspy_field_type")

                    # Based on the '__dspy_field_type', append to the respective list
                    if dspy_field_type == "input":
                        inputs.append(field_name)
                    else:
                        outputs.append(field_name)

                module_code = f"{program.predictors()[pred_i].__class__.__name__}({', '.join(inputs)}) -> {', '.join(outputs)}"

                module_description = self.describe_module(
                    program_code=self.program_code_string,
                    program_description=program_description,
                    program_example=task_demos,
                    module=module_code,
                    max_depth=10,
                ).module_description
            except Exception as e:
                if self.verbose:
                    print(f"Error getting program description. Running without program aware proposer. Error: {e}")
                self.program_aware = False

        # Generate an instruction for our chosen module
        if self.verbose:
            print(f"task_demos {task_demos}")

        instruct = self.generate_module_instruction(
            dataset_description=data_summary,
            program_code=self.program_code_string,
            module=module_code,
            program_description=program_description,
            module_description=module_description,
            task_demos=task_demos,
            tip=tip,
            basic_instruction=basic_instruction,
            previous_instructions=previous_instructions,
        )

        proposed_instruction = strip_prefix(instruct.proposed_instruction)

        return dspy.Prediction(proposed_instruction=proposed_instruction)

### CLASS USED TO GENERATE THE FULL SET OF INSTRUCTIONS GIVEN THE SPECIFIED CRITERIA ###

class GroundedProposer(Proposer):
    def __init__(
        self,
        prompt_model,
        program,
        trainset,
        view_data_batch_size=10,
        use_dataset_summary=True,
        program_aware=True,
        use_task_demos=True,
        num_demos_in_context = 3,
        use_instruct_history=True,
        use_tip=True,
        set_tip_randomly=True,
        set_history_randomly=True,
        verbose=False,
        rng=None
    ):
        super().__init__()
        self.program_aware = program_aware
        self.use_dataset_summary = use_dataset_summary
        self.use_task_demos = use_task_demos
        self.num_demos_in_context = num_demos_in_context
        self.use_instruct_history = use_instruct_history
        self.use_tip = use_tip
        self.set_tip_randomly=set_tip_randomly
        self.set_history_randomly=set_history_randomly
        self.verbose = verbose
        self.rng = rng or random

        self.prompt_model = get_prompt_model(prompt_model)

        self.program_code_string = None
        if self.program_aware:
            try:
                self.program_code_string = get_dspy_source_code(program)
                if self.verbose:
                    print("SOURCE CODE:",self.program_code_string)
            except Exception as e:
                print(f"Error getting source code: {e}.\n\nRunning without program aware proposer.")
                self.program_aware = False

        self.data_summary  = None
        if self.use_dataset_summary:
            try:
                self.data_summary = create_dataset_summary(
                    trainset=trainset, view_data_batch_size=view_data_batch_size, prompt_model=prompt_model,
                )
                if self.verbose:
                    print(f"DATA SUMMARY: {self.data_summary}")
            except Exception as e:
                print(f"Error getting data summary: {e}.\n\nRunning without data aware proposer.")
                self.use_dataset_summary = False
                print("")

    def propose_instructions_for_program(
        self,
        trainset,
        program,
        demo_candidates,
        trial_logs,
        N, # noqa: N803
        T, # noqa: N803
    ) -> list[str]:
        """This method is responsible for returning the full set of new instructions for our program, given the specified criteria."""

        proposed_instructions = {}

        if self.set_history_randomly:
            # Randomly select whether or not we're using instruction history
            use_history = self.rng.random() < 0.5
            self.use_instruct_history = use_history
            if self.verbose:
                print(f"Use history T/F: {self.use_instruct_history}")

        if not demo_candidates:
            if self.verbose:
                print("No demo candidates provided. Running without task demos.")
            self.use_task_demos = False
            # When no demo candidates are provided, defailt to N
            num_demos = N
        else:
            num_demos = max(len(demo_candidates[0]), 1)

        # Create an instruction for each predictor
        for pred_i, predictor in enumerate(program.predictors()):
            for demo_set_i in range(num_demos)[:min(N, num_demos)]:
                if pred_i not in proposed_instructions:
                    proposed_instructions[pred_i] = []
                selected_tip = None
                if self.set_tip_randomly:
                    if self.verbose:
                        print("Using a randomly generated configuration for our grounded proposer.")
                    # Randomly select the tip
                    selected_tip_key = self.rng.choice(list(TIPS.keys()))
                    selected_tip = TIPS[selected_tip_key]
                    self.use_tip = bool(
                        selected_tip,
                    )
                    if self.verbose:
                        print(f"Selected tip: {selected_tip_key}")

                proposed_instructions[pred_i].append(
                    self.propose_instruction_for_predictor(
                        program=program,
                        predictor=predictor,
                        pred_i=pred_i,
                        T=T,
                        demo_candidates=demo_candidates,
                        demo_set_i=demo_set_i,
                        trial_logs=trial_logs,
                        tip=selected_tip,
                    ),
                )

        return proposed_instructions

    def propose_instruction_for_predictor(
        self,
        program,
        predictor,
        pred_i,
        T, # noqa: N803
        demo_candidates,
        demo_set_i,
        trial_logs,
        tip=None,
    ) -> str:
        """This method is responsible for returning a single instruction for a given predictor, using the specified criteria."""

        # Create an instruction history string for our predictor
        instruction_history = create_predictor_level_history_string(
            program, pred_i, trial_logs, MAX_INSTRUCT_IN_HISTORY,
        )

        # Create our instruction generator class (given specific criteria for this round of proposal)
        instruction_generator = GenerateModuleInstruction(
            program_code_string=self.program_code_string,
            use_dataset_summary=self.use_dataset_summary,
            program_aware=self.program_aware,
            use_task_demos=self.use_task_demos and demo_candidates,
            use_instruct_history=self.use_instruct_history and instruction_history,
            use_tip=self.use_tip,
            verbose=self.verbose
        )

        # Generate a new instruction for our predictor, using the temperature specified for this round
        original_temp = self.prompt_model.kwargs["temperature"]

        epsilon = self.rng.uniform(0.01, 0.05)
        modified_temp = T + epsilon

        with dspy.settings.context(lm=self.prompt_model):
            self.prompt_model.kwargs["temperature"] = modified_temp
            proposed_instruction = instruction_generator.forward(
                demo_candidates=demo_candidates,
                pred_i=pred_i,
                demo_set_i=demo_set_i,
                program=program,
                data_summary=self.data_summary,
                previous_instructions=instruction_history,
                num_demos_in_context = self.num_demos_in_context,
                tip=tip,
            ).proposed_instruction
        self.prompt_model.kwargs["temperature"] = original_temp

        # Log the trace used to generate the new instruction, along with the new instruction itself
        if self.verbose:
            self.prompt_model.inspect_history(n=1)
            print(f"PROPOSED INSTRUCTION: {proposed_instruction}")

        return strip_prefix(proposed_instruction)
```

## File: `propose/propose_base.py`
```
from abc import ABC, abstractmethod


class Proposer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def propose_instructions_for_program(self):
        pass

    def propose_instruction_for_predictor(self):
        pass
```

## File: `propose/utils.py`
```
import inspect
import json
import re

import dspy

try:
    from IPython.core.magics.code import extract_symbols
except ImportError:
    # Won't be able to read code from juptyer notebooks
    extract_symbols = None

from dspy.predict.parameter import Parameter
from dspy.teleprompt.utils import get_signature, new_getfile


def strip_prefix(text):
    pattern = r"^[\*\s]*(([\w\'\-]+\s+){0,4}[\w\'\-]+):\s*"
    modified_text = re.sub(pattern, "", text)
    return modified_text.strip('"')

def create_instruction_set_history_string(base_program, trial_logs, top_n):
    program_history = []
    for trial_num in trial_logs:
        trial = trial_logs[trial_num]
        if "program_path" in trial:
            trial_program = base_program.deepcopy()
            trial_program.load(trial["program_path"])
            program_history.append({
                "program": trial_program,
                "score": trial["score"],
            })

    # Deduplicate program history based on the program's instruction set
    seen_programs = set()
    unique_program_history = []
    for entry in program_history:
        program = entry["program"]
        instruction_set = get_program_instruction_set_string(program)
        if instruction_set not in seen_programs:
            seen_programs.add(instruction_set)
            unique_program_history.append(entry)

    # Get the top n programs from program history
    top_n_program_history = sorted(unique_program_history, key=lambda x: x["score"], reverse=True)[:top_n]
    top_n_program_history.reverse()

    # Create formatted string
    instruction_set_history_string = ""
    for entry in top_n_program_history:
        program = entry["program"]
        score = entry["score"]
        instruction_set = get_program_instruction_set_string(program)
        instruction_set_history_string += instruction_set + f" | Score: {score}\n\n"

    return instruction_set_history_string

def parse_list_of_instructions(instruction_string):
    # Try to convert the string representation of a list to an actual list using JSON
    try:
        instructions = json.loads(instruction_string)
        return instructions
    except json.JSONDecodeError:
        pass

    # If JSON decoding fails, extract strings within quotes
    instructions = re.findall(r'"([^"]*)"', instruction_string)
    return instructions

def get_program_instruction_set_string(program):
    instruction_list = []
    for _, pred in enumerate(program.predictors()):
        pred_instructions = get_signature(pred).instructions
        instruction_list.append(f'"{pred_instructions}"')
    # Joining the list into a single string that looks like a list
    return f"[{', '.join(instruction_list)}]"

def create_predictor_level_history_string(base_program, predictor_i, trial_logs, top_n):
    instruction_aggregate = {}
    instruction_history = []

    # Load trial programs
    for trial_num in trial_logs:
        trial = trial_logs[trial_num]
        if "program_path" in trial:
            trial_program = base_program.deepcopy()
            trial_program.load(trial["program_path"])
            instruction_history.append({
                "program": trial_program,
                "score": trial["score"],
            })

    # Aggregate scores for each instruction
    for history_item in instruction_history:
        predictor = history_item["program"].predictors()[predictor_i]
        instruction = get_signature(predictor).instructions
        score = history_item["score"]

        if instruction in instruction_aggregate:
            instruction_aggregate[instruction]["total_score"] += score
            instruction_aggregate[instruction]["count"] += 1
        else:
            instruction_aggregate[instruction] = {"total_score": score, "count": 1}

    # Calculate average score for each instruction and prepare for sorting
    predictor_history = []
    for instruction, data in instruction_aggregate.items():
        average_score = data["total_score"] / data["count"]
        predictor_history.append((instruction, average_score))

    # Deduplicate and sort by average score, then select top N
    seen_instructions = set()
    unique_predictor_history = []
    for instruction, score in predictor_history:
        if instruction not in seen_instructions:
            seen_instructions.add(instruction)
            unique_predictor_history.append((instruction, score))

    top_instructions = sorted(unique_predictor_history, key=lambda x: x[1], reverse=True)[:top_n]
    top_instructions.reverse()

    # Create formatted history string
    predictor_history_string = ""
    for instruction, score in top_instructions:
        predictor_history_string += instruction + f" | Score: {score}\n\n"

    return predictor_history_string

def create_example_string(fields, example):

    # Building the output string
    output = []
    for field_name, field_values in fields.items():
        name = field_values.json_schema_extra["prefix"]

        # Determine the value from input_data or prediction_data
        value = example.get(field_name)

        # Construct the string for the current field
        field_str = f"{name} {value}"
        output.append(field_str)

    # Joining all the field strings
    return "\n".join(output)

def get_dspy_source_code(module):
    header = []
    base_code = ""

    # Don't get source code for Predict or ChainOfThought modules (NOTE we will need to extend this list as more DSPy.modules are added)
    # TODO: if type(module).__name__ not in ["Predict", "ChainOfThought", "ReAct"]:
    if not type(module).__name__ == "Predict" and not type(module).__name__ == "ChainOfThought":
        try:
            base_code = inspect.getsource(type(module))
        except TypeError:
            obj = type(module)
            cell_code = "".join(inspect.linecache.getlines(new_getfile(obj)))
            class_code = extract_symbols(cell_code, obj.__name__)[0][0]
            base_code = str(class_code)

    completed_set = set()
    for attribute in module.__dict__.keys():
        try:
            iterable = iter(getattr(module, attribute))
        except TypeError:
            iterable = [getattr(module, attribute)]

        for item in iterable:
            if item in completed_set:
                continue
            if isinstance(item, Parameter):
                if hasattr(item, "signature") and item.signature is not None and item.signature.__pydantic_parent_namespace__["signature_name"] + "_sig" not in completed_set:
                    try:
                        header.append(inspect.getsource(item.signature))
                        print(inspect.getsource(item.signature))
                    except (TypeError, OSError):
                        header.append(str(item.signature))
                    completed_set.add(item.signature.__pydantic_parent_namespace__["signature_name"] + "_sig")
            if isinstance(item, dspy.Module):
                code = get_dspy_source_code(item).strip()
                if code not in completed_set:
                    header.append(code)
                    completed_set.add(code)
            completed_set.add(item)

    return "\n\n".join(header) + "\n\n" + base_code
```

## File: `retrievers/__init__.py`
```
from dspy.retrievers.embeddings import Embeddings
from dspy.retrievers.retrieve import Retrieve

__all__ = ["Embeddings", "Retrieve"]
```

## File: `retrievers/databricks_rm.py`
```
import json
import os
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

import requests

import dspy
from dspy.primitives.prediction import Prediction

_databricks_sdk_installed = find_spec("databricks.sdk") is not None


@dataclass
class Document:
    page_content: str
    metadata: dict[str, Any]
    type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_content": self.page_content,
            "metadata": self.metadata,
            "type": self.type,
        }


class DatabricksRM(dspy.Retrieve):
    """
    A retriever module that uses a Databricks Mosaic AI Vector Search Index to return the top-k
    embeddings for a given query.

    Examples:
        Below is a code snippet that shows how to set up a Databricks Vector Search Index
        and configure a DatabricksRM DSPy retriever module to query the index.

        (example adapted from "Databricks: How to create and query a Vector Search Index:
        https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-index)

        ```python
        from databricks.vector_search.client import VectorSearchClient

        # Create a Databricks Vector Search Endpoint
        client = VectorSearchClient()
        client.create_endpoint(
            name="your_vector_search_endpoint_name",
            endpoint_type="STANDARD"
        )

        # Create a Databricks Direct Access Vector Search Index
        index = client.create_direct_access_index(
            endpoint_name="your_vector_search_endpoint_name",
            index_name="your_index_name",
            primary_key="id",
            embedding_dimension=1024,
            embedding_vector_column="text_vector",
            schema={
              "id": "int",
              "field2": "str",
              "field3": "float",
              "text_vector": "array<float>"
            }
        )

        # Create a DatabricksRM retriever module to query the Databricks Direct Access Vector
        # Search Index
        retriever = DatabricksRM(
            databricks_index_name = "your_index_name",
            docs_id_column_name="id",
            text_column_name="field2",
            k=3
        )
        ```

        Below is a code snippet that shows how to query the Databricks Direct Access Vector
        Search Index using the DatabricksRM retriever module:

        ```python
        retrieved_results = DatabricksRM(query="Example query text"))
        ```
    """

    def __init__(
        self,
        databricks_index_name: str,
        databricks_endpoint: str | None = None,
        databricks_token: str | None = None,
        databricks_client_id: str | None = None,
        databricks_client_secret: str | None = None,
        columns: list[str] | None = None,
        filters_json: str | None = None,
        k: int = 3,
        docs_id_column_name: str = "id",
        docs_uri_column_name: str | None = None,
        text_column_name: str = "text",
        use_with_databricks_agent_framework: bool = False,
    ):
        """
        Args:
            databricks_index_name (str): The name of the Databricks Vector Search Index to query.
            databricks_endpoint (Optional[str]): The URL of the Databricks Workspace containing
                the Vector Search Index. Defaults to the value of the ``DATABRICKS_HOST``
                environment variable. If unspecified, the Databricks SDK is used to identify the
                endpoint based on the current environment.
            databricks_token (Optional[str]): The Databricks Workspace authentication token to use
                when querying the Vector Search Index. Defaults to the value of the
                ``DATABRICKS_TOKEN`` environment variable. If unspecified, the Databricks SDK is
                used to identify the token based on the current environment.
            databricks_client_id (str): Databricks service principal id. If not specified,
                the token is resolved from the current environment (DATABRICKS_CLIENT_ID).
            databricks_client_secret (str): Databricks service principal secret. If not specified,
                the endpoint is resolved from the current environment (DATABRICKS_CLIENT_SECRET).
            columns (Optional[list[str]]): Extra column names to include in response,
                in addition to the document id and text columns specified by
                ``docs_id_column_name`` and ``text_column_name``.
            filters_json (Optional[str]): A JSON string specifying additional query filters.
                Example filters: ``{"id <": 5}`` selects records that have an ``id`` column value
                less than 5, and ``{"id >=": 5, "id <": 10}`` selects records that have an ``id``
                column value greater than or equal to 5 and less than 10.
            k (int): The number of documents to retrieve.
            docs_id_column_name (str): The name of the column in the Databricks Vector Search Index
                containing document IDs.
            docs_uri_column_name (Optional[str]): The name of the column in the Databricks Vector Search Index
                containing document URI.
            text_column_name (str): The name of the column in the Databricks Vector Search Index
                containing document text to retrieve.
            use_with_databricks_agent_framework (bool): Whether to use the `DatabricksRM` in a way that is
                compatible with the Databricks Mosaic Agent Framework.
        """
        super().__init__(k=k)
        self.databricks_token = databricks_token if databricks_token is not None else os.environ.get("DATABRICKS_TOKEN")
        self.databricks_endpoint = (
            databricks_endpoint if databricks_endpoint is not None else os.environ.get("DATABRICKS_HOST")
        )
        self.databricks_client_id = (
            databricks_client_id if databricks_client_id is not None else os.environ.get("DATABRICKS_CLIENT_ID")
        )
        self.databricks_client_secret = (
            databricks_client_secret
            if databricks_client_secret is not None
            else os.environ.get("DATABRICKS_CLIENT_SECRET")
        )
        if not _databricks_sdk_installed and (self.databricks_token, self.databricks_endpoint).count(None) > 0:
            raise ValueError(
                "To retrieve documents with Databricks Vector Search, you must install the"
                " databricks-sdk Python library, supply the databricks_token and"
                " databricks_endpoint parameters, or set the DATABRICKS_TOKEN and DATABRICKS_HOST"
                " environment variables. You may also supply a service principal the databricks_client_id and"
                " databricks_client_secret parameters, or set the DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET"
            )
        self.databricks_index_name = databricks_index_name
        self.columns = list({docs_id_column_name, text_column_name, *(columns or [])})
        self.filters_json = filters_json
        self.k = k
        self.docs_id_column_name = docs_id_column_name
        self.docs_uri_column_name = docs_uri_column_name
        self.text_column_name = text_column_name
        self.use_with_databricks_agent_framework = use_with_databricks_agent_framework
        if self.use_with_databricks_agent_framework:
            try:
                import mlflow

                mlflow.models.set_retriever_schema(
                    primary_key="doc_id",
                    text_column="page_content",
                    doc_uri="doc_uri",
                )
            except ImportError:
                raise ValueError(
                    "To use the `DatabricksRM` retriever module with the Databricks Mosaic Agent Framework, "
                    "you must install the mlflow Python library. Please install mlflow via `pip install mlflow`."
                )

    def _extract_doc_ids(self, item: dict[str, Any]) -> str:
        """Extracts the document id from a search result

        Args:
            item: dict[str, Any]: a record from the search results.
        Returns:
            str: document id.
        """
        if self.docs_id_column_name == "metadata":
            docs_dict = json.loads(item["metadata"])
            return docs_dict["document_id"]
        return item[self.docs_id_column_name]

    def _get_extra_columns(self, item: dict[str, Any]) -> dict[str, Any]:
        """Extracts search result column values, excluding the "text" and not "id" columns

        Args:
            item: dict[str, Any]: a record from the search results.
        Returns:
            dict[str, Any]: Search result column values, excluding the "text", "id" and "uri" columns.
        """
        extra_columns = {
            k: v
            for k, v in item.items()
            if k not in [self.docs_id_column_name, self.text_column_name, self.docs_uri_column_name]
        }
        if self.docs_id_column_name == "metadata":
            extra_columns = {
                **extra_columns,
                **{"metadata": {k: v for k, v in json.loads(item["metadata"]).items() if k != "document_id"}},
            }
        return extra_columns

    def forward(
        self,
        query: str | list[float],
        query_type: str = "ANN",
        filters_json: str | None = None,
    ) -> dspy.Prediction | list[dict[str, Any]]:
        """
        Retrieve documents from a Databricks Mosaic AI Vector Search Index that are relevant to the
        specified query.

        Args:
            query (Union[str, list[float]]): The query text or numeric query vector for which to
                retrieve relevant documents.
            query_type (str): The type of search query to perform against the Databricks Vector
                Search Index. Must be either 'ANN' (approximate nearest neighbor) or 'HYBRID'
                (hybrid search).
            filters_json (Optional[str]): A JSON string specifying additional query filters.
                Example filters: ``{"id <": 5}`` selects records that have an ``id`` column value
                less than 5, and ``{"id >=": 5, "id <": 10}`` selects records that have an ``id``
                column value greater than or equal to 5 and less than 10. If specified, this
                parameter overrides the `filters_json` parameter passed to the constructor.

        Returns:
            A list of dictionaries when ``use_with_databricks_agent_framework`` is ``True``,
            or a ``dspy.Prediction`` object when ``use_with_databricks_agent_framework`` is
            ``False``.
        """
        if query_type in ["vector", "text"]:
            # Older versions of DSPy used a `query_type` argument to disambiguate between text
            # and vector queries, rather than checking the type of the `query` argument. This
            # differs from the Databricks Vector Search definition of `query_type`, which
            # specifies the search algorithm to use (e.g. "ANN" or "HYBRID"). To maintain
            # backwards compatibility with older versions of DSPy, we map the old `query_type`
            # values to the Databricks Vector Search default query type of "ANN".
            query_type = "ANN"

        if isinstance(query, str):
            query_text = query
            query_vector = None
        elif isinstance(query, list):
            query_vector = query
            query_text = None
        else:
            raise ValueError("Query must be a string or a list of floats.")

        if _databricks_sdk_installed:
            results = self._query_via_databricks_sdk(
                index_name=self.databricks_index_name,
                k=self.k,
                columns=self.columns,
                query_type=query_type,
                query_text=query_text,
                query_vector=query_vector,
                databricks_token=self.databricks_token,
                databricks_endpoint=self.databricks_endpoint,
                databricks_client_id=self.databricks_client_id,
                databricks_client_secret=self.databricks_client_secret,
                filters_json=filters_json or self.filters_json,
            )
        else:
            results = self._query_via_requests(
                index_name=self.databricks_index_name,
                k=self.k,
                columns=self.columns,
                databricks_token=self.databricks_token,
                databricks_endpoint=self.databricks_endpoint,
                query_type=query_type,
                query_text=query_text,
                query_vector=query_vector,
                filters_json=filters_json or self.filters_json,
            )

        # Checking if defined columns are present in the index columns
        col_names = [column["name"] for column in results["manifest"]["columns"]]

        if self.docs_id_column_name not in col_names:
            raise Exception(
                f"docs_id_column_name: '{self.docs_id_column_name}' is not in the index columns: \n {col_names}"
            )

        if self.text_column_name not in col_names:
            raise Exception(f"text_column_name: '{self.text_column_name}' is not in the index columns: \n {col_names}")

        # Extracting the results
        items = []
        if "data_array" in results["result"]:
            for _, data_row in enumerate(results["result"]["data_array"]):
                item = {}
                for col_name, val in zip(col_names, data_row, strict=False):
                    item[col_name] = val
                items += [item]

        # Sorting results by score in descending order
        sorted_docs = sorted(items, key=lambda x: x["score"], reverse=True)[: self.k]

        if self.use_with_databricks_agent_framework:
            return [
                Document(
                    page_content=doc[self.text_column_name],
                    metadata={
                        "doc_id": self._extract_doc_ids(doc),
                        "doc_uri": doc[self.docs_uri_column_name] if self.docs_uri_column_name else None,
                    }
                    | self._get_extra_columns(doc),
                    type="Document",
                ).to_dict()
                for doc in sorted_docs
            ]
        else:
            # Returning the prediction
            return Prediction(
                docs=[doc[self.text_column_name] for doc in sorted_docs],
                doc_ids=[self._extract_doc_ids(doc) for doc in sorted_docs],
                doc_uris=[doc[self.docs_uri_column_name] for doc in sorted_docs] if self.docs_uri_column_name else None,
                extra_columns=[self._get_extra_columns(item) for item in sorted_docs],
            )

    @staticmethod
    def _query_via_databricks_sdk(
        index_name: str,
        k: int,
        columns: list[str],
        query_type: str,
        query_text: str | None,
        query_vector: list[float] | None,
        databricks_token: str | None,
        databricks_endpoint: str | None,
        databricks_client_id: str | None,
        databricks_client_secret: str | None,
        filters_json: str | None,
    ) -> dict[str, Any]:
        """
        Query a Databricks Vector Search Index via the Databricks SDK.
        Assumes that the databricks-sdk Python library is installed.

        Args:
            index_name (str): Name of the Databricks vector search index to query
            k (int): Number of relevant documents to retrieve.
            columns (list[str]): Column names to include in response.
            query_text (Optional[str]): Text query for which to find relevant documents. Exactly
                one of query_text or query_vector must be specified.
            query_vector (Optional[list[float]]): Numeric query vector for which to find relevant
                documents. Exactly one of query_text or query_vector must be specified.
            filters_json (Optional[str]): JSON string representing additional query filters.
            databricks_token (str): Databricks authentication token. If not specified,
                the token is resolved from the current environment.
            databricks_endpoint (str): Databricks index endpoint url. If not specified,
                the endpoint is resolved from the current environment.
            databricks_client_id (str): Databricks service principal id. If not specified,
                the token is resolved from the current environment (DATABRICKS_CLIENT_ID).
            databricks_client_secret (str): Databricks service principal secret. If not specified,
                the endpoint is resolved from the current environment (DATABRICKS_CLIENT_SECRET).
        Returns:
        Returns:
            dict[str, Any]: Parsed JSON response from the Databricks Vector Search Index query.
        """

        from databricks.sdk import WorkspaceClient

        if (query_text, query_vector).count(None) != 1:
            raise ValueError("Exactly one of query_text or query_vector must be specified.")

        if databricks_client_secret and databricks_client_id:
            # Use client ID and secret for authentication if they are provided
            databricks_client = WorkspaceClient(
                client_id=databricks_client_id,
                client_secret=databricks_client_secret,
            )
            print("Creating Databricks workspace client using service principal authentication.")

        else:
            # Fallback for token-based authentication
            databricks_client = WorkspaceClient(
                host=databricks_endpoint,
                token=databricks_token,
            )
            print("Creating Databricks workspace client using token authentication.")

        return databricks_client.vector_search_indexes.query_index(
            index_name=index_name,
            query_type=query_type,
            query_text=query_text,
            query_vector=query_vector,
            columns=columns,
            filters_json=filters_json,
            num_results=k,
        ).as_dict()

    @staticmethod
    def _query_via_requests(
        index_name: str,
        k: int,
        columns: list[str],
        databricks_token: str,
        databricks_endpoint: str,
        query_type: str,
        query_text: str | None,
        query_vector: list[float] | None,
        filters_json: str | None,
    ) -> dict[str, Any]:
        """
        Query a Databricks Vector Search Index via the Python requests library.

        Args:
            index_name (str): Name of the Databricks vector search index to query
            k (int): Number of relevant documents to retrieve.
            columns (list[str]): Column names to include in response.
            databricks_token (str): Databricks authentication token.
            databricks_endpoint (str): Databricks index endpoint url.
            query_text (Optional[str]): Text query for which to find relevant documents. Exactly
                one of query_text or query_vector must be specified.
            query_vector (Optional[list[float]]): Numeric query vector for which to find relevant
                documents. Exactly one of query_text or query_vector must be specified.
            filters_json (Optional[str]): JSON string representing additional query filters.

        Returns:
            dict[str, Any]: Parsed JSON response from the Databricks Vector Search Index query.
        """
        if (query_text, query_vector).count(None) != 1:
            raise ValueError("Exactly one of query_text or query_vector must be specified.")

        headers = {
            "Authorization": f"Bearer {databricks_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "columns": columns,
            "num_results": k,
            "query_type": query_type,
        }
        if filters_json is not None:
            payload["filters_json"] = filters_json
        if query_text is not None:
            payload["query_text"] = query_text
        elif query_vector is not None:
            payload["query_vector"] = query_vector
        response = requests.post(
            f"{databricks_endpoint}/api/2.0/vector-search/indexes/{index_name}/query",
            json=payload,
            headers=headers,
        )
        results = response.json()
        if "error_code" in results:
            raise Exception(f"ERROR: {results['error_code']} -- {results['message']}")
        return results
```

## File: `retrievers/embeddings.py`
```
from typing import Any

import numpy as np

from dspy.utils.unbatchify import Unbatchify

# TODO: Add .save and .load methods!


class Embeddings:
    def __init__(
        self,
        corpus: list[str],
        embedder,
        k: int = 5,
        callbacks: list[Any] | None = None,
        cache: bool = False,
        brute_force_threshold: int = 20_000,
        normalize: bool = True,
    ):
        assert cache is False, "Caching is not supported for embeddings-based retrievers"

        self.embedder = embedder
        self.k = k
        self.corpus = corpus
        self.normalize = normalize

        self.corpus_embeddings = self.embedder(self.corpus)
        self.corpus_embeddings = self._normalize(self.corpus_embeddings) if self.normalize else self.corpus_embeddings

        self.index = self._build_faiss() if len(corpus) >= brute_force_threshold else None
        self.search_fn = Unbatchify(self._batch_forward)

    def __call__(self, query: str):
        return self.forward(query)

    def forward(self, query: str):
        import dspy

        passages, indices = self.search_fn(query)
        return dspy.Prediction(passages=passages, indices=indices)

    def _batch_forward(self, queries: list[str]):
        q_embeds = self.embedder(queries)
        q_embeds = self._normalize(q_embeds) if self.normalize else q_embeds

        pids = self._faiss_search(q_embeds, self.k * 10) if self.index else None
        pids = np.tile(np.arange(len(self.corpus)), (len(queries), 1)) if pids is None else pids

        return self._rerank_and_predict(q_embeds, pids)

    def _build_faiss(self):
        nbytes = 32
        partitions = int(2 * np.sqrt(len(self.corpus)))
        dim = self.corpus_embeddings.shape[1]

        try:
            import faiss
        except ImportError:
            raise ImportError("Please `pip install faiss-cpu` or increase `brute_force_threshold` to avoid FAISS.")

        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, partitions, nbytes, 8)

        print(
            f"Training a {nbytes}-byte FAISS index with {partitions} partitions, based on "
            f"{len(self.corpus)} x {dim}-dim embeddings"
        )
        index.train(self.corpus_embeddings)
        index.add(self.corpus_embeddings)
        index.nprobe = min(16, partitions)

        return index

    def _faiss_search(self, query_embeddings: np.ndarray, num_candidates: int):
        return self.index.search(query_embeddings, num_candidates)[1]

    def _rerank_and_predict(self, q_embeds: np.ndarray, candidate_indices: np.ndarray):
        candidate_embeddings = self.corpus_embeddings[candidate_indices]
        scores = np.einsum("qd,qkd->qk", q_embeds, candidate_embeddings)

        top_k_indices = np.argsort(-scores, axis=1)[:, : self.k]
        top_indices = candidate_indices[np.arange(len(q_embeds))[:, None], top_k_indices]

        return [([self.corpus[idx] for idx in indices], [idx for idx in indices]) for indices in top_indices]  # noqa: C416

    def _normalize(self, embeddings: np.ndarray):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-10)
```

## File: `retrievers/retrieve.py`
```
import random

from dspy.predict.parameter import Parameter
from dspy.primitives.prediction import Prediction
from dspy.utils.callback import with_callbacks


def single_query_passage(passages):
    passages_dict = {key: [] for key in list(passages[0].keys())}
    for docs in passages:
        for key, value in docs.items():
            passages_dict[key].append(value)
    if "long_text" in passages_dict:
        passages_dict["passages"] = passages_dict.pop("long_text")
    return Prediction(**passages_dict)


class Retrieve(Parameter):
    name = "Search"
    input_variable = "query"
    desc = "takes a search query and returns one or more potentially relevant passages from a corpus"

    def __init__(self, k=3, callbacks=None):
        self.stage = random.randbytes(8).hex()
        self.k = k
        self.callbacks = callbacks or []

    def reset(self):
        pass

    def dump_state(self):
        state_keys = ["k"]
        return {k: getattr(self, k) for k in state_keys}

    def load_state(self, state):
        for name, value in state.items():
            setattr(self, name, value)

    @with_callbacks
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        query: str,
        k: int | None = None,
        **kwargs,
    ) -> list[str] | Prediction | list[Prediction]:
        k = k if k is not None else self.k

        import dspy

        if not dspy.settings.rm:
            raise AssertionError("No RM is loaded.")

        passages = dspy.settings.rm(query, k=k, **kwargs)

        from collections.abc import Iterable
        if not isinstance(passages, Iterable):
            # it's not an iterable yet; make it one.
            # TODO: we should unify the type signatures of dspy.Retriever
            passages = [passages]
        passages = [psg.long_text for psg in passages]

        return Prediction(passages=passages)

# TODO: Consider doing Prediction.from_completions with the individual sets of passages (per query) too.
```

## File: `retrievers/weaviate_rm.py`
```

import dspy
from dspy.dsp.utils import dotdict
from dspy.primitives.prediction import Prediction

try:
    from uuid import uuid4

    import weaviate
    from weaviate.util import get_valid_uuid
except ImportError as err:
    raise ImportError(
        "The 'weaviate' extra is required to use WeaviateRM. Install it with `pip install dspy-ai[weaviate]`",
    ) from err


class WeaviateRM(dspy.Retrieve):
    """A retrieval module that uses Weaviate to return the top passages for a given query.

    Assumes that a Weaviate collection has been created and populated with the following payload:
        - content: The text of the passage

    Args:
        weaviate_collection_name (str): The name of the Weaviate collection.
        weaviate_client (WeaviateClient): An instance of the Weaviate client.
        k (int, optional): The default number of top passages to retrieve. Default to 3.
        tenant_id (str, optional): The tenant to retrieve objects from.

    Examples:
        Below is a code snippet that shows how to use Weaviate as the default retriever:
        ```python
        import weaviate

        llm = dspy.Cohere(model="command-r-plus", api_key=api_key)
        weaviate_client = weaviate.connect_to_[local, wcs, custom, embedded]("your-path-here")
        retriever_model = WeaviateRM("my_collection_name", weaviate_client=weaviate_client)
        dspy.settings.configure(lm=llm, rm=retriever_model)

        retrieve = dspy.Retrieve(k=1)
        topK_passages = retrieve("what are the stages in planning, sanctioning and execution of public works").passages
        ```

        Below is a code snippet that shows how to use Weaviate in the forward() function of a module
        ```python
        self.retrieve = WeaviateRM("my_collection_name", weaviate_client=weaviate_client, k=num_passages)
        ```
    """

    def __init__(
        self,
        weaviate_collection_name: str,
        weaviate_client: weaviate.WeaviateClient | weaviate.Client,
        weaviate_collection_text_key: str | None = "content",
        k: int = 3,
        tenant_id: str | None = None,
    ):
        self._weaviate_collection_name = weaviate_collection_name
        self._weaviate_client = weaviate_client
        self._weaviate_collection = self._weaviate_client.collections.get(self._weaviate_collection_name)
        self._weaviate_collection_text_key = weaviate_collection_text_key
        self._tenant_id = tenant_id

        # Check the type of weaviate_client (this is added to support v3 and v4)
        if hasattr(weaviate_client, "collections"):
            self._client_type = "WeaviateClient"
        elif hasattr(weaviate_client, "query"):
            self._client_type = "Client"
        else:
            raise ValueError("Unsupported Weaviate client type")

        super().__init__(k=k)

    def forward(self, query_or_queries: str | list[str], k: int | None = None, **kwargs) -> Prediction:
        """Search with Weaviate for self.k top passages for query or queries.

        Args:
            query_or_queries (Union[str, list[str]]): The query or queries to search for.
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k.
            kwargs :

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        k = k if k is not None else self.k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]
        passages, parsed_results = [], []
        tenant = kwargs.pop("tenant_id", self._tenant_id)
        for query in queries:
            if self._client_type == "WeaviateClient":
                if tenant:
                    results = self._weaviate_collection.query.with_tenant(tenant).hybrid(query=query, limit=k, **kwargs)
                else:
                    results = self._weaviate_collection.query.hybrid(query=query, limit=k, **kwargs)

                parsed_results = [result.properties[self._weaviate_collection_text_key] for result in results.objects]

            elif self._client_type == "Client":
                q = self._weaviate_client.query.get(
                        self._weaviate_collection_name, [self._weaviate_collection_text_key]
                    )
                if tenant:
                    q = q.with_tenant(tenant)
                results = q.with_hybrid(query=query).with_limit(k).do()

                results = results["data"]["Get"][self._weaviate_collection_name]
                parsed_results = [result[self._weaviate_collection_text_key] for result in results]

            passages.extend(dotdict({"long_text": d}) for d in parsed_results)

        return passages

    def get_objects(self, num_samples: int, fields: list[str]) -> list[dict]:
        """Get objects from Weaviate using the cursor API."""
        if self._client_type == "WeaviateClient":
            objects = []
            counter = 0
            for item in self._weaviate_collection.iterator(): # TODO: add tenancy scoping
                if counter >= num_samples:
                    break
                new_object = {}
                for key in item.properties.keys():
                    if key in fields:
                      new_object[key] = item.properties[key]
                objects.append(new_object)
                counter += 1
            return objects
        else:
            raise ValueError("`get_objects` is not supported for the v3 Weaviate Python client, please upgrade to v4.")

    def insert(self, new_object_properties: dict):
        if self._client_type == "WeaviateClient":
            self._weaviate_collection.data.insert(
                properties=new_object_properties,
                uuid=get_valid_uuid(uuid4())
            ) # TODO: add tenancy scoping
        else:
            raise AttributeError("`insert` is not supported for the v3 Weaviate Python client, please upgrade to v4.")
```

## File: `signatures/__init__.py`
```
from dspy.signatures.field import InputField, OldField, OldInputField, OldOutputField, OutputField
from dspy.signatures.signature import (
    Signature,
    SignatureMeta,
    ensure_signature,
    infer_prefix,
    make_signature,
)

__all__ = [
    "InputField",
    "OutputField",
    "OldField",
    "OldInputField",
    "OldOutputField",
    "SignatureMeta",
    "Signature",
    "infer_prefix",
    "ensure_signature",
    "make_signature",
]
```

## File: `signatures/field.py`
```
import pydantic

# The following arguments can be used in DSPy InputField and OutputField in addition
# to the standard pydantic.Field arguments. We just hope pydanitc doesn't add these,
# as it would give a name clash.
DSPY_FIELD_ARG_NAMES = ["desc", "prefix", "format", "parser", "__dspy_field_type"]

PYDANTIC_CONSTRAINT_MAP = {
    "gt": "greater than: ",
    "ge": "greater than or equal to: ",
    "lt": "less than: ",
    "le": "less than or equal to: ",
    "min_length": "minimum length: ",
    "max_length": "maximum length: ",
    "multiple_of": "a multiple of the given number: ",
    "allow_inf_nan": "allow 'inf', '-inf', 'nan' values: ",
}


def move_kwargs(**kwargs):
    # Pydantic doesn't allow arbitrary arguments to be given to fields,
    # but asks that
    # > any extra data you want to add to the JSON schema should be passed
    # > as a dictionary to the json_schema_extra keyword argument.
    # See: https://docs.pydantic.dev/2.6/migration/#changes-to-pydanticfield
    pydantic_kwargs = {}
    json_schema_extra = {}
    for k, v in kwargs.items():
        if k in DSPY_FIELD_ARG_NAMES:
            json_schema_extra[k] = v
        else:
            pydantic_kwargs[k] = v
    # Also copy over the pydantic "description" if no dspy "desc" is given.
    if "description" in kwargs and "desc" not in json_schema_extra:
        json_schema_extra["desc"] = kwargs["description"]
    constraints = _translate_pydantic_field_constraints(**kwargs)
    if constraints:
        json_schema_extra["constraints"] = constraints
    pydantic_kwargs["json_schema_extra"] = json_schema_extra
    return pydantic_kwargs


def _translate_pydantic_field_constraints(**kwargs):
    """Extracts Pydantic constraints and translates them into human-readable format."""

    constraints = []
    for key, value in kwargs.items():
        if key in PYDANTIC_CONSTRAINT_MAP:
            constraints.append(f"{PYDANTIC_CONSTRAINT_MAP[key]}{value}")

    return ", ".join(constraints)


def InputField(**kwargs): # noqa: N802
    return pydantic.Field(**move_kwargs(**kwargs, __dspy_field_type="input"))


def OutputField(**kwargs): # noqa: N802
    return pydantic.Field(**move_kwargs(**kwargs, __dspy_field_type="output"))


def new_to_old_field(field):
    return (OldInputField if field.json_schema_extra["__dspy_field_type"] == "input" else OldOutputField)(
        prefix=field.json_schema_extra["prefix"],
        desc=field.json_schema_extra["desc"],
        format=field.json_schema_extra.get("format"),
    )


class OldField:
    """A more ergonomic datatype that infers prefix and desc if omitted."""

    def __init__(self, *, prefix=None, desc=None, input, format=None):
        self.prefix = prefix  # This can be None initially and set later
        self.desc = desc
        self.format = format

    def finalize(self, key, inferred_prefix):
        """Set the prefix if it's not provided explicitly."""
        if self.prefix is None:
            self.prefix = inferred_prefix + ":"

        if self.desc is None:
            self.desc = f"${{{key}}}"

    def __repr__(self):
        return f"{self.__class__.__name__}(prefix={self.prefix}, desc={self.desc})"

    def __eq__(self, __value: object) -> bool:
        return self.__dict__ == __value.__dict__


class OldInputField(OldField):
    def __init__(self, *, prefix=None, desc=None, format=None):
        super().__init__(prefix=prefix, desc=desc, input=True, format=format)


class OldOutputField(OldField):
    def __init__(self, *, prefix=None, desc=None, format=None):
        super().__init__(prefix=prefix, desc=desc, input=False, format=format)
```

## File: `signatures/signature.py`
```
"""Signature class for DSPy.

You typically subclass the Signature class, like this:
    class MySignature(dspy.Signature):
        input: str = InputField(desc="...")
        output: int = OutputField(desc="...")

You can call Signature("input1, input2 -> output1, output2") to create a new signature type.
You can also include instructions, Signature("input -> output", "This is a test").
But it's generally better to use the make_signature function.

If you are not sure if your input is a string representation, (like "input1, input2 -> output1, output2"),
or a signature, you can use the ensure_signature function.

For compatibility with the legacy dsp format, you can use the signature_to_template function.
"""

import ast
import importlib
import inspect
import re
import sys
import types
import typing
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from dspy.signatures.field import InputField, OutputField


def _default_instructions(cls) -> str:
    inputs_ = ", ".join([f"`{field}`" for field in cls.input_fields])
    outputs_ = ", ".join([f"`{field}`" for field in cls.output_fields])
    return f"Given the fields {inputs_}, produce the fields {outputs_}."


class SignatureMeta(type(BaseModel)):
    def __call__(cls, *args, **kwargs):
        if cls is Signature:
            # We don't create an actual Signature instance, instead, we create a new Signature class.
            custom_types = kwargs.pop("custom_types", None)

            if custom_types is None and args and isinstance(args[0], str):
                custom_types = cls._detect_custom_types_from_caller(args[0])

            return make_signature(*args, custom_types=custom_types, **kwargs)
        return super().__call__(*args, **kwargs)

    @staticmethod
    def _detect_custom_types_from_caller(signature_str):
        """Detect custom types from the caller's frame based on the signature string.

        Note: This method relies on Python's frame introspection which has some limitations:
        1. May not work in all Python implementations (e.g., compiled with optimizations)
        2. Looks up a limited number of frames in the call stack
        3. Cannot find types that are imported but not in the caller's namespace

        For more reliable custom type resolution, explicitly provide types using the
        `custom_types` parameter when creating a Signature.
        """

        # Extract potential type names from the signature string, including dotted names
        # Match both simple types like 'MyType' and dotted names like 'Module.Type'
        type_pattern = r":\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)"
        type_names = re.findall(type_pattern, signature_str)
        if not type_names:
            return None

        # Get type references from caller frames by walking the stack
        found_types = {}

        needed_types = set()
        dotted_types = {}

        for type_name in type_names:
            parts = type_name.split(".")
            base_name = parts[0]

            if base_name not in typing.__dict__ and base_name not in __builtins__:
                if len(parts) > 1:
                    dotted_types[type_name] = base_name
                    needed_types.add(base_name)
                else:
                    needed_types.add(type_name)

        if not needed_types:
            return None

        frame = None
        try:
            frame = sys._getframe(1)  # Start one level up (skip this function)

            max_frames = 100
            frame_count = 0

            while frame and needed_types and frame_count < max_frames:
                frame_count += 1

                for type_name in list(needed_types):
                    if type_name in frame.f_locals:
                        found_types[type_name] = frame.f_locals[type_name]
                        needed_types.remove(type_name)
                    elif frame.f_globals and type_name in frame.f_globals:
                        found_types[type_name] = frame.f_globals[type_name]
                        needed_types.remove(type_name)

                # If we found all needed types, stop looking
                if not needed_types:
                    break

                frame = frame.f_back

            if needed_types and frame_count >= max_frames:
                import logging

                logging.getLogger("dspy").warning(
                    f"Reached maximum frame search depth ({max_frames}) while looking for types: {needed_types}. "
                    "Consider providing custom_types explicitly to Signature."
                )
        except (AttributeError, ValueError):
            # Handle environments where frame introspection is not available
            import logging

            logging.getLogger("dspy").debug(
                "Frame introspection failed while trying to resolve custom types. "
                "Consider providing custom_types explicitly to Signature."
            )
        finally:
            if frame:
                del frame

        return found_types or None

    def __new__(mcs, signature_name, bases, namespace, **kwargs):
        # At this point, the orders have been swapped already.
        field_order = [name for name, value in namespace.items() if isinstance(value, FieldInfo)]
        # Set `str` as the default type for all fields
        raw_annotations = namespace.get("__annotations__", {})
        for name, field in namespace.items():
            if not isinstance(field, FieldInfo):
                continue  # Don't add types to non-field attributes
            if not name.startswith("__") and name not in raw_annotations:
                raw_annotations[name] = str
        # Create ordered annotations dictionary that preserves field order
        ordered_annotations = {name: raw_annotations[name] for name in field_order if name in raw_annotations}
        # Add any remaining annotations that weren't in field_order
        ordered_annotations.update({k: v for k, v in raw_annotations.items() if k not in ordered_annotations})
        namespace["__annotations__"] = ordered_annotations

        # Let Pydantic do its thing
        cls = super().__new__(mcs, signature_name, bases, namespace, **kwargs)

        # If we don't have instructions, it might be because we are a derived generic type.
        # In that case, we should inherit the instructions from the base class.
        if cls.__doc__ is None:
            for base in bases:
                if isinstance(base, SignatureMeta):
                    doc = getattr(base, "__doc__", "")
                    if doc != "":
                        cls.__doc__ = doc

        # The more likely case is that the user has just not given us a type.
        # In that case, we should default to the input/output format.
        if cls.__doc__ is None:
            cls.__doc__ = _default_instructions(cls)

        # Ensure all fields are declared with InputField or OutputField
        cls._validate_fields()

        # Ensure all fields have a prefix
        for name, field in cls.model_fields.items():
            if "prefix" not in field.json_schema_extra:
                field.json_schema_extra["prefix"] = infer_prefix(name) + ":"
            if "desc" not in field.json_schema_extra:
                field.json_schema_extra["desc"] = f"${{{name}}}"

        return cls

    def _validate_fields(cls):
        for name, field in cls.model_fields.items():
            extra = field.json_schema_extra or {}
            field_type = extra.get("__dspy_field_type")
            if field_type not in ["input", "output"]:
                raise TypeError(
                    f"Field `{name}` in `{cls.__name__}` must be declared with InputField or OutputField, but "
                    f"field `{name}` has `field.json_schema_extra={field.json_schema_extra}`",
                )

    @property
    def instructions(cls) -> str:
        return inspect.cleandoc(getattr(cls, "__doc__", ""))

    @instructions.setter
    def instructions(cls, instructions: str) -> None:
        cls.__doc__ = instructions

    @property
    def input_fields(cls) -> dict[str, FieldInfo]:
        return cls._get_fields_with_type("input")

    @property
    def output_fields(cls) -> dict[str, FieldInfo]:
        return cls._get_fields_with_type("output")

    @property
    def fields(cls) -> dict[str, FieldInfo]:
        # Make sure to give input fields before output fields
        return {**cls.input_fields, **cls.output_fields}

    @property
    def signature(cls) -> str:
        """The string representation of the signature."""
        input_fields = ", ".join(cls.input_fields.keys())
        output_fields = ", ".join(cls.output_fields.keys())
        return f"{input_fields} -> {output_fields}"

    def _get_fields_with_type(cls, field_type) -> dict[str, FieldInfo]:
        return {k: v for k, v in cls.model_fields.items() if v.json_schema_extra["__dspy_field_type"] == field_type}

    def __repr__(cls):
        """Output a representation of the signature.

        Uses the form:
        Signature(question, context -> answer
            question: str = InputField(desc="..."),
            context: list[str] = InputField(desc="..."),
            answer: int = OutputField(desc="..."),
        ).
        """
        field_reprs = []
        for name, field in cls.fields.items():
            field_reprs.append(f"{name} = Field({field})")
        field_repr = "\n    ".join(field_reprs)
        return f"{cls.__name__}({cls.signature}\n    instructions={cls.instructions!r}\n    {field_repr}\n)"


class Signature(BaseModel, metaclass=SignatureMeta):
    ""

    # Note: Don't put a docstring here, as it will become the default instructions
    # for any signature that doesn't define it's own instructions.

    @classmethod
    def with_instructions(cls, instructions: str) -> type["Signature"]:
        return Signature(cls.fields, instructions)

    @classmethod
    def with_updated_fields(cls, name: str, type_: type | None = None, **kwargs: dict[str, Any]) -> type["Signature"]:
        """Create a new Signature class with the updated field information.

        Returns a new Signature class with the field, name, updated
        with fields[name].json_schema_extra[key] = value.

        Args:
            name: The name of the field to update.
            type_: The new type of the field.
            kwargs: The new values for the field.

        Returns:
            A new Signature class (not an instance) with the updated field information.
        """
        fields_copy = deepcopy(cls.fields)
        # Update `fields_copy[name].json_schema_extra` with the new kwargs, on conflicts
        # we use the new value in kwargs.
        fields_copy[name].json_schema_extra = {
            **fields_copy[name].json_schema_extra,
            **kwargs,
        }
        if type_ is not None:
            fields_copy[name].annotation = type_
        return Signature(fields_copy, cls.instructions)

    @classmethod
    def prepend(cls, name, field, type_=None) -> type["Signature"]:
        return cls.insert(0, name, field, type_)

    @classmethod
    def append(cls, name, field, type_=None) -> type["Signature"]:
        return cls.insert(-1, name, field, type_)

    @classmethod
    def delete(cls, name) -> type["Signature"]:
        fields = dict(cls.fields)

        fields.pop(name, None)

        return Signature(fields, cls.instructions)

    @classmethod
    def insert(cls, index: int, name: str, field, type_: type | None = None) -> type["Signature"]:
        # It's possible to set the type as annotation=type in pydantic.Field(...)
        # But this may be annoying for users, so we allow them to pass the type
        if type_ is None:
            type_ = field.annotation
        if type_ is None:
            type_ = str

        input_fields = list(cls.input_fields.items())
        output_fields = list(cls.output_fields.items())

        # Choose the list to insert into based on the field type
        lst = input_fields if field.json_schema_extra["__dspy_field_type"] == "input" else output_fields
        # We support negative insert indices
        if index < 0:
            index += len(lst) + 1
        if index < 0 or index > len(lst):
            raise ValueError(
                f"Invalid index to insert: {index}, index must be in the range of [{len(lst) - 1}, {len(lst)}] for "
                f"{field.json_schema_extra['__dspy_field_type']} fields, but received: {index}.",
            )
        lst.insert(index, (name, (type_, field)))

        new_fields = dict(input_fields + output_fields)
        return Signature(new_fields, cls.instructions)

    @classmethod
    def equals(cls, other) -> bool:
        """Compare the JSON schema of two Signature classes."""
        if not isinstance(other, type) or not issubclass(other, BaseModel):
            return False
        if cls.instructions != other.instructions:
            return False
        for name in cls.fields.keys() | other.fields.keys():
            if name not in other.fields or name not in cls.fields:
                return False
            if cls.fields[name].json_schema_extra != other.fields[name].json_schema_extra:
                return False
        return True

    @classmethod
    def dump_state(cls):
        state = {"instructions": cls.instructions, "fields": []}
        for field in cls.fields:
            state["fields"].append(
                {
                    "prefix": cls.fields[field].json_schema_extra["prefix"],
                    "description": cls.fields[field].json_schema_extra["desc"],
                }
            )

        return state

    @classmethod
    def load_state(cls, state):
        signature_copy = Signature(deepcopy(cls.fields), cls.instructions)

        signature_copy.instructions = state["instructions"]
        for field, saved_field in zip(signature_copy.fields.values(), state["fields"], strict=False):
            field.json_schema_extra["prefix"] = saved_field["prefix"]
            field.json_schema_extra["desc"] = saved_field["description"]

        return signature_copy


def ensure_signature(signature: str | type[Signature], instructions=None) -> type[Signature]:
    if signature is None:
        return None
    if isinstance(signature, str):
        return Signature(signature, instructions)
    if instructions is not None:
        raise ValueError("Don't specify instructions when initializing with a Signature")
    return signature


def make_signature(
    signature: str | dict[str, tuple[type, FieldInfo]],
    instructions: str | None = None,
    signature_name: str = "StringSignature",
    custom_types: dict[str, type] | None = None,
) -> type[Signature]:
    """Create a new Signature subclass with the specified fields and instructions.

    Args:
        signature: Either a string in the format "input1, input2 -> output1, output2"
            or a dictionary mapping field names to tuples of (type, FieldInfo).
        instructions: Optional string containing instructions/prompt for the signature.
            If not provided, defaults to a basic description of inputs and outputs.
        signature_name: Optional string to name the generated Signature subclass.
            Defaults to "StringSignature".
        custom_types: Optional dictionary mapping type names to their actual type objects.
            Useful for resolving custom types that aren't built-ins or in the typing module.

    Returns:
        A new signature class with the specified fields and instructions.

    Examples:

    ```
    # Using string format
    sig1 = make_signature("question, context -> answer")

    # Using dictionary format
    sig2 = make_signature({
        "question": (str, InputField()),
        "answer": (str, OutputField())
    })

    # Using custom types
    class MyType:
        pass

    sig3 = make_signature("input: MyType -> output", custom_types={"MyType": MyType})
    ```
    """
    # Prepare the names dictionary for type resolution
    names = None
    if custom_types:
        names = dict(typing.__dict__)
        names.update(custom_types)

    fields = _parse_signature(signature, names) if isinstance(signature, str) else signature

    # Validate the fields, this is important because we sometimes forget the
    # slightly unintuitive syntax with tuples of (type, Field)
    fixed_fields = {}
    for name, type_field in fields.items():
        if not isinstance(name, str):
            raise ValueError(f"Field names must be strings, but received: {name}.")
        if isinstance(type_field, FieldInfo):
            type_ = type_field.annotation
            field = type_field
        else:
            if not isinstance(type_field, tuple):
                raise ValueError(f"Field values must be tuples, but received: {type_field}.")
            type_, field = type_field
        # It might be better to be explicit about the type, but it currently would break
        # program of thought and teleprompters, so we just silently default to string.
        if type_ is None:
            type_ = str
        if not isinstance(type_, (type, typing._GenericAlias, types.GenericAlias, typing._SpecialForm, types.UnionType)):
            raise ValueError(f"Field types must be types, but received: {type_} of type {type(type_)}.")
        if not isinstance(field, FieldInfo):
            raise ValueError(f"Field values must be Field instances, but received: {field}.")
        fixed_fields[name] = (type_, field)

    # Default prompt when no instructions are provided
    if instructions is None:
        sig = Signature(signature, "")  # Simple way to parse input/output fields
        instructions = _default_instructions(sig)

    return create_model(
        signature_name,
        __base__=Signature,
        __doc__=instructions,
        **fixed_fields,
    )


def _parse_signature(signature: str, names=None) -> dict[str, tuple[type, Field]]:
    if signature.count("->") != 1:
        raise ValueError(f"Invalid signature format: '{signature}', must contain exactly one '->'.")

    inputs_str, outputs_str = signature.split("->")

    fields = {}
    for field_name, field_type in _parse_field_string(inputs_str, names):
        fields[field_name] = (field_type, InputField())
    for field_name, field_type in _parse_field_string(outputs_str, names):
        fields[field_name] = (field_type, OutputField())

    return fields


def _parse_field_string(field_string: str, names=None) -> dict[str, str]:
    """Extract the field name and type from field string in the string-based Signature.

    It takes a string like "x: int, y: str" and returns a dictionary mapping field names to their types.
    For example, "x: int, y: str" -> [("x", int), ("y", str)]. This function utitlizes the Python AST to parse the
    fields and types.
    """

    args = ast.parse(f"def f({field_string}): pass").body[0].args.args
    field_names = [arg.arg for arg in args]
    types = [str if arg.annotation is None else _parse_type_node(arg.annotation, names) for arg in args]
    return zip(field_names, types, strict=False)


def _parse_type_node(node, names=None) -> Any:
    """Recursively parse an AST node representing a type annotation.

    This function converts Python's Abstract Syntax Tree (AST) nodes into actual Python types.
    It's used to parse type annotations in signature strings like "x: list[int] -> y: str".

    Examples:
        - For "x: int", the AST node represents 'int' and returns the int type
        - For "x: list[str]", it processes a subscript node to return typing.list[str]
        - For "x: Optional[int]", it handles the Union type to return Optional[int]
        - For "x: MyModule.CustomType", it processes attribute access to return the actual type

    Args:
        node: An AST node from Python's ast module, representing a type annotation.
            Common node types include:
            - ast.Name: Simple types like 'int', 'str'
            - ast.Attribute: Nested types like 'typing.List'
            - ast.Subscript: Generic types like 'list[int]'
        names: Optional dictionary mapping type names to their actual type objects.
            Defaults to Python's typing module contents plus NoneType.

    Returns:
        The actual Python type represented by the AST node.

    Raises:
        ValueError: If the AST node represents an unknown or invalid type annotation.
    """

    if names is None:
        names = dict(typing.__dict__)
        names["NoneType"] = type(None)

    def resolve_name(type_name: str):
        # Check if it's a built-in known type or in the provided names
        if type_name in names:
            return names[type_name]
        # Common built-in types
        builtin_types = [int, str, float, bool, list, tuple, dict, set, frozenset, complex, bytes, bytearray]

        # Check if it matches any known built-in type by name
        for t in builtin_types:
            if t.__name__ == type_name:
                return t

        # Attempt to import a module with this name dynamically
        # This allows handling of module-based annotations like `dspy.Image`.
        try:
            mod = importlib.import_module(type_name)
            names[type_name] = mod
            return mod
        except ImportError:
            pass

        # If we don't know the type or module, raise an error
        raise ValueError(f"Unknown name: {type_name}")

    if isinstance(node, ast.Module):
        if len(node.body) != 1:
            raise ValueError(f"Code is not syntactically valid: {ast.dump(node)}")
        return _parse_type_node(node.body[0], names)

    if isinstance(node, ast.Expr):
        return _parse_type_node(node.value, names)

    if isinstance(node, ast.Name):
        return resolve_name(node.id)

    if isinstance(node, ast.Attribute):
        base = _parse_type_node(node.value, names)
        attr_name = node.attr

        if hasattr(base, attr_name):
            return getattr(base, attr_name)

        if isinstance(node.value, ast.Name):
            full_name = f"{node.value.id}.{attr_name}"
            if full_name in names:
                return names[full_name]

        raise ValueError(f"Unknown attribute: {attr_name} on {base}")

    if isinstance(node, ast.Subscript):
        base_type = _parse_type_node(node.value, names)
        slice_node = node.slice
        if isinstance(slice_node, ast.Index):  # For older Python versions
            slice_node = slice_node.value

        if isinstance(slice_node, ast.Tuple):
            arg_types = tuple(_parse_type_node(elt, names) for elt in slice_node.elts)
        else:
            arg_types = (_parse_type_node(slice_node, names),)

        # Special handling for Union, Optional
        if base_type is typing.Union:
            return typing.Union[arg_types]
        if base_type is typing.Optional:
            if len(arg_types) != 1:
                raise ValueError("Optional must have exactly one type argument")
            return typing.Optional[arg_types[0]]

        return base_type[arg_types]

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        # Handle PEP 604: int | None, str | float, etc.
        left = _parse_type_node(node.left, names)
        right = _parse_type_node(node.right, names)

        # Optional[X] is Union[X, NoneType]
        if right is type(None):
            return typing.Optional[left]
        if left is type(None):
            return typing.Optional[right]
        return typing.Union[left, right]

    if isinstance(node, ast.Tuple):
        return tuple(_parse_type_node(elt, names) for elt in node.elts)

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Field":
        keys = [kw.arg for kw in node.keywords]
        values = []
        for kw in node.keywords:
            if isinstance(kw.value, ast.Constant):
                values.append(kw.value.value)
            else:
                values.append(_parse_type_node(kw.value, names))
        return Field(**dict(zip(keys, values, strict=False)))

    raise ValueError(
        f"Failed to parse string-base Signature due to unhandled AST node type in annotation: {ast.dump(node)}. "
        "Please consider using class-based DSPy Signatures instead."
    )


def infer_prefix(attribute_name: str) -> str:
    """Infer a prefix from an attribute name by converting it to a human-readable format.

    Examples:
        "camelCaseText" -> "Camel Case Text"
        "snake_case_text" -> "Snake Case Text"
        "text2number" -> "Text 2 Number"
        "HTMLParser" -> "HTML Parser"
    """
    # Step 1: Convert camelCase to snake_case
    # Example: "camelCase" -> "camel_Case"
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", attribute_name)

    # Handle consecutive capitals
    # Example: "camel_Case" -> "camel_case"
    intermediate_name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)

    # Step 2: Handle numbers by adding underscores around them
    # Example: "text2number" -> "text_2_number"
    with_underscores_around_numbers = re.sub(
        r"([a-zA-Z])(\d)",  # Match letter followed by number
        r"\1_\2",  # Add underscore between them
        intermediate_name,
    )
    # Example: "2text" -> "2_text"
    with_underscores_around_numbers = re.sub(
        r"(\d)([a-zA-Z])",  # Match number followed by letter
        r"\1_\2",  # Add underscore between them
        with_underscores_around_numbers,
    )

    # Step 3: Convert to Title Case while preserving acronyms
    words = with_underscores_around_numbers.split("_")
    title_cased_words = []
    for word in words:
        if word.isupper():
            # Preserve acronyms like 'HTML', 'API' as-is
            title_cased_words.append(word)
        else:
            # Capitalize first letter: 'text' -> 'Text'
            title_cased_words.append(word.capitalize())

    # Join words with spaces
    # Example: ["Text", "2", "Number"] -> "Text 2 Number"
    return " ".join(title_cased_words)
```

## File: `signatures/utils.py`
```
from typing import Literal

from pydantic.fields import FieldInfo


def get_dspy_field_type(field: FieldInfo) -> Literal["input", "output"]:
    field_type = field.json_schema_extra.get("__dspy_field_type")
    if field_type is None:
        raise ValueError(f"Field {field} does not have a __dspy_field_type")
    return field_type
```

## File: `streaming/__init__.py`
```
from dspy.streaming.messages import StatusMessage, StatusMessageProvider, StreamResponse
from dspy.streaming.streamify import apply_sync_streaming, streamify, streaming_response
from dspy.streaming.streaming_listener import StreamListener

__all__ = [
    "StatusMessage",
    "StatusMessageProvider",
    "streamify",
    "StreamListener",
    "StreamResponse",
    "streaming_response",
    "apply_sync_streaming",
]
```

## File: `streaming/messages.py`
```
import asyncio
import concurrent.futures
from dataclasses import dataclass
from typing import Any

from asyncer import syncify

from dspy.dsp.utils.settings import settings
from dspy.utils.callback import BaseCallback


@dataclass
class StreamResponse:
    predict_name: str
    signature_field_name: str
    chunk: str


@dataclass
class StatusMessage:
    """Dataclass that wraps a status message for status streaming."""

    message: str


def sync_send_to_stream(stream, message):
    """Send message to stream in a sync context, regardless of event loop state."""

    async def _send():
        await stream.send(message)

    try:
        asyncio.get_running_loop()

        # If we're in an event loop, offload to a new thread with its own event loop
        def run_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(_send())
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_new_loop)
            return future.result()
    except RuntimeError:
        # Not in an event loop, safe to use a new event loop in this thread
        return syncify(_send)()


class StatusMessageProvider:
    """Provides customizable status message streaming for DSPy programs.

    This class serves as a base for creating custom status message providers. Users can subclass
    and override its methods to define specific status messages for different stages of program execution,
    each method must return a string.

    Example:
    ```python
    class MyStatusMessageProvider(StatusMessageProvider):
        def lm_start_status_message(self, instance, inputs):
            return f"Calling LM with inputs {inputs}..."

        def module_end_status_message(self, outputs):
            return f"Module finished with output: {outputs}!"

    program = dspy.streamify(dspy.Predict("q->a"), status_message_provider=MyStatusMessageProvider())
    ```
    """

    def tool_start_status_message(self, instance: Any, inputs: dict[str, Any]):
        """Status message before a `dspy.Tool` is called."""
        return f"Calling tool {instance.name}..."

    def tool_end_status_message(self, outputs: Any):
        """Status message after a `dspy.Tool` is called."""
        return "Tool calling finished! Querying the LLM with tool calling results..."

    def module_start_status_message(self, instance: Any, inputs: dict[str, Any]):
        """Status message before a `dspy.Module` or `dspy.Predict` is called."""
        pass

    def module_end_status_message(self, outputs: Any):
        """Status message after a `dspy.Module` or `dspy.Predict` is called."""
        pass

    def lm_start_status_message(self, instance: Any, inputs: dict[str, Any]):
        """Status message before a `dspy.LM` is called."""
        pass

    def lm_end_status_message(self, outputs: Any):
        """Status message after a `dspy.LM` is called."""
        pass


class StatusStreamingCallback(BaseCallback):
    def __init__(self, status_message_provider: StatusMessageProvider | None = None):
        self.status_message_provider = status_message_provider or StatusMessageProvider()

    def on_tool_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        stream = settings.send_stream
        if stream is None or instance.name == "finish":
            return

        status_message = self.status_message_provider.tool_start_status_message(instance, inputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))

    def on_tool_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ):
        stream = settings.send_stream
        if stream is None or outputs == "Completed.":
            return

        status_message = self.status_message_provider.tool_end_status_message(outputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))

    def on_lm_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        stream = settings.send_stream
        if stream is None:
            return

        status_message = self.status_message_provider.lm_start_status_message(instance, inputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))

    def on_lm_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ):
        stream = settings.send_stream
        if stream is None:
            return

        status_message = self.status_message_provider.lm_end_status_message(outputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))

    def on_module_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        stream = settings.send_stream
        if stream is None:
            return

        status_message = self.status_message_provider.module_start_status_message(instance, inputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))

    def on_module_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ):
        stream = settings.send_stream
        if stream is None:
            return

        status_message = self.status_message_provider.module_end_status_message(outputs)
        if status_message:
            sync_send_to_stream(stream, StatusMessage(status_message))
```

## File: `streaming/streamify.py`
```
import asyncio
import contextvars
import logging
import threading
from asyncio import iscoroutinefunction
from queue import Queue
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Generator

import litellm
import ujson
from anyio import create_memory_object_stream, create_task_group
from anyio.streams.memory import MemoryObjectSendStream
from litellm import ModelResponseStream

from dspy.dsp.utils.settings import settings
from dspy.primitives.prediction import Prediction
from dspy.streaming.messages import StatusMessage, StatusMessageProvider, StatusStreamingCallback
from dspy.streaming.streaming_listener import StreamListener, find_predictor_for_stream_listeners
from dspy.utils.asyncify import asyncify

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dspy.primitives.module import Module


def streamify(
    program: "Module",
    status_message_provider: StatusMessageProvider | None = None,
    stream_listeners: list[StreamListener] | None = None,
    include_final_prediction_in_output_stream: bool = True,
    is_async_program: bool = False,
    async_streaming: bool = True,
) -> Callable[[Any, Any], Awaitable[Any]]:
    """
    Wrap a DSPy program so that it streams its outputs incrementally, rather than returning them
    all at once. It also provides status messages to the user to indicate the progress of the program, and users
    can implement their own status message provider to customize the status messages and what module to generate
    status messages for.

    Args:
        program: The DSPy program to wrap with streaming functionality.
        status_message_provider: A custom status message generator to use instead of the default one. Users can
            implement their own status message generator to customize the status messages and what module to generate
            status messages for.
        stream_listeners: A list of stream listeners to capture the streaming output of specific fields of sub predicts
            in the program. When provided, only the target fields in the target predict will be streamed to the user.
        include_final_prediction_in_output_stream: Whether to include the final prediction in the output stream, only
            useful when `stream_listeners` is provided. If `False`, the final prediction will not be included in the
            output stream. When the program hit cache, or no listeners captured anything, the final prediction will
            still be included in the output stream even if this is `False`.
        is_async_program: Whether the program is async. If `False`, the program will be wrapped with `asyncify`,
            otherwise the program will be called with `acall`.
        async_streaming: Whether to return an async generator or a sync generator. If `False`, the streaming will be
            converted to a sync generator.

    Returns:
        A function that takes the same arguments as the original program, but returns an async
            generator that yields the program's outputs incrementally.

    Example:

    ```python
    import asyncio
    import dspy

    dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))
    # Create the program and wrap it with streaming functionality
    program = dspy.streamify(dspy.Predict("q->a"))

    # Use the program with streaming output
    async def use_streaming():
        output = program(q="Why did a chicken cross the kitchen?")
        return_value = None
        async for value in output:
            if isinstance(value, dspy.Prediction):
                return_value = value
            else:
                print(value)
        return return_value

    output = asyncio.run(use_streaming())
    print(output)
    ```

    Example with custom status message provider:
    ```python
    import asyncio
    import dspy

    dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

    class MyStatusMessageProvider(StatusMessageProvider):
        def module_start_status_message(self, instance, inputs):
            return f"Predicting..."

        def tool_end_status_message(self, outputs):
            return f"Tool calling finished with output: {outputs}!"

    # Create the program and wrap it with streaming functionality
    program = dspy.streamify(dspy.Predict("q->a"), status_message_provider=MyStatusMessageProvider())

    # Use the program with streaming output
    async def use_streaming():
        output = program(q="Why did a chicken cross the kitchen?")
        return_value = None
        async for value in output:
            if isinstance(value, dspy.Prediction):
                return_value = value
            else:
                print(value)
        return return_value

    output = asyncio.run(use_streaming())
    print(output)
    ```

    Example with stream listeners:

    ```python
    import asyncio
    import dspy

    dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False))

    # Create the program and wrap it with streaming functionality
    predict = dspy.Predict("question->answer, reasoning")
    stream_listeners = [
        dspy.streaming.StreamListener(signature_field_name="answer"),
        dspy.streaming.StreamListener(signature_field_name="reasoning"),
    ]
    stream_predict = dspy.streamify(predict, stream_listeners=stream_listeners)

    async def use_streaming():
        output = stream_predict(
            question="why did a chicken cross the kitchen?",
            include_final_prediction_in_output_stream=False,
        )
        return_value = None
        async for value in output:
            if isinstance(value, dspy.Prediction):
                return_value = value
            else:
                print(value)
        return return_value

    output = asyncio.run(use_streaming())
    print(output)
    ```

    You should see the streaming chunks (in the format of `dspy.streaming.StreamResponse`) in the console output.
    """
    stream_listeners = stream_listeners or []
    if len(stream_listeners) > 0:
        predict_id_to_listener = find_predictor_for_stream_listeners(program, stream_listeners)
    else:
        predict_id_to_listener = {}

    if is_async_program:
        program = program.acall
    elif not iscoroutinefunction(program):
        program = asyncify(program)

    callbacks = settings.callbacks
    status_streaming_callback = StatusStreamingCallback(status_message_provider)
    if not any(isinstance(c, StatusStreamingCallback) for c in callbacks):
        callbacks.append(status_streaming_callback)

    async def generator(args, kwargs, stream: MemoryObjectSendStream):
        with settings.context(send_stream=stream, callbacks=callbacks, stream_listeners=stream_listeners):
            prediction = await program(*args, **kwargs)

        await stream.send(prediction)

    async def async_streamer(*args, **kwargs):
        send_stream, receive_stream = create_memory_object_stream(16)
        async with create_task_group() as tg, send_stream, receive_stream:
            tg.start_soon(generator, args, kwargs, send_stream)

            async for value in receive_stream:
                if isinstance(value, ModelResponseStream):
                    if len(predict_id_to_listener) == 0:
                        # No listeners are configured, yield the chunk directly for backwards compatibility.
                        yield value
                    else:
                        # We are receiving a chunk from the LM's response stream, delegate it to the listeners to
                        # determine if we should yield a value to the user.
                        output = None
                        for listener in predict_id_to_listener[value.predict_id]:
                            # There should be at most one listener provides a return value.
                            output = listener.receive(value) or output
                        if output:
                            yield output
                elif isinstance(value, StatusMessage):
                    yield value
                elif isinstance(value, Prediction):
                    if include_final_prediction_in_output_stream:
                        yield value
                    elif (
                        len(stream_listeners) == 0
                        or any(listener.cache_hit for listener in stream_listeners)
                        or not any(listener.stream_start for listener in stream_listeners)
                    ):
                        yield value
                    return

    if async_streaming:
        return async_streamer
    else:

        def sync_streamer(*args, **kwargs):
            output = async_streamer(*args, **kwargs)
            return apply_sync_streaming(output)

        return sync_streamer


def apply_sync_streaming(async_generator: AsyncGenerator) -> Generator:
    """Convert the async streaming generator to a sync generator."""
    queue = Queue()  # Queue to hold items from the async generator
    stop_sentinel = object()  # Sentinel to signal the generator is complete

    # To propagate prediction request ID context to the child thread
    context = contextvars.copy_context()

    def producer():
        """Runs in a background thread to fetch items asynchronously."""

        async def runner():
            try:
                async for item in async_generator:
                    queue.put(item)
            finally:
                # Signal completion
                queue.put(stop_sentinel)

        context.run(asyncio.run, runner())

    # Start the producer in a background thread
    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    # Consume items from the queue
    while True:
        item = queue.get()  # Block until an item is available
        if item is stop_sentinel:
            break
        yield item


async def streaming_response(streamer: AsyncGenerator) -> AsyncGenerator:
    """
    Convert a DSPy program output stream to an OpenAI-compatible output stream that can be
    used by a service as an API response to a streaming request.

    Args:
        streamer: An async generator that yields values from a DSPy program output stream.
    Returns:
        An async generator that yields OpenAI-compatible streaming response chunks.
    """
    async for value in streamer:
        if isinstance(value, Prediction):
            data = {"prediction": dict(value.items(include_dspy=False))}
            yield f"data: {ujson.dumps(data)}\n\n"
        elif isinstance(value, litellm.ModelResponseStream):
            data = {"chunk": value.json()}
            yield f"data: {ujson.dumps(data)}\n\n"
        elif isinstance(value, str) and value.startswith("data:"):
            # The chunk value is an OpenAI-compatible streaming chunk value,
            # e.g. "data: {"finish_reason": "stop", "index": 0, "is_finished": True, ...}",
            # so yield it directly
            yield value
        else:
            raise ValueError(f"Unknown chunk value type: {value}")
    yield "data: [DONE]\n\n"
```

## File: `streaming/streaming_listener.py`
```
import re
from collections import defaultdict
from queue import Queue
from typing import TYPE_CHECKING, Any

from litellm import ModelResponseStream

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.xml_adapter import XMLAdapter
from dspy.dsp.utils.settings import settings
from dspy.streaming.messages import StreamResponse

if TYPE_CHECKING:
    from dspy.primitives.module import Module

ADAPTER_SUPPORT_STREAMING = [ChatAdapter, XMLAdapter, JSONAdapter]


class StreamListener:
    """Class that listens to the stream to capture the streeaming of a specific output field of a predictor."""

    def __init__(
        self,
        signature_field_name: str,
        predict: Any = None,
        predict_name: str | None = None,
        allow_reuse: bool = False,
    ):
        """
        Args:
            signature_field_name: The name of the field to listen to.
            predict: The predictor to listen to. If None, when calling `streamify()` it will automatically look for
                the predictor that has the `signature_field_name` in its signature.
            predict_name: The name of the predictor to listen to. If None, when calling `streamify()` it will
                automatically look for the predictor that has the `signature_field_name` in its signature.
            allow_reuse: If True, the stream listener can be reused for multiple streams. Please note that this could
                hurt the performance because the same stream chunk is sent to multiple listeners.
        """
        self.signature_field_name = signature_field_name
        self.predict = predict
        self.predict_name = predict_name

        self.field_start_queue = []
        self.field_end_queue = Queue()
        self.stream_start = False
        self.stream_end = False
        self.cache_hit = False
        self.allow_reuse = allow_reuse

        self.adapter_identifiers = {
            "ChatAdapter": {
                "start_identifier": f"[[ ## {self.signature_field_name} ## ]]",
                "end_identifier": re.compile(r"\[\[ ## (\w+) ## \]\]"),
                "start_indicator": "[",
            },
            "JSONAdapter": {
                "start_identifier": f'"{self.signature_field_name}":',
                "end_identifier": re.compile(r"\w*\"(,|\s*})"),
                "start_indicator": '"',
            },
            "XMLAdapter": {
                "start_identifier": f"<{self.signature_field_name}>",
                "end_identifier": re.compile(rf"</{self.signature_field_name}>"),
                "start_indicator": "<",
            },
        }

    def _buffered_message_end_with_start_identifier(self, concat_message: str, start_identifier: str) -> str:
        for i in range(len(concat_message)):
            if start_identifier.startswith(concat_message[len(concat_message) - i - 1 :]):
                return True
        return False

    def receive(self, chunk: ModelResponseStream):
        adapter_name = settings.adapter.__class__.__name__ if settings.adapter else "ChatAdapter"
        if adapter_name not in self.adapter_identifiers:
            raise ValueError(
                f"Unsupported adapter for streaming: {adapter_name}, please use one of the following adapters: "
                f"{', '.join([a.__name__ for a in ADAPTER_SUPPORT_STREAMING])}"
            )
        start_identifier = self.adapter_identifiers[adapter_name]["start_identifier"]
        end_identifier = self.adapter_identifiers[adapter_name]["end_identifier"]
        start_indicator = self.adapter_identifiers[adapter_name]["start_indicator"]

        if self.stream_end:
            if self.allow_reuse:
                # Clear up the state for the next stream.
                self.stream_end = False
                self.cache_hit = False
                self.field_start_queue = []
                self.field_end_queue = Queue()
                self.stream_start = False
            else:
                return

        try:
            chunk_message = chunk.choices[0].delta.content
            if chunk_message is None:
                return
        except Exception:
            return

        if chunk_message and start_identifier in chunk_message:
            # If the cache is hit, the chunk_message could be the full response. When it happens we can
            # directly end the stream listening. In some models like gemini, each stream chunk can be multiple
            # tokens, so it's posible that response only has one chunk, we also fall back to this logic.
            message_after_start_identifier = chunk_message[
                chunk_message.find(start_identifier) + len(start_identifier) :
            ]
            if re.search(end_identifier, message_after_start_identifier):
                self.cache_hit = True
                self.stream_start = True
                self.stream_end = True
                return

        if len(self.field_start_queue) == 0 and not self.stream_start and start_indicator in chunk_message:
            # We look for the pattern of start_identifier, i.e., "[[ ## {self.signature_field_name} ## ]]" for
            # ChatAdapter to identify the start of the stream of our target field. Once the start_indicator, i.e., "[["
            # for ChatAdapter, is found, we start checking the next tokens
            self.field_start_queue.append(chunk_message)
            return

        if len(self.field_start_queue) > 0 and not self.stream_start:
            # We keep appending the tokens to the queue until we have a full identifier or the concanated
            # tokens no longer match our expected identifier.
            self.field_start_queue.append(chunk_message)
            concat_message = "".join(self.field_start_queue)

            if start_identifier in concat_message:
                # We have a full identifier, we can start the stream.
                self.stream_start = True
                self.field_start_queue = []
                # Keep the part after the start_identifier from the concat_message, we need to write it to the buffer.
                value_start_index = concat_message.find(start_identifier) + len(start_identifier)
                chunk_message = concat_message[value_start_index:].lstrip()
                if isinstance(settings.adapter, JSONAdapter) and chunk_message.startswith('"'):
                    # For JSONAdapter, we need to remove the leading ". We cannot do this with the start_identifier
                    # because there could be a few splitters between ':' and '"', e.g., '"name": "value"'.
                    chunk_message = chunk_message[1:]

            elif self._buffered_message_end_with_start_identifier(concat_message.strip(), start_identifier):
                # If the buffered message ends with part of the start_identifier, we keep looking for the
                # start_identifier from the token stream.
                return
            else:
                # Doesn't match the expected identifier, reset the queue.
                self.field_start_queue = []
                return

        if self.stream_start:
            # The stream is started, we keep returning the token until we see the start of the next field.
            token = None
            self.field_end_queue.put(chunk_message)
            if self.field_end_queue.qsize() > 10:
                # We keep the last 10 tokens in the buffer to check if they form a valid identifier for end_identifier,
                # i.e., "[[ ## {next_field_name} ## ]]" for ChatAdapter to identify the end of the current field.
                # In most cases 10 tokens are enough to cover the end_identifier for all adapters.
                token = self.field_end_queue.get()
            concat_message = "".join(self.field_end_queue.queue).strip()
            if re.search(end_identifier, concat_message):
                # The next field is identified, we can end the stream and flush out all tokens in the buffer.
                self.stream_end = True
                last_token = self.flush()
                token = token + last_token if token else last_token
                token = token.rstrip()  # Remove the trailing \n\n

            if token:
                return StreamResponse(self.predict_name, self.signature_field_name, token)

    def flush(self) -> str:
        """Flush all tokens in the field end queue.

        This method is called to flush out the last a few tokens when the stream is ended. These tokens
        are in the buffer because we don't directly yield the tokens received by the stream listener
        with the purpose to not yield the end_identifier tokens, e.g., "[[ ## ... ## ]]" for ChatAdapter.
        """
        last_tokens = "".join(self.field_end_queue.queue)
        self.field_end_queue = Queue()
        if isinstance(settings.adapter, JSONAdapter):
            match = re.search(r'",|"\s*}', last_tokens)
            if match:
                boundary_index = match.start()
            else:
                boundary_index = len(last_tokens)
            return last_tokens[:boundary_index]
        elif isinstance(settings.adapter, XMLAdapter):
            boundary_index = last_tokens.find(f"</{self.signature_field_name}>")
            if boundary_index == -1:
                boundary_index = len(last_tokens)
            return last_tokens[:boundary_index]
        elif isinstance(settings.adapter, ChatAdapter) or settings.adapter is None:
            boundary_index = last_tokens.find("[[")
            return last_tokens[:boundary_index]
        else:
            raise ValueError(
                f"Unsupported adapter for streaming: {settings.adapter}, please use one of the following adapters: "
                f"{', '.join([a.__name__ for a in ADAPTER_SUPPORT_STREAMING])}"
            )


def find_predictor_for_stream_listeners(program: "Module", stream_listeners: list[StreamListener]):
    """Find the predictor for each stream listener.

    This is a utility function to automatically find the predictor for each stream listener. It is used when some
    listeners don't specify the predictor they want to listen to. If a listener's `signature_field_name` is not
    unique in the program, this function will raise an error.
    """
    predictors = program.named_predictors()

    field_name_to_named_predictor = {}
    for listener in stream_listeners:
        if listener.predict:
            continue
        field_name_to_named_predictor[listener.signature_field_name] = None

    for name, predictor in predictors:
        for field_name, field_info in predictor.signature.output_fields.items():
            if field_name not in field_name_to_named_predictor:
                continue

            if field_name_to_named_predictor[field_name] is not None:
                raise ValueError(
                    f"Signature field {field_name} is not unique in the program, cannot automatically determine which "
                    "predictor to use for streaming. Please specify the predictor to listen to."
                )

            if field_info.annotation is not str:
                raise ValueError(
                    f"Stream listener can only be applied to string output field, but your field {field_name} is of "
                    f"type {field_info.annotation}."
                )

            field_name_to_named_predictor[field_name] = (name, predictor)

    predict_id_to_listener = defaultdict(list)
    for listener in stream_listeners:
        if listener.predict:
            predict_id_to_listener[id(listener.predict)].append(listener)
            continue
        if listener.signature_field_name not in field_name_to_named_predictor:
            raise ValueError(
                f"Signature field {listener.signature_field_name} is not a field of any predictor in the program, "
                "cannot automatically determine which predictor to use for streaming. Please verify your field name or "
                "specify the predictor to listen to."
            )
        listener.predict_name, listener.predict = field_name_to_named_predictor[listener.signature_field_name]
        predict_id_to_listener[id(listener.predict)].append(listener)
    return predict_id_to_listener
```

## File: `teleprompt/__init__.py`
```
from dspy.teleprompt.avatar_optimizer import AvatarOptimizer
from dspy.teleprompt.bettertogether import BetterTogether
from dspy.teleprompt.bootstrap import BootstrapFewShot
from dspy.teleprompt.bootstrap_finetune import BootstrapFinetune
from dspy.teleprompt.copro_optimizer import COPRO
from dspy.teleprompt.ensemble import Ensemble
from dspy.teleprompt.infer_rules import InferRules
from dspy.teleprompt.knn_fewshot import KNNFewShot
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2
from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch
from dspy.teleprompt.simba import SIMBA
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.teleprompt.teleprompt_optuna import BootstrapFewShotWithOptuna
from dspy.teleprompt.vanilla import LabeledFewShot

__all__ = [
    "AvatarOptimizer",
    "BetterTogether",
    "BootstrapFewShot",
    "BootstrapFinetune",
    "COPRO",
    "Ensemble",
    "KNNFewShot",
    "MIPROv2",
    "BootstrapFewShotWithRandomSearch",
    "BootstrapFewShotWithOptuna",
    "LabeledFewShot",
    "InferRules",
    "SIMBA",
]
```

## File: `teleprompt/avatar_optimizer.py`
```
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from random import sample
from typing import Callable

from pydantic import BaseModel
from tqdm import tqdm

import dspy
from dspy.predict.avatar import ActionOutput
from dspy.teleprompt.teleprompt import Teleprompter

DEFAULT_MAX_EXAMPLES = 10


class EvalResult(BaseModel):
    example: dict
    score: float
    actions: list[ActionOutput] | None = None


class Comparator(dspy.Signature):
    """After executing the given actions on user inputs using the given instruction, some inputs have yielded good, results, while others have not. I'll provide you the inputs along with their, corresponding evaluation metrics:

Task:
(1) Firstly, identify and contrast the patterns of inputs that have achieved good results with those that have not.
(2) Then, review the computational logic for any inconsistencies in the previous actions.
(3) Lastly, specify the modification in tools used that can lead to improved performance on the negative inputs."""

    instruction: str = dspy.InputField(
        prefix="Instruction: ",
        desc="Instruction for the actor to execute the task",
    )
    actions: list[str] = dspy.InputField(
        prefix="Actions: ",
        desc="Actions actor can take to complete the task",
    )
    pos_input_with_metrics: list[EvalResult] = dspy.InputField(
        prefix="Positive Inputs: ",
        desc="Positive inputs along with their score on a evaluation metric and actions taken",
    )
    neg_input_with_metrics: list[EvalResult] = dspy.InputField(
        prefix="Negative Inputs: ",
        desc="Negative inputs along with their score on a evaluation metric and actions taken",
    )
    feedback: str = dspy.OutputField(
        prefix="Feedback: ",
        desc="Feedback for the actor to improve the performance of negative inputs",
    )


class FeedbackBasedInstruction(dspy.Signature):
    """There is a task that needs to be completed for which one can use multiple tools to achieve the desired outcome. A group's performance was evaluated on a dataset of inputs, the inputs that did well are positive inputs, and the inputs that did not do well are negative inputs.

You received feedback on how they can better use the tools to improve your performance on the negative inputs. You have been provided with the previous instruction, that they followed to use tools to complete the task, and the feedback on your performance.

Your task is to incorporate the feedback and generate a detailed instruction for the group to follow to improve their performance on the task.

Make sure that the new instruction talks about how to use the tools effectively and should be no more than 3 paragraphs long. The previous instruction contains general guidelines that you must retain in the new instruction."""

    previous_instruction: str = dspy.InputField(
        prefix="Previous Instruction: ",
        desc="Previous instruction for the actor to execute the task",
    )
    feedback: str = dspy.InputField(
        prefix="Feedback: ",
        desc="Feedback for the actor to improve the performance of negative inputs",
    )
    new_instruction: str = dspy.OutputField(
        prefix="New Instruction: ",
        desc="New instruction for the actor to execute the task",
    )


class AvatarOptimizer(Teleprompter):
    def __init__(
        self,
        metric: Callable,
        max_iters: int = 10,
        lower_bound: int = 0,
        upper_bound: int = 1,
        max_positive_inputs: int | None = None,
        max_negative_inputs: int | None = None,
        optimize_for: str = "max",
    ):
        assert metric is not None, "`metric` argument cannot be None. Please provide a metric function."
        self.metric = metric
        self.optimize_for = optimize_for

        self.max_iters = max_iters

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.max_positive_inputs = max_positive_inputs or DEFAULT_MAX_EXAMPLES
        self.max_negative_inputs = max_negative_inputs or DEFAULT_MAX_EXAMPLES

        self.comparator = dspy.TypedPredictor(Comparator)
        self.feedback_instruction = dspy.Predict(FeedbackBasedInstruction)

    def process_example(self, actor, example, return_outputs):
        actor = deepcopy(actor)

        try:
            prediction = actor(**example.inputs().toDict())
            score = self.metric(example, prediction)

            if return_outputs:
                return example, prediction, score
            else:
                return score

        except Exception as e:
            print(e)

            if return_outputs:
                return example, None, 0
            else:
                return 0


    def thread_safe_evaluator(self, devset, actor, return_outputs=False, num_threads=None):
        total_score = 0
        total_examples = len(devset)
        results = []
        num_threads = num_threads or dspy.settings.num_threads

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(self.process_example, actor, example, return_outputs) for example in devset]

            for future in tqdm(futures, total=total_examples, desc="Processing examples"):
                result = future.result()
                if return_outputs:
                    example, prediction, score = result
                    total_score += score
                    results.append((example, prediction, score))
                else:
                    total_score += result

        avg_metric = total_score / total_examples

        if return_outputs:
            return avg_metric, results
        else:
            return avg_metric


    def _get_pos_neg_results(
        self,
        actor: dspy.Module,
        trainset: list[dspy.Example]
    ) -> tuple[float, list[EvalResult], list[EvalResult]]:
        pos_inputs = []
        neg_inputs = []

        avg_score, results = self.thread_safe_evaluator(trainset, actor, return_outputs=True)
        print(f"Average Score: {avg_score}")

        for example, prediction, score in results:
            if score >= self.upper_bound:
                pos_inputs.append(
                    EvalResult(
                        example=example.inputs().toDict(),
                        score=score,
                        actions=prediction.actions if prediction else None
                    )
                )
            elif score <= self.lower_bound:
                neg_inputs.append(
                    EvalResult(
                        example=example.inputs().toDict(),
                        score=score,
                        actions=prediction.actions if prediction else None
                    )
                )

        if len(pos_inputs) == 0:
            raise ValueError("No positive examples found, try lowering the upper_bound or providing more training data")
        if len(neg_inputs) == 0:
            raise ValueError("No negative examples found, try raising the lower_bound or providing more training data")

        return (avg_score, pos_inputs, neg_inputs)


    def compile(self, student, *, trainset):
        best_actor = deepcopy(student)
        best_score = -999 if self.optimize_for == "max" else 999

        for i in range(self.max_iters):
            print(20*"=")
            print(f"Iteration {i+1}/{self.max_iters}")

            score, pos_inputs, neg_inputs = self._get_pos_neg_results(best_actor, trainset)
            print(f"Positive examples: {len(pos_inputs)}")
            print(f"Negative examples: {len(neg_inputs)}")
            print(f"Sampling {self.max_positive_inputs} positive examples and {self.max_negative_inputs} negative examples")

            if self.max_positive_inputs and len(pos_inputs) > self.max_positive_inputs:
                pos_inputs = sample(pos_inputs, self.max_positive_inputs)

            if self.max_negative_inputs and len(neg_inputs) > self.max_negative_inputs:
                neg_inputs = sample(neg_inputs, self.max_negative_inputs)

            feedback = self.comparator(
                instruction=best_actor.actor.signature.instructions,
                actions=[str(tool) for tool in best_actor.tools],
                pos_input_with_metrics=pos_inputs,
                neg_input_with_metrics=neg_inputs
            ).feedback

            new_instruction = self.feedback_instruction(
                previous_instruction=best_actor.actor.signature.instructions,
                feedback=feedback
            ).new_instruction

            print(f"Generated new instruction: {new_instruction}")

            if (self.optimize_for == "max" and best_score < score) or (self.optimize_for == "min" and best_score > score):
                best_actor.actor.signature = best_actor.actor.signature.with_instructions(new_instruction)
                best_actor.actor_clone = deepcopy(best_actor.actor)
                best_score = score

        print(f"Best Actor: {best_actor}")

        return best_actor
```

## File: `teleprompt/bettertogether.py`
```
import logging
import random
from typing import Callable

import dspy
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.teleprompt.bootstrap_finetune import (
    BootstrapFinetune,
    all_predictors_have_lms,
    kill_lms,
    launch_lms,
    prepare_student,
)
from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch
from dspy.teleprompt.teleprompt import Teleprompter

logger = logging.getLogger(__name__)


class BetterTogether(Teleprompter):

    STRAT_SEP = " -> "

    def __init__(self,
        metric: Callable,
        prompt_optimizer: Teleprompter | None = None,
        weight_optimizer: Teleprompter | None = None,
        seed: int | None = None,
      ):
        if not dspy.settings.experimental:
            raise ValueError("This is an experimental optimizer. Set `dspy.settings.experimental` to `True` to use it.")

        # TODO: Note that the BetterTogether optimizer is meaningful when
        # BootstrapFinetune uses a metric to filter the training data before
        # fine-tuning. However, one can also choose to run this optimizer with
        # a BoostrapFinetune without a metric, say, if there aren't labels
        # available for the training data. Should this be noted somewhere?
        # TODO: We should re-consider if the metric should be required.
        self.prompt_optimizer = prompt_optimizer if prompt_optimizer else BootstrapFewShotWithRandomSearch(metric=metric)
        self.weight_optimizer = weight_optimizer if weight_optimizer else BootstrapFinetune(metric=metric)

        is_supported_prompt = isinstance(self.prompt_optimizer, BootstrapFewShotWithRandomSearch)
        is_supported_weight = isinstance(self.weight_optimizer, BootstrapFinetune)
        if not is_supported_prompt or not is_supported_weight:
            raise ValueError(
                "The BetterTogether optimizer only supports the following optimizers for now: BootstrapFinetune, "
                "BootstrapFewShotWithRandomSearch."
            )

        self.rng = random.Random(seed)

    def compile(
        self,
        student: Module,
        trainset: list[Example],
        strategy: str = "p -> w -> p",
        valset_ratio = 0.1,
    ) -> Module:
        # TODO: We could record acc on a different valset to pick the best
        # strategy within the provided strategy
        logger.info("Validating the strategy")
        parsed_strategy = strategy.lower().split(self.STRAT_SEP)

        if not all(s in ["p", "w"] for s in parsed_strategy):
            raise ValueError(
                f"The strategy should be a sequence of 'p' and 'w' separated by '{self.STRAT_SEP}', but "
                f"found: {strategy}"
            )

        logger.info("Preparing the student program...")
        # TODO: Prepare student returns student.reset_copy(), which is what gets
        # optimized. We should make this clear in the doc comments.
        student = prepare_student(student)
        all_predictors_have_lms(student)

        # Make a shallow copy of the trainset, so that we don't change the order
        # of the examples in the original trainset
        trainset = trainset[:]
        logger.info("Compiling the student program...")
        student = self._run_strategies(parsed_strategy, student, trainset, valset_ratio)

        logger.info("BetterTogether has finished compiling the student program")
        return student

    def _run_strategies(self, parsed_strategy, student, trainset, valset_ratio) -> Module:
        # Keep track of all the partial strategies/programs in parsed_strategy
        # "" corresponds to the initial student program
        candidate_programs = []
        candidate_programs.append(("", student))
        launched_flag = False

        for ind, step_code in enumerate(parsed_strategy):
            current_strategy = self.STRAT_SEP.join(parsed_strategy[:ind + 1])
            logger.info(
                f"\n########## Step {ind + 1} of {len(parsed_strategy)} - Strategy "
                f"'{current_strategy}' ##########"
            )

            logger.info("Shuffling the trainset...")
            self.rng.shuffle(trainset)
            if not launched_flag:
                launch_lms(student)
                launched_flag = True

            # TODO: Should we reset or just deepcopy? How does resetting affect
            # the predictor LMs?
            student = student.deepcopy()
            student._compiled = False
            if step_code == "p":
                student = self._compile_prompt_optimizer(student, trainset, valset_ratio)
            elif step_code == "w":
                student = self._compile_weight_optimizer(student, trainset)
                launched_flag = False

            # Record the program corresponding to the current strategy
            candidate_programs.append((current_strategy, student))

        if launched_flag:
            kill_lms(student)

        student.candidate_programs = candidate_programs
        return student

    def _compile_prompt_optimizer(self, student, trainset, valset_ratio) -> Module:
        logger.info("Preparing for prompt optimization...")

        # Sampling a validation set from the trainset for the prompt optimizer
        # We drop the hints for prompt optimization
        trainset = [x.with_inputs(*list(set(x.inputs().keys()) - {"hint"})) for x in trainset]
        num_val = int(valset_ratio * len(trainset))
        prompt_valset = trainset[:num_val]
        prompt_trainset = trainset[num_val:]

        # TODO: To make this optimizer general, we need to ensure that all the
        # prompt optimizers are accepting a valset or encode a way to check if
        # a valset should be passed to an optimizer's compile method.
        # TODO: We should ensure that the prompt optimizers in DSPy respect the
        # predictor.lm attributes. In particular,
        # BootstrapFewShotWithRandomSearch seems to be resetting these. We are
        # manually re-setting the LMs here to circumvent this issue, but we
        # should consider addressing it in BFRS.
        logger.info("Compiling the prompt optimizer...")
        pred_lms = [pred.lm for pred in student.predictors()]
        student = self.prompt_optimizer.compile(student, trainset=prompt_trainset, valset=prompt_valset)
        for pred, lm in zip(student.predictors(), pred_lms, strict=False):
            pred.lm = lm

        return student

    def _compile_weight_optimizer(self, student, trainset) -> Module:
        logger.info("Preparing for weight optimization...")

        # Saving the LMs before compiling the weight optimizer
        original_lms = [pred.lm for pred in student.predictors()]

        # TODO: To make this optimizer general, we need to ensure that all the
        # prompt optimizers are accepting a valset or encode a way to check if
        # a valset should be passed to an optimizer's compile.
        logger.info("Compiling the weight optimizer...")
        student = self.weight_optimizer.compile(student, trainset=trainset)

        # Updating the train kwargs for the new LMs. This is needed because the
        # train_kwargs of the optimizer is configured for the original LMs.
        new_lms = [pred.lm for pred in student.predictors()]
        for original_lm, new_lm in zip(original_lms, new_lms, strict=False):
            original_params = self.weight_optimizer.train_kwargs[original_lm]
            self.weight_optimizer.train_kwargs[new_lm] = original_params

        return student
```

## File: `teleprompt/bootstrap_finetune.py`
```
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

import dspy
from dspy.adapters.base import Adapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.clients.lm import LM
from dspy.clients.utils_finetune import infer_data_format
from dspy.dsp.utils.settings import settings
from dspy.evaluate.evaluate import Evaluate
from dspy.predict.predict import Predict
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)


class FinetuneTeleprompter(Teleprompter):
    def __init__(
        self,
        train_kwargs: dict[str, Any] | dict[LM, dict[str, Any]] | None = None,
    ):
        self.train_kwargs: dict[LM, Any] = self.convert_to_lm_dict(train_kwargs or {})

    @staticmethod
    def convert_to_lm_dict(arg) -> dict[LM, Any]:
        non_empty_dict = arg and isinstance(arg, dict)
        if non_empty_dict and all(isinstance(k, LM) for k in arg.keys()):
            return arg
        # Default to using the same value for all LMs
        return defaultdict(lambda: arg)


class BootstrapFinetune(FinetuneTeleprompter):
    def __init__(
        self,
        metric: Callable | None = None,
        multitask: bool = True,
        train_kwargs: dict[str, Any] | dict[LM, dict[str, Any]] | None = None,
        adapter: Adapter | dict[LM, Adapter] | None = None,
        exclude_demos: bool = False,
        num_threads: int | None = None,
    ):
        # TODO(feature): Inputs train_kwargs (a dict with string keys) and
        # adapter (Adapter) can depend on the LM they are used with. We are
        # takingthese as parameters for the time being. However, they can be
        # attached to LMs themselves -- an LM could know which adapter it should
        # be used with along with the train_kwargs. This will lead the only
        # required argument for LM.finetune() to be the train dataset.

        super().__init__(train_kwargs=train_kwargs)
        self.metric = metric
        self.multitask = multitask
        self.adapter: dict[LM, Adapter] = self.convert_to_lm_dict(adapter)
        self.exclude_demos = exclude_demos
        self.num_threads = num_threads

    def compile(
        self, student: Module, trainset: list[Example], teacher: Module | list[Module] | None = None
    ) -> Module:
        # TODO: Print statements can be converted to logger.info if we ensure
        # that the default DSPy logger logs info level messages in notebook
        # environments.
        logger.info("Preparing the student and teacher programs...")
        all_predictors_have_lms(student)

        logger.info("Bootstrapping data...")
        trace_data = []

        teachers = teacher if isinstance(teacher, list) else [teacher]
        teachers = [prepare_teacher(student, t) for t in teachers]
        num_threads = self.num_threads or dspy.settings.num_threads
        for t in teachers:
            trace_data += bootstrap_trace_data(program=t, dataset=trainset, metric=self.metric, num_threads=num_threads)

        logger.info("Preparing the train data...")
        key_to_data = {}
        for pred_ind, pred in enumerate(student.predictors()):
            data_pred_ind = None if self.multitask else pred_ind
            training_key = (pred.lm, data_pred_ind)
            if training_key not in key_to_data:
                train_data, data_format = self._prepare_finetune_data(
                    trace_data=trace_data, lm=pred.lm, pred_ind=data_pred_ind
                )
                logger.info(f"Using {len(train_data)} data points for fine-tuning the model: {pred.lm.model}")
                finetune_kwargs = {
                    "lm": pred.lm,
                    "train_data": train_data,
                    "train_data_format": data_format,
                    "train_kwargs": self.train_kwargs[pred.lm],
                }
                key_to_data[training_key] = finetune_kwargs

        logger.info("Starting LM fine-tuning...")
        # TODO(feature): We could run batches of fine-tuning jobs in sequence
        # to avoid exceeding the number of threads.
        if len(key_to_data) > num_threads:
            raise ValueError(
                "BootstrapFinetune requires `num_threads` to be bigger than or equal to the number of fine-tuning "
                f"jobs. There are {len(key_to_data)} fine-tuning jobs to start, but the number of threads is: "
                f"{num_threads}! If the `multitask` flag is set to False, the number of fine-tuning jobs will "
                "be equal to the number of predictors in the student program. If the `multitask` flag is set to True, "
                "the number of fine-tuning jobs will be equal to: 1 if there is only a context LM, or the number of "
                "unique LMs attached to the predictors in the student program. In any case, the number of fine-tuning "
                "jobs will be less than or equal to the number of predictors."
            )
        logger.info(f"{len(key_to_data)} fine-tuning job(s) to start")
        key_to_lm = self.finetune_lms(key_to_data)

        logger.info("Updating the student program with the fine-tuned LMs...")
        for pred_ind, pred in enumerate(student.predictors()):
            data_pred_ind = None if self.multitask else pred_ind
            training_key = (pred.lm, data_pred_ind)
            finetuned_lm = key_to_lm[training_key]
            if isinstance(finetuned_lm, Exception):
                raise RuntimeError(f"Finetuned LM for predictor {pred_ind} failed.") from finetuned_lm
            pred.lm = finetuned_lm
            # TODO: What should the correct behavior be here? Should
            # BootstrapFinetune modify the prompt demos according to the
            # train data?
            pred.demos = [] if self.exclude_demos else pred.demos

        logger.info("BootstrapFinetune has finished compiling the student program")
        student._compiled = True
        return student

    @staticmethod
    def finetune_lms(finetune_dict) -> dict[Any, LM]:
        num_jobs = len(finetune_dict)
        logger.info(f"Starting {num_jobs} fine-tuning job(s)...")
        # TODO(nit) Pass an identifier to the job so that we can tell the logs
        # coming from different fine-tune threads.

        key_to_job = {}
        for key, finetune_kwargs in finetune_dict.items():
            lm: LM = finetune_kwargs.pop("lm")
            # TODO: The following line is a hack. We should re-think how to free
            # up resources for fine-tuning. This might mean introducing a new
            # provider method (e.g. prepare_for_finetune) that can be called
            # before fine-tuning is started.
            logger.info(
                "Calling lm.kill() on the LM to be fine-tuned to free up resources. This won't have any effect if the "
                "LM is not running."
            )
            lm.kill()
            key_to_job[key] = lm.finetune(**finetune_kwargs)

        key_to_lm = {}
        for ind, (key, job) in enumerate(key_to_job.items()):
            result = job.result()
            if isinstance(result, Exception):
                raise result
            key_to_lm[key] = result
            job.thread.join()
            logger.info(f"Job {ind + 1}/{num_jobs} is done")

        return key_to_lm

    def _prepare_finetune_data(self, trace_data: list[dict[str, Any]], lm: LM, pred_ind: int | None = None):
        # TODO(nit) Log dataset details/size; make logs nicer
        if self.metric:
            logger.info(f"Collected data for {len(trace_data)} examples")
            trace_data = [d for d in trace_data if d["score"]]
            logger.info(f"After filtering with the metric, {len(trace_data)} examples remain")

        data = []
        adapter = self.adapter[lm] or settings.adapter or ChatAdapter()
        data_format = infer_data_format(adapter)
        for item in trace_data:
            for pred_ind, _ in enumerate(item["trace"]):
                include_data = pred_ind is None or pred_ind == pred_ind
                if include_data:
                    call_data = build_call_data_from_trace(
                        trace=item["trace"], pred_ind=pred_ind, adapter=adapter, exclude_demos=self.exclude_demos
                    )
                    data.append(call_data)

        import random

        random.Random(0).shuffle(data)

        return data, data_format


# Note: Shared below are useful functions for preparing student/teacher programs
# Similar methods are implemented separately and used by other DSPy
# teleprompters. These can be moved to shared locations.
def build_call_data_from_trace(
    trace: list[dict],
    pred_ind: int,
    adapter: Adapter,
    exclude_demos: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    # Find data that's relevant to the predictor
    pred, inputs, outputs = trace[pred_ind]  # assuming that the order is kept

    demos = [] if exclude_demos else pred.demos
    call_data = adapter.format_finetune_data(
        signature=pred.signature,
        demos=demos,
        inputs=inputs,
        outputs=outputs,
    )
    return call_data


@dataclass
class FailedPrediction:
    completion_text: str
    format_reward: float | None = None


def bootstrap_trace_data(
    program: Module,
    dataset: list[Example],
    metric: Callable | None = None,
    num_threads: int | None = None,
    raise_on_error=True,
    capture_failed_parses=False,
    failure_score: float = 0,
    format_failure_score: float = -1,
    log_format_failures: bool = False,
) -> list[dict[str, Any]]:
    # Return a list of dicts with the following keys: example_ind, example, prediction, trace, and score
    # (if metric != None)
    evaluator = Evaluate(
        devset=dataset,
        num_threads=num_threads,
        display_progress=True,
        provide_traceback=False,  # TODO(check with team)
        max_errors=len(dataset) * 10,  # TODO(check with team)
        failure_score=failure_score,
    )

    def wrapped_metric(example, prediction, trace=None):
        prediction, _ = prediction
        if isinstance(prediction, FailedPrediction):
            return prediction.format_reward or format_failure_score
        return metric(example, prediction, trace) if metric else True

    def wrapped_program(**kwargs):
        with dspy.context(trace=[]):
            try:
                return program(**kwargs), dspy.settings.trace.copy()
            except AdapterParseError as e:
                completion_str = e.lm_response
                parsed_result = e.parsed_result
                failed_signature = e.signature
                failed_inputs = kwargs

                present = list(parsed_result.keys()) if parsed_result else None
                expected = list(failed_signature.output_fields.keys())

                found_pred = None
                for pred in program.predictors():
                    if pred.signature == failed_signature:
                        found_pred = pred
                        break
                if found_pred is None:
                    raise ValueError(f"Failed to find the predictor for the failed signature: {failed_signature}")

                trace = dspy.settings.trace.copy()
                # Trace is Tuple[signature, inputs, prediction outputs]
                if present:
                    failed_pred = FailedPrediction(
                        completion_text=completion_str,
                        format_reward=format_failure_score
                        + (failure_score - format_failure_score) * (present / expected),
                    )
                else:
                    failed_pred = FailedPrediction(completion_text=completion_str, format_reward=format_failure_score)

                trace.append(
                    (
                        found_pred,
                        failed_inputs,
                        failed_pred,
                    )
                )

                if log_format_failures:
                    logging.warning(
                        "Failed to parse output for example. This is likely due to the LLM response not following the adapter's formatting."
                    )

                return failed_pred, trace

    results = evaluator(wrapped_program, metric=wrapped_metric).results

    data = []
    for example_ind, (example, prediction, score) in enumerate(results):
        try:
            prediction, trace = prediction
        except ValueError as ve:
            # TODO(GRPO Team): Often during GRPO bootstrapping, the LLM response does not follow dspy formatting. This leads to a value error.
            # To reproduce this issue, try Qwen/Qwen2.5-Coder-0.5B-Instruct with MATH dataset
            # Proposal(Lakshya): We should capture the incorrectly-formatted LLM response, and store it in the trace, and pass it to in the GRPO group
            # with a high-negative user-configurable score.
            logger.warning(
                "Failed to unpack prediction and trace. This is likely due to the LLM response not following dspy formatting."
            )
            if raise_on_error:
                raise ve
            else:
                continue
        data_dict = {"example": example, "prediction": prediction, "trace": trace, "example_ind": example_ind}
        if metric:
            data_dict["score"] = score
        data.append(data_dict)

    return data


# # TODO(PR) check with team
# def bootstrap_trace_data_one_example(
#     example: Example,
#     program: Program,
#     metric: Optional[Callable] = None
# ) -> dict[str, Any]:
#     # Return a dict with the following keys:
#     #     example, prediction, trace, and score (if metric != None)
#     with dspy.context(trace=[]):
#         prediction = program(**example.inputs())
#         trace = dspy.settings.trace
#         score = metric(example, prediction, trace) if metric else None

#     data_dict = dict(
#         example=example,
#         prediction=prediction,
#         trace=trace,
#     )
#     if metric:
#         data_dict["score"] = score

#     return data_dict


# Note: Shared below are useful functions for preparing student/teacher programs
# Similar methods are implemented separately and used by other DSPy
# teleprompters. These can be moved to shared locations.
def all_predictors_have_lms(program: Module) -> bool:
    """Return True if all predictors in the program have an LM set."""
    return all(pred.lm for pred in program.predictors())


def copy_program_with_lms(program: Module) -> Module:
    pred_lms = [pred.lm for pred in program.predictors()]
    program = program.deepcopy()
    for ind, pred in enumerate(program.predictors()):
        pred.lm = pred_lms[ind]
    return program


def prepare_student(student: Module) -> Module:
    if getattr(student, "_compiled", False):
        raise ValueError("The student program should not be compiled.")

    # TODO: Should we use reset_copy here? How would it affect the student
    # program's predictor LMs, if they are set?

    # TODO: Should there be a deepcopy here?
    # student = student.deepcopy()
    return student


def prepare_teacher(student: Module, teacher: Module | None = None) -> Module:
    if teacher is None:
        return student

    # Ensuring that the student and teacher are are structurally equivalent
    assert_structural_equivalency(student, teacher)

    # Ensuring that the student and teacher programs do not share predictors
    assert_no_shared_predictor(student, teacher)

    return teacher


def assert_structural_equivalency(program1: object, program2: object):
    assert isinstance(program1, Module)
    assert isinstance(program2, Module)

    num1 = len(program1.predictors())
    num2 = len(program2.predictors())
    err = f"Structurally equivalent programs must have the the number of predictors. The number of predictors for the two modules do not match: {num1} != {num2}"
    assert num1 == num2, err

    pzip = zip(program1.named_predictors(), program2.named_predictors(), strict=False)
    for ind, ((name1, pred1), (name2, pred2)) in enumerate(pzip):
        err = f"Program predictor names must match at  corresponding indices for structural equivalency. The predictor names for the programs do not match at index {ind}: '{name1}' != '{name2}'"
        assert name1 == name2, err
        assert isinstance(pred1, Predict)
        assert isinstance(pred2, Predict)


def assert_no_shared_predictor(program1: Module, program2: Module):
    id_to_name1 = {id(p): n for n, p in program1.named_predictors()}
    id_to_name2 = {id(p): n for n, p in program2.named_predictors()}
    shared_ids = set(id_to_name1.keys()) & set(id_to_name2.keys())

    pred_names = ", ".join(id_to_name1[id] for id in shared_ids)
    err = f"The programs share the following predictor(s) with each other: {pred_names}"
    assert not shared_ids, err


def get_unique_lms(program: Module) -> list[LM]:
    lms = [pred.lm for pred in program.predictors()]
    return list(set(lms))


def launch_lms(program: Module):
    lms = get_unique_lms(program)
    for lm in lms:
        lm.launch()


def kill_lms(program: Module):
    lms = get_unique_lms(program)
    for lm in lms:
        lm.kill()
```

## File: `teleprompt/bootstrap.py`
```
import logging
import random
import threading

import tqdm

import dspy
from dspy.teleprompt.teleprompt import Teleprompter

from .vanilla import LabeledFewShot

# TODO: metrics should return an object with __bool__ basically, but fine if they're more complex.
# They can also be sortable.

# TODO: Switch here from dspy.dsp.Example to dspy.Example. Right now, it's okay because it's internal only (predictors).
# NOTE: Notice the places where we don't shuffle examples. I do like that this one doesn't shuffle.
# Other ones that consider options may want to use both unshuffled and then shuffle a few times, when
# considering candidates.

# TODO: the max_rounds via branch_idx to get past the cache, not just temperature.
# In principle, we can also sample multiple outputs from the final generation step
# (or even each step, in case the validation function just wants *one* thing that works, but nah)
# and try them all. Having a pretty solid guess on the "final step" of each example isn't hard by the second round,
# in the sense that we have the trace from the first round. (Yes it may change but that's an edge case that
# won't hurt our "best effort" guarantees.)

# TODO: When this bootstraps for another teleprompter like finetune, we want all demos we gather.
# But when it's for direct use we may want to sample ONE demo per predictor--example pair.
# This is important for "multi-use" modules.

# TODO: Add baselines=[...]

logger = logging.getLogger(__name__)


class BootstrapFewShot(Teleprompter):
    def __init__(
        self,
        metric=None,
        metric_threshold=None,
        teacher_settings: dict | None = None,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_rounds=1,
        max_errors=None,
    ):
        """A Teleprompter class that composes a set of demos/examples to go into a predictor's prompt.
        These demos come from a combination of labeled examples in the training set, and bootstrapped demos.

        Args:
            metric (Callable): A function that compares an expected value and predicted value,
                outputting the result of that comparison.
            metric_threshold (float, optional): If the metric yields a numerical value, then check it
                against this threshold when deciding whether or not to accept a bootstrap example.
                Defaults to None.
            teacher_settings (dict, optional): Settings for the `teacher` model.
                Defaults to None.
            max_bootstrapped_demos (int): Maximum number of bootstrapped demonstrations to include.
                Defaults to 4.
            max_labeled_demos (int): Maximum number of labeled demonstrations to include.
                Defaults to 16.
            max_rounds (int): Number of iterations to attempt generating the required bootstrap
                examples. If unsuccessful after `max_rounds`, the program ends. Defaults to 1.
            max_errors (Optional[int]): Maximum number of errors until program ends.
                If ``None``, inherits from ``dspy.settings.max_errors``.
        """
        self.metric = metric
        self.metric_threshold = metric_threshold
        self.teacher_settings = {} if teacher_settings is None else teacher_settings

        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_rounds = max_rounds
        self.max_errors = max_errors
        self.error_count = 0
        self.error_lock = threading.Lock()

    def compile(self, student, *, teacher=None, trainset):
        self.trainset = trainset

        self._prepare_student_and_teacher(student, teacher)
        self._prepare_predictor_mappings()
        self._bootstrap()

        self.student = self._train()
        self.student._compiled = True

        # set assert_failures and suggest_failures as attributes of student w/ value 0
        self.student._assert_failures = 0
        self.student._suggest_failures = 0

        return self.student

    def _prepare_student_and_teacher(self, student, teacher):
        self.student = student.reset_copy()

        # NOTE: behavior change on Oct 28, 2024. Deep copy instead of reset copy for the student-as-teacher.
        self.teacher = teacher.deepcopy() if teacher is not None else student.deepcopy()

        assert getattr(self.student, "_compiled", False) is False, "Student must be uncompiled."

        if self.max_labeled_demos and getattr(self.teacher, "_compiled", False) is False:
            teleprompter = LabeledFewShot(k=self.max_labeled_demos)
            self.teacher = teleprompter.compile(self.teacher.reset_copy(), trainset=self.trainset)

    def _prepare_predictor_mappings(self):
        name2predictor, predictor2name = {}, {}
        student, teacher = self.student, self.teacher

        assert len(student.predictors()) == len(
            teacher.predictors(),
        ), "Student and teacher must have the same number of predictors."

        for (name1, predictor1), (name2, predictor2) in zip(student.named_predictors(), teacher.named_predictors(), strict=False):
            assert name1 == name2, "Student and teacher must have the same program structure."
            if hasattr(predictor1.signature, "equals"):
                assert predictor1.signature.equals(
                    predictor2.signature,
                ), (
                    f"Student and teacher must have the same signatures. "
                    f"{type(predictor1.signature)} != {type(predictor2.signature)}"
                )
            else:
                # fallback in case if .equals is not implemented (e.g. dsp.Prompt)
                assert predictor1.signature == predictor2.signature, (
                    f"Student and teacher must have the same signatures. "
                    f"{type(predictor1.signature)} != {type(predictor2.signature)}"
                )
            assert id(predictor1) != id(predictor2), "Student and teacher must be different objects."

            name2predictor[name1] = None  # dict(student=predictor1, teacher=predictor2)
            predictor2name[id(predictor1)] = name1

            # FIXME(shangyint): This is an ugly hack to bind traces of
            # retry.module to retry
            # if isinstance(predictor1, Retry):
            #     predictor2name[id(predictor1.module)] = name1

            predictor2name[id(predictor2)] = name2

        self.name2predictor = name2predictor
        self.predictor2name = predictor2name

    def _bootstrap(self, *, max_bootstraps=None):
        max_bootstraps = max_bootstraps or self.max_bootstrapped_demos
        bootstrap_attempts = 0

        bootstrapped = {}
        self.name2traces = {name: [] for name in self.name2predictor}

        for example_idx, example in enumerate(tqdm.tqdm(self.trainset)):
            if len(bootstrapped) >= max_bootstraps:
                break

            for round_idx in range(self.max_rounds):
                bootstrap_attempts += 1

                if self._bootstrap_one_example(example, round_idx):
                    bootstrapped[example_idx] = True
                    break

        print(
            f"Bootstrapped {len(bootstrapped)} full traces after {example_idx} examples "
            f"for up to {self.max_rounds} rounds, amounting to {bootstrap_attempts} attempts."
        )

        # Unbootstrapped training examples

        self.validation = [x for idx, x in enumerate(self.trainset) if idx not in bootstrapped]
        random.Random(0).shuffle(self.validation)

        self.validation = self.validation

        # NOTE: Can't yet use evaluate because we need to trace *per example*
        # evaluate = Evaluate(program=self.teacher, metric=self.metric, num_threads=12)
        # score = evaluate(self.metric, display_table=False, display_progress=True)

    def _bootstrap_one_example(self, example, round_idx=0):
        name2traces = {}
        teacher = self.teacher
        predictor_cache = {}

        try:
            with dspy.settings.context(trace=[], **self.teacher_settings):
                lm = dspy.settings.lm
                lm = lm.copy(temperature=0.7 + 0.001 * round_idx) if round_idx > 0 else lm
                new_settings = {"lm": lm} if round_idx > 0 else {}

                with dspy.settings.context(**new_settings):
                    for name, predictor in teacher.named_predictors():
                        predictor_cache[name] = predictor.demos
                        predictor.demos = [x for x in predictor.demos if x != example]

                    prediction = teacher(**example.inputs())
                    trace = dspy.settings.trace

                    for name, predictor in teacher.named_predictors():
                        predictor.demos = predictor_cache[name]

                if self.metric:
                    metric_val = self.metric(example, prediction, trace)
                    if self.metric_threshold:
                        success = metric_val >= self.metric_threshold
                    else:
                        success = metric_val
                else:
                    success = True
        except Exception as e:
            success = False
            with self.error_lock:
                self.error_count += 1
                current_error_count = self.error_count
            effective_max_errors = (
                self.max_errors
                if self.max_errors is not None
                else dspy.settings.max_errors
            )
            if current_error_count >= effective_max_errors:
                raise e
            logger.error(f"Failed to run or to evaluate example {example} with {self.metric} due to {e}.")

        if success:
            for step in trace:
                predictor, inputs, outputs = step
                demo = dspy.Example(augmented=True, **inputs, **outputs)

                try:
                    predictor_name = self.predictor2name[id(predictor)]
                except KeyError:
                    continue  # FIXME: !

                    # # TODO: Look closer into this. It's a bit tricky to reproduce.
                    # print(f"Failed to find predictor {predictor} in {self.predictor2name}.")
                    # print(
                    #     "Are you doing this in a notebook (Jupyter)? This might be caused by redefining values by rerunning cells.",
                    # )
                    # print("Try restarting the notebook, or open an issue.")
                    # raise KeyError(
                    #     f"Failed to find predictor {id(predictor)} {predictor} in {self.predictor2name}.",
                    # ) from e

                name2traces[predictor_name] = name2traces.get(predictor_name, [])
                name2traces[predictor_name].append(demo)

            # Update the traces
            for name, demos in name2traces.items():

                # If there are multiple traces for the same predictor in the sample example,
                # sample 50/50 from the first N-1 traces or the last trace.
                if len(demos) > 1:
                    from datasets.fingerprint import Hasher

                    rng = random.Random(Hasher.hash(tuple(demos)))
                    demos = [rng.choice(demos[:-1]) if rng.random() < 0.5 else demos[-1]]
                self.name2traces[name].extend(demos)

        return success

    def _train(self):
        rng = random.Random(0)
        raw_demos = self.validation

        for name, predictor in self.student.named_predictors():
            augmented_demos = self.name2traces[name][: self.max_bootstrapped_demos]

            sample_size = min(self.max_labeled_demos - len(augmented_demos), len(raw_demos))
            sample_size = max(0, sample_size)

            raw_demos = rng.sample(raw_demos, sample_size)
            predictor.demos = augmented_demos + raw_demos

        return self.student
```

## File: `teleprompt/copro_optimizer.py`
```
import logging
from collections import defaultdict

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.signatures import Signature
from dspy.teleprompt.teleprompt import Teleprompter

logger = logging.getLogger(__name__)

"""
USAGE SUGGESTIONS:

The following code can be used to compile a optimized signature teleprompter, and evaluate it on an end task:

teleprompter = COPRO(prompt_model=prompt_model, metric=metric, breadth=BREADTH, depth=DEPTH, init_temperature=INIT_TEMPERATURE)
kwargs = dict(num_threads=NUM_THREADS, display_progress=True, display_table=0)
compiled_prompt_opt = teleprompter.compile(program.deepcopy(), trainset=trainset[:DEV_NUM], eval_kwargs=kwargs)
eval_score = evaluate(compiled_prompt_opt, devset=evalset[:EVAL_NUM], **kwargs)

Note that this teleprompter takes in the following parameters:

* prompt_model: The model used for prompt generation. When unspecified, defaults to the model set in settings (ie. dspy.settings.configure(lm=task_model)).
* metric: The task metric used for optimization.
* breadth: The number of new prompts to generate at each iteration. Default=10.
* depth: The number of times we should ask our prompt model to generate new prompts, with the history of the past prompts as input. Default=3.
* init_temperature: The temperature used to generate new prompts. Higher roughly equals more creative. Default=1.4.
* track_stats: Tells the method whether or not to track statistics about the optimization process.
                If True, the method will track the following statistics:
                    * results_best: The min,max,avg,stddev of top 10 scores for each predictor at each depth.
                    * results_latest: The min,max,avg,stddev of newest prompt scores for each predictor at each depth.
                    * total_calls: The total number of calls to the task metric.
                These statistics will be returned as attributes of the best program.
"""


class BasicGenerateInstruction(Signature):
    """You are an instruction optimizer for large language models. I will give you a ``signature`` of fields (inputs and outputs) in English. Your task is to propose an instruction that will lead a good language model to perform the task well. Don't be afraid to be creative."""

    basic_instruction = dspy.InputField(desc="The initial instructions before optimization")
    proposed_instruction = dspy.OutputField(desc="The improved instructions for the language model")
    proposed_prefix_for_output_field = dspy.OutputField(
        desc="The string at the end of the prompt, which will help the model start solving the task",
    )


class GenerateInstructionGivenAttempts(dspy.Signature):
    """You are an instruction optimizer for large language models. I will give some task instructions I've tried, along with their corresponding validation scores. The instructions are arranged in increasing order based on their scores, where higher scores indicate better quality.

    Your task is to propose a new instruction that will lead a good language model to perform the task even better. Don't be afraid to be creative."""

    attempted_instructions = dspy.InputField()
    proposed_instruction = dspy.OutputField(desc="The improved instructions for the language model")
    proposed_prefix_for_output_field = dspy.OutputField(
        desc="The string at the end of the prompt, which will help the model start solving the task",
    )


class COPRO(Teleprompter):
    def __init__(
        self,
        prompt_model=None,
        metric=None,
        breadth=10,
        depth=3,
        init_temperature=1.4,
        track_stats=False,
        **_kwargs,
    ):
        if breadth <= 1:
            raise ValueError("Breadth must be greater than 1")
        self.metric = metric
        self.breadth = breadth
        self.depth = depth
        self.init_temperature = init_temperature
        self.prompt_model = prompt_model
        self.track_stats = track_stats

    def _check_candidates_equal(self, candidate1, candidate2):
        for p1, p2 in zip(candidate1["program"].predictors(), candidate2["program"].predictors(), strict=False):
            if self._get_signature(p1).instructions != self._get_signature(p2).instructions:
                return False
            *_, p1_last_field = self._get_signature(p1).fields.values()
            *_, p2_last_field = self._get_signature(p2).fields.values()
            if p1_last_field != p2_last_field:
                return False
        return True

    def _drop_duplicates(self, candidates):
        final_candidates = []
        last_batch = []
        last_batch_score = -1
        for c in candidates:
            repeat = False
            if c["score"] == last_batch_score:
                for c2 in last_batch:
                    if self._check_candidates_equal(c, c2):
                        repeat = True
                        break
                if not repeat:
                    last_batch.append(c)
            else:
                last_batch = [c]
                last_batch_score = c["score"]
            if not repeat:
                final_candidates.append(c)
        return final_candidates

    def _print_signature(self, predictor):
        signature = self._get_signature(predictor)

        logger.debug(f"i: {signature.instructions}")
        logger.debug(f"p: {list(signature.fields.values())[-1].json_schema_extra['prefix']}")

    def _get_signature(self, predictor):
        assert hasattr(predictor, "signature")
        return predictor.signature

    def _set_signature(self, predictor, updated_signature):
        assert hasattr(predictor, "signature")
        predictor.signature = updated_signature

    def compile(self, student, *, trainset, eval_kwargs):
        """
        optimizes `signature` of `student` program - note that it may be zero-shot or already pre-optimized (demos already chosen - `demos != []`)

        parameters:
        student: program to optimize and left modified.
        trainset: iterable of `Example`s
        eval_kwargs: optional, dict
           Additional keywords to go into `Evaluate` for the metric.

        Returns optimized version of `student`.
        """
        module = student.deepcopy()
        evaluate = Evaluate(devset=trainset, metric=self.metric, **eval_kwargs)
        total_calls = 0
        results_best = {
            id(p): {"depth": [], "max": [], "average": [], "min": [], "std": []} for p in module.predictors()
        }
        results_latest = {
            id(p): {"depth": [], "max": [], "average": [], "min": [], "std": []} for p in module.predictors()
        }

        if self.track_stats:
            import numpy as np

        candidates = {}
        evaluated_candidates = defaultdict(dict)

        # Seed the prompt optimizer zero shot with just the instruction, generate BREADTH new prompts
        for predictor in module.predictors():
            basic_instruction = None
            basic_prefix = None
            *_, last_key = self._get_signature(predictor).fields.keys()
            basic_instruction = self._get_signature(predictor).instructions
            basic_prefix = self._get_signature(predictor).fields[last_key].json_schema_extra["prefix"]
            if self.prompt_model:
                with dspy.settings.context(lm=self.prompt_model):
                    instruct = dspy.Predict(
                        BasicGenerateInstruction,
                        n=self.breadth - 1,
                        temperature=self.init_temperature,
                    )(basic_instruction=basic_instruction)
            else:
                instruct = dspy.Predict(
                    BasicGenerateInstruction,
                    n=self.breadth - 1,
                    temperature=self.init_temperature,
                )(basic_instruction=basic_instruction)
            # Add in our initial prompt as a candidate as well
            instruct.completions.proposed_instruction.append(basic_instruction)
            instruct.completions.proposed_prefix_for_output_field.append(basic_prefix)
            candidates[id(predictor)] = instruct.completions
            evaluated_candidates[id(predictor)] = {}

        if self.prompt_model:
            logger.debug(f"{self.prompt_model.inspect_history(n=1)}")

        latest_candidates = candidates
        all_candidates = candidates

        module_clone = module.deepcopy()

        # For each iteration in depth...
        for d in range(
            self.depth,
        ):  # TODO: fix this so that we eval the new batch of predictors with the new best following predictors
            logger.info(f"Iteration Depth: {d+1}/{self.depth}.")

            latest_scores = []

            # Go through our module's predictors
            for p_i, (p_old, p_new) in enumerate(zip(module.predictors(), module_clone.predictors(), strict=False)):
                candidates_ = latest_candidates[id(p_old)]  # Use the most recently generated candidates for evaluation
                if len(module.predictors()) > 1:
                    # Unless our program has multiple predictors, in which case we need to reevaluate all prompts with
                    # the new prompt(s) for the other predictor(s).
                    candidates_ = all_candidates[
                        id(p_old)
                    ]

                # For each candidate
                for c_i, c in enumerate(candidates_):
                    # Get the candidate instruction and prefix
                    instruction, prefix = (
                        c.proposed_instruction.strip('"').strip(),
                        c.proposed_prefix_for_output_field.strip('"').strip(),
                    )

                    # Set this new module with our instruction / prefix
                    *_, last_key = self._get_signature(p_new).fields.keys()
                    updated_signature = (
                        self._get_signature(p_new)
                        .with_instructions(instruction)
                        .with_updated_fields(last_key, prefix=prefix)
                    )
                    self._set_signature(p_new, updated_signature)

                    # Score the instruction / prefix
                    for i, predictor in enumerate(module_clone.predictors()):
                        logger.debug(f"Predictor {i+1}")
                        self._print_signature(predictor)
                    logger.info(
                        f"At Depth {d+1}/{self.depth}, Evaluating Prompt Candidate #{c_i+1}/{len(candidates_)} for "
                        f"Predictor {p_i+1} of {len(module.predictors())}.",
                    )
                    score = evaluate(module_clone, devset=trainset, **eval_kwargs).score
                    if self.prompt_model:
                        logger.debug(f"prompt_model.inspect_history(n=1) {self.prompt_model.inspect_history(n=1)}")
                    total_calls += 1

                    replace_entry = True
                    logger.debug(f"(instruction, prefix) {(instruction, prefix)}")
                    if (instruction, prefix) in evaluated_candidates[id(p_old)]:
                        if evaluated_candidates[id(p_old)][(instruction, prefix)]["score"] >= score:
                            replace_entry = False

                    if replace_entry:
                        # Add it to our evaluated candidates list
                        evaluated_candidates[id(p_old)][(instruction, prefix)] = {
                            "score": score,
                            "program": module_clone.deepcopy(),
                            "instruction": instruction,
                            "prefix": prefix,
                            "depth": d,
                        }

                    if len(candidates_) - self.breadth <= c_i:
                        latest_scores.append(score)

                if self.track_stats:
                    results_latest[id(p_old)]["depth"].append(d)
                    results_latest[id(p_old)]["max"].append(max(latest_scores))
                    results_latest[id(p_old)]["average"].append(sum(latest_scores) / len(latest_scores))
                    results_latest[id(p_old)]["min"].append(min(latest_scores))
                    results_latest[id(p_old)]["std"].append(np.std(latest_scores))

                # Now that we've evaluated the candidates, set this predictor to the best performing version
                # to ensure the next round of scores reflect the best possible version
                best_candidate = max(evaluated_candidates[id(p_old)].values(), key=lambda candidate: candidate["score"])
                *_, last_key = self._get_signature(p_old).fields.keys()
                updated_signature = (
                    self._get_signature(p_new)
                    .with_instructions(best_candidate["instruction"])
                    .with_updated_fields(last_key, prefix=best_candidate["prefix"])
                )
                self._set_signature(p_new, updated_signature)

                logger.debug(
                    f"Updating Predictor {id(p_old)} to:\ni: {best_candidate['instruction']}\n"
                    f"p: {best_candidate['prefix']}",
                )
                logger.debug("Full predictor with update: ")
                for i, predictor in enumerate(module_clone.predictors()):
                    logger.debug(f"Predictor {i}")
                    self._print_signature(predictor)

            if d == self.depth - 1:
                break

            new_candidates = {}
            for p_base in module.predictors():
                # Build Few-Shot Example of Optimized Prompts
                attempts = []
                shortest_len = self.breadth
                shortest_len = min(len(evaluated_candidates[id(p_base)]), shortest_len)
                best_predictors = list(evaluated_candidates[id(p_base)].values())

                # best_predictors = evaluated_candidates[id(p_base)].values()[:]
                best_predictors.sort(key=lambda x: x["score"], reverse=True)

                if self.track_stats:
                    scores = [x["score"] for x in best_predictors][:10]
                    results_best[id(p_base)]["depth"].append(d)
                    results_best[id(p_base)]["max"].append(max(scores))
                    results_best[id(p_base)]["average"].append(sum(scores) / len(scores))
                    results_best[id(p_base)]["min"].append(min(scores))
                    results_best[id(p_base)]["std"].append(np.std(scores))

                for i in range(shortest_len - 1, -1, -1):
                    # breakpoint()
                    attempts.append(f'Instruction #{shortest_len-i}: {best_predictors[i]["instruction"]}')
                    attempts.append(f'Prefix #{shortest_len-i}: {best_predictors[i]["prefix"]}')
                    attempts.append(f'Resulting Score #{shortest_len-i}: {best_predictors[i]["score"]}')

                # Generate next batch of potential prompts to optimize, with previous attempts as input
                if self.prompt_model:
                    with dspy.settings.context(lm=self.prompt_model):
                        instr = dspy.Predict(
                            GenerateInstructionGivenAttempts,
                            n=self.breadth,
                            temperature=self.init_temperature,
                        )(attempted_instructions=attempts)
                else:
                    instr = dspy.Predict(
                        GenerateInstructionGivenAttempts,
                        n=self.breadth,
                        temperature=self.init_temperature,
                    )(attempted_instructions=attempts)

                # Get candidates for each predictor
                new_candidates[id(p_base)] = instr.completions
                all_candidates[id(p_base)].proposed_instruction.extend(instr.completions.proposed_instruction)
                all_candidates[id(p_base)].proposed_prefix_for_output_field.extend(
                    instr.completions.proposed_prefix_for_output_field,
                )

            latest_candidates = new_candidates

        candidates = []
        for predictor in module.predictors():
            candidates.extend(list(evaluated_candidates[id(predictor)].values()))

            if self.track_stats:
                best_predictors = list(evaluated_candidates[id(predictor)].values())
                best_predictors.sort(key=lambda x: x["score"], reverse=True)

                scores = [x["score"] for x in best_predictors][:10]
                results_best[id(predictor)]["depth"].append(d)
                results_best[id(predictor)]["max"].append(max(scores))
                results_best[id(predictor)]["average"].append(sum(scores) / len(scores))
                results_best[id(predictor)]["min"].append(min(scores))
                results_best[id(predictor)]["std"].append(np.std(scores))

        candidates.sort(key=lambda x: x["score"], reverse=True)

        candidates = self._drop_duplicates(candidates)

        best_program = candidates[0]["program"]
        best_program.candidate_programs = candidates
        best_program.total_calls = total_calls
        if self.track_stats:
            best_program.results_best = results_best
            best_program.results_latest = results_latest

        return best_program
```

## File: `teleprompt/ensemble.py`
```
import random

from dspy.teleprompt.teleprompt import Teleprompter

"""
TODO: The EnsembledProgram should actually imitate the structure of the individual programs (IF they are all compatible). This allows compiling with an ensemble program as a (singular) teacher. Basically the top majority-compatible trace will end up being used, if dspy.majority is the reduce_fn.
"""


class Ensemble(Teleprompter):
    def __init__(self, *, reduce_fn=None, size=None, deterministic=False):
        """A common reduce_fn is dspy.majority."""

        assert deterministic is False, "TODO: Implement example hashing for deterministic ensemble."

        self.reduce_fn = reduce_fn
        self.size = size
        self.deterministic = deterministic

    def compile(self, programs):
        size = self.size
        reduce_fn = self.reduce_fn

        import dspy

        class EnsembledProgram(dspy.Module):
            def __init__(self):
                super().__init__()
                self.programs = programs

            def forward(self, *args, **kwargs):
                programs = random.sample(self.programs, size) if size else self.programs
                outputs = [prog(*args, **kwargs) for prog in programs]

                if reduce_fn:
                    return reduce_fn(outputs)

                return outputs

        return EnsembledProgram()
```

## File: `teleprompt/grpo.py`
```
import logging
import random
from collections import Counter
from typing import Any, Callable, Literal

from dspy.adapters.base import Adapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.clients.lm import LM
from dspy.clients.utils_finetune import GRPOGroup, TrainDataFormat
from dspy.dsp.utils.settings import settings
from dspy.evaluate.evaluate import Evaluate
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.teleprompt.bootstrap_finetune import (
    FailedPrediction,
    FinetuneTeleprompter,
    all_predictors_have_lms,
    assert_structural_equivalency,
    bootstrap_trace_data,
)

logger = logging.getLogger(__name__)


class GRPO(FinetuneTeleprompter):
    def __init__(
        self,
        metric: Callable | None = None,
        multitask: bool = True,
        train_kwargs: dict[str, Any] | dict[LM, dict[str, Any]] | None = None,
        adapter: Adapter | dict[LM, Adapter] | None = None,
        exclude_demos: bool = False,
        num_threads: int = 6,
        num_train_steps: int = 100,
        seed: int = 0,
        num_dspy_examples_per_grpo_step: int = 1,
        num_rollouts_per_grpo_step: int = 1,
        use_train_as_val: bool = False,
        num_steps_for_val: int = 5,
        report_train_scores: bool = False,
        failure_score: float = 0,
        format_failure_score: float = -1,
        variably_invoked_predictor_grouping_mode: Literal["truncate"] | Literal["fill"] | Literal["ragged"] = "truncate",
        variably_invoked_predictor_fill_strategy: Literal["randint"] | Literal["max"] | None = None,
    ):
        super().__init__(train_kwargs=train_kwargs)
        self.metric = metric
        self.multitask = multitask
        self.adapter: dict[LM, Adapter] = self.convert_to_lm_dict(adapter)
        self.exclude_demos = exclude_demos
        self.num_threads = num_threads
        self.num_train_steps = num_train_steps
        self.rng = random.Random(seed)
        self.num_dspy_examples_per_grpo_step = num_dspy_examples_per_grpo_step
        self.num_rollouts_per_grpo_step = num_rollouts_per_grpo_step
        self.use_train_as_val = use_train_as_val
        self.num_steps_for_val = num_steps_for_val
        self.report_train_scores = report_train_scores
        self.failure_score = failure_score
        self.format_failure_score = format_failure_score

        assert failure_score > format_failure_score, "failure_score must be greater than format_failure_score since the range [format_failure_score, failure_score] is used to provide dspy formatting rewards"

        if self.use_train_as_val:
            assert report_train_scores, "If use_train_as_val is True, report_train_scores must be True."

        assert exclude_demos, "exclude_demos==False is not supported yet. Please set it to True."
        assert multitask, "independent GRPO training jobs for each predictor in the student program is not supported yet. Please set multitask=True."

        # The backend will be called with a batch of (num_dspy_examples_per_grpo_step * num_rollouts_per_grpo_step * num_predictors) per training set if multitask is True
        # If multitask is False, the backend will be called with a batch of (num_dspy_examples_per_grpo_step * num_rollouts_per_grpo_step) per training job
        self.variably_invoked_predictor_grouping_mode = variably_invoked_predictor_grouping_mode
        if variably_invoked_predictor_grouping_mode == "fill":
            assert variably_invoked_predictor_fill_strategy is not None, "variably_invoked_predictor_fill_strategy must be set when variably_invoked_predictor_grouping_mode is 'fill'"
            assert variably_invoked_predictor_fill_strategy in ["randint", "max"], "variably_invoked_predictor_fill_strategy must be either 'randint' or 'max'"
        self.variably_invoked_predictor_fill_strategy = variably_invoked_predictor_fill_strategy

        self.shuffled_trainset_ids = []
        self.epoch = -1
        self.id_freqs = Counter()

    def validate_trace_data_and_log_issues(
        self,
        trace_data: list[list[list[dict[str, Any]]]],
        subsample_training_dataset: list[Example],
        num_teachers: int,
        num_samples_per_input: int,
        pred_signature_hash_to_ind: dict[int, int],
    ):
        # At this point, trace_data: list[example_idx -> list[teacher_idx -> [num_samples_per_input * Dict(example, prediction, trace, example_ind, score)]]]
        # Shape of trace is: [dspy_module_invocation_idx -> Tuple[Predictor, PredictorInputs, Prediction]]
        assert len(trace_data) == len(subsample_training_dataset), f"Trace data length {len(trace_data)} does not match the number of examples {len(subsample_training_dataset)}"
        assert len(trace_data[0]) == num_teachers, f"Trace data length {len(trace_data[0])} does not match the number of teachers {num_teachers}"
        # TODO(GRPO Team): Ideally, once the dspy format issue is fixed, this change should be reverted back to being a normal assert.
        if len(trace_data[0][0]) == 0:
            logger.warning(f"Trace data for example {0} and teacher {0} is empty. This is likely due to all examples in the training set input, resulting in the model generating output not following the dspy response format.")
        elif len(trace_data[0][0]) != num_samples_per_input:
            logger.warning(f"Trace data length {len(trace_data[0][0])} does not match the expected number of samples per input {num_samples_per_input}")
            assert "trace" in trace_data[0][0][0], "Trace data does not contain the 'trace' key"
            assert len(trace_data[0][0][0]["trace"]) > 0, "Trace data is empty"
            assert len(trace_data[0][0][0]["trace"][0]) == 3, f"Trace tuple length {len(trace_data[0][0][0]['trace'][0])} does not match the expected length 3"

        for example_data in trace_data:
            for teacher_data in example_data:
                for sample in teacher_data:
                    for t in sample["trace"]:
                        assert hash(t[0].signature) in pred_signature_hash_to_ind

    def report_validation_metrics(self, student, trainset, valset, logger, step_idx=-1):
        if step_idx == -1 or step_idx == self.num_train_steps - 1 or (step_idx + 1) % self.num_steps_for_val == 0:
            pass
        else:
            return

        if valset is not None:
            # Validation set provided by user
            assert not self.use_train_as_val, "If valset is provided, use_train_as_val must be False."
            assert isinstance(self.num_steps_for_val, int) and self.num_steps_for_val > 0, "num_steps_for_val must be a positive integer."
            if self.report_train_scores:
                if step_idx == -1:
                    logger.info("Using user provided validation set and reporting train scores for every validation step in addition.")
                valset_evaluator = Evaluate(
                    devset=valset + trainset,
                    num_threads=self.num_threads,
                    display_progress=True,
                    provide_traceback=False,  # TODO(check with team)
                    max_errors=len(valset)*10,  # TODO(check with team)
                    failure_score=self.failure_score
                )
                if step_idx == -1:
                    logger.info("Evaluating the student program on the train+validation set before training loop...")
                else:
                    logger.info(f"Evaluating the student program on the validation set after training step {step_idx + 1}/{self.num_train_steps}")
                valset_evaluation = valset_evaluator(student, metric=self.metric)
                trainset_scores = [r[-1] for r in valset_evaluation.results[len(valset):]]
                valset_scores = [r[-1] for r in valset_evaluation.results[:len(valset)]]
                trainset_agg = sum(trainset_scores) / len(trainset_scores)
                valset_agg = sum(valset_scores) / len(valset_scores)
                if step_idx == -1:
                    logger.info(f"Student program training set score before training loop: {trainset_agg}")
                    logger.info(f"Student program validation set score before training loop: {valset_agg}")
                else:
                    logger.info(f"Student program training set score after training step {step_idx + 1}/{self.num_train_steps}: {trainset_agg}")
                    logger.info(f"Student program validation set score after training step {step_idx + 1}/{self.num_train_steps}: {valset_agg}")
            else:
                if step_idx == -1:
                    logger.info("Using user provided validation set and not reporting train scores.")
                valset_evaluator = Evaluate(
                    devset=valset,
                    num_threads=self.num_threads,
                    display_progress=True,
                    provide_traceback=False,  # TODO(check with team)
                    max_errors=len(valset)*10,  # TODO(check with team)
                    failure_score=self.failure_score
                )
                if step_idx == -1:
                    logger.info("Evaluating the student program on the validation set before training loop...")
                else:
                    logger.info(f"Evaluating the student program on the validation set after training step {step_idx + 1}/{self.num_train_steps}")
                valset_evaluation = valset_evaluator(student, metric=self.metric)
                if step_idx == -1:
                    logger.info(f"Student program validation set score before training loop: {valset_evaluation.score}")
                else:
                    logger.info(f"Student program validation set score after training step {step_idx + 1}/{self.num_train_steps}: {valset_evaluation.score}")
        else:
            # No validation set provided by user
            if self.report_train_scores:
                assert self.use_train_as_val, "If report_train_scores is True, use_train_as_val must be True when valset is not provided explicitly."
                assert isinstance(self.num_steps_for_val, int) and self.num_steps_for_val > 0, "num_steps_for_val must be a positive integer."
                if step_idx == -1:
                    logger.info("Using trainset as validation set.")
                valset_evaluator = Evaluate(
                    devset=trainset,
                    num_threads=self.num_threads,
                    display_progress=True,
                    provide_traceback=False,  # TODO(check with team)
                    max_errors=len(trainset)*10,  # TODO(check with team)
                    failure_score=self.failure_score
                )
                if step_idx == -1:
                    logger.info("Evaluating the student program on the validation set before training loop...")
                else:
                    logger.info(f"Evaluating the student program on the validation set after training step {step_idx + 1}/{self.num_train_steps}")
                valset_evaluation = valset_evaluator(student, metric=self.metric)
                if step_idx == -1:
                    logger.info(f"Student program training set score before training loop: {valset_evaluation.score}")
                else:
                    logger.info(f"Student program training set score after training step {step_idx + 1}/{self.num_train_steps}: {valset_evaluation.score}")
            else:
                # No valset provided, and not using train as val
                assert not self.use_train_as_val, "If report_train_scores is False, use_train_as_val must be False."
                if step_idx == -1:
                    logger.info("Not using any validation set and not reporting train scores.")

    def update_shuffled_trainset(self, original_trainset):
        self.shuffled_trainset_ids = list(range(len(original_trainset)))
        self.rng.shuffle(self.shuffled_trainset_ids)
        for id in self.shuffled_trainset_ids:
            self.id_freqs[id] += 1

        num_to_pad = self.num_dspy_examples_per_grpo_step - (len(original_trainset) % self.num_dspy_examples_per_grpo_step)
        if num_to_pad > 0:
            # Select ids based on least frequent ids
            for _ in range(num_to_pad):
                selected_id = self.id_freqs.most_common()[::-1][0][0]
                self.shuffled_trainset_ids.append(selected_id)
                self.id_freqs[selected_id] += 1

    def select_training_sample_and_update_shuffled_trainset(
        self,
        original_trainset: list[Example],
        train_step_idx: int,
    ) -> list[Example]:
        base_idx = train_step_idx * self.num_dspy_examples_per_grpo_step
        if self.epoch == -1:
            curr_epoch = 0
        else:
            curr_epoch = base_idx // len(self.shuffled_trainset_ids)
        if curr_epoch > self.epoch:
            logger.info(f"Updating shuffled trainset for epoch {curr_epoch}...")
            self.epoch = curr_epoch
            self.update_shuffled_trainset(original_trainset)

        assert len(self.shuffled_trainset_ids) >= self.num_dspy_examples_per_grpo_step, f"Shuffled trainset length {len(self.shuffled_trainset_ids)} is less than num_dspy_examples_per_grpo_step {self.num_dspy_examples_per_grpo_step}"
        assert len(self.shuffled_trainset_ids) % self.num_dspy_examples_per_grpo_step == 0, f"Shuffled trainset length {len(self.shuffled_trainset_ids)} is not divisible by num_dspy_examples_per_grpo_step {self.num_dspy_examples_per_grpo_step}"

        base_idx = base_idx % len(self.shuffled_trainset_ids)
        end_idx = base_idx + self.num_dspy_examples_per_grpo_step
        assert end_idx <= len(self.shuffled_trainset_ids), f"End index {end_idx} is out of bounds for shuffled trainset length {len(self.shuffled_trainset_ids)}"
        selected_ids = self.shuffled_trainset_ids[base_idx:end_idx]
        selected_trainset = [original_trainset[i] for i in selected_ids]
        return selected_trainset

    def compile(
        self,
        student: Module,
        trainset: list[Example],
        teacher: Module | list[Module] | None = None,
        valset: list[Example] | None = None,
        **kwargs,
    ) -> Module:
        logger.info("Starting the GRPO compilation process... The LM(s) for the student program will be updated in place at the end of the training.")
        logger.info("Validating the inputs...")

        assert len(trainset) > 0, "Training set is empty. Please provide a non-empty training set."

        if len(trainset) < self.num_dspy_examples_per_grpo_step:
            logger.warning(
            f"Number of training examples {len(trainset)} is less than the number of examples per GRPO step {self.num_dspy_examples_per_grpo_step}. "
                "Repeating the training set to fill the GRPO step. This could lead to overfitting and training instability."
            )
            multiplier = (self.num_dspy_examples_per_grpo_step + len(trainset) - 1) // len(trainset)
            if multiplier > 1:
                logger.warning(
                    f"Repeating the training set {multiplier} times to fill the GRPO step. This could lead to overfitting and training instability."
                )
                trainset = trainset * multiplier

        # TODO(GRPO Team): Following checks are for unimplemented features.
        # Consider if we want to eventually implement them or remove. We don't
        # yet support:
        # * multitask == False
        # * student program with multiple predictor LMs
        # The main reason for these is that we update the LMs in place. If these
        # LMs are shared between the different predictors of the student
        # program and we have multitask == False, we need to decide which steps
        # will use new LM copies and we need to ensure our decision is
        # consistent with any teacher LMs that share the same LMs.
        # TODO(GRPO Team): We want to make it possible to continue GRPO runs in
        # the future by saving the state of the GRPO run in the event of a
        # process failure.
        if not self.multitask:
            raise ValueError(
                "Independent GRPO training jobs for each predictor in the student program "
                "are not supported yet. Please set multitask=True."
            )

        student_lms = {id(pred.lm) for pred in student.predictors()}
        assert len(student_lms) == 1, (
            f"Student program has multiple LMs: {student_lms}. "
            "GRPO only supports student programs with a single LM."
            "You can set the LM for a program with `program.set_lm(...)`"
        )

        # Our regular input validation starts here
        if self.use_train_as_val:
            assert valset is None, "If use_train_as_val is True, valset must be None."

        logger.info("Preparing the student program...")
        all_predictors_have_lms(student)
        pred_signature_hash_to_ind = {hash(pred.signature): ind for ind, pred in enumerate(student.predictors())}
        num_student_predictors = len(student.predictors())

        logging.info("Preparing the teacher program(s)... We will ensure that the provided programs have the same program structure as the student program.")
        if (isinstance(teacher, list) and len(teacher) == 0) or teacher is None:
            teacher = student
        teachers = teacher if isinstance(teacher, list) else [teacher]
        for t in teachers:
            assert_structural_equivalency(student, t)
            all_predictors_have_lms(t)

        # Ensure that the teachers list contain the student program
        assert student in teachers, f"Student program {student} is not in the list of teachers {teachers}. Please provide the student program as one of the teachers. Alternatively, you can leave the teacher argument as None, and the student program will be used as the teacher program."
        assert self.num_rollouts_per_grpo_step % len(teachers) == 0, (
            f"The GRPO group size (num_rollouts_per_grpo_step) {self.num_rollouts_per_grpo_step} is not divisible by the number of teachers {len(teachers)}. "
            "This is required to ensure that each teacher gets the same number of examples."
            "Please provide a number of examples that is divisible by the number of teachers."
        )
        num_samples_per_input = self.num_rollouts_per_grpo_step // len(teachers)

        # We will disable the LM cache for all programs (student and teachers)
        # These will be reverted to their original state at the end of the
        # training
        lm_cache_dict = {}
        disable_lm_cache(program=student, lm_cache_dict=lm_cache_dict)
        for t in teachers:
            disable_lm_cache(program=t, lm_cache_dict=lm_cache_dict)

        # Update train_kwargs
        for pred in student.predictors():
            train_kwargs = self.train_kwargs[pred.lm]
            train_kwargs = {} if train_kwargs is None else train_kwargs
            train_kwargs["num_generations"] = self.num_rollouts_per_grpo_step
            self.train_kwargs[pred.lm] = train_kwargs

        # We need to have a separate job for each unique LM x the data
        # collection strategy. This properly handles all combinations of
        # multitask and predictor LMs
        logger.info("Preparing the GRPO training job(s)...")
        grpo_training_jobs = {}
        for pred_ind, pred in enumerate(student.predictors()):
            data_key = None if self.multitask else pred_ind
            job_key = (pred.lm, data_key)
            if job_key not in grpo_training_jobs:
                train_kwargs = self.train_kwargs[pred.lm]
                job = pred.lm.reinforce(train_kwargs=train_kwargs)
                grpo_training_jobs[job_key] = job

        self.report_validation_metrics(
            student=student,
            trainset=trainset,
            valset=valset,
            logger=logger,
            step_idx=-1,
        )

        logger.info("Starting the GRPO training loop...")
        for train_step_idx in range(self.num_train_steps):
            logger.info(f"GRPO training step {train_step_idx + 1}/{self.num_train_steps}...")

            subsample_training_dataset = self.select_training_sample_and_update_shuffled_trainset(
                original_trainset=trainset,
                train_step_idx=train_step_idx,
            )

            logger.info("Bootstrapping data...")
            trace_data = [[[] for _ in range(len(teachers))] for _ in range(len(subsample_training_dataset))]
            for tind, teacher in enumerate(teachers):
                subsample_training_dataset_repeated = [example for _ in range(num_samples_per_input) for example in subsample_training_dataset]
                round_data = bootstrap_trace_data(
                    program=teacher,
                    dataset=subsample_training_dataset_repeated,
                    metric=self.metric,
                    num_threads=self.num_threads,
                    raise_on_error=False, # TODO(GRPO Team): This should be True, once the dspy format issue is fixed
                    capture_failed_parses=True,
                    failure_score=self.failure_score,
                    format_failure_score=self.format_failure_score,
                )
                for data_dict in round_data:
                    example_ind_in_subsample = data_dict["example_ind"] % len(subsample_training_dataset)
                    data_dict["example_ind"] = example_ind_in_subsample
                    trace_data[example_ind_in_subsample][tind].append(data_dict)

            # The trace_data for examples with FailedPrediction cases will have the signature at index 0, instead of the predictor
            # We need to replace the signature with the predictor

            # At this point, trace_data: list[example_idx -> list[teacher_idx -> [num_samples_per_input * Dict(example, prediction, trace, example_ind, score)]]]
            # Shape of trace is: [dspy_module_invocation_idx -> Tuple[Predictor, PredictorInputs, Prediction]]
            self.validate_trace_data_and_log_issues(
                trace_data=trace_data,
                subsample_training_dataset=subsample_training_dataset,
                num_teachers=len(teachers),
                num_samples_per_input=num_samples_per_input,
                pred_signature_hash_to_ind=pred_signature_hash_to_ind,
            )

            logger.info("Preparing the training data batch from bootstrapped examples for GRPO...")
            # Now, we need to prepare batches of data to be sent for training
            # Shape of train_batch_per_predictor: list[num_student_predictors -> list[ ]]
            train_batch_per_predictor: list[list[GRPOGroup]] = [[] for _ in range(num_student_predictors)]
            for pred_id in range(num_student_predictors):
                for example_ind, example_data in enumerate(trace_data):
                    # Each example_data is a list of teacher_idx -> [num_samples_per_input * Dict(example, prediction, trace, example_ind, score)]
                    # We need to flatten this list and create a batch for each predictor

                    # TODO(Lakshya, Omar, Noah): Discuss what to do with the same module being invoked multiple times within a single dspy.Example
                    predictor_example_invocations: list[list[tuple]] = []

                    for teacher_data in example_data:
                        for sample in teacher_data:
                            # Each sample is a Dict(example, prediction, trace, example_ind, score)
                            # sample['prediction'] is module_level prediction
                            assert sample["example_ind"] == example_ind, f"Example index {sample['example_ind']} does not match the expected index {example_ind}"

                            trace_instances_for_current_pred = [(*t, sample["score"]) for t in sample["trace"] if hash(t[0].signature) == hash(student.predictors()[pred_id].signature)]

                            predictor_example_invocations.append(trace_instances_for_current_pred)

                    if len(predictor_example_invocations) == 0:
                        logger.warning(f"Skipping example {example_ind} for predictor {pred_id} as it has no invocations. This is likely due to all examples in the training set input, resulting in the model generating output not following the dspy response format.")
                        continue
                    elif len(predictor_example_invocations) != self.num_rollouts_per_grpo_step:
                        logger.warning(f"Number of predictor example invocations {len(predictor_example_invocations)} does not match the expected batch size {self.num_rollouts_per_grpo_step}. This is likely due to all examples in the training set input, resulting in the model generating output not following the dspy response format.")

                    min_len = min([len(predictor_example_invocations[i]) for i in range(len(predictor_example_invocations))])
                    max_len = max([len(predictor_example_invocations[i]) for i in range(len(predictor_example_invocations))])
                    if min_len == 0:
                        logger.warning(f"Skipping example {example_ind} for predictor {pred_id} as it has no invocations.")
                        continue

                    if self.variably_invoked_predictor_grouping_mode == "truncate":
                        predictor_example_invocations = [invocation[:min_len] for invocation in predictor_example_invocations]
                    elif self.variably_invoked_predictor_grouping_mode == "fill":
                        if self.variably_invoked_predictor_fill_strategy == "randint":
                            selector = lambda l: self.rng.choice(l) # noqa: E731, E741
                        else:
                            selector = lambda l: l[-1] # noqa: E731, E741
                        predictor_example_invocations = [
                            invocation + [selector(invocation) for _ in range(max_len - len(invocation))]
                            for invocation in predictor_example_invocations
                        ]
                    else:
                        assert self.variably_invoked_predictor_grouping_mode == "ragged", f"Unknown variably invoked predictor grouping mode {self.variably_invoked_predictor_grouping_mode}"
                    max_len = max([len(predictor_example_invocations[i]) for i in range(len(predictor_example_invocations))])

                    example_training_data: list[GRPOGroup] = [[] for _ in range(max_len)]

                    for group_idx in range(max_len):
                        for rollout_idx in range(len(predictor_example_invocations)):
                            trace_instance = predictor_example_invocations[rollout_idx][group_idx]
                            score = trace_instance[3]
                            # for module_invocation_idx, trace_instance in enumerate(trace_instances_for_current_pred):
                            # Each trace is a tuple of (Predictor, PredictorInputs, Prediction)
                            trace_pred_id = pred_signature_hash_to_ind.get(hash(trace_instance[0].signature))
                            assert trace_pred_id == pred_id

                            predictor = trace_instance[0]
                            pred_lm = predictor.lm
                            adapter = self.adapter[pred_lm] or settings.adapter or ChatAdapter()
                            assert isinstance(adapter, ChatAdapter), f"Adapter {adapter} is not a ChatAdapter. GRPO training is not supported for this adapter."
                            # TODO(Lakshya): Currently we exclude demos from the training data
                            # TODO(GRPO Team): Use build_call_data_from_trace (from bootstrap_finetune) instead of
                            # dealing with the message formatting ourselves.
                            inp_messages = adapter.format(
                                signature=trace_instance[0].signature,
                                inputs=trace_instance[1],
                                demos=[] # TODO: Add support for demos
                            )

                            if isinstance(trace_instance[2], FailedPrediction):
                                score = trace_instance[2].format_reward or self.format_failure_score
                                example_training_data[group_idx].append({
                                    "messages": inp_messages,
                                    "completion": {
                                        "role": "assistant",
                                        "content": trace_instance[2].completion_text,
                                    },
                                    "reward": float(score),
                                })
                                logger.warning(f"Adding a format failure example to the training data for predictor {pred_id} and example {example_ind}.")
                            else:
                                all_messages = adapter.format_finetune_data(
                                    signature=trace_instance[0].signature,
                                    inputs=trace_instance[1],
                                    outputs=trace_instance[2],
                                    demos=[] # TODO: Add support for demos
                                )["messages"]

                                assert all_messages[:-1] == inp_messages, f"Input messages {inp_messages} do not match the expected messages {all_messages[:-1]}"

                                example_training_data[group_idx].append({
                                    "messages": inp_messages,
                                    "completion": {
                                        "role": all_messages[-1]["role"],
                                        "content": all_messages[-1]["content"],
                                    },
                                    "reward": float(score),
                                })

                    train_batch_per_predictor[pred_id].extend(example_training_data)

            if not any(train_batch_per_predictor):
                logger.warning("No training data found for this training step. This means that the model did not generate valid formatted responses for any of the examples in the training set. This is a critical error. Please check the model and the training set.")
                continue

            for predictor_train_batch in train_batch_per_predictor:
                for grpo_train_group in predictor_train_batch:
                    if len(grpo_train_group) != self.num_rollouts_per_grpo_step:
                        logger.warning(f"Number of completions {len(grpo_train_group)} does not match the expected number num_rollouts_per_grpo_step={self.num_rollouts_per_grpo_step}")
                        assert len(grpo_train_group) <= self.num_rollouts_per_grpo_step, f"Number of completions {len(grpo_train_group)} is greater than the expected number num_rollouts_per_grpo_step={self.num_rollouts_per_grpo_step}"
                    if len(set(map(repr, grpo_train_group))) < 2:
                        # TODO(GRPO Team): How can we avoid this warning?
                        logger.warning(f"GRPOGroup has no diversity. This could be due to low temperature, or low number of rollouts, or the cache could be enabled inadvertently. The GRPOGroup is {grpo_train_group}.")

            # We now run the GRPO step. Notes:
            # * The job here has a reference to a particular M that's attached
            #   to the student program. We update the .model field of this LM
            #   inside the job, which also updates the LM in the student program
            #   since these point to the same reference (along with any teacher
            #   program that shares the same LM).
            # * TODO(GRPO Team): This is inconsistent with how
            #   BootstrapFinetune works, which creates new LM instances post
            #   training. We should decide whether the LMs should be updated in
            #   place or new LMs should be created, and standardize our approach
            #   for both. If we decide to create new LMs, we should find a way
            #   to update self.adapter and self.train_kwargs accordingly, in
            #   addition to updating any teacher programs that share the same
            #   LM.
            logger.info("Invoking GRPO training step...")
            for (_, data_key), job in grpo_training_jobs.items():
                train_data: list[GRPOGroup] = sum(train_batch_per_predictor, []) if data_key is None else train_batch_per_predictor[data_key] #noqa: RUF017
                for group in train_data:
                    if len(group) != self.num_rollouts_per_grpo_step:
                        # TODO(GRPO Team): This is very undesirable. This occurs only because in some of the generations, the model does not follow the correct dspy format.
                        # The ideal solution is to identify the full response string in that predictor's group, and then assign  a high-negative (user-configurable) reward to that group.

                        # Pad the group to the expected number of generations by repeating the whole group, might require multiple iterations
                        while len(group) < self.num_rollouts_per_grpo_step:
                            group.extend(group[:min(self.num_rollouts_per_grpo_step - len(group), len(group))])
                    assert len(group) == self.num_rollouts_per_grpo_step, f"Number of completions {len(group)} does not match the expected number self.num_rollouts_per_grpo_step={self.num_rollouts_per_grpo_step}"

                job.step(train_data=train_data, train_data_format=TrainDataFormat.GRPO_CHAT)

            logger.info(f"GRPO training step {train_step_idx + 1}/{self.num_train_steps} completed.")

            self.report_validation_metrics(
                student=student,
                trainset=trainset,
                valset=valset,
                logger=logger,
                step_idx=train_step_idx,
            )

        logger.info("Done with the iterations! Retrieving the final model(s)...")
        for _, job in grpo_training_jobs.items():
            job.terminate()

        # Revert cache states to their initial values
        recover_lm_cache(program=student, lm_cache_dict=lm_cache_dict)
        for t in teachers:
            recover_lm_cache(program=t, lm_cache_dict=lm_cache_dict)

        logger.info("GRPO compiler has finished compiling the student program")
        student._compiled = True
        return student


def disable_lm_cache(program: Module, lm_cache_dict: dict):
    """Disable the LM cache for all predictors in the program."""
    for pred in program.predictors():
        if not pred.lm:
            raise ValueError(f"Cannot disable cache: predictor {pred} does not have an LM set.")
        if pred.lm not in lm_cache_dict:  # Check to avoid overwriting the cache
            lm_cache_dict[pred.lm] = pred.lm.cache
        pred.lm.cache = False


def recover_lm_cache(program: Module, lm_cache_dict: dict):
    """Recover the LM caches for all predictors in the program to their original state."""
    for pred in program.predictors():
        if pred.lm in lm_cache_dict:
            pred.lm.cache = lm_cache_dict[pred.lm]
        else:
            # We do not expect this branch to execute at all since all the LMs
            # are modified in place and no new LMs are created during training.
            # However, we do not complain if this happens since this is a
            # relatively minor feature. We default the LM cache to True.
            pred.lm.cache = True
```

## File: `teleprompt/infer_rules.py`
```
import logging
import random

import numpy as np

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot

logger = logging.getLogger(__name__)


class InferRules(BootstrapFewShot):
    def __init__(self, num_candidates=10, num_rules=10, num_threads=None, teacher_settings=None, **kwargs):
        super().__init__(teacher_settings=teacher_settings, **kwargs)

        self.num_candidates = num_candidates
        self.num_rules = num_rules
        self.num_threads = num_threads
        self.rules_induction_program = RulesInductionProgram(num_rules, teacher_settings=teacher_settings)
        self.metric = kwargs.get("metric")
        self.max_errors = kwargs.get("max_errors")

    def compile(self, student, *, teacher=None, trainset, valset=None):
        if valset is None:
            train_size = int(0.5 * len(trainset))
            trainset, valset = trainset[:train_size], trainset[train_size:]

        super().compile(student, teacher=teacher, trainset=trainset)

        original_program = self.student.deepcopy()
        all_predictors = [p for p in original_program.predictors() if hasattr(p, "signature")]
        instructions_list = [p.signature.instructions for p in all_predictors]

        best_score = -np.inf
        best_program = None

        for candidate_idx in range(self.num_candidates):
            candidate_program = original_program.deepcopy()
            candidate_predictors = [p for p in candidate_program.predictors() if hasattr(p, "signature")]

            for i, predictor in enumerate(candidate_predictors):
                predictor.signature.instructions = instructions_list[i]

            for i, predictor in enumerate(candidate_predictors):
                rules = self.induce_natural_language_rules(predictor, trainset)
                predictor.signature.instructions = instructions_list[i]
                self.update_program_instructions(predictor, rules)

            score = self.evaluate_program(candidate_program, valset)

            if score > best_score:
                best_score = score
                best_program = candidate_program

            logger.info(f"Evaluated Candidate {candidate_idx + 1} with score {score}. Current best score: {best_score}")

        logger.info(f"Final best score: {best_score}")

        return best_program

    def induce_natural_language_rules(self, predictor, trainset):
        demos = self.get_predictor_demos(trainset, predictor)
        signature = predictor.signature
        while True:
            examples_text = self.format_examples(demos, signature)
            try:
                return self.rules_induction_program(examples_text)
            except Exception as e:
                assert (
                    isinstance(e, ValueError)
                    or e.__class__.__name__ == "BadRequestError"
                    or "ContextWindowExceededError" in str(e)
                )
                if len(demos) > 1:
                    demos = demos[:-1]
                else:
                    raise RuntimeError(
                        "Failed to generate natural language rules since a single example couldn't fit in the model's "
                        "context window."
                    ) from e

    def update_program_instructions(self, predictor, natural_language_rules):
        predictor.signature.instructions = (
            f"{predictor.signature.instructions}\n\n"
            f"Please adhere to the following rules when making your prediction:\n{natural_language_rules}"
        )

    def format_examples(self, demos, signature):
        examples_text = ""
        for demo in demos:
            input_fields = {k: v for k, v in demo.items() if k in signature.input_fields}
            output_fields = {k: v for k, v in demo.items() if k in signature.output_fields}
            input_text = "\n".join(f"{k}: {v}" for k, v in input_fields.items())
            output_text = "\n".join(f"{k}: {v}" for k, v in output_fields.items())
            examples_text += f"Input Fields:\n{input_text}\n\n=========\nOutput Fields:\n{output_text}\n\n"
        return examples_text

    def get_predictor_demos(self, trainset, predictor):
        # TODO: Consider how this handled "incomplete" demos.
        signature = predictor.signature
        return [
            {
                key: value
                for key, value in example.items()
                if key in signature.input_fields or key in signature.output_fields
            }
            for example in trainset
        ]

    def evaluate_program(self, program, dataset):
        effective_max_errors = (
            self.max_errors if self.max_errors is not None else dspy.settings.max_errors
        )
        evaluate = Evaluate(
            devset=dataset,
            metric=self.metric,
            num_threads=self.num_threads,
            max_errors=effective_max_errors,
            display_table=False,
            display_progress=True,
        )
        score = evaluate(program, metric=self.metric).score
        return score


class RulesInductionProgram(dspy.Module):
    def __init__(self, num_rules, teacher_settings=None):
        super().__init__()

        class CustomRulesInduction(dspy.Signature):
            __doc__ = (
                f"Given a set of examples, extract a list of {num_rules} concise and non-redundant natural language "
                "rules that provide clear guidance for performing the task. All rules should be actionable for a "
                "well-specified scope of examples of this general kind of task."
            )
            examples_text = dspy.InputField(desc="Text containing examples")
            natural_language_rules = dspy.OutputField(desc="Induced natural language rules")

        self.rules_induction = dspy.ChainOfThought(CustomRulesInduction)
        self.teacher_settings = teacher_settings or {}
        self.rng = random.Random(0)

    def forward(self, examples_text):
        with dspy.settings.context(**self.teacher_settings):
            lm = dspy.settings.lm.copy(temperature=self.rng.uniform(0.9, 1.0))
            with dspy.settings.context(lm=lm):
                rules = self.rules_induction(examples_text=examples_text).natural_language_rules

        return rules.strip()
```

## File: `teleprompt/knn_fewshot.py`
```
import types
from typing import Any

from dspy.clients import Embedder
from dspy.predict.knn import KNN
from dspy.primitives import Example
from dspy.teleprompt import BootstrapFewShot
from dspy.teleprompt.teleprompt import Teleprompter


class KNNFewShot(Teleprompter):
    def __init__(self, k: int, trainset: list[Example], vectorizer: Embedder, **few_shot_bootstrap_args: dict[str, Any]):
        """
        KNNFewShot is an optimizer that uses an in-memory KNN retriever to find the k nearest neighbors
        in a trainset at test time. For each input example in a forward call, it identifies the k most
        similar examples from the trainset and attaches them as demonstrations to the student module.

        Args:
            k: The number of nearest neighbors to attach to the student model.
            trainset: The training set to use for few-shot prompting.
            vectorizer: The `Embedder` to use for vectorization
            **few_shot_bootstrap_args: Additional arguments for the `BootstrapFewShot` optimizer.

        Example:
            ```python
            import dspy
            from sentence_transformers import SentenceTransformer

            # Define a QA module with chain of thought
            qa = dspy.ChainOfThought("question -> answer")

            # Create a training dataset with examples
            trainset = [
                dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
                # ... more examples ...
            ]

            # Initialize KNNFewShot with a sentence transformer model
            knn_few_shot = KNNFewShot(
                k=3,
                trainset=trainset,
                vectorizer=dspy.Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode)
            )

            # Compile the QA module with few-shot learning
            compiled_qa = knn_few_shot.compile(qa)

            # Use the compiled module
            result = compiled_qa("What is the capital of Belgium?")
            ```
        """
        self.KNN = KNN(k, trainset, vectorizer=vectorizer)
        self.few_shot_bootstrap_args = few_shot_bootstrap_args

    def compile(self, student, *, teacher=None):
        student_copy = student.reset_copy()

        def forward_pass(_, **kwargs):
            knn_trainset = self.KNN(**kwargs)
            few_shot_bootstrap = BootstrapFewShot(**self.few_shot_bootstrap_args)
            compiled_program = few_shot_bootstrap.compile(
                student,
                teacher=teacher,
                trainset=knn_trainset,
            )
            return compiled_program(**kwargs)

        student_copy.forward = types.MethodType(forward_pass, student_copy)
        return student_copy
```

## File: `teleprompt/mipro_optimizer_v2.py`
```
import logging
import random
import sys
import textwrap
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.propose import GroundedProposer
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.teleprompt.utils import (
    create_minibatch,
    create_n_fewshot_demo_sets,
    eval_candidate_program,
    get_program_with_highest_avg_score,
    get_signature,
    print_full_program,
    save_candidate_program,
    set_signature,
)

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)

# Constants
BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT = 3
LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT = 0
MIN_MINIBATCH_SIZE = 50

AUTO_RUN_SETTINGS = {
    "light": {"n": 6, "val_size": 100},
    "medium": {"n": 12, "val_size": 300},
    "heavy": {"n": 18, "val_size": 1000},
}

# ANSI escape codes for colors
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
BOLD = "\033[1m"
ENDC = "\033[0m"  # Resets the color to default


class MIPROv2(Teleprompter):
    def __init__(
        self,
        metric: Callable,
        prompt_model: Any | None = None,
        task_model: Any | None = None,
        teacher_settings: dict | None = None,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        auto: Literal["light", "medium", "heavy"] | None = "light",
        num_candidates: int | None = None,
        num_threads: int | None = None,
        max_errors: int | None = None,
        seed: int = 9,
        init_temperature: float = 0.5,
        verbose: bool = False,
        track_stats: bool = True,
        log_dir: str | None = None,
        metric_threshold: float | None = None,
    ):
        # Validate 'auto' parameter
        allowed_modes = {None, "light", "medium", "heavy"}
        if auto not in allowed_modes:
            raise ValueError(f"Invalid value for auto: {auto}. Must be one of {allowed_modes}.")
        self.auto = auto
        self.num_fewshot_candidates = num_candidates
        self.num_instruct_candidates = num_candidates
        self.num_candidates = num_candidates
        self.metric = metric
        self.init_temperature = init_temperature
        self.task_model = task_model if task_model else dspy.settings.lm
        self.prompt_model = prompt_model if prompt_model else dspy.settings.lm
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.verbose = verbose
        self.track_stats = track_stats
        self.log_dir = log_dir
        self.teacher_settings = teacher_settings or {}
        self.prompt_model_total_calls = 0
        self.total_calls = 0
        self.num_threads = num_threads
        self.max_errors = max_errors
        self.metric_threshold = metric_threshold
        self.seed = seed
        self.rng = None

    def compile(
        self,
        student: Any,
        *,
        trainset: list,
        teacher: Any = None,
        valset: list | None = None,
        num_trials: int | None = None,
        max_bootstrapped_demos: int | None = None,
        max_labeled_demos: int | None = None,
        seed: int | None = None,
        minibatch: bool = True,
        minibatch_size: int = 35,
        minibatch_full_eval_steps: int = 5,
        program_aware_proposer: bool = True,
        data_aware_proposer: bool = True,
        view_data_batch_size: int = 10,
        tip_aware_proposer: bool = True,
        fewshot_aware_proposer: bool = True,
        requires_permission_to_run: bool = True,
        provide_traceback: bool | None = None,
    ) -> Any:
        effective_max_errors = (
            self.max_errors
            if self.max_errors is not None
            else dspy.settings.max_errors
        )
        zeroshot_opt = (self.max_bootstrapped_demos == 0) and (self.max_labeled_demos == 0)

        # If auto is None, and num_trials is not provided (but num_candidates is), raise an error that suggests a good num_trials value
        if self.auto is None and (self.num_candidates is not None and num_trials is None):
            raise ValueError(
                f"If auto is None, num_trials must also be provided. Given num_candidates={self.num_candidates}, we'd recommend setting num_trials to ~{self._set_num_trials_from_num_candidates(student, zeroshot_opt, self.num_candidates)}."
            )

        # If auto is None, and num_candidates or num_trials is None, raise an error
        if self.auto is None and (self.num_candidates is None or num_trials is None):
            raise ValueError("If auto is None, num_candidates must also be provided.")

        # If auto is provided, and either num_candidates or num_trials is not None, raise an error
        if self.auto is not None and (self.num_candidates is not None or num_trials is not None):
            raise ValueError(
                "If auto is not None, num_candidates and num_trials cannot be set, since they would be overrided by the auto settings. Please either set auto to None, or do not specify num_candidates and num_trials."
            )

        # Set random seeds
        seed = seed or self.seed
        self._set_random_seeds(seed)

        # Update max demos if specified
        if max_bootstrapped_demos is not None:
            self.max_bootstrapped_demos = max_bootstrapped_demos
        if max_labeled_demos is not None:
            self.max_labeled_demos = max_labeled_demos

        # Set training & validation sets
        trainset, valset = self._set_and_validate_datasets(trainset, valset)

        # Set hyperparameters based on run mode (if set)
        num_trials, valset, minibatch = self._set_hyperparams_from_run_mode(
            student, num_trials, minibatch, zeroshot_opt, valset
        )

        if self.auto:
            self._print_auto_run_settings(num_trials, minibatch, valset)

        if minibatch and minibatch_size > len(valset):
            raise ValueError(f"Minibatch size cannot exceed the size of the valset. Valset size: {len(valset)}.")

        # Estimate LM calls and get user confirmation
        if requires_permission_to_run:
            if not self._get_user_confirmation(
                student,
                num_trials,
                minibatch,
                minibatch_size,
                minibatch_full_eval_steps,
                valset,
                program_aware_proposer,
            ):
                logger.info("Compilation aborted by the user.")
                return student  # Return the original student program

        # Initialize program and evaluator
        program = student.deepcopy()
        evaluate = Evaluate(
            devset=valset,
            metric=self.metric,
            num_threads=self.num_threads,
            max_errors=effective_max_errors,
            display_table=False,
            display_progress=True,
            provide_traceback=provide_traceback,
        )

        # Step 1: Bootstrap few-shot examples
        demo_candidates = self._bootstrap_fewshot_examples(program, trainset, seed, teacher)

        # Step 2: Propose instruction candidates
        instruction_candidates = self._propose_instructions(
            program,
            trainset,
            demo_candidates,
            view_data_batch_size,
            program_aware_proposer,
            data_aware_proposer,
            tip_aware_proposer,
            fewshot_aware_proposer,
        )

        # If zero-shot, discard demos
        if zeroshot_opt:
            demo_candidates = None

        # Step 3: Find optimal prompt parameters
        best_program = self._optimize_prompt_parameters(
            program,
            instruction_candidates,
            demo_candidates,
            evaluate,
            valset,
            num_trials,
            minibatch,
            minibatch_size,
            minibatch_full_eval_steps,
            seed,
        )

        return best_program

    def _set_random_seeds(self, seed):
        self.rng = random.Random(seed)
        np.random.seed(seed)

    def _set_num_trials_from_num_candidates(self, program, zeroshot_opt, num_candidates):
        num_vars = len(program.predictors())
        if not zeroshot_opt:
            num_vars *= 2  # Account for few-shot examples + instruction variables
        # Trials = MAX(c*M*log(N), c=2, 3/2*N)
        num_trials = int(max(2 * num_vars * np.log2(num_candidates), 1.5 * num_candidates))

        return num_trials

    def _set_hyperparams_from_run_mode(
        self,
        program: Any,
        num_trials: int,
        minibatch: bool,
        zeroshot_opt: bool,
        valset: list,
    ) -> tuple[int, list, bool]:
        if self.auto is None:
            return num_trials, valset, minibatch

        auto_settings = AUTO_RUN_SETTINGS[self.auto]

        valset = create_minibatch(valset, batch_size=auto_settings["val_size"], rng=self.rng)
        minibatch = len(valset) > MIN_MINIBATCH_SIZE

        # Set num instruct candidates to 1/2 of N if optimizing with few-shot examples, otherwise set to N
        # This is because we've found that it's generally better to spend optimization budget on few-shot examples
        # When they are allowed.
        self.num_instruct_candidates = auto_settings["n"] if zeroshot_opt else int(auto_settings["n"] * 0.5)
        self.num_fewshot_candidates = auto_settings["n"]

        num_trials = self._set_num_trials_from_num_candidates(program, zeroshot_opt, auto_settings["n"])

        return num_trials, valset, minibatch

    def _set_and_validate_datasets(self, trainset: list, valset: list | None):
        if not trainset:
            raise ValueError("Trainset cannot be empty.")

        if valset is None:
            if len(trainset) < 2:
                raise ValueError("Trainset must have at least 2 examples if no valset specified.")
            valset_size = min(1000, max(1, int(len(trainset) * 0.80)))
            cutoff = len(trainset) - valset_size
            valset = trainset[cutoff:]
            trainset = trainset[:cutoff]
        else:
            if len(valset) < 1:
                raise ValueError("Validation set must have at least 1 example.")

        return trainset, valset

    def _print_auto_run_settings(self, num_trials: int, minibatch: bool, valset: list):
        logger.info(
            f"\nRUNNING WITH THE FOLLOWING {self.auto.upper()} AUTO RUN SETTINGS:"
            f"\nnum_trials: {num_trials}"
            f"\nminibatch: {minibatch}"
            f"\nnum_fewshot_candidates: {self.num_fewshot_candidates}"
            f"\nnum_instruct_candidates: {self.num_instruct_candidates}"
            f"\nvalset size: {len(valset)}\n"
        )

    def _estimate_lm_calls(
        self,
        program: Any,
        num_trials: int,
        minibatch: bool,
        minibatch_size: int,
        minibatch_full_eval_steps: int,
        valset: list,
        program_aware_proposer: bool,
    ) -> tuple[str, str]:
        num_predictors = len(program.predictors())

        # Estimate prompt model calls
        estimated_prompt_model_calls = (
            10  # Data summarizer calls
            + self.num_instruct_candidates * num_predictors  # Candidate generation
            + (num_predictors + 1 if program_aware_proposer else 0)  # Program-aware proposer
        )
        prompt_model_line = (
            f"{YELLOW}- Prompt Generation: {BLUE}{BOLD}10{ENDC}{YELLOW} data summarizer calls + "
            f"{BLUE}{BOLD}{self.num_instruct_candidates}{ENDC}{YELLOW} * "
            f"{BLUE}{BOLD}{num_predictors}{ENDC}{YELLOW} lm calls in program "
            f"+ ({BLUE}{BOLD}{num_predictors + 1}{ENDC}{YELLOW}) lm calls in program-aware proposer "
            f"= {BLUE}{BOLD}{estimated_prompt_model_calls}{ENDC}{YELLOW} prompt model calls{ENDC}"
        )

        # Estimate task model calls
        if not minibatch:
            estimated_task_model_calls = len(valset) * num_trials
            task_model_line = (
                f"{YELLOW}- Program Evaluation: {BLUE}{BOLD}{len(valset)}{ENDC}{YELLOW} examples in val set * "
                f"{BLUE}{BOLD}{num_trials}{ENDC}{YELLOW} batches = "
                f"{BLUE}{BOLD}{estimated_task_model_calls}{ENDC}{YELLOW} LM program calls{ENDC}"
            )
        else:
            full_eval_steps = num_trials // minibatch_full_eval_steps + 1
            estimated_task_model_calls = minibatch_size * num_trials + len(valset) * full_eval_steps
            task_model_line = (
                f"{YELLOW}- Program Evaluation: {BLUE}{BOLD}{minibatch_size}{ENDC}{YELLOW} examples in minibatch * "
                f"{BLUE}{BOLD}{num_trials}{ENDC}{YELLOW} batches + "
                f"{BLUE}{BOLD}{len(valset)}{ENDC}{YELLOW} examples in val set * "
                f"{BLUE}{BOLD}{full_eval_steps}{ENDC}{YELLOW} full evals = "
                f"{BLUE}{BOLD}{estimated_task_model_calls}{ENDC}{YELLOW} LM Program calls{ENDC}"
            )

        return prompt_model_line, task_model_line

    def _get_user_confirmation(
        self,
        program: Any,
        num_trials: int,
        minibatch: bool,
        minibatch_size: int,
        minibatch_full_eval_steps: int,
        valset: list,
        program_aware_proposer: bool,
    ) -> bool:
        prompt_model_line, task_model_line = self._estimate_lm_calls(
            program,
            num_trials,
            minibatch,
            minibatch_size,
            minibatch_full_eval_steps,
            valset,
            program_aware_proposer,
        )

        user_message = textwrap.dedent(
            f"""\
            {YELLOW}{BOLD}Projected Language Model (LM) Calls{ENDC}

            Based on the parameters you have set, the maximum number of LM calls is projected as follows:

            {prompt_model_line}
            {task_model_line}

            {YELLOW}{BOLD}Estimated Cost Calculation:{ENDC}

            {YELLOW}Total Cost = (Number of calls to task model * (Avg Input Token Length per Call * Task Model Price per Input Token + Avg Output Token Length per Call * Task Model Price per Output Token)
                        + (Number of program calls * (Avg Input Token Length per Call * Task Prompt Price per Input Token + Avg Output Token Length per Call * Prompt Model Price per Output Token).{ENDC}

            For a preliminary estimate of potential costs, we recommend you perform your own calculations based on the task
            and prompt models you intend to use. If the projected costs exceed your budget or expectations, you may consider:

            {YELLOW}- Reducing the number of trials (`num_trials`), the size of the valset, or the number of LM calls in your program.{ENDC}
            {YELLOW}- Using a cheaper task model to optimize the prompt.{ENDC}
            {YELLOW}- Setting `minibatch=True` if you haven't already.{ENDC}\n"""
        )

        user_confirmation_message = textwrap.dedent(
            f"""\
            To proceed with the execution of this program, please confirm by typing {BLUE}'y'{ENDC} for yes or {BLUE}'n'{ENDC} for no.
            If no input is received within 20 seconds, the program will proceed automatically.

            If you would like to bypass this confirmation step in future executions, set the {YELLOW}`requires_permission_to_run`{ENDC} flag to {YELLOW}`False`{ENDC} when calling compile.

            {YELLOW}Awaiting your input...{ENDC}
        """
        )

        print(f"{user_message}\n{user_confirmation_message}\nDo you wish to continue? (y/n): ", end="", flush=True)

        # Wait for input with timeout
        start_time = time.time()
        while time.time() - start_time < 20:
            if sys.platform == "win32":
                import msvcrt
                if msvcrt.kbhit():
                    user_input = msvcrt.getch().decode("utf-8").strip().lower()
                    print(user_input)  # Echo the input
                    return user_input == "y"
            else:
                import select
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    user_input = sys.stdin.readline().strip().lower()
                    return user_input == "y"
            time.sleep(0.1)

        print("\nNo input received within 20 seconds. Proceeding with execution...")
        return True

    def _bootstrap_fewshot_examples(self, program: Any, trainset: list, seed: int, teacher: Any) -> list | None:
        logger.info("\n==> STEP 1: BOOTSTRAP FEWSHOT EXAMPLES <==")
        if self.max_bootstrapped_demos > 0:
            logger.info(
                "These will be used as few-shot example candidates for our program and for creating instructions.\n"
            )
        else:
            logger.info("These will be used for informing instruction proposal.\n")

        logger.info(f"Bootstrapping N={self.num_fewshot_candidates} sets of demonstrations...")

        zeroshot = self.max_bootstrapped_demos == 0 and self.max_labeled_demos == 0

        try:
            effective_max_errors = (
                self.max_errors if self.max_errors is not None else dspy.settings.max_errors
            )

            demo_candidates = create_n_fewshot_demo_sets(
                student=program,
                num_candidate_sets=self.num_fewshot_candidates,
                trainset=trainset,
                max_labeled_demos=(LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT if zeroshot else self.max_labeled_demos),
                max_bootstrapped_demos=(
                    BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT if zeroshot else self.max_bootstrapped_demos
                ),
                metric=self.metric,
                max_errors=effective_max_errors,
                teacher=teacher,
                teacher_settings=self.teacher_settings,
                seed=seed,
                metric_threshold=self.metric_threshold,
                rng=self.rng,
            )
        except Exception as e:
            logger.info(f"Error generating few-shot examples: {e}")
            logger.info("Running without few-shot examples.")
            demo_candidates = None

        return demo_candidates

    def _propose_instructions(
        self,
        program: Any,
        trainset: list,
        demo_candidates: list | None,
        view_data_batch_size: int,
        program_aware_proposer: bool,
        data_aware_proposer: bool,
        tip_aware_proposer: bool,
        fewshot_aware_proposer: bool,
    ) -> dict[int, list[str]]:
        logger.info("\n==> STEP 2: PROPOSE INSTRUCTION CANDIDATES <==")
        logger.info(
            "We will use the few-shot examples from the previous step, a generated dataset summary, a summary of the program code, and a randomly selected prompting tip to propose instructions."
        )

        proposer = GroundedProposer(
            program=program,
            trainset=trainset,
            prompt_model=self.prompt_model,
            view_data_batch_size=view_data_batch_size,
            program_aware=program_aware_proposer,
            use_dataset_summary=data_aware_proposer,
            use_task_demos=fewshot_aware_proposer,
            num_demos_in_context=BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT,
            use_tip=tip_aware_proposer,
            set_tip_randomly=tip_aware_proposer,
            use_instruct_history=False,
            set_history_randomly=False,
            verbose=self.verbose,
            rng=self.rng,
        )

        logger.info(f"\nProposing N={self.num_instruct_candidates} instructions...\n")
        instruction_candidates = proposer.propose_instructions_for_program(
            trainset=trainset,
            program=program,
            demo_candidates=demo_candidates,
            N=self.num_instruct_candidates,
            T=self.init_temperature,
            trial_logs={},
        )

        for i, pred in enumerate(program.predictors()):
            logger.info(f"Proposed Instructions for Predictor {i}:\n")
            instruction_candidates[i][0] = get_signature(pred).instructions
            for j, instruction in enumerate(instruction_candidates[i]):
                logger.info(f"{j}: {instruction}\n")
            logger.info("\n")

        return instruction_candidates

    def _optimize_prompt_parameters(
        self,
        program: Any,
        instruction_candidates: dict[int, list[str]],
        demo_candidates: list | None,
        evaluate: Evaluate,
        valset: list,
        num_trials: int,
        minibatch: bool,
        minibatch_size: int,
        minibatch_full_eval_steps: int,
        seed: int,
    ) -> Any | None:
        import optuna

        # Run optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        logger.info("==> STEP 3: FINDING OPTIMAL PROMPT PARAMETERS <==")
        logger.info(
            "We will evaluate the program over a series of trials with different combinations of instructions and few-shot examples to find the optimal combination using Bayesian Optimization.\n"
        )

        # Compute the adjusted total trials that we will run (including full evals)
        run_additional_full_eval_at_end = 1 if num_trials % minibatch_full_eval_steps != 0 else 0
        adjusted_num_trials = int(
            (num_trials + num_trials // minibatch_full_eval_steps + 1 + run_additional_full_eval_at_end)
            if minibatch
            else num_trials
        )
        logger.info(f"== Trial {1} / {adjusted_num_trials} - Full Evaluation of Default Program ==")

        default_score = eval_candidate_program(len(valset), valset, program, evaluate, self.rng).score
        logger.info(f"Default program score: {default_score}\n")

        trial_logs = {}
        trial_logs[1] = {}
        trial_logs[1]["full_eval_program_path"] = save_candidate_program(program, self.log_dir, -1)
        trial_logs[1]["full_eval_score"] = default_score
        trial_logs[1]["total_eval_calls_so_far"] = len(valset)
        trial_logs[1]["full_eval_program"] = program.deepcopy()

        # Initialize optimization variables
        best_score = default_score
        best_program = program.deepcopy()
        total_eval_calls = len(valset)
        score_data = [{"score": best_score, "program": program.deepcopy(), "full_eval": True}]
        param_score_dict = defaultdict(list)
        fully_evaled_param_combos = {}

        # Define the objective function
        def objective(trial):
            nonlocal program, best_program, best_score, trial_logs, total_eval_calls, score_data

            trial_num = trial.number + 1
            if minibatch:
                logger.info(f"== Trial {trial_num} / {adjusted_num_trials} - Minibatch ==")
            else:
                logger.info(f"===== Trial {trial_num} / {num_trials} =====")

            trial_logs[trial_num] = {}

            # Create a new candidate program
            candidate_program = program.deepcopy()

            # Choose instructions and demos, insert them into the program
            chosen_params, raw_chosen_params = self._select_and_insert_instructions_and_demos(
                candidate_program,
                instruction_candidates,
                demo_candidates,
                trial,
                trial_logs,
                trial_num,
            )

            # Log assembled program
            if self.verbose:
                logger.info("Evaluating the following candidate program...\n")
                print_full_program(candidate_program)

            # Evaluate the candidate program (on minibatch if minibatch=True)
            batch_size = minibatch_size if minibatch else len(valset)
            score = eval_candidate_program(batch_size, valset, candidate_program, evaluate, self.rng).score
            total_eval_calls += batch_size

            # Update best score and program
            if not minibatch and score > best_score:
                best_score = score
                best_program = candidate_program.deepcopy()
                logger.info(f"{GREEN}Best full score so far!{ENDC} Score: {score}")

            # Log evaluation results
            score_data.append(
                {"score": score, "program": candidate_program, "full_eval": batch_size >= len(valset)}
            )  # score, prog, full_eval
            if minibatch:
                self._log_minibatch_eval(
                    score,
                    best_score,
                    batch_size,
                    chosen_params,
                    score_data,
                    trial,
                    adjusted_num_trials,
                    trial_logs,
                    trial_num,
                    candidate_program,
                    total_eval_calls,
                )
            else:
                self._log_normal_eval(
                    score,
                    best_score,
                    chosen_params,
                    score_data,
                    trial,
                    num_trials,
                    trial_logs,
                    trial_num,
                    valset,
                    batch_size,
                    candidate_program,
                    total_eval_calls,
                )
            categorical_key = ",".join(map(str, chosen_params))
            param_score_dict[categorical_key].append(
                (score, candidate_program, raw_chosen_params),
            )

            # If minibatch, perform full evaluation at intervals (and at the very end)
            if minibatch and (
                (trial_num % (minibatch_full_eval_steps + 1) == 0) or (trial_num == (adjusted_num_trials - 1))
            ):
                best_score, best_program, total_eval_calls = self._perform_full_evaluation(
                    trial_num,
                    adjusted_num_trials,
                    param_score_dict,
                    fully_evaled_param_combos,
                    evaluate,
                    valset,
                    trial_logs,
                    total_eval_calls,
                    score_data,
                    best_score,
                    best_program,
                    study,
                    instruction_candidates,
                    demo_candidates,
                )

            return score

        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        default_params = {f"{i}_predictor_instruction": 0 for i in range(len(program.predictors()))}
        if demo_candidates:
            default_params.update({f"{i}_predictor_demos": 0 for i in range(len(program.predictors()))})

        # Add default run as a baseline in optuna (TODO: figure out how to weight this by # of samples evaluated on)
        trial = optuna.trial.create_trial(
            params=default_params,
            distributions=self._get_param_distributions(program, instruction_candidates, demo_candidates),
            value=default_score,
        )
        study.add_trial(trial)
        study.optimize(objective, n_trials=num_trials)

        # Attach logs to best program
        if best_program is not None and self.track_stats:
            best_program.trial_logs = trial_logs
            best_program.score = best_score
            best_program.prompt_model_total_calls = self.prompt_model_total_calls
            best_program.total_calls = self.total_calls
            sorted_candidate_programs = sorted(score_data, key=lambda x: x["score"], reverse=True)
            # Attach all minibatch programs
            best_program.mb_candidate_programs = [
                score_data for score_data in sorted_candidate_programs if not score_data["full_eval"]
            ]
            # Attach all programs that were evaluated on the full trainset, in descending order of score
            best_program.candidate_programs = [
                score_data for score_data in sorted_candidate_programs if score_data["full_eval"]
            ]

        logger.info(f"Returning best identified program with score {best_score}!")

        return best_program

    def _log_minibatch_eval(
        self,
        score,
        best_score,
        batch_size,
        chosen_params,
        score_data,
        trial,
        adjusted_num_trials,
        trial_logs,
        trial_num,
        candidate_program,
        total_eval_calls,
    ):
        trial_logs[trial_num]["mb_program_path"] = save_candidate_program(candidate_program, self.log_dir, trial_num)
        trial_logs[trial_num]["mb_score"] = score
        trial_logs[trial_num]["total_eval_calls_so_far"] = total_eval_calls
        trial_logs[trial_num]["mb_program"] = candidate_program.deepcopy()

        logger.info(f"Score: {score} on minibatch of size {batch_size} with parameters {chosen_params}.")
        minibatch_scores = ", ".join([f"{s['score']}" for s in score_data if not s["full_eval"]])
        logger.info(f"Minibatch scores so far: {'[' + minibatch_scores + ']'}")
        full_eval_scores = ", ".join([f"{s['score']}" for s in score_data if s["full_eval"]])
        trajectory = "[" + full_eval_scores + "]"
        logger.info(f"Full eval scores so far: {trajectory}")
        logger.info(f"Best full score so far: {best_score}")
        logger.info(
            f"{'=' * len(f'== Trial {trial.number + 1} / {adjusted_num_trials} - Minibatch Evaluation ==')}\n\n"
        )

    def _log_normal_eval(
        self,
        score,
        best_score,
        chosen_params,
        score_data,
        trial,
        num_trials,
        trial_logs,
        trial_num,
        valset,
        batch_size,
        candidate_program,
        total_eval_calls,
    ):
        trial_logs[trial_num]["full_eval_program_path"] = save_candidate_program(
            candidate_program, self.log_dir, trial_num
        )
        trial_logs[trial_num]["full_eval_score"] = score
        trial_logs[trial_num]["total_eval_calls_so_far"] = total_eval_calls
        trial_logs[trial_num]["full_eval_program"] = candidate_program.deepcopy()

        logger.info(f"Score: {score} with parameters {chosen_params}.")
        full_eval_scores = ", ".join([f"{s['score']}" for s in score_data if s["full_eval"]])
        logger.info(f"Scores so far: {'[' + full_eval_scores + ']'}")
        logger.info(f"Best score so far: {best_score}")
        logger.info(f"{'=' * len(f'===== Trial {trial.number + 1} / {num_trials} =====')}\n\n")

    def _select_and_insert_instructions_and_demos(
        self,
        candidate_program: Any,
        instruction_candidates: dict[int, list[str]],
        demo_candidates: list | None,
        trial: "optuna.trial.Trial",
        trial_logs: dict,
        trial_num: int,
    ) -> list[str]:
        chosen_params = []
        raw_chosen_params = {}

        for i, predictor in enumerate(candidate_program.predictors()):
            # Select instruction
            instruction_idx = trial.suggest_categorical(
                f"{i}_predictor_instruction", range(len(instruction_candidates[i]))
            )
            selected_instruction = instruction_candidates[i][instruction_idx]
            updated_signature = get_signature(predictor).with_instructions(selected_instruction)
            set_signature(predictor, updated_signature)
            trial_logs[trial_num][f"{i}_predictor_instruction"] = instruction_idx
            chosen_params.append(f"Predictor {i}: Instruction {instruction_idx}")
            raw_chosen_params[f"{i}_predictor_instruction"] = instruction_idx
            # Select demos if available
            if demo_candidates:
                demos_idx = trial.suggest_categorical(f"{i}_predictor_demos", range(len(demo_candidates[i])))
                predictor.demos = demo_candidates[i][demos_idx]
                trial_logs[trial_num][f"{i}_predictor_demos"] = demos_idx
                chosen_params.append(f"Predictor {i}: Few-Shot Set {demos_idx}")
                raw_chosen_params[f"{i}_predictor_demos"] = instruction_idx

        return chosen_params, raw_chosen_params

    def _get_param_distributions(self, program, instruction_candidates, demo_candidates):
        from optuna.distributions import CategoricalDistribution

        param_distributions = {}

        for i in range(len(instruction_candidates)):
            param_distributions[f"{i}_predictor_instruction"] = CategoricalDistribution(
                range(len(instruction_candidates[i]))
            )
            if demo_candidates:
                param_distributions[f"{i}_predictor_demos"] = CategoricalDistribution(range(len(demo_candidates[i])))

        return param_distributions

    def _perform_full_evaluation(
        self,
        trial_num: int,
        adjusted_num_trials: int,
        param_score_dict: dict,
        fully_evaled_param_combos: dict,
        evaluate: Evaluate,
        valset: list,
        trial_logs: dict,
        total_eval_calls: int,
        score_data,
        best_score: float,
        best_program: Any,
        study: "optuna.Study",
        instruction_candidates: list,
        demo_candidates: list,
    ):
        import optuna

        logger.info(f"===== Trial {trial_num + 1} / {adjusted_num_trials} - Full Evaluation =====")

        # Identify best program to evaluate fully
        highest_mean_program, mean_score, combo_key, params = get_program_with_highest_avg_score(
            param_score_dict, fully_evaled_param_combos
        )
        logger.info(f"Doing full eval on next top averaging program (Avg Score: {mean_score}) from minibatch trials...")
        full_eval_score = eval_candidate_program(len(valset), valset, highest_mean_program, evaluate, self.rng).score
        score_data.append({"score": full_eval_score, "program": highest_mean_program, "full_eval": True})

        # Log full eval as a trial so that optuna can learn from the new results
        trial = optuna.trial.create_trial(
            params=params,
            distributions=self._get_param_distributions(best_program, instruction_candidates, demo_candidates),
            value=full_eval_score,
        )
        study.add_trial(trial)

        # Log full evaluation results
        fully_evaled_param_combos[combo_key] = {
            "program": highest_mean_program,
            "score": full_eval_score,
        }
        total_eval_calls += len(valset)
        trial_logs[trial_num + 1] = {}
        trial_logs[trial_num + 1]["total_eval_calls_so_far"] = total_eval_calls
        trial_logs[trial_num + 1]["full_eval_program_path"] = save_candidate_program(
            program=highest_mean_program,
            log_dir=self.log_dir,
            trial_num=trial_num + 1,
            note="full_eval",
        )
        trial_logs[trial_num + 1]["full_eval_program"] = highest_mean_program
        trial_logs[trial_num + 1]["full_eval_score"] = full_eval_score

        # Update best score and program if necessary
        if full_eval_score > best_score:
            logger.info(f"{GREEN}New best full eval score!{ENDC} Score: {full_eval_score}")
            best_score = full_eval_score
            best_program = highest_mean_program.deepcopy()
        full_eval_scores = ", ".join([f"{s['score']}" for s in score_data if s["full_eval"]])
        trajectory = "[" + full_eval_scores + "]"
        logger.info(f"Full eval scores so far: {trajectory}")
        logger.info(f"Best full score so far: {best_score}")
        logger.info(len(f"===== Full Eval {len(fully_evaled_param_combos) + 1} =====") * "=")
        logger.info("\n")

        return best_score, best_program, total_eval_calls
```

## File: `teleprompt/random_search.py`
```
import random

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt.teleprompt import Teleprompter

from .bootstrap import BootstrapFewShot
from .vanilla import LabeledFewShot

# TODO: Don't forget dealing with the raw demos.
# TODO: Deal with the (pretty common) case of having a metric for filtering and a separate metric for eval.
# The metric itself may tell though by the presence of trace.

# TODO: This function should take a max_budget and max_teacher_budget. That's in the number of program calls.
# In this case, max_student_budget is max_budget - max_teacher_budget.
# For max_teacher_budget, this will just limit the total number of things we bootstrap.
# This can end up implicitly defining the number of candidate programs (i.e., stop when runs out). Cap at 16.
# For max_student_budget, this will be a more upfront calculation.
# Right now, it can also just induce the number of candidate programs. Later, it could be used more interestingly
# for selective early stopping.
# Progressive elimination sounds about right: after 50 examples, drop bottom third, after 100, another third, etc.
# until only 3--5 are left for the end. Could also be systematic and add (earlier) stopping based on error bounds.
# In general, though, the early filtering is just saying: either there are some really bad ones, or some really really
# good ones, or most things are pretty close. In all of these cases, dropping the bottom third is not going to hurt.


class BootstrapFewShotWithRandomSearch(Teleprompter):
    def __init__(
        self,
        metric,
        teacher_settings=None,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_rounds=1,
        num_candidate_programs=16,
        num_threads=None,
        max_errors=None,
        stop_at_score=None,
        metric_threshold=None,
    ):
        self.metric = metric
        self.teacher_settings = teacher_settings or {}
        self.max_rounds = max_rounds

        self.num_threads = num_threads
        self.stop_at_score = stop_at_score
        self.metric_threshold = metric_threshold
        self.min_num_samples = 1
        self.max_num_samples = max_bootstrapped_demos
        self.max_errors = max_errors
        self.num_candidate_sets = num_candidate_programs
        self.max_labeled_demos = max_labeled_demos

        print(f"Going to sample between {self.min_num_samples} and {self.max_num_samples} traces per predictor.")
        print(f"Will attempt to bootstrap {self.num_candidate_sets} candidate sets.")

    def compile(self, student, *, teacher=None, trainset, valset=None, restrict=None, labeled_sample=True):
        self.trainset = trainset
        self.valset = valset or trainset  # TODO: FIXME: Note this choice.

        effective_max_errors = (
            self.max_errors
            if self.max_errors is not None
            else dspy.settings.max_errors
        )

        scores = []
        all_subscores = []
        score_data = []

        for seed in range(-3, self.num_candidate_sets):
            if (restrict is not None) and (seed not in restrict):
                continue

            trainset_copy = list(self.trainset)

            if seed == -3:
                # zero-shot
                program = student.reset_copy()

            elif seed == -2:
                # labels only
                teleprompter = LabeledFewShot(k=self.max_labeled_demos)
                program = teleprompter.compile(student, trainset=trainset_copy, sample=labeled_sample)

            elif seed == -1:
                # unshuffled few-shot
                optimizer = BootstrapFewShot(
                    metric=self.metric,
                    metric_threshold=self.metric_threshold,
                    max_bootstrapped_demos=self.max_num_samples,
                    max_labeled_demos=self.max_labeled_demos,
                    teacher_settings=self.teacher_settings,
                    max_rounds=self.max_rounds,
                    max_errors=effective_max_errors,
                )
                program = optimizer.compile(student, teacher=teacher, trainset=trainset_copy)

            else:
                assert seed >= 0, seed

                random.Random(seed).shuffle(trainset_copy)
                size = random.Random(seed).randint(self.min_num_samples, self.max_num_samples)

                optimizer = BootstrapFewShot(
                    metric=self.metric,
                    metric_threshold=self.metric_threshold,
                    max_bootstrapped_demos=size,
                    max_labeled_demos=self.max_labeled_demos,
                    teacher_settings=self.teacher_settings,
                    max_rounds=self.max_rounds,
                    max_errors=effective_max_errors,
                )

                program = optimizer.compile(student, teacher=teacher, trainset=trainset_copy)

            evaluate = Evaluate(
                devset=self.valset,
                metric=self.metric,
                num_threads=self.num_threads,
                max_errors=effective_max_errors,
                display_table=False,
                display_progress=True,
            )

            result = evaluate(program)

            score, subscores = result.score, [output[2] for output in result.results]

            all_subscores.append(subscores)

            ############ Assertion-aware Optimization ############
            if hasattr(program, "_suggest_failures"):
                score = score - program._suggest_failures * 0.2
            if hasattr(program, "_assert_failures"):
                score = 0 if program._assert_failures > 0 else score
            ######################################################

            if len(scores) == 0 or score > max(scores):
                print("New best score:", score, "for seed", seed)
                best_program = program

            scores.append(score)
            print(f"Scores so far: {scores}")
            print(f"Best score so far: {max(scores)}")

            score_data.append({"score": score, "subscores": subscores, "seed": seed, "program": program})

            if self.stop_at_score is not None and score >= self.stop_at_score:
                print(f"Stopping early because score {score} is >= stop_at_score {self.stop_at_score}")
                break

        # To best program, attach all program candidates in decreasing average score
        best_program.candidate_programs = score_data
        best_program.candidate_programs = sorted(
            best_program.candidate_programs, key=lambda x: x["score"], reverse=True
        )

        print(f"{len(best_program.candidate_programs)} candidate programs found.")

        return best_program


# sample between 4 and 10 examples from traces
# TODO: FIXME: The max number of demos should be determined in part by the LM's tokenizer + max_length.
# This does require executing the program, or at least the predictor.
# # # # # # (Actually we can just combine the token counts of the traces, when formatted via signature/adapter).
# Alternatively, we can keep track of the (zero-shot) number of tokens when we bootstrap.
# As another option, we can just try a wide range and handle failures as penalties on the score.
# The number "24" of traces to collect can also be affected. If we only need 3x10, some overlap is ok.
# We can also consider having short_demos and long_demos.
```

## File: `teleprompt/signature_opt.py`
```
from .copro_optimizer import COPRO

"""
===============================================================
DEPRECATED!!!
PLEASE USE COPRO INSTEAD.
===============================================================

USAGE SUGGESTIONS:

The following code can be used to compile a optimized signature teleprompter, and evaluate it on an end task:

teleprompter = SignatureOptimizer(prompt_model=prompt_model, metric=metric, breadth=BREADTH, depth=DEPTH, init_temperature=INIT_TEMPERATURE)
kwargs = dict(num_threads=NUM_THREADS, display_progress=True, display_table=0)
compiled_prompt_opt = teleprompter.compile(program.deepcopy(), devset=devset[:DEV_NUM], eval_kwargs=kwargs)
eval_score = evaluate(compiled_prompt_opt, devset=evalset[:EVAL_NUM], **kwargs)

Note that this teleprompter takes in the following parameters:

* prompt_model: The model used for prompt generation. When unspecified, defaults to the model set in settings (ie. dspy.settings.configure(lm=task_model)).
* metric: The task metric used for optimization.
* breadth: The number of new prompts to generate at each iteration. Default=10.
* depth: The number of times we should ask our prompt model to generate new prompts, with the history of the past prompts as input. Default=3.
* init_temperature: The temperature used to generate new prompts. Higher roughly equals more creative. Default=1.4.
* verbose: Tells the method whether or not to print intermediate steps.
* track_stats: Tells the method whether or not to track statistics about the optimization process.
                If True, the method will track the following statistics:
                    * results_best: The min,max,avg,stddev of top 10 scores for each predictor at each depth.
                    * results_latest: The min,max,avg,stddev of newest prompt scores for each predictor at each depth.
                    * total_calls: The total number of calls to the task metric.
                These statistics will be returned as attributes of the best program.
"""


class SignatureOptimizer(COPRO):
    def __init__(
        self,
        prompt_model=None,
        metric=None,
        breadth=10,
        depth=3,
        init_temperature=1.4,
        verbose=False,
        track_stats=False,
    ):
        print(
            "\u001b[31m[WARNING] SignatureOptimizer has been deprecated and replaced with COPRO.  SignatureOptimizer will be removed in a future release. \u001b[31m",
        )
        super().__init__(prompt_model, metric, breadth, depth, init_temperature, verbose, track_stats)

    def compile(self, student, *, devset, eval_kwargs):
        return super().compile(student, trainset=devset, eval_kwargs=eval_kwargs)
```

## File: `teleprompt/simba_utils.py`
```
import inspect
import logging
import textwrap
from typing import Callable

import ujson

import dspy
from dspy.adapters.utils import get_field_description_string
from dspy.signatures import InputField, OutputField

logger = logging.getLogger(__name__)


def prepare_models_for_resampling(program: dspy.Module, n: int):
    lm = program.get_lm() or dspy.settings.lm
    temps = [lm.kwargs["temperature"]] + [0.5 + i * (0.5 / n) for i in range(n)]
    temps = list(dict.fromkeys(temps))[:n]
    return [lm.copy(temperature=t) for t in temps]


def wrap_program(program: dspy.Module, metric: Callable):
    def wrapped_program(example):
        with dspy.context(trace=[]):
            prediction, trace, score = None, None, 0.0
            try:
                prediction = program(**example.inputs())
            except Exception as e:
                print(e)
            trace = dspy.settings.trace.copy()

        try:
            score = metric(example, prediction)
        except Exception as e:
            print(e)

        # Include the `example` in the output for subsequent usage in buckets/strategies.
        return {
            "prediction": prediction,
            "trace": trace,
            "score": score,
            "example": example
        }

    return wrapped_program



def append_a_demo(demo_input_field_maxlen):
    def append_a_demo_(bucket, system, **kwargs):
        predictor2name, name2predictor = kwargs["predictor2name"], kwargs["name2predictor"]

        trace = bucket[0]["trace"]
        name2demo = {}

        for step in trace:
            predictor, _inputs, _outputs = step

            for k, v in _inputs.items():
                if demo_input_field_maxlen and len(str(v)) > demo_input_field_maxlen:
                    _inputs[k] = f"{str(v)[:demo_input_field_maxlen]}\n\t\t... <TRUNCATED FOR BREVITY>"

            demo = dspy.Example(augmented=True, **_inputs, **_outputs)
            name = predictor2name[id(predictor)]
            name2demo[name] = demo  # keep the last demo for each predictor

        for name, demo in name2demo.items():
            predictor = name2predictor[name]
            predictor.demos.append(demo)

        logger.info(f"Added {len(name2demo)} demos (one each) across all predictors.")
        return True

    return append_a_demo_


def append_a_rule(bucket, system, **kwargs):
    predictor2name = kwargs["predictor2name"]
    batch_10p_score, batch_90p_score = kwargs["batch_10p_score"], kwargs["batch_90p_score"]

    module_names = [name for name, _ in system.named_predictors()]
    good, bad = bucket[0], bucket[-1]
    example = good["example"]

    if good["score"] < batch_10p_score or bad["score"] > batch_90p_score:
        logger.info(f"Skipping rule generation as good score {good['score']} is below the 10th percentile "
                    f"*or* bad score {bad['score']} is above the 90th percentile.")
        return False

    if good["score"] <= bad["score"]:
        if good["score"] > batch_90p_score:
            bad["trace"] = []
            bad["score"] = "N/A"
            bad["prediction"] = {"N/A": "Prediction not available"}
        else:
            good["trace"] = []
            good["score"] = "N/A"
            good["prediction"] = {"N/A": "Prediction not available"}

    better_trajectory = [
        {"module_name": predictor2name[id(p)], "inputs": i, "outputs": dict(o)}
        for p, i, o in good["trace"]
    ]
    worse_trajectory = [
        {"module_name": predictor2name[id(p)], "inputs": i, "outputs": dict(o)}
        for p, i, o in bad["trace"]
    ]

    kwargs = {
        "program_code": inspect.getsource(system.__class__),
        "modules_defn": inspect_modules(system),
        "program_inputs": {**example.inputs()},
        "oracle_metadata": {**example.labels()},
        "better_program_trajectory": better_trajectory,
        "better_program_outputs": dict(good["prediction"]),
        "worse_program_trajectory": worse_trajectory,
        "worse_program_outputs": dict(bad["prediction"] or {}),
        "worse_reward_value": bad["score"],
        "better_reward_value": good["score"],
        "module_names": module_names,
    }

    kwargs = {k: v if isinstance(v, str) else ujson.dumps(recursive_mask(v), indent=2)
              for k, v in kwargs.items()}
    advice = dspy.Predict(OfferFeedback)(**kwargs).module_advice

    for name, predictor in system.named_predictors():
        if name in advice:
            logger.info(f"Advice for {name}: {advice[name]}")
            instructions = predictor.signature.instructions + "\n\n" + advice[name]
            predictor.signature = predictor.signature.with_instructions(instructions)

    return True

class OfferFeedback(dspy.Signature):
    """
    You will be given two trajectories of an LLM-driven program's execution. Your goal is to help the program's modules
    build up experience on how to maximize the reward value assigned to the program's outputs if it were to receive
    similar inputs in the future.

    The module won't see its own history. It will rely on your advice balancing being concrete and being generalizable.

    In your advice:
    - Avoid boilerplate. Offer advice that would change the module's behavior for the better in the future.
    - Ensure that advice offered to a module M is specific to that M's specific sub-task, not the overall program.
    - Rely on contrasting the behavior of the worse trajectory against the better trajectory in making recommendations.
    - Ensure each unique module name appears exactly once as a key in the advice dictionary.
    """

    program_code: str = InputField(desc="The code of the program that we are analyzing")
    modules_defn: str = InputField(desc="The definition of each module in the program, including its I/O")
    program_inputs: str = InputField(desc="The inputs to the program that we are analyzing")
    oracle_metadata: str = InputField(desc="Any (hidden) metadata about the training set instance we're analyzing")
    worse_program_trajectory: str = InputField(
        desc="The trajectory of the program's execution, showing each module's I/O"
    )
    worse_program_outputs: str = InputField(desc="The outputs of the program that we are analyzing")
    worse_reward_value: float = InputField(desc="The reward value assigned to the program's outputs")
    better_program_trajectory: str = InputField(
        desc="The trajectory of the program's execution, showing each module's I/O"
    )
    better_program_outputs: str = InputField(desc="The outputs of the program that we are analyzing")
    better_reward_value: float = InputField(desc="The reward value assigned to the program's outputs")
    module_names: list[str] = InputField(desc="The names of the modules in the program, for which we seek advice")
    discussion: str = OutputField(desc="Discussing blame of where each module went wrong, if it did")
    module_advice: dict[str, str] = OutputField(
        desc="For each module, describe very concretely: If the module receives ${description of input or patterns "
        "therein}, then it should ${description of content, behavior, or strategies to adopt and/or others to avoid}. "
        "Basically, your advice be such that if the module has access to your tip, it would be much more likely to act "
        "like the successful trajectory rather than the lower-scoring trajectory."
    )


def inspect_modules(program):
    separator = "-" * 80
    output = [separator]

    for name, predictor in program.named_predictors():
        signature = predictor.signature
        instructions = textwrap.dedent(signature.instructions)
        instructions = ("\n" + "\t" * 2).join([""] + instructions.splitlines())

        output.append(f"Module {name}")
        output.append("\n\tInput Fields:")
        output.append(("\n" + "\t" * 2).join([""] + get_field_description_string(signature.input_fields).splitlines()))
        output.append("\tOutput Fields:")
        output.append(("\n" + "\t" * 2).join([""] + get_field_description_string(signature.output_fields).splitlines()))
        output.append(f"\tOriginal Instructions: {instructions}")
        output.append(separator)

    return "\n".join([o.strip("\n") for o in output])


def recursive_mask(o):
    # If the object is already serializable, return it.
    try:
        ujson.dumps(o)
        return o
    except TypeError:
        pass

    # If it's a dictionary, apply recursively to its values.
    if isinstance(o, dict):
        return {k: recursive_mask(v) for k, v in o.items()}
    # If it's a list, apply recursively.
    elif isinstance(o, list):
        return [recursive_mask(v) for v in o]
    # If it's a tuple, apply recursively.
    elif isinstance(o, tuple):
        return tuple(recursive_mask(v) for v in o)
    # Otherwise, replace it with a placeholder string (or use repr(o)).
    else:
        return f"<non-serializable: {type(o).__name__}>"
```

## File: `teleprompt/simba.py`
```
import logging
import random
from typing import Callable

import numpy as np

import dspy
from dspy.teleprompt.simba_utils import append_a_demo, append_a_rule, prepare_models_for_resampling, wrap_program
from dspy.teleprompt.teleprompt import Teleprompter

logger = logging.getLogger(__name__)


# Stochastic Introspective Mini-Batch Ascent
class SIMBA(Teleprompter):
    def __init__(
        self,
        *,
        metric: Callable,
        bsize=32,
        num_candidates=6,
        max_steps=8,
        max_demos=4,
        demo_input_field_maxlen=100_000,
        num_threads=None,
        temperature_for_sampling=0.2,
        temperature_for_candidates=0.2,
    ):
        """
        Initializes SIMBA.

        Args:
            metric (Callable): A function that takes an Example and a prediction_dict
                as input and returns a float.
            bsize (int, optional): Mini-batch size. Defaults to 32.
            num_candidates (int, optional): Number of new candidate programs to produce
                per iteration. Defaults to 6.
            max_steps (int, optional): Number of optimization steps to run. Defaults to 8.
            max_demos (int, optional): Maximum number of demos a predictor can hold
                before dropping some. Defaults to 4.
            demo_input_field_maxlen (int, optional): Maximum number of characters to keep
                in an input field when building a new demo. Defaults to 100,000.
            num_threads (int, optional): Number of threads for parallel execution.
                Defaults to None.
            temperature_for_sampling (float, optional): Temperature used for picking
                programs during the trajectory-sampling step. Defaults to 0.2.
            temperature_for_candidates (float, optional): Temperature used for picking
                the source program for building new candidates. Defaults to 0.2.
        """
        self.metric = metric
        self.bsize = bsize
        self.num_candidates = num_candidates
        self.max_steps = max_steps
        self.max_demos = max_demos
        self.demo_input_field_maxlen = demo_input_field_maxlen
        self.num_threads = num_threads

        self.temperature_for_sampling = temperature_for_sampling
        self.temperature_for_candidates = temperature_for_candidates

        if self.max_demos > 0:
            self.strategies = [append_a_demo(demo_input_field_maxlen), append_a_rule]
        else:
            self.strategies = [append_a_rule]

    def compile(self, student: dspy.Module, *, trainset: list[dspy.Example], seed: int = 0):
        # Basic checks
        assert len(trainset) >= self.bsize, f"Trainset too small: {len(trainset)} < {self.bsize}"

        # Initialize RNG
        rng = random.Random(seed)
        rng_np = np.random.default_rng(seed)

        programs = []
        program_scores = {}
        next_program_idx = 0

        # Helper functions
        def calc_average_score(prog_idx: int) -> float:
            scores = program_scores.get(prog_idx, [])
            if not scores:
                return 0.0
            return sum(scores) / len(scores)

        def top_k_plus_baseline(k: int) -> list[int]:
            # Sort all programs by descending average score
            scored_programs = sorted(programs, key=lambda p: calc_average_score(p.simba_idx), reverse=True)
            top_k = [p.simba_idx for p in scored_programs[:k]]
            # Ensure baseline=0 is in there:
            if 0 not in top_k and len(top_k) > 0:
                top_k[-1] = 0
            return list(dict.fromkeys(top_k))

        def softmax_sample(rng_obj: random.Random, program_idxs: list[int], temperature: float) -> int:
            if not program_idxs:
                raise ValueError("No programs available for softmax sampling.")

            # Unnormalized weights
            scores = [calc_average_score(idx) for idx in program_idxs]
            exps = [np.exp(s / temperature) for s in scores]
            sum_exps = sum(exps)
            if sum_exps <= 0:
                # Fallback: uniform if all exps are zero
                return rng_obj.choice(program_idxs)

            # Weighted random choice
            probs = [val / sum_exps for val in exps]
            return rng_obj.choices(program_idxs, weights=probs, k=1)[0]

        def register_new_program(prog: dspy.Module, score_list: list[float]):
            nonlocal next_program_idx
            next_program_idx += 1
            new_idx = next_program_idx
            prog.simba_idx = new_idx
            programs.append(prog)
            program_scores[new_idx] = score_list

        # Initialize the baseline program: index=0
        student = student.deepcopy()
        student.simba_idx = 0
        programs.append(student)
        program_scores[0] = []

        winning_programs = [student]

        # Data shuffling
        data_indices = list(range(len(trainset)))
        rng.shuffle(data_indices)
        instance_idx = 0

        # Parallel runner
        run_parallel = dspy.Parallel(access_examples=False, num_threads=self.num_threads)

        trial_logs = {}
        for batch_idx in range(self.max_steps):
            trial_logs[batch_idx] = {}

            logger.info(f"Starting batch {batch_idx+1} of {self.max_steps}.")

            # STEP 1: Get next batch
            if instance_idx + self.bsize > len(trainset):
                rng.shuffle(data_indices)
                instance_idx = 0

            batch_indices = data_indices[instance_idx : instance_idx + self.bsize]
            batch = [trainset[i] for i in batch_indices]
            instance_idx += self.bsize

            # We'll generate (program, model) pairs for the trajectory sampling.
            # Prepare distinct LMs (with different temperatures, etc.) from the baseline=programs[0].
            models = prepare_models_for_resampling(programs[0], self.num_candidates)
            top_programs = top_k_plus_baseline(self.num_candidates)

            exec_pairs = []
            predictor2name = {}

            # For each model, for each example, pick a program from the pool via softmax
            for model in models:
                for example in batch:
                    chosen_prog_idx = softmax_sample(rng, top_programs, self.temperature_for_sampling)
                    candidate_system = programs[chosen_prog_idx].deepcopy()
                    candidate_system.set_lm(model)

                    for name, predictor in candidate_system.named_predictors():
                        predictor2name[id(predictor)] = name

                    # Use the special wrap that includes the 'example' in the output
                    wrapped_candidate_system = wrap_program(candidate_system, self.metric)
                    exec_pairs.append((wrapped_candidate_system, example))

            # STEP 2: Execute
            logger.info(f"Sampling program trajectories on {self.bsize} examples x {self.num_candidates} samples.")
            outputs = run_parallel(exec_pairs)
            assert len(outputs) == len(exec_pairs) == self.bsize * self.num_candidates

            # STEP 3: Sort the training buckets by (max-to-min gap, max score, and max-to-avg gap).
            buckets = []
            largest_max_to_avg_gap = float("-inf")
            batch_10th_percentile_score = np.percentile([float(o["score"]) for o in outputs], 10)
            batch_90th_percentile_score = np.percentile([float(o["score"]) for o in outputs], 90)

            # We'll chunk `outputs` by example index, each chunk has length = num_candidates
            for idx, _ in enumerate(batch):
                # gather all results for this example
                bucket = [outputs[i] for i in range(idx, len(outputs), self.bsize)]
                bucket.sort(key=lambda x: x["score"], reverse=True)

                max_score = float(bucket[0]["score"])
                min_score = float(bucket[-1]["score"])
                avg_score = sum(x["score"] for x in bucket) / len(bucket)
                max_to_min_gap = max_score - min_score
                max_to_avg_gap = max_score - avg_score
                if max_to_avg_gap > largest_max_to_avg_gap:
                    largest_max_to_avg_gap = max_to_avg_gap

                buckets.append((bucket, (max_to_min_gap, max_score, max_to_avg_gap)))

            # sort the buckets
            buckets.sort(key=lambda x: x[1], reverse=True)

            # Baseline for the batch is just the average of all runs
            all_scores_in_this_batch = [o["score"] for o in outputs]
            baseline_score = sum(all_scores_in_this_batch) / len(all_scores_in_this_batch)
            logger.info(f"Batch {batch_idx+1}: Baseline mini-batch score: {baseline_score}\n")

            # STEP 4: Build new candidate programs by applying a strategy to some top buckets.
            system_candidates = []
            for bucket_idx, (bucket, bucket_stats) in enumerate(buckets):
                max_to_min_gap, max_score, max_to_avg_gap = bucket_stats
                logger.info(
                    f"Batch {batch_idx+1}: Processing bucket #{bucket_idx+1}, with max score {max_score}, "
                    f"max-to-min gap {max_to_min_gap}, and max-to-avg gap {max_to_avg_gap}."
                )

                # pick source program
                src_prog_idx = softmax_sample(
                    rng, top_k_plus_baseline(self.num_candidates), self.temperature_for_candidates
                )
                system_candidate = programs[src_prog_idx].deepcopy()

                # Drop some demos from each predictor
                name2predictor = {}
                num_demos_list = []

                max_demos_tmp = self.max_demos if self.max_demos > 0 else 3

                for name, predictor in system_candidate.named_predictors():
                    name2predictor[name] = predictor
                    num_demos_list.append(len(predictor.demos))

                num_demos = max(num_demos_list) if num_demos_list else 0
                num_demos_to_drop = max(rng_np.poisson(num_demos / max_demos_tmp), int(num_demos >= max_demos_tmp))
                num_demos_to_drop = min(num_demos_to_drop, num_demos)
                demos_to_drop = [rng.randrange(num_demos) for _ in range(num_demos_to_drop)]

                for _, predictor in name2predictor.items():
                    predictor.demos = [demo for idxd, demo in enumerate(predictor.demos) if idxd not in demos_to_drop]

                # Pick a strategy
                strategy = rng.choice(self.strategies)
                logger.info(
                    f"Batch {batch_idx+1}: Invoking strategy: {strategy.__name__}"
                    + (f", having dropped {num_demos_to_drop} demos per predictor" if num_demos_to_drop else "")
                )

                try:
                    strategy(
                        bucket,
                        system_candidate,
                        predictor2name=predictor2name,
                        name2predictor=name2predictor,
                        batch_10p_score=batch_10th_percentile_score,
                        batch_90p_score=batch_90th_percentile_score,
                    )
                except Exception as e:
                    logger.error(f"Strategy failed with error: {e}")
                    continue

                system_candidates.append(system_candidate)
                logger.info("\n")

                if len(system_candidates) >= self.num_candidates + 1:
                    break

            # STEP 5: Evaluate these new system_candidates on the same mini-batch
            logger.info(f"Batch {batch_idx+1}: Evaluating {len(system_candidates)} programs on {self.bsize} examples.")

            exec_pairs = [(wrap_program(sys, self.metric), ex) for sys in system_candidates for ex in batch]
            outputs = run_parallel(exec_pairs)
            assert len(outputs) == len(exec_pairs) == len(system_candidates) * self.bsize

            # STEP 6: Compute average mini-batch scores for each new candidate
            candidate_scores = []
            for idx_cand, _ in enumerate(system_candidates):
                start = idx_cand * self.bsize
                end = (idx_cand + 1) * self.bsize
                sys_scores = [outputs[i]["score"] for i in range(start, end)]
                avg_sys_score = sum(sys_scores) / len(sys_scores)
                candidate_scores.append(avg_sys_score)

            logger.info(
                f"Scores after {batch_idx+1} batches: {candidate_scores}, "
                f"Best: {max(candidate_scores) if candidate_scores else 'N/A'}\n"
            )

            # STEP 7: Select the best among these new ones for "winning" record
            if candidate_scores:
                best_idx_among_candidates = candidate_scores.index(max(candidate_scores))
                best_program = system_candidates[best_idx_among_candidates]
                winning_programs.append(best_program.deepcopy())

            # STEP 8: Register all new candidate systems in our global pool
            for idx_cand, cand_sys in enumerate(system_candidates):
                start = idx_cand * self.bsize
                end = (idx_cand + 1) * self.bsize
                sys_scores = [outputs[i]["score"] for i in range(start, end)]
                register_new_program(cand_sys, sys_scores)

        M = len(winning_programs) - 1  # noqa: N806
        N = self.num_candidates + 1  # noqa: N806
        if M < 1:
            program_idxs = [0] * N
        else:
            program_idxs = [round(i * M / (N - 1)) for i in range(N)]

        program_idxs = list(dict.fromkeys(program_idxs))

        candidate_programs = [winning_programs[i].deepcopy() for i in program_idxs]
        logger.info(f"VALIDATION: Evaluating {len(candidate_programs)} programs on the full trainset.")
        exec_pairs = [(wrap_program(sys, self.metric), ex) for sys in candidate_programs for ex in trainset]
        outputs = run_parallel(exec_pairs)

        scores = []
        for idx_prog, _ in enumerate(candidate_programs):
            start = idx_prog * len(trainset)
            end = (idx_prog + 1) * len(trainset)
            sys_scores = [outputs[i]["score"] for i in range(start, end)]
            avg_score = sum(sys_scores) / len(sys_scores) if sys_scores else 0.0
            scores.append(avg_score)
            if idx_prog != 0:
                trial_logs[idx_prog - 1]["train_score"] = avg_score

        # Build sorted list of {"score", "program"} dicts
        assert len(scores) == len(candidate_programs)
        candidate_data = [{"score": s, "program": p} for s, p in zip(scores, candidate_programs, strict=False)]
        candidate_data.sort(key=lambda x: x["score"], reverse=True)

        best_idx = scores.index(max(scores)) if scores else 0
        best_program = candidate_programs[best_idx].deepcopy()
        logger.info(
            f"Final trainset scores: {scores}, Best: {max(scores) if scores else 'N/A'} "
            f"(at index {best_idx if scores else 'N/A'})\n\n\n"
        )

        # Attach sorted, scored candidates & logs
        best_program.candidate_programs = candidate_data
        best_program.trial_logs = trial_logs

        return best_program
```

## File: `teleprompt/teleprompt_optuna.py`
```
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt.teleprompt import Teleprompter

from .bootstrap import BootstrapFewShot


class BootstrapFewShotWithOptuna(Teleprompter):
    def __init__(
        self,
        metric,
        teacher_settings=None,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_rounds=1,
        num_candidate_programs=16,
        num_threads=None,
    ):
        self.metric = metric
        self.teacher_settings = teacher_settings or {}
        self.max_rounds = max_rounds
        self.num_threads = num_threads
        self.min_num_samples = 1
        self.max_num_samples = max_bootstrapped_demos
        self.num_candidate_sets = num_candidate_programs
        # self.max_num_traces = 1 + int(max_bootstrapped_demos / 2.0 * self.num_candidate_sets)

        # Semi-hacky way to get the parent class's _bootstrap function to stop early.
        # self.max_bootstrapped_demos = self.max_num_traces
        self.max_labeled_demos = max_labeled_demos

        print("Going to sample between", self.min_num_samples, "and", self.max_num_samples, "traces per predictor.")
        # print("Going to sample", self.max_num_traces, "traces in total.")
        print("Will attempt to train", self.num_candidate_sets, "candidate sets.")

    def objective(self, trial):
        program2 = self.student.reset_copy()
        for (name, compiled_predictor), (_, program2_predictor) in zip(
            self.compiled_teleprompter.named_predictors(), program2.named_predictors(), strict=False,
        ):
            all_demos = compiled_predictor.demos
            demo_index = trial.suggest_int(f"demo_index_for_{name}", 0, len(all_demos) - 1)
            selected_demo = dict(all_demos[demo_index])
            program2_predictor.demos = [selected_demo]
        evaluate = Evaluate(
            devset=self.valset,
            metric=self.metric,
            num_threads=self.num_threads,
            display_table=False,
            display_progress=True,
        )
        result = evaluate(program2)
        trial.set_user_attr("program", program2)
        return result.score

    def compile(self, student, *, teacher=None, max_demos, trainset, valset=None):
        import optuna
        self.trainset = trainset
        self.valset = valset or trainset
        self.student = student.reset_copy()
        self.teacher = teacher.deepcopy() if teacher is not None else student.reset_copy()
        teleprompter_optimize = BootstrapFewShot(
            metric=self.metric,
            max_bootstrapped_demos=max_demos,
            max_labeled_demos=self.max_labeled_demos,
            teacher_settings=self.teacher_settings,
            max_rounds=self.max_rounds,
        )
        self.compiled_teleprompter = teleprompter_optimize.compile(
            self.student, teacher=self.teacher, trainset=self.trainset,
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.num_candidate_sets)
        best_program = study.trials[study.best_trial.number].user_attrs["program"]
        print("Best score:", study.best_value)
        print("Best program:", best_program)
        return best_program
```

## File: `teleprompt/teleprompt.py`
```
from typing import Any

from dspy.primitives import Example, Module


class Teleprompter:
    def __init__(self):
        pass

    def compile(self, student: Module, *, trainset: list[Example], teacher: Module | None = None, valset: list[Example] | None = None, **kwargs) -> Module:
        """
        Optimize the student program.

        Args:
            student: The student program to optimize.
            trainset: The training set to use for optimization.
            teacher: The teacher program to use for optimization.
            valset: The validation set to use for optimization.

        Returns:
            The optimized student program.
        """
        raise NotImplementedError

    def get_params(self) -> dict[str, Any]:
        """
        Get the parameters of the teleprompter.

        Returns:
            The parameters of the teleprompter.
        """
        return self.__dict__
```

## File: `teleprompt/utils.py`
```
import inspect
import logging
import math
import os
import random
import shutil
import sys

import numpy as np

try:
    from IPython.core.magics.code import extract_symbols
except ImportError:
    # Won't be able to read code from juptyer notebooks
    extract_symbols = None

import dspy
from dspy.teleprompt.bootstrap import BootstrapFewShot, LabeledFewShot

"""
This file consists of helper functions for our variety of optimizers.
"""

### OPTIMIZER TRAINING UTILS ###

logger = logging.getLogger(__name__)


def create_minibatch(trainset, batch_size=50, rng=None):
    """Create a minibatch from the trainset."""

    # Ensure batch_size isn't larger than the size of the dataset
    batch_size = min(batch_size, len(trainset))

    # If no RNG is provided, fall back to the global random instance
    rng = rng or random

    # Randomly sample indices for the mini-batch using the provided rng
    sampled_indices = rng.sample(range(len(trainset)), batch_size)

    # Create the mini-batch using the sampled indices
    minibatch = [trainset[i] for i in sampled_indices]

    return minibatch


def eval_candidate_program(batch_size, trainset, candidate_program, evaluate, rng=None):
    """Evaluate a candidate program on the trainset, using the specified batch size."""

    try:
        # Evaluate on the full trainset
        if batch_size >= len(trainset):
            return evaluate(candidate_program, devset=trainset, callback_metadata={"metric_key": "eval_full"})
        # Or evaluate on a minibatch
        else:
            return evaluate(
                candidate_program,
                devset=create_minibatch(trainset, batch_size, rng),
                callback_metadata={"metric_key": "eval_minibatch"}
            )
    except Exception:
        logger.error("An exception occurred during evaluation", exc_info=True)
        # TODO: Handle this better, as -ve scores are possible
        return dspy.Prediction(score=0.0, results=[])


def eval_candidate_program_with_pruning(
    trial,
    trial_logs,
    trainset,
    candidate_program,
    evaluate,
    trial_num,
    batch_size=100,
):
    """Evaluation of candidate_program with pruning implemented"""

    # Evaluate with the new prompts
    total_score = 0
    num_batches = math.ceil(len(trainset) / batch_size)
    total_eval_size = 0

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(trainset))
        split_trainset = trainset[start_index:end_index]
        split_score = evaluate(
            candidate_program,
            devset=split_trainset,
            display_table=0,
        )
        print(f"{i}st split score: {split_score}")
        total_eval_size += len(split_trainset)

        total_score += split_score * len(split_trainset)
        curr_weighted_avg_score = total_score / min((i + 1) * batch_size, len(trainset))
        print(f"curr average score: {curr_weighted_avg_score}")

        trial.report(curr_weighted_avg_score, i)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            print("Trial pruned.")
            trial_logs[trial_num]["score"] = curr_weighted_avg_score
            trial_logs[trial_num]["num_eval_calls"] = total_eval_size
            trial_logs[trial_num]["pruned"] = True
            return curr_weighted_avg_score, trial_logs, total_eval_size, True

    print(f"Fully evaled score: {curr_weighted_avg_score}")
    score = curr_weighted_avg_score

    trial_logs[trial_num]["full_eval"] = False
    trial_logs[trial_num]["score"] = score
    trial_logs[trial_num]["pruned"] = False
    return score, trial_logs, total_eval_size, False


def get_program_with_highest_avg_score(param_score_dict, fully_evaled_param_combos):
    """Used as a helper function for bayesian + minibatching optimizers. Returns the program with the highest average score from the batches evaluated so far."""

    # Calculate the mean for each combination of categorical parameters, based on past trials
    results = []
    for key, values in param_score_dict.items():
        scores = np.array([v[0] for v in values])
        mean = np.average(scores)
        program = values[0][1]
        params = values[0][2]
        results.append((key, mean, program, params))

    # Sort results by the mean
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    # Find the combination with the highest mean, skip fully evaluated ones
    for combination in sorted_results:
        key, mean, program, params = combination

        if key in fully_evaled_param_combos:
            continue

        return program, mean, key, params

    # If no valid program is found, we return the last valid one that we found
    return program, mean, key, params


def calculate_last_n_proposed_quality(
    base_program,
    trial_logs,
    evaluate,
    trainset,
    devset,
    n,
):
    """
    Calculate the average and best quality of the last n programs proposed. This is useful for seeing if our proposals
    are actually 'improving' overtime or not.
    """
    # Get the trials from the last n keys in trial logs
    last_n_trial_nums = list(trial_logs.keys())[-n:]

    # Calculate the average and best score of these trials
    # if num_eval_calls in the trial is less than the trainset, throw a not-implemented error for now
    total_train_score = 0
    best_train_score = 0
    total_dev_score = 0
    best_dev_score = 0
    for trial_num in last_n_trial_nums:
        full_eval = trial_logs[trial_num]["full_eval"]
        if not full_eval:
            raise NotImplementedError(
                "Still need to implement non full eval handling in calculate_last_n_proposed_quality",
            )
        train_score = trial_logs[trial_num]["score"]
        program = base_program.deepcopy()
        program.load(trial_logs[trial_num]["program_path"])

        dev_score = evaluate(program, devset=devset)

        total_train_score += train_score
        total_dev_score += dev_score
        if train_score > best_train_score:
            best_train_score = train_score
            best_dev_score = dev_score

    return best_train_score, total_train_score / n, best_dev_score, total_dev_score / n


### LOGGING UTILS ###


def get_task_model_history_for_full_example(
    candidate_program,
    task_model,
    devset,
    evaluate,
):
    """Get a full trace of the task model's history for a given candidate program."""
    _ = evaluate(candidate_program, devset=devset[:1])
    _ = task_model.inspect_history(n=len(candidate_program.predictors()))
    return task_model.inspect_history(n=len(candidate_program.predictors()))


def print_full_program(program):
    """Print out the program's instructions & prefixes for each module."""
    for i, predictor in enumerate(program.predictors()):
        print(f"Predictor {i}")
        print(f"i: {get_signature(predictor).instructions}")
        *_, last_field = get_signature(predictor).fields.values()
        print(f"p: {last_field.json_schema_extra['prefix']}")
    print("\n")


def save_candidate_program(program, log_dir, trial_num, note=None):
    """Save the candidate program to the log directory."""

    if log_dir is None:
        return None

    # Ensure the directory exists
    eval_programs_dir = os.path.join(log_dir, "evaluated_programs")
    os.makedirs(eval_programs_dir, exist_ok=True)

    # Define the save path for the program
    if note:
        save_path = os.path.join(eval_programs_dir, f"program_{trial_num}_{note}.json")
    else:
        save_path = os.path.join(eval_programs_dir, f"program_{trial_num}.json")

    # Save the program
    program.save(save_path)

    return save_path


def save_file_to_log_dir(source_file_path, log_dir):
    if log_dir is None:
        return
    """Save a file to our log directory"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    destination_file_path = os.path.join(log_dir, os.path.basename(source_file_path))

    # Copy the file
    shutil.copy(source_file_path, destination_file_path)


def setup_logging(log_dir):
    """Setup logger, which will log our print statements to a txt file at our log_dir for later viewing"""
    if log_dir is None:
        return
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    # Create a file handler that logs debug and higher level messages
    file_handler = logging.FileHandler(f"{log_dir}/logs.txt")
    file_handler.setLevel(logging.WARNING)
    file_formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Create a console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)


def get_token_usage(model) -> tuple[int, int]:
    """
    Extract total input tokens and output tokens from a model's interaction history.
    Returns (total_input_tokens, total_output_tokens).
    """
    if not hasattr(model, "history"):
        return 0, 0

    input_tokens = []
    output_tokens = []
    for interaction in model.history:
        usage = interaction.get("usage", {})
        _input_tokens = usage.get("prompt_tokens", 0)
        _output_tokens = usage.get("completion_tokens", 0)
        input_tokens.append(_input_tokens)
        output_tokens.append(_output_tokens)

    total_input_tokens = int(np.sum(input_tokens))
    total_output_tokens = int(np.sum(output_tokens))

    return total_input_tokens, total_output_tokens


def log_token_usage(trial_logs, trial_num, model_dict):
    """
    Extract total input and output tokens used by each model and log to trial_logs[trial_num]["token_usage"].
    """

    token_usage_dict = {}

    for model_name, model in model_dict.items():
        in_tokens, out_tokens = get_token_usage(model)
        token_usage_dict[model_name] = {"total_input_tokens": in_tokens, "total_output_tokens": out_tokens}

    # Store token usage info in trial logs
    trial_logs[trial_num]["token_usage"] = token_usage_dict


### OTHER UTILS ###


def get_prompt_model(prompt_model):
    if prompt_model:
        return prompt_model
    else:
        return dspy.settings.lm


def get_signature(predictor):
    assert hasattr(predictor, "signature")
    return predictor.signature


def set_signature(predictor, updated_signature):
    assert hasattr(predictor, "signature")
    predictor.signature = updated_signature


def create_n_fewshot_demo_sets(
    student,
    num_candidate_sets,
    trainset,
    max_labeled_demos,
    max_bootstrapped_demos,
    metric,
    teacher_settings,
    max_errors=None,
    max_rounds=1,
    labeled_sample=True,
    min_num_samples=1,
    metric_threshold=None,
    teacher=None,
    include_non_bootstrapped=True,
    seed=0,
    rng=None,
):
    """
    This function is copied from random_search.py, and creates fewshot examples in the same way that random search does.
    This allows us to take advantage of using the same fewshot examples when we use the same random seed in our optimizers.
    """
    max_errors = dspy.settings.max_errors if max_errors is None else max_errors
    demo_candidates = {}

    # Account for confusing way this is set up, where we add in 3 more candidate sets to the N specified
    num_candidate_sets -= 3

    # Initialize demo_candidates dictionary
    for i, _ in enumerate(student.predictors()):
        demo_candidates[i] = []

    rng = rng or random.Random(seed)

    # Go through and create each candidate set
    for seed in range(-3, num_candidate_sets):
        print(f"Bootstrapping set {seed + 4}/{num_candidate_sets + 3}")

        trainset_copy = list(trainset)

        if seed == -3 and include_non_bootstrapped:
            # zero-shot
            program2 = student.reset_copy()

        elif seed == -2 and max_labeled_demos > 0 and include_non_bootstrapped:
            # labels only
            teleprompter = LabeledFewShot(k=max_labeled_demos)
            program2 = teleprompter.compile(
                student,
                trainset=trainset_copy,
                sample=labeled_sample,
            )

        elif seed == -1:
            # unshuffled few-shot
            program = BootstrapFewShot(
                metric=metric,
                max_errors=max_errors,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                teacher_settings=teacher_settings,
                max_rounds=max_rounds,
            )
            program2 = program.compile(student, teacher=teacher, trainset=trainset_copy)

        else:
            # shuffled few-shot
            rng.shuffle(trainset_copy)
            size = rng.randint(min_num_samples, max_bootstrapped_demos)

            teleprompter = BootstrapFewShot(
                metric=metric,
                max_errors=max_errors,
                metric_threshold=metric_threshold,
                max_bootstrapped_demos=size,
                max_labeled_demos=max_labeled_demos,
                teacher_settings=teacher_settings,
                max_rounds=max_rounds,
            )

            program2 = teleprompter.compile(
                student,
                teacher=teacher,
                trainset=trainset_copy,
            )

        for i, _ in enumerate(student.predictors()):
            demo_candidates[i].append(program2.predictors()[i].demos)

    return demo_candidates


def old_getfile(object):
    """Work out which source or compiled file an object was defined in."""
    if inspect.ismodule(object):
        if getattr(object, "__file__", None):
            return object.__file__
        raise TypeError(f"{object!r} is a built-in module")
    if inspect.isclass(object):
        if hasattr(object, "__module__"):
            module = sys.modules.get(object.__module__)
            if getattr(module, "__file__", None):
                return module.__file__
            if object.__module__ == "__main__":
                raise OSError("source code not available")
        raise TypeError(f"{object!r} is a built-in class")
    if inspect.ismethod(object):
        object = object.__func__
    if inspect.isfunction(object):
        object = object.__code__
    if inspect.istraceback(object):
        object = object.tb_frame
    if inspect.isframe(object):
        object = object.f_code
    if inspect.iscode(object):
        return object.co_filename
    raise TypeError(
        f"module, class, method, function, traceback, frame, or code object was expected, got {type(object).__name__}"
    )


def new_getfile(object):
    if not inspect.isclass(object):
        return old_getfile(object)

    # Lookup by parent module (as in current inspect)
    if hasattr(object, "__module__"):
        object_ = sys.modules.get(object.__module__)
        if hasattr(object_, "__file__"):
            return object_.__file__

    # If parent module is __main__, lookup by methods (NEW)
    for _, member in inspect.getmembers(object):
        if inspect.isfunction(member) and object.__qualname__ + "." + member.__name__ == member.__qualname__:
            return inspect.getfile(member)
    raise TypeError(f"Source for {object!r} not found")


inspect.getfile = new_getfile
```

## File: `teleprompt/vanilla.py`
```
import random

from dspy.teleprompt.teleprompt import Teleprompter


class LabeledFewShot(Teleprompter):
    def __init__(self, k=16):
        self.k = k

    def compile(self, student, *, trainset, sample=True):
        self.student = student.reset_copy()
        self.trainset = trainset

        if len(self.trainset) == 0:
            return self.student

        rng = random.Random(0)

        for predictor in self.student.predictors():
            if sample:
                predictor.demos = rng.sample(self.trainset, min(self.k, len(self.trainset)))
            else:
                predictor.demos = self.trainset[: min(self.k, len(self.trainset))]

        return self.student


# NOTE: I believe templatev2 keeps rdemos as long as they have the last field.
# This may change later, especially with the introduction of required vs optional fields.
# NOTE: Since we're relying on downstream code to handle the demos, this sampling may be sub-sampled.
```

## File: `utils/__init__.py`
```
import os

import requests

from dspy.streaming.messages import StatusMessage, StatusMessageProvider
from dspy.utils import exceptions
from dspy.utils.callback import BaseCallback, with_callbacks
from dspy.utils.dummies import DummyLM, DummyVectorizer, dummy_rm
from dspy.utils.inspect_history import pretty_print_history
from dspy.utils.syncify import syncify


def download(url):
    filename = os.path.basename(url)
    remote_size = int(requests.head(url, allow_redirects=True).headers.get("Content-Length", 0))
    local_size = os.path.getsize(filename) if os.path.exists(filename) else 0

    if local_size != remote_size:
        print(f"Downloading '{filename}'...")
        with requests.get(url, stream=True) as r, open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


__all__ = [
    "download",
    "exceptions",
    "BaseCallback",
    "with_callbacks",
    "DummyLM",
    "DummyVectorizer",
    "dummy_rm",
    "StatusMessage",
    "StatusMessageProvider",
    "pretty_print_history",
]
```

## File: `utils/asyncify.py`
```
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import asyncer
from anyio import CapacityLimiter

if TYPE_CHECKING:
    from dspy.primitives.module import Module

_limiter = None


def get_async_max_workers():
    import dspy

    return dspy.settings.async_max_workers


def get_limiter():
    async_max_workers = get_async_max_workers()

    global _limiter
    if _limiter is None:
        _limiter = CapacityLimiter(async_max_workers)
    elif _limiter.total_tokens != async_max_workers:
        _limiter.total_tokens = async_max_workers

    return _limiter


def asyncify(program: "Module") -> Callable[[Any, Any], Awaitable[Any]]:
    """
    Wraps a DSPy program so that it can be called asynchronously. This is useful for running a
    program in parallel with another task (e.g., another DSPy program).

    This implementation propagates the current thread's configuration context to the worker thread.

    Args:
        program: The DSPy program to be wrapped for asynchronous execution.

    Returns:
        An async function: An async function that, when awaited, runs the program in a worker thread.
            The current thread's configuration context is inherited for each call.
    """

    async def async_program(*args, **kwargs) -> Any:
        # Capture the current overrides at call-time.
        from dspy.dsp.utils.settings import thread_local_overrides

        parent_overrides = thread_local_overrides.get().copy()

        def wrapped_program(*a, **kw):
            from dspy.dsp.utils.settings import thread_local_overrides

            original_overrides = thread_local_overrides.get()
            token = thread_local_overrides.set({**original_overrides, **parent_overrides.copy()})
            try:
                return program(*a, **kw)
            finally:
                thread_local_overrides.reset(token)

        # Create a fresh asyncified callable each time, ensuring the latest context is used.
        call_async = asyncer.asyncify(wrapped_program, abandon_on_cancel=True, limiter=get_limiter())
        return await call_async(*args, **kwargs)

    return async_program
```

## File: `utils/caching.py`
```
import os
from pathlib import Path

_DEFAULT_CACHE_DIR = os.path.join(Path.home(), ".dspy_cache")
DSPY_CACHEDIR = os.environ.get("DSPY_CACHEDIR") or _DEFAULT_CACHE_DIR


def create_subdir_in_cachedir(subdir: str) -> str:
    """Create a subdirectory in the DSPy cache directory."""
    subdir = os.path.join(DSPY_CACHEDIR, subdir)
    subdir = os.path.abspath(subdir)
    os.makedirs(subdir, exist_ok=True)
    return subdir
```

## File: `utils/callback.py`
```
import functools
import inspect
import logging
import uuid
from contextvars import ContextVar
from typing import Any, Callable

import dspy

ACTIVE_CALL_ID = ContextVar("active_call_id", default=None)

logger = logging.getLogger(__name__)


class BaseCallback:
    """A base class for defining callback handlers for DSPy components.

    To use a callback, subclass this class and implement the desired handlers. Each handler
    will be called at the appropriate time before/after the execution of the corresponding component.  For example, if
    you want to print a message before and after an LM is called, implement `the on_llm_start` and `on_lm_end` handler.
    Users can set the callback globally using `dspy.settings.configure` or locally by passing it to the component
    constructor.


    Example 1: Set a global callback using `dspy.settings.configure`.

    ```
    import dspy
    from dspy.utils.callback import BaseCallback

    class LoggingCallback(BaseCallback):

        def on_lm_start(self, call_id, instance, inputs):
            print(f"LM is called with inputs: {inputs}")

        def on_lm_end(self, call_id, outputs, exception):
            print(f"LM is finished with outputs: {outputs}")

    dspy.settings.configure(
        callbacks=[LoggingCallback()]
    )

    cot = dspy.ChainOfThought("question -> answer")
    cot(question="What is the meaning of life?")

    # > LM is called with inputs: {'question': 'What is the meaning of life?'}
    # > LM is finished with outputs: {'answer': '42'}
    ```

    Example 2: Set a local callback by passing it to the component constructor.

    ```
    lm_1 = dspy.LM("gpt-3.5-turbo", callbacks=[LoggingCallback()])
    lm_1(question="What is the meaning of life?")

    # > LM is called with inputs: {'question': 'What is the meaning of life?'}
    # > LM is finished with outputs: {'answer': '42'}

    lm_2 = dspy.LM("gpt-3.5-turbo")
    lm_2(question="What is the meaning of life?")
    # No logging here because only `lm_1` has the callback set.
    ```
    """

    def on_module_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """A handler triggered when forward() method of a module (subclass of dspy.Module) is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Module instance.
            inputs: The inputs to the module's forward() method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        pass

    def on_module_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ):
        """A handler triggered after forward() method of a module (subclass of dspy.Module) is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the module's forward() method. If the method is interrupted by
                an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass

    def on_lm_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """A handler triggered when __call__ method of dspy.LM instance is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The LM instance.
            inputs: The inputs to the LM's __call__ method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        pass

    def on_lm_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ):
        """A handler triggered after __call__ method of dspy.LM instance is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the LM's __call__ method. If the method is interrupted by
                an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass

    def on_adapter_format_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """A handler triggered when format() method of an adapter (subclass of dspy.Adapter) is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Adapter instance.
            inputs: The inputs to the Adapter's format() method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        pass

    def on_adapter_format_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ):
        """A handler triggered after format() method of an adapter (subclass of dspy.Adapter) is called..

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the Adapter's format() method. If the method is interrupted
                by an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass

    def on_adapter_parse_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """A handler triggered when parse() method of an adapter (subclass of dspy.Adapter) is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Adapter instance.
            inputs: The inputs to the Adapter's parse() method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        pass

    def on_adapter_parse_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ):
        """A handler triggered after parse() method of an adapter (subclass of dspy.Adapter) is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the Adapter's parse() method. If the method is interrupted
                by an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass

    def on_tool_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """A handler triggered when a tool is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Tool instance.
            inputs: The inputs to the Tool's __call__ method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        pass

    def on_tool_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ):
        """A handler triggered after a tool is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the Tool's __call__ method. If the method is interrupted by
                an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass

    def on_evaluate_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """A handler triggered when evaluation is started.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Evaluate instance.
            inputs: The inputs to the Evaluate's __call__ method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        pass

    def on_evaluate_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ):
        """A handler triggered after evaluation is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the Evaluate's __call__ method. If the method is interrupted by
                an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass


def with_callbacks(fn):
    """Decorator to add callback functionality to instance methods."""

    def _execute_start_callbacks(instance, fn, call_id, callbacks, args, kwargs):
        """Execute all start callbacks for a function call."""
        inputs = inspect.getcallargs(fn, instance, *args, **kwargs)
        if "self" in inputs:
            inputs.pop("self")
        elif "instance" in inputs:
            inputs.pop("instance")
        for callback in callbacks:
            try:
                _get_on_start_handler(callback, instance, fn)(call_id=call_id, instance=instance, inputs=inputs)
            except Exception as e:
                logger.warning(f"Error when calling callback {callback}: {e}")

    def _execute_end_callbacks(instance, fn, call_id, results, exception, callbacks):
        """Execute all end callbacks for a function call."""
        for callback in callbacks:
            try:
                _get_on_end_handler(callback, instance, fn)(
                    call_id=call_id,
                    outputs=results,
                    exception=exception,
                )
            except Exception as e:
                logger.warning(f"Error when applying callback {callback}'s end handler on function {fn.__name__}: {e}.")

    def _get_active_callbacks(instance):
        """Get combined global and instance-level callbacks."""
        return dspy.settings.get("callbacks", []) + getattr(instance, "callbacks", [])

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(instance, *args, **kwargs):
            callbacks = _get_active_callbacks(instance)
            if not callbacks:
                return await fn(instance, *args, **kwargs)

            call_id = uuid.uuid4().hex

            _execute_start_callbacks(instance, fn, call_id, callbacks, args, kwargs)

            # Active ID must be set right before the function is called, not before calling the callbacks.
            parent_call_id = ACTIVE_CALL_ID.get()
            ACTIVE_CALL_ID.set(call_id)

            results = None
            exception = None
            try:
                results = await fn(instance, *args, **kwargs)
                return results
            except Exception as e:
                exception = e
                raise exception
            finally:
                ACTIVE_CALL_ID.set(parent_call_id)
                _execute_end_callbacks(instance, fn, call_id, results, exception, callbacks)

        return async_wrapper

    else:

        @functools.wraps(fn)
        def sync_wrapper(instance, *args, **kwargs):
            callbacks = _get_active_callbacks(instance)
            if not callbacks:
                return fn(instance, *args, **kwargs)

            call_id = uuid.uuid4().hex

            _execute_start_callbacks(instance, fn, call_id, callbacks, args, kwargs)

            # Active ID must be set right before the function is called, not before calling the callbacks.
            parent_call_id = ACTIVE_CALL_ID.get()
            ACTIVE_CALL_ID.set(call_id)

            results = None
            exception = None
            try:
                results = fn(instance, *args, **kwargs)
                return results
            except Exception as e:
                exception = e
                raise exception
            finally:
                ACTIVE_CALL_ID.set(parent_call_id)
                _execute_end_callbacks(instance, fn, call_id, results, exception, callbacks)

        return sync_wrapper


def _get_on_start_handler(callback: BaseCallback, instance: Any, fn: Callable) -> Callable:
    """Selects the appropriate on_start handler of the callback based on the instance and function name."""
    if isinstance(instance, dspy.LM):
        return callback.on_lm_start
    elif isinstance(instance, dspy.Evaluate):
        return callback.on_evaluate_start

    if isinstance(instance, dspy.Adapter):
        if fn.__name__ == "format":
            return callback.on_adapter_format_start
        elif fn.__name__ == "parse":
            return callback.on_adapter_parse_start
        else:
            raise ValueError(f"Unsupported adapter method for using callback: {fn.__name__}.")

    if isinstance(instance, dspy.Tool):
        return callback.on_tool_start

    # We treat everything else as a module.
    return callback.on_module_start


def _get_on_end_handler(callback: BaseCallback, instance: Any, fn: Callable) -> Callable:
    """Selects the appropriate on_end handler of the callback based on the instance and function name."""
    if isinstance(instance, (dspy.LM)):
        return callback.on_lm_end
    elif isinstance(instance, dspy.Evaluate):
        return callback.on_evaluate_end

    if isinstance(instance, (dspy.Adapter)):
        if fn.__name__ == "format":
            return callback.on_adapter_format_end
        elif fn.__name__ == "parse":
            return callback.on_adapter_parse_end
        else:
            raise ValueError(f"Unsupported adapter method for using callback: {fn.__name__}.")

    if isinstance(instance, dspy.Tool):
        return callback.on_tool_end

    # We treat everything else as a module.
    return callback.on_module_end
```

## File: `utils/dummies.py`
```
import random
from collections import defaultdict
from typing import Any

import numpy as np

from dspy.adapters.chat_adapter import ChatAdapter, FieldInfoWithName, field_header_pattern
from dspy.clients.lm import LM
from dspy.dsp.utils.utils import dotdict
from dspy.signatures.field import OutputField
from dspy.utils.callback import with_callbacks


class DummyLM(LM):
    """Dummy language model for unit testing purposes.

    Three modes of operation:

    Mode 1: List of dictionaries

    If a list of dictionaries is provided, the dummy model will return the next dictionary
    in the list for each request, formatted according to the `format_field_with_value` function.

    Example:

    ```
    lm = DummyLM([{"answer": "red"}, {"answer": "blue"}])
    dspy.settings.configure(lm=lm)
    predictor("What color is the sky?")
    # Output:
    # [[## answer ##]]
    # red
    predictor("What color is the sky?")
    # Output:
    # [[## answer ##]]
    # blue
    ```

    Mode 2: Dictionary of dictionaries

    If a dictionary of dictionaries is provided, the dummy model will return the value
    corresponding to the key which is contained with the final message of the prompt,
    formatted according to the `format_field_with_value` function from the chat adapter.

    ```
    lm = DummyLM({"What color is the sky?": {"answer": "blue"}})
    dspy.settings.configure(lm=lm)
    predictor("What color is the sky?")
    # Output:
    # [[## answer ##]]
    # blue
    ```

    Mode 3: Follow examples

    If `follow_examples` is set to True, and the prompt contains an example input exactly equal to the prompt,
    the dummy model will return the output from that example.

    ```
    lm = DummyLM([{"answer": "red"}], follow_examples=True)
    dspy.settings.configure(lm=lm)
    predictor("What color is the sky?, demos=dspy.Example(input="What color is the sky?", output="blue"))
    # Output:
    # [[## answer ##]]
    # blue
    ```

    """

    def __init__(self, answers: list[dict[str, str]] | dict[str, dict[str, str]], follow_examples: bool = False):
        super().__init__("dummy", "chat", 0.0, 1000, True)
        self.answers = answers
        if isinstance(answers, list):
            self.answers = iter(answers)
        self.follow_examples = follow_examples

    def _use_example(self, messages):
        # find all field names
        fields = defaultdict(int)
        for message in messages:
            if "content" in message:
                if ma := field_header_pattern.match(message["content"]):
                    fields[message["content"][ma.start() : ma.end()]] += 1
        # find the fields which are missing from the final turns
        max_count = max(fields.values())
        output_fields = [field for field, count in fields.items() if count != max_count]

        # get the output from the last turn that has the output fields as headers
        final_input = messages[-1]["content"].split("\n\n")[0]
        for input, output in zip(reversed(messages[:-1]), reversed(messages), strict=False):
            if any(field in output["content"] for field in output_fields) and final_input in input["content"]:
                return output["content"]

    @with_callbacks
    def __call__(self, prompt=None, messages=None, **kwargs):
        def format_answer_fields(field_names_and_values: dict[str, Any]):
            return ChatAdapter().format_field_with_value(
                fields_with_values={
                    FieldInfoWithName(name=field_name, info=OutputField()): value
                    for field_name, value in field_names_and_values.items()
                }
            )

        # Build the request.
        outputs = []
        for _ in range(kwargs.get("n", 1)):
            messages = messages or [{"role": "user", "content": prompt}]
            kwargs = {**self.kwargs, **kwargs}

            if self.follow_examples:
                outputs.append(self._use_example(messages))
            elif isinstance(self.answers, dict):
                outputs.append(
                    next(
                        (format_answer_fields(v) for k, v in self.answers.items() if k in messages[-1]["content"]),
                        "No more responses",
                    )
                )
            else:
                outputs.append(format_answer_fields(next(self.answers, {"answer": "No more responses"})))

            # Logging, with removed api key & where `cost` is None on cache hit.
            kwargs = {k: v for k, v in kwargs.items() if not k.startswith("api_")}
            entry = {"prompt": prompt, "messages": messages, "kwargs": kwargs}
            entry = {**entry, "outputs": outputs, "usage": 0}
            entry = {**entry, "cost": 0}
            self.history.append(entry)
            self.update_global_history(entry)

        return outputs

    async def acall(self, prompt=None, messages=None, **kwargs):
        return self.__call__(prompt=prompt, messages=messages, **kwargs)

    def get_convo(self, index):
        """Get the prompt + answer from the ith message."""
        return self.history[index]["messages"], self.history[index]["outputs"]


def dummy_rm(passages=()) -> callable:
    if not passages:

        def inner(query: str, *, k: int, **kwargs):
            raise ValueError("No passages defined")

        return inner
    max_length = max(map(len, passages)) + 100
    vectorizer = DummyVectorizer(max_length)
    passage_vecs = vectorizer(passages)

    def inner(query: str, *, k: int, **kwargs):
        assert k <= len(passages)
        query_vec = vectorizer([query])[0]
        scores = passage_vecs @ query_vec
        largest_idx = (-scores).argsort()[:k]

        return [dotdict(long_text=passages[i]) for i in largest_idx]

    return inner


class DummyVectorizer:
    """Simple vectorizer based on n-grams."""

    def __init__(self, max_length=100, n_gram=2):
        self.max_length = max_length
        self.n_gram = n_gram
        self.P = 10**9 + 7  # A large prime number
        random.seed(123)
        self.coeffs = [random.randrange(1, self.P) for _ in range(n_gram)]

    def _hash(self, gram):
        """Hashes a string using a polynomial hash function."""
        h = 1
        for coeff, c in zip(self.coeffs, gram, strict=False):
            h = h * coeff + ord(c)
            h %= self.P
        return h % self.max_length

    def __call__(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for text in texts:
            grams = [text[i : i + self.n_gram] for i in range(len(text) - self.n_gram + 1)]
            vec = [0] * self.max_length
            for gram in grams:
                vec[self._hash(gram)] += 1
            vecs.append(vec)

        vecs = np.array(vecs, dtype=np.float32)
        vecs -= np.mean(vecs, axis=1, keepdims=True)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10  # Added epsilon to avoid division by zero
        return vecs
```

## File: `utils/exceptions.py`
```

from dspy.signatures.signature import Signature


class AdapterParseError(Exception):
    """Exception raised when adapter cannot parse the LM response."""

    def __init__(
        self,
        adapter_name: str,
        signature: Signature,
        lm_response: str,
        message: str | None = None,
        parsed_result: str | None = None,
    ):
        self.adapter_name = adapter_name
        self.signature = signature
        self.lm_response = lm_response
        self.parsed_result = parsed_result

        message = f"{message}\n\n" if message else ""
        message = (
            f"{message}"
            f"Adapter {adapter_name} failed to parse the LM response. \n\n"
            f"LM Response: {lm_response} \n\n"
            f"Expected to find output fields in the LM response: [{', '.join(signature.output_fields.keys())}] \n\n"
        )

        if parsed_result is not None:
            message += f"Actual output fields parsed from the LM response: [{', '.join(parsed_result.keys())}] \n\n"

        super().__init__(message)
```

## File: `utils/inspect_history.py`
```
def _green(text: str, end: str = "\n"):
    return "\x1b[32m" + str(text).lstrip() + "\x1b[0m" + end


def _red(text: str, end: str = "\n"):
    return "\x1b[31m" + str(text) + "\x1b[0m" + end


def _blue(text: str, end: str = "\n"):
    return "\x1b[34m" + str(text) + "\x1b[0m" + end


def pretty_print_history(history, n: int = 1):
    """Prints the last n prompts and their completions."""

    for item in history[-n:]:
        messages = item["messages"] or [{"role": "user", "content": item["prompt"]}]
        outputs = item["outputs"]
        timestamp = item.get("timestamp", "Unknown time")

        print("\n\n\n")
        print("\x1b[34m" + f"[{timestamp}]" + "\x1b[0m" + "\n")

        for msg in messages:
            print(_red(f"{msg['role'].capitalize()} message:"))
            if isinstance(msg["content"], str):
                print(msg["content"].strip())
            else:
                if isinstance(msg["content"], list):
                    for c in msg["content"]:
                        if c["type"] == "text":
                            print(c["text"].strip())
                        elif c["type"] == "image_url":
                            image_str = ""
                            if "base64" in c["image_url"].get("url", ""):
                                len_base64 = len(c["image_url"]["url"].split("base64,")[1])
                                image_str = (
                                    f"<{c['image_url']['url'].split('base64,')[0]}base64,"
                                    f"<IMAGE BASE 64 ENCODED({len_base64!s})>"
                                )
                            else:
                                image_str = f"<image_url: {c['image_url']['url']}>"
                            print(_blue(image_str.strip()))
                        elif c["type"] == "input_audio":
                            audio_format = c["input_audio"]["format"]
                            len_audio = len(c["input_audio"]["data"])
                            audio_str = f"<audio format='{audio_format}' base64-encoded, length={len_audio}>"
                            print(_blue(audio_str.strip()))
            print("\n")

        if isinstance(outputs[0], dict):
            if outputs[0]["text"]:
                print(_red("Response:"))
                print(_green(outputs[0]["text"].strip()))

            if outputs[0].get("tool_calls"):
                print(_red("Tool calls:"))
                for tool_call in outputs[0]["tool_calls"]:
                    print(_green(f"{tool_call['function']['name']}: {tool_call['function']['arguments']}"))
        else:
            print(_red("Response:"))
            print(_green(outputs[0].strip()))

        if len(outputs) > 1:
            choices_text = f" \t (and {len(outputs) - 1} other completions)"
            print(_red(choices_text, end=""))

    print("\n\n\n")
```

## File: `utils/langchain_tool.py`
```
from typing import TYPE_CHECKING, Any

from dspy.adapters.types.tool import Tool, convert_input_schema_to_tool_args

if TYPE_CHECKING:
    from langchain.tools import BaseTool



def convert_langchain_tool(tool: "BaseTool") -> Tool:
    """Build a DSPy tool from a LangChain tool.
    
    This function converts a LangChain tool (either created with @tool decorator
    or by subclassing BaseTool) into a DSPy Tool.

    Args:
        tool: The LangChain tool to convert.

    Returns:
        A DSPy Tool object.
    """
    async def func(**kwargs):
        try:
            result = await tool.ainvoke(kwargs)
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to call LangChain tool {tool.name}: {e!s}")

    # Get args_schema from the tool
    # https://python.langchain.com/api_reference/core/tools/langchain_core.tools.base.BaseTool.html#langchain_core.tools.base.BaseTool.args_schema
    args_schema = tool.args_schema
    args, _, arg_desc = convert_input_schema_to_tool_args(args_schema.model_json_schema())

    # The args_schema of Langchain tool is a pydantic model, so we can get the type hints from the model fields
    arg_types = {
        key: field.annotation if field.annotation is not None else Any
        for key, field in args_schema.model_fields.items()
    }

    return Tool(
        func=func,
        name=tool.name,
        desc=tool.description,
        args=args,
        arg_types=arg_types,
        arg_desc=arg_desc
    )
```

## File: `utils/logging_utils.py`
```
import logging
import logging.config
import sys

LOGGING_LINE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOGGING_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"


class DSPyLoggingStream:
    """
    A Python stream for use with event logging APIs throughout DSPy (`eprint()`,
    `logger.info()`, etc.). This stream wraps `sys.stderr`, forwarding `write()` and
    `flush()` calls to the stream referred to by `sys.stderr` at the time of the call.
    It also provides capabilities for disabling the stream to silence event logs.
    """

    def __init__(self):
        self._enabled = True

    def write(self, text):
        if self._enabled:
            sys.stderr.write(text)

    def flush(self):
        if self._enabled:
            sys.stderr.flush()

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value


DSPY_LOGGING_STREAM = DSPyLoggingStream()


def disable_logging():
    """
    Disables the `DSPyLoggingStream` used by event logging APIs throughout DSPy
    (`eprint()`, `logger.info()`, etc), silencing all subsequent event logs.
    """
    DSPY_LOGGING_STREAM.enabled = False


def enable_logging():
    """
    Enables the `DSPyLoggingStream` used by event logging APIs throughout DSPy
    (`eprint()`, `logger.info()`, etc), emitting all subsequent event logs. This
    reverses the effects of `disable_logging()`.
    """
    DSPY_LOGGING_STREAM.enabled = True


def configure_dspy_loggers(root_module_name):
    formatter = logging.Formatter(fmt=LOGGING_LINE_FORMAT, datefmt=LOGGING_DATETIME_FORMAT)

    dspy_handler_name = "dspy_handler"
    handler = logging.StreamHandler(stream=DSPY_LOGGING_STREAM)
    handler.setFormatter(formatter)
    handler.set_name(dspy_handler_name)

    logger = logging.getLogger(root_module_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for existing_handler in logger.handlers[:]:
        if getattr(existing_handler, "name", None) == dspy_handler_name:
            logger.removeHandler(existing_handler)

    logger.addHandler(handler)
```

## File: `utils/mcp.py`
```
from typing import TYPE_CHECKING, Any

from dspy.adapters.types.tool import Tool, convert_input_schema_to_tool_args

if TYPE_CHECKING:
    import mcp


def _convert_mcp_tool_result(call_tool_result: "mcp.types.CallToolResult") -> str | list[Any]:
    from mcp.types import TextContent

    text_contents: list[TextContent] = []
    non_text_contents = []
    for content in call_tool_result.content:
        if isinstance(content, TextContent):
            text_contents.append(content)
        else:
            non_text_contents.append(content)

    tool_content = [content.text for content in text_contents]
    if len(text_contents) == 1:
        tool_content = tool_content[0]

    if call_tool_result.isError:
        raise RuntimeError(f"Failed to call a MCP tool: {tool_content}")

    return tool_content or non_text_contents


def convert_mcp_tool(session: "mcp.client.session.ClientSession", tool: "mcp.types.Tool") -> Tool:
    """Build a DSPy tool from an MCP tool.

    Args:
        session: The MCP session to use.
        tool: The MCP tool to convert.

    Returns:
        A dspy Tool object.
    """
    args, arg_types, arg_desc = convert_input_schema_to_tool_args(tool.inputSchema)

    # Convert the MCP tool and Session to a single async method
    async def func(*args, **kwargs):
        result = await session.call_tool(tool.name, arguments=kwargs)
        return _convert_mcp_tool_result(result)

    return Tool(func=func, name=tool.name, desc=tool.description, args=args, arg_types=arg_types, arg_desc=arg_desc)
```

## File: `utils/parallelizer.py`
```
import contextlib
import copy
import logging
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import tqdm

logger = logging.getLogger(__name__)


class ParallelExecutor:
    def __init__(
        self,
        num_threads=None,
        max_errors=None,
        disable_progress_bar=False,
        provide_traceback=None,
        compare_results=False,
        timeout=120,
        straggler_limit=3,
    ):
        """
        Offers isolation between the tasks (dspy.settings) irrespective of whether num_threads == 1 or > 1.
        Handles also straggler timeouts.
        """
        from dspy.dsp.utils.settings import settings

        self.num_threads = num_threads or settings.num_threads
        self.max_errors = settings.max_errors if max_errors is None else max_errors
        self.disable_progress_bar = disable_progress_bar
        self.provide_traceback = provide_traceback if provide_traceback is not None else settings.provide_traceback
        self.compare_results = compare_results
        self.timeout = timeout
        self.straggler_limit = straggler_limit

        self.error_count = 0
        self.error_lock = threading.Lock()
        self.cancel_jobs = threading.Event()

    def execute(self, function, data):
        tqdm.tqdm._instances.clear()
        wrapped = self._wrap_function(function)
        return self._execute_parallel(wrapped, data)

    def _wrap_function(self, user_function):
        def safe_func(item):
            if self.cancel_jobs.is_set():
                return None
            try:
                return user_function(item)
            except Exception as e:
                with self.error_lock:
                    self.error_count += 1
                    if self.error_count >= self.max_errors:
                        self.cancel_jobs.set()
                if self.provide_traceback:
                    logger.error(f"Error for {item}: {e}\n{traceback.format_exc()}")
                else:
                    logger.error(f"Error for {item}: {e}. Set `provide_traceback=True` for traceback.")
                return None

        return safe_func

    def _execute_parallel(self, function, data):
        results = [None] * len(data)
        job_cancelled = "cancelled"

        # We resubmit at most once per item.
        start_time_map = {}
        start_time_lock = threading.Lock()
        resubmitted = set()

        # This is the worker function each thread will run.
        def worker(parent_overrides, submission_id, index, item):
            if self.cancel_jobs.is_set():
                return index, job_cancelled
            # Record actual start time
            with start_time_lock:
                start_time_map[submission_id] = time.time()

            # Apply parent's thread-local overrides
            from dspy.dsp.utils.settings import thread_local_overrides

            original = thread_local_overrides.get()
            token = thread_local_overrides.set({**original, **parent_overrides.copy()})
            if parent_overrides.get("usage_tracker"):
                # Usage tracker needs to be deep copied across threads so that each thread tracks its own usage
                thread_local_overrides.overrides["usage_tracker"] = copy.deepcopy(parent_overrides["usage_tracker"])

            try:
                return index, function(item)
            finally:
                thread_local_overrides.reset(token)

        # Handle Ctrl-C in the main thread
        @contextlib.contextmanager
        def interrupt_manager():
            if threading.current_thread() is threading.main_thread():
                orig_handler = signal.getsignal(signal.SIGINT)

                def handler(sig, frame):
                    self.cancel_jobs.set()
                    logger.warning("SIGINT received. Cancelling.")
                    orig_handler(sig, frame)

                signal.signal(signal.SIGINT, handler)
                try:
                    yield
                finally:
                    signal.signal(signal.SIGINT, orig_handler)
            else:
                yield

        executor = ThreadPoolExecutor(max_workers=self.num_threads)
        try:
            with interrupt_manager():
                from dspy.dsp.utils.settings import thread_local_overrides

                parent_overrides = thread_local_overrides.get().copy()

                futures_map = {}
                futures_set = set()
                submission_counter = 0

                for idx, item in enumerate(data):
                    f = executor.submit(worker, parent_overrides, submission_counter, idx, item)
                    futures_map[f] = (submission_counter, idx, item)
                    futures_set.add(f)
                    submission_counter += 1

                pbar = tqdm.tqdm(
                    total=len(data),
                    dynamic_ncols=True,
                    disable=self.disable_progress_bar,
                    file=sys.stdout,
                )

                def all_done():
                    return all(r is not None for r in results)

                while futures_set and not self.cancel_jobs.is_set():
                    if all_done():
                        break
                    done, not_done = wait(futures_set, timeout=1, return_when=FIRST_COMPLETED)
                    for f in done:
                        futures_set.remove(f)
                        try:
                            index, outcome = f.result()
                        except Exception:
                            pass
                        else:
                            if outcome != job_cancelled and results[index] is None:
                                results[index] = outcome

                            # Update progress
                            if self.compare_results:
                                vals = [r[-1] for r in results if r is not None]
                                self._update_progress(pbar, sum(vals), len(vals))
                            else:
                                self._update_progress(
                                    pbar,
                                    len([r for r in results if r is not None]),
                                    len(data),
                                )

                    if all_done():
                        break

                    # Check stragglers if few remain
                    if 0 < self.timeout and len(not_done) <= self.straggler_limit:
                        now = time.time()
                        for f in list(not_done):
                            if f not in resubmitted:
                                sid, idx, item = futures_map[f]
                                with start_time_lock:
                                    st = start_time_map.get(sid, None)
                                if st and (now - st) >= self.timeout:
                                    resubmitted.add(f)
                                    nf = executor.submit(
                                        worker,
                                        parent_overrides,
                                        submission_counter,
                                        idx,
                                        item,
                                    )
                                    futures_map[nf] = (submission_counter, idx, item)
                                    futures_set.add(nf)
                                    submission_counter += 1

                pbar.close()

        finally:
            # Avoid waiting on leftover tasks that no longer matter
            executor.shutdown(wait=False)

        if self.cancel_jobs.is_set():
            logger.warning("Execution cancelled due to errors or interruption.")
            raise Exception("Execution cancelled due to errors or interruption.")

        return results

    def _update_progress(self, pbar, nresults, ntotal):
        if self.compare_results:
            pct = round(100 * nresults / ntotal, 1) if ntotal else 0
            pbar.set_description(f"Average Metric: {nresults:.2f} / {ntotal} ({pct}%)")
        else:
            pbar.set_description(f"Processed {nresults} / {ntotal} examples")
        pbar.update()
```

## File: `utils/saving.py`
```
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cloudpickle
import ujson

if TYPE_CHECKING:
    from dspy.primitives.module import Module

logger = logging.getLogger(__name__)


def get_dependency_versions():
    import dspy

    cloudpickle_version = ".".join(cloudpickle.__version__.split(".")[:2])

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "dspy": dspy.__version__,
        "cloudpickle": cloudpickle_version,
    }


def load(path: str) -> "Module":
    """Load saved DSPy model.

    This method is used to load a saved DSPy model with `save_program=True`, i.e., the model is saved with cloudpickle.

    Args:
        path (str): Path to the saved model.

    Returns:
        The loaded model, a `dspy.Module` instance.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"The path '{path}' does not exist.")

    with open(path / "metadata.json") as f:
        metadata = ujson.load(f)

    dependency_versions = get_dependency_versions()
    saved_dependency_versions = metadata["dependency_versions"]
    for key, saved_version in saved_dependency_versions.items():
        if dependency_versions[key] != saved_version:
            logger.warning(
                f"There is a mismatch of {key} version between saved model and current environment. You saved with "
                f"`{key}=={saved_version}`, but now you have `{key}=={dependency_versions[key]}`. This might cause "
                "errors or performance downgrade on the loaded model, please consider loading the model in the same "
                "environment as the saving environment."
            )

    with open(path / "program.pkl", "rb") as f:
        return cloudpickle.load(f)
```

## File: `utils/syncify.py`
```
import asyncio
from types import MethodType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dspy.primitives.module import Module


def run_async(coro):
    """Run an async coroutine from a synchronous context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # If we're in a running event loop (e.g., Jupyter), use asyncio.create_task and run until done
        import nest_asyncio

        nest_asyncio.apply()
        return asyncio.get_event_loop().run_until_complete(coro)
    else:
        return asyncio.run(coro)


def syncify(program: "Module", in_place: bool = True) -> "Module":
    """Convert an async DSPy module to a sync program.

    There are two modes of this function:

    - `in_place=True` (recommended): Modify the module in place. But this may not work if you already have a `forward`
        method which does different things from `aforward`.
    - `in_place=False`: Return a wrapper module. This changes the module's architecture, but it's more robust.

    Args:
        program: The async program to convert, must have an `aforward` method implemented.
        in_place: If True, modify the module in place. Otherwise, return a wrapper module.

    Returns:
        The sync program, which has a `forward` method that can be called from a synchronous context.
    """
    if in_place:

        def forward(self, *args, **kwargs):
            return run_async(self.aforward(*args, **kwargs))

        # Create the `forward` method in place.
        program.forward = MethodType(forward, program)
        return program
    else:
        from dspy.primitives.module import Module

        class SyncWrapper(Module):
            def __init__(self, program: "Module"):
                self.program = program

            def forward(self, *args, **kwargs):
                return run_async(self.program.aforward(*args, **kwargs))

        return SyncWrapper(program)
```

## File: `utils/unbatchify.py`
```
import queue
import threading
import time
from concurrent.futures import Future
from typing import Any, Callable


class Unbatchify:
    def __init__(
        self,
        batch_fn: Callable[[list[Any]], list[Any]],
        max_batch_size: int = 32,
        max_wait_time: float = 0.1
    ):
        """
        Initializes the Unbatchify.

        Args:
            batch_fn: The batch-processing function that accepts a list of inputs and returns a list of outputs.
            max_batch_size: The maximum number of items to include in a batch.
            max_wait_time: The maximum time (in seconds) to wait for batch to fill before processing.
        """

        self.batch_fn = batch_fn
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.input_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker)
        self.worker_thread.daemon = True  # Ensures thread exits when main program exits
        self.worker_thread.start()

    def __call__(self, input_item: Any) -> Any:
        """
        Thread-safe function that accepts a single input and returns the corresponding output.

        Args:
            input_item: The single input item to process.

        Returns:
            The output corresponding to the input_item after processing through batch_fn.
        """
        future = Future()
        self.input_queue.put((input_item, future))
        try:
            result = future.result()
        except Exception as e:
            raise e
        return result

    def _worker(self):
        """
        Worker thread that batches inputs and processes them using batch_fn.
        """
        while not self.stop_event.is_set():
            batch = []
            futures = []
            start_time = time.time()
            while len(batch) < self.max_batch_size and (time.time() - start_time) < self.max_wait_time:
                try:
                    input_item, future = self.input_queue.get(timeout=self.max_wait_time)
                    batch.append(input_item)
                    futures.append(future)
                except queue.Empty:
                    break

            if batch:
                try:
                    outputs = self.batch_fn(batch)
                    for output, future in zip(outputs, futures, strict=False):
                        future.set_result(output)
                except Exception as e:
                    for future in futures:
                        future.set_exception(e)
            else:
                time.sleep(0.01)

        # Clean up remaining items when stopping
        while True:
            try:
                _, future = self.input_queue.get_nowait()
                future.set_exception(RuntimeError("Unbatchify is closed"))
            except queue.Empty:
                break

        print("Worker thread has been terminated.")

    def close(self):
        """
        Stops the worker thread and cleans up resources.
        """
        if not self.stop_event.is_set():
            self.stop_event.set()
            self.worker_thread.join()

    def __enter__(self):
        """
        Enables use as a context manager.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Ensures resources are cleaned up when exiting context.
        """
        self.close()

    def __del__(self):
        """
        Ensures the worker thread is terminated when the object is garbage collected.
        """
        self.close()
```

## File: `utils/usage_tracker.py`
```
"""Usage tracking utilities for DSPy."""

from collections import defaultdict
from contextlib import contextmanager
from typing import Any

from dspy.dsp.utils.settings import settings


class UsageTracker:
    """Tracks LM usage data within a context."""

    def __init__(self):
        # Map of LM name to list of usage entries. For example:
        # {
        #     "openai/gpt-4o-mini": [
        #         {"prompt_tokens": 100, "completion_tokens": 200},
        #         {"prompt_tokens": 300, "completion_tokens": 400},
        #     ],
        # }
        self.usage_data = defaultdict(list)

    def _flatten_usage_entry(self, usage_entry) -> dict[str, dict[str, Any]]:
        result = dict(usage_entry)

        if result.get("completion_tokens_details"):
            result["completion_tokens_details"] = dict(result["completion_tokens_details"])
        if result.get("prompt_tokens_details"):
            result["prompt_tokens_details"] = dict(result["prompt_tokens_details"])
        return result

    def _merge_usage_entries(self, usage_entry1, usage_entry2) -> dict[str, dict[str, Any]]:
        if usage_entry1 is None or len(usage_entry1) == 0:
            return dict(usage_entry2)
        if usage_entry2 is None or len(usage_entry2) == 0:
            return dict(usage_entry1)

        result = dict(usage_entry2)
        for k, v in usage_entry1.items():
            current_v = result.get(k)
            if isinstance(v, dict) or isinstance(current_v, dict):
                result[k] = self._merge_usage_entries(current_v, v)
            else:
                result[k] = (current_v or 0) + (v or 0)
        return result

    def add_usage(self, lm: str, usage_entry: dict):
        """Add a usage entry to the tracker."""
        if len(usage_entry) > 0:
            self.usage_data[lm].append(self._flatten_usage_entry(usage_entry))

    def get_total_tokens(self) -> dict[str, dict[str, Any]]:
        """Calculate total tokens from all tracked usage."""
        total_usage_by_lm = {}
        for lm, usage_entries in self.usage_data.items():
            total_usage = {}
            for usage_entry in usage_entries:
                total_usage = self._merge_usage_entries(total_usage, usage_entry)
            total_usage_by_lm[lm] = total_usage
        return total_usage_by_lm


@contextmanager
def track_usage():
    """Context manager for tracking LM usage."""
    tracker = UsageTracker()

    with settings.context(usage_tracker=tracker):
        yield tracker
```
