#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import io
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    UserImageRawFrame,
    VisionImageRawFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.google.frames import LLMSearchResponseFrame
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.llm import (
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
)

# Suppress gRPC fork warnings (may or may not be needed with the new HTTP-based SDK,
# but safer to keep if other underlying Google libraries might still use gRPC)
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

try:
    import google.genai as genai
    import google.genai.types as genai_types
    from google.auth.exceptions import GoogleAuthError
    # DeadlineExceeded might still be raised by underlying libraries like google-auth or google-api-core
    from google.api_core import exceptions as api_core_exceptions
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Google AI, you need to `pip install pipecat-ai[google]` (which includes google-generativeai, now google-genai)."
    )
    raise Exception(f"Missing module: {e}")


class GoogleUserContextAggregator(OpenAIUserContextAggregator):
    async def push_aggregation(self):
        if len(self._aggregation) > 0:
            self._context.add_message(
                genai_types.Content(role="user", parts=[
                                    genai_types.Part(text=self._aggregation)])
            )

            # Reset the aggregation. Reset it before pushing it down, otherwise
            # if the tasks gets cancelled we won't be able to clear things up.
            self._aggregation = ""

            # Push context frame
            frame = OpenAILLMContextFrame(self._context)
            await self.push_frame(frame)

            # Reset our accumulator state.
            self.reset()


class GoogleAssistantContextAggregator(OpenAIAssistantContextAggregator):
    async def handle_aggregation(self, aggregation: str):
        self._context.add_message(
            genai_types.Content(role="model", parts=[
                                genai_types.Part(text=aggregation)])
        )

    async def handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        self._context.add_message(
            genai_types.Content(
                role="model",
                parts=[
                    genai_types.Part(
                        function_call=genai_types.FunctionCall(
                            name=frame.function_name, args=frame.arguments
                        )
                    )
                ],
            )
        )
        # For Google AI, FunctionResponse needs a name, but it's the function name, not tool_call_id.
        # The response part of FunctionResponse is a dict.
        # Simulating "IN_PROGRESS" as a structured response.
        self._context.add_message(
            genai_types.Content(
                role="function",  # Or "user" if that's how Google expects tool responses. The old code used "user".
                                 # The new genai.types.Content role for function responses is "function".
                parts=[
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=frame.function_name, # Corresponds to FunctionCall.name
                            response={"status": "IN_PROGRESS"},
                        )
                    )
                ],
            )
        )

    async def handle_function_call_result(self, frame: FunctionCallResultFrame):
        result_payload = frame.result if frame.result else {"status": "COMPLETED"}
        await self._update_function_call_result(
            frame.function_name, frame.tool_call_id, result_payload # tool_call_id is pipecat internal
        )

    async def handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
        await self._update_function_call_result(
            frame.function_name, frame.tool_call_id, {"status": "CANCELLED"}
        )

    async def _update_function_call_result(
        self, function_name: str, tool_call_id: str, result: Any
    ):
        # The new SDK uses `role="function"` for responses.
        # The matching of tool_call_id is a pipecat internal mechanism,
        # Google's FunctionResponse matches by `name`.
        # We need to find the "IN_PROGRESS" response and update it.
        # The old logic iterated through messages, found a user role part with function_response.id == tool_call_id.
        # This needs to be adapted. `tool_call_id` is not part of Google's FunctionCall/Response model.
        # Pipecat's `OpenAIAssistantContextAggregator` has a more robust way of handling this by matching message contents.
        # For Google, the FunctionResponse part itself needs to be updated.
        # We search for the specific FunctionResponse part related to this function name that was marked IN_PROGRESS.
        # Since `tool_call_id` is not in Google's model, we rely on function_name and the assumption
        # that the "IN_PROGRESS" message is the one to update.
        for message in reversed(self._context.messages): # Iterate backwards to find the most recent
            if message.role == "function": # Or "user" if that was the choice for "IN_PROGRESS"
                for part in message.parts:
                    if part.function_response and part.function_response.name == function_name:
                        # Check if this is the one we marked as IN_PROGRESS
                        if isinstance(part.function_response.response, dict) and \
                           part.function_response.response.get("status") == "IN_PROGRESS":
                            part.function_response.response = {"value": json.dumps(result)}
                            return # Found and updated

        # If not found (e.g., if IN_PROGRESS was structured differently or not added),
        # we might need to add a new function response message.
        # This part needs to be robust. For now, assuming update of a pre-existing placeholder.
        logger.warning(f"Could not find IN_PROGRESS placeholder for function {function_name} to update result.")


    async def handle_user_image_frame(self, frame: UserImageRawFrame):
        # This assumes a tool call requested the image. Update its status.
        await self._update_function_call_result(
            frame.request.function_name, frame.request.tool_call_id, {"status": "COMPLETED_WITH_IMAGE"}
        )
        self._context.add_image_frame_message(
            format=frame.format,
            size=frame.size,
            image=frame.image,
            text=frame.request.context,
        )


@dataclass
class GoogleContextAggregatorPair:
    _user: GoogleUserContextAggregator
    _assistant: GoogleAssistantContextAggregator

    def user(self) -> GoogleUserContextAggregator:
        return self._user

    def assistant(self) -> GoogleAssistantContextAggregator:
        return self._assistant


class GoogleLLMContext(OpenAILLMContext):
    def __init__(
        self,
        messages: Optional[List[genai_types.Content]] = None, # Expect new types
        tools: Optional[List[genai_types.Tool]] = None,      # Expect new types
        tool_choice: Optional[Any] = None, # tool_choice format for genai?
                                           # genai_types.ToolConfig or specific string
    ):
        # Superclass expects List[ChatCompletionMessageParam], etc.
        # We need to handle this initial conversion if `messages` are passed in OpenAI format.
        # For now, let's assume messages are either already genai_types.Content or will be converted.
        # This init might need more sophisticated handling if mixed types are common at instantiation.
        super().__init__(messages=[], tools=tools, tool_choice=tool_choice) # Start with empty and add
        if messages:
            self.add_messages(messages)
        self.system_message_text: Optional[str] = None # Store the raw system message string

    @property
    def system_instruction_content(self) -> Optional[genai_types.Content]:
        if self.system_message_text:
            return genai_types.Content(parts=[genai_types.Part(text=self.system_message_text)], role="system") # Role "system" is conceptual here
        return None

    @staticmethod
    def upgrade_to_google(obj: OpenAILLMContext) -> "GoogleLLMContext":
        if not isinstance(obj, GoogleLLMContext):
            logger.debug(f"Upgrading OpenAI context to GoogleLLMContext: {obj}")
            # Create a new GoogleLLMContext and transfer relevant data
            # This is safer than changing __class__
            new_google_context = GoogleLLMContext()
            # Convert messages from OpenAI format to Google format
            standard_messages = obj.get_messages_for_persistent_storage() # Gets messages in standard OpenAI dict format
            converted_google_messages = []
            for std_msg in standard_messages:
                # Handle system message separately
                if std_msg.get("role") == "system" and isinstance(std_msg.get("content"), str):
                    new_google_context.system_message_text = std_msg["content"]
                    continue
                google_msg = new_google_context.from_standard_message(std_msg)
                if google_msg:
                    converted_google_messages.append(google_msg)
            new_google_context._messages = converted_google_messages

            # Transfer tools and tool_choice, adapting format if necessary
            # For now, assume they are compatible or handled by adapter
            if obj.tools and obj.tools is not genai_types.NOT_GIVEN: # NOT_GIVEN is from openai._types
                new_google_context._tools = obj.tools # This might need conversion
            if obj.tool_choice and obj.tool_choice is not genai_types.NOT_GIVEN:
                new_google_context._tool_choice = obj.tool_choice # This might need conversion

            new_google_context._llm_adapter = obj.get_llm_adapter()
            # new_google_context._restructure_from_openai_messages() # Call if still needed after conversion
            return new_google_context
        return obj


    def set_messages(self, messages: List[Any]): # Can be list of dicts or list of Content
        self._messages.clear()
        self.add_messages(messages)
        # self._restructure_from_openai_messages() # May need adjustment

    def add_messages(self, messages: List[Any]): # Can be list of dicts or list of Content
        converted_messages = []
        for msg in messages:
            if isinstance(msg, genai_types.Content):
                converted_messages.append(msg)
            elif isinstance(msg, dict): # Assuming it's OpenAI standard message format
                 # Handle system message separately
                if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                    self.system_message_text = msg["content"]
                    # System messages are not added to the main message list for Gemini API
                    continue
                converted = self.from_standard_message(msg)
                if converted:
                    converted_messages.append(converted)
            else:
                logger.warning(f"Skipping unknown message type during add_messages: {type(msg)}")

        self._messages.extend(converted_messages)
        # self._restructure_from_openai_messages()

    def get_messages_for_logging(self) -> List[Dict]:
        msgs = []
        for message in self.messages:
            # genai_types.Content are Pydantic models
            obj = message.model_dump(exclude_none=True)
            try:
                if "parts" in obj:
                    for part in obj["parts"]:
                        if "inline_data" in part and isinstance(part["inline_data"], dict):
                            # inline_data itself is a Blob, its 'data' field is bytes
                            # For logging, we might want to truncate or indicate presence of bytes
                            if "data" in part["inline_data"]:
                                 part["inline_data"]["data"] = f"<bytes len={len(part['inline_data']['data'])}>"
            except Exception as e:
                logger.debug(f"Error redacting message for logging: {e}")
            msgs.append(obj)
        return msgs

    def add_image_frame_message(
        self, *, format: str, size: tuple[int, int], image: bytes, text: Optional[str] = None
    ):
        buffer = io.BytesIO()
        Image.frombytes(format, size, image).save(buffer, format="JPEG")

        parts = []
        if text:
            parts.append(genai_types.Part(text=text))
        parts.append(genai_types.Part(inline_data=genai_types.Blob(
            mime_type="image/jpeg", data=buffer.getvalue())))

        # Images are typically user messages
        self.add_messages([genai_types.Content(role="user", parts=parts)])


    def add_audio_frames_message(
        self, *, audio_frames: list[AudioRawFrame], text: Optional[str] = None
    ):
        if not audio_frames:
            return

        sample_rate = audio_frames[0].sample_rate
        num_channels = audio_frames[0].num_channels

        audio_data = b"".join(frame.audio for frame in audio_frames)
        wav_data = self.create_wav_header(sample_rate, num_channels, 16, len(audio_data)) + audio_data

        parts = []
        if text:
            parts.append(genai_types.Part(text=text))
        parts.append(
            genai_types.Part(
                inline_data=genai_types.Blob(
                    mime_type="audio/wav",
                    data=wav_data,
                )
            )
        )
        self.add_messages([genai_types.Content(role="user", parts=parts)])

    def from_standard_message(self, message: Dict[str, Any]) -> Optional[genai_types.Content]:
        role = message.get("role")
        content = message.get("content") # Can be str or list of dicts

        # System messages are handled by setting self.system_message_text
        # and not converted to a genai_types.Content in the main list
        if role == "system":
            if isinstance(content, str):
                self.system_message_text = content
            elif isinstance(content, list) and content and isinstance(content[0].get("text"), str):
                self.system_message_text = content[0]["text"] # Take first text part for simplicity
            return None

        # Map roles: "assistant" -> "model", "tool" -> "function" (for responses)
        # "user" remains "user"
        if role == "assistant":
            google_role = "model"
        elif role == "tool":
            google_role = "function" # Role for function responses
        elif role == "user":
            google_role = "user"
        else:
            logger.warning(f"Unknown role '{role}' in standard message, defaulting to 'user'.")
            google_role = "user"

        parts = []
        if message.get("tool_calls"): # This is from an assistant/model
            google_role = "model" # Ensure role is model for function calls
            for tc in message["tool_calls"]:
                function_data = tc.get("function", {})
                try:
                    args = json.loads(function_data.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                    logger.warning(f"Could not parse arguments for function call {function_data.get('name')}")

                parts.append(
                    genai_types.Part(
                        function_call=genai_types.FunctionCall(
                            name=function_data.get("name"),
                            args=args,
                        )
                    )
                )
        elif role == "tool": # This is a response from a tool/function
            # Pipecat's standard "tool" role message has "content" (json string of result)
            # and "tool_call_id" (maps to function name for Google if simple).
            # Google expects FunctionResponse part with name and response (dict).
            try:
                response_content = json.loads(content) if isinstance(content, str) else content
            except json.JSONDecodeError:
                response_content = {"error": "failed to parse content", "original_content": content}
            parts.append(
                genai_types.Part(
                    function_response=genai_types.FunctionResponse(
                        name=message.get("name") or message.get("tool_call_id", "unknown_function"), # OpenAI uses name, Pipecat might use tool_call_id
                        response=response_content
                    )
                )
            )
        elif isinstance(content, str):
            parts.append(genai_types.Part(text=content))
        elif isinstance(content, list): # Multi-part content (e.g., text and image)
            for item in content:
                item_type = item.get("type")
                if item_type == "text":
                    parts.append(genai_types.Part(text=item.get("text")))
                elif item_type == "image_url":
                    image_url_data = item.get("image_url", {}).get("url", "")
                    if image_url_data.startswith("data:image"):
                        try:
                            mime_type, b64_data = image_url_data.split(';base64,')
                            mime_type = mime_type.split(':')[1]
                            image_bytes = base64.b64decode(b64_data)
                            parts.append(genai_types.Part(inline_data=genai_types.Blob(
                                mime_type=mime_type, data=image_bytes
                            )))
                        except Exception as e:
                            logger.error(f"Error decoding image_url data: {e}")
                elif item_type == "audio_url": # Assuming similar base64 data URI
                    audio_url_data = item.get("audio_url", {}).get("url", "")
                    if audio_url_data.startswith("data:audio"):
                        try:
                            mime_type, b64_data = audio_url_data.split(';base64,')
                            mime_type = mime_type.split(':')[1]
                            audio_bytes = base64.b64decode(b64_data)
                            parts.append(genai_types.Part(inline_data=genai_types.Blob(
                                mime_type=mime_type, data=audio_bytes
                            )))
                        except Exception as e:
                            logger.error(f"Error decoding audio_url data: {e}")
        if not parts:
            return None
        return genai_types.Content(role=google_role, parts=parts)

    def to_standard_messages(self, google_content: genai_types.Content) -> List[Dict[str, Any]]:
        # Converts a single google.genai.types.Content object to Pipecat's standard list-of-dicts format.
        # A single Google Content can map to one standard message.
        std_msg: Dict[str, Any] = {}

        if google_content.role == "model":
            std_msg["role"] = "assistant"
        elif google_content.role == "function": # Google's role for function responses
            std_msg["role"] = "tool"
        else: # "user"
            std_msg["role"] = google_content.role

        content_parts = []
        tool_calls = []

        for part in google_content.parts:
            if part.text:
                content_parts.append({"type": "text", "text": part.text})
            elif part.inline_data:
                mime_type = part.inline_data.mime_type
                encoded_data = base64.b64encode(part.inline_data.data).decode("utf-8")
                data_url = f"data:{mime_type};base64,{encoded_data}"
                if mime_type.startswith("image/"):
                    content_parts.append({"type": "image_url", "image_url": {"url": data_url}})
                elif mime_type.startswith("audio/"):
                    content_parts.append({"type": "audio_url", "audio_url": {"url": data_url}})
            elif part.function_call:
                # This implies the assistant (model) is requesting a function call
                std_msg["role"] = "assistant" # Ensure role is assistant
                tool_calls.append({
                    "id": part.function_call.name, # Use function name as ID, as OpenAI does
                    "type": "function",
                    "function": {
                        "name": part.function_call.name,
                        "arguments": json.dumps(part.function_call.args or {}),
                    },
                })
            elif part.function_response:
                # This is a response from a tool/function
                std_msg["role"] = "tool" # Standard role for tool responses
                std_msg["tool_call_id"] = part.function_response.name # Name of the function that was called
                # The response from google.genai.types.FunctionResponse is already a dict.
                # Pipecat expects the "content" of a tool message to be a JSON string of this response.
                std_msg["content"] = json.dumps(part.function_response.response or {})
                # Clear other content parts if this is a function response message
                content_parts = []
                break # A function_response part defines the whole message as a tool response

        if tool_calls:
            std_msg["tool_calls"] = tool_calls
        
        if content_parts:
            # If only one text part, simplify to string content, else list of dicts
            if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                std_msg["content"] = content_parts[0]["text"]
            else:
                std_msg["content"] = content_parts
        elif not tool_calls and "content" not in std_msg : # Ensure content field exists if no tool_calls and not a tool response
             std_msg["content"] = None if std_msg["role"] == "assistant" else ""


        return [std_msg]


    def _restructure_from_openai_messages(self):
        # This method might need to be re-evaluated.
        # The primary goal is to ensure messages are in Google's format.
        # System messages are handled by self.system_message_text.
        # The main self._messages list should only contain user/model/function messages.
        
        openai_formatted_messages = self._messages # These might be dicts from OpenAI context
        self._messages = [] # Reset to store converted genai_types.Content

        temp_system_message_text = None

        for msg_data in openai_formatted_messages:
            if isinstance(msg_data, genai_types.Content): # Already converted
                self._messages.append(msg_data)
                continue
            
            # Assuming msg_data is a dict in OpenAI standard format
            role = msg_data.get("role")
            content = msg_data.get("content")

            if role == "system":
                if isinstance(content, str):
                    temp_system_message_text = content
                elif isinstance(content, list) and content and isinstance(content[0].get("text"), str):
                    temp_system_message_text = content[0]["text"]
                continue # System messages are not added to self._messages directly

            converted_msg = self.from_standard_message(msg_data)
            if converted_msg:
                self._messages.append(converted_msg)
        
        if temp_system_message_text:
            self.system_message_text = temp_system_message_text
        
        # Remove any empty messages (e.g. if a conversion resulted in None and wasn't filtered)
        self._messages = [m for m in self._messages if m and getattr(m, 'parts', None)]


class GoogleLLMService(LLMService):
    adapter_class = GeminiLLMAdapter

    class InputParams(BaseModel):
        max_tokens: Optional[int] = Field(default=None, ge=1) # Defaulting to None, let model decide or be set by GenerationConfig
        temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
        top_k: Optional[int] = Field(default=None, ge=0)
        top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
        extra: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-1.5-flash", # Updated default model
        params: InputParams = InputParams(),
        system_instruction: Optional[str] = None, # This is the string form
        tools: Optional[List[Dict[str, Any]]] = None, # Standard tool format
        tool_config: Optional[Dict[str, Any]] = None, # Standard tool_config format
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Configure the genai client. API key is passed here.
        # The genai.Client handles underlying API configuration.
        try:
            self._google_client = genai.Client(api_key=api_key)
        except GoogleAuthError as e:
            logger.error(f"Google Authentication Error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Google GenAI Client: {e}")
            raise

        self._google_model_service = self._google_client.models # Service to call generate_content_stream

        self.set_model_name(model) # Stores self._model_name
        self._system_instruction_text = system_instruction # Store raw string
        
        self._settings = {
            "max_output_tokens": params.max_tokens, # genai uses max_output_tokens
            "temperature": params.temperature,
            "top_k": params.top_k,
            "top_p": params.top_p,
            "extra": params.extra if isinstance(params.extra, dict) else {},
        }
        
        # Tools and tool_config are stored in their standard format.
        # They will be converted by the adapter before being sent to the API.
        self._tools_standard_format = tools
        self._tool_config_standard_format = tool_config

    def can_generate_metrics(self) -> bool:
        return True

    # _create_client is no longer needed as self._google_model_service is initialized in __init__

    async def _process_context(self, context: GoogleLLMContext): # Expects GoogleLLMContext
        await self.push_frame(LLMFullResponseStartFrame())

        prompt_tokens = 0
        completion_tokens = 0
        # total_tokens will be derived

        grounding_metadata_dict: Optional[Dict[str, Any]] = None
        search_result_text = ""

        try:
            logger.debug(f"{self}: Generating chat for model {self._model_name} with context: [{context.get_messages_for_logging()}]")

            messages_for_api = context.messages # These should be List[genai_types.Content]
            
            # Prepare GenerationConfig
            generation_config_params = {k: v for k, v in self._settings.items() if v is not None and k != "extra"}
            
            # Handle system instruction
            current_system_instruction_text = context.system_message_text if context.system_message_text else self._system_instruction_text
            if current_system_instruction_text:
                generation_config_params["system_instruction"] = genai_types.Content(
                    parts=[genai_types.Part(text=current_system_instruction_text)],
                    # role="system" is not directly part of genai_types.Content for system_instruction,
                    # it's implicitly understood by its placement in GenerationConfig.
                )

            final_generation_config = genai_types.GenerationConfig(**generation_config_params) if generation_config_params else None

            # Prepare tools and tool_config using the adapter
            # The context.tools and context.tool_choice should be in the target format or adaptable
            api_tools = None
            if context.tools not in [None, genai_types.NOT_GIVEN]: # NOT_GIVEN is from openai._types
                api_tools = self.get_llm_adapter().from_standard_tools(context.tools)
            elif self._tools_standard_format:
                api_tools = self.get_llm_adapter().from_standard_tools(self._tools_standard_format)

            api_tool_config = None
            # Assuming tool_choice from context or self._tool_config_standard_format needs to be converted
            # to genai_types.ToolConfig. This is also adapter's job.
            # For example, context.tool_choice could be a string like "auto" or a specific function dict.
            # This needs careful mapping in the GeminiLLMAdapter.
            # For now, let's assume adapter handles it or it's passed if compatible.
            raw_tool_choice = context.tool_choice if context.tool_choice is not genai_types.NOT_GIVEN else self._tool_config_standard_format

            if raw_tool_choice:
                 api_tool_config = self.get_llm_adapter().from_standard_tool_choice(raw_tool_choice, api_tools)


            await self.start_ttfb_metrics()
            
            stream_response = await self._google_model_service.generate_content_stream(
                model=self._model_name,
                contents=messages_for_api,
                tools=api_tools, # Expects List[genai_types.Tool]
                generation_config=final_generation_config,
                tool_config=api_tool_config, # Expects genai_types.ToolConfig
            )
            await self.stop_ttfb_metrics()

            final_usage_metadata = None
            async for chunk in stream_response:
                if chunk.usage_metadata:
                    # Per google-genai docs, usage_metadata is only in the first chunk for streams.
                    # If so, this will capture it. If it's in the last, this will also capture the last.
                    final_usage_metadata = chunk.usage_metadata
                
                if chunk.candidates:
                    for candidate_idx, candidate in enumerate(chunk.candidates):
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if part.text:
                                    search_result_text += part.text
                                    await self.push_frame(LLMTextFrame(part.text))
                                elif part.function_call:
                                    logger.debug(f"Function call from model: {part.function_call.name} with args {part.function_call.args}")
                                    await self.call_function(
                                        context=context, # Pass GoogleLLMContext
                                        tool_call_id=str(uuid.uuid4()), # Pipecat internal ID
                                        function_name=part.function_call.name,
                                        arguments=part.function_call.args or {}, # args is already a dict
                                    )
                        
                        # Grounding metadata processing (needs careful adaptation)
                        if candidate.grounding_metadata:
                            gm = candidate.grounding_metadata
                            origins_list = []
                            rendered_web_content = None
                            if gm.search_entry_point:
                                rendered_web_content = gm.search_entry_point.rendered_content

                            if gm.grounding_attributions:
                                for attr in gm.grounding_attributions:
                                    origin_entry: Dict[str, Any] = {"results": []}
                                    source_uri = None
                                    source_title = None
                                    if attr.web:
                                        source_uri = attr.web.uri
                                        source_title = attr.web.title
                                    elif attr.retrieved_context: # For RAG
                                        source_uri = attr.retrieved_context.uri
                                        source_title = attr.retrieved_context.title
                                    
                                    origin_entry["site_uri"] = source_uri
                                    origin_entry["site_title"] = source_title

                                    # Simplified result extraction:
                                    # Take text from attr.content if available, and confidence score
                                    result_text = ""
                                    if attr.content and attr.content.parts:
                                        for p in attr.content.parts:
                                            if p.text:
                                                result_text += p.text + " "
                                    
                                    origin_entry["results"].append({
                                        "text": result_text.strip(),
                                        "confidence": attr.confidence_score if hasattr(attr, 'confidence_score') else None
                                    })
                                    origins_list.append(origin_entry)
                            
                            grounding_metadata_dict = {
                                "rendered_content": rendered_web_content,
                                "origins": origins_list
                            }
                
                # Handle finish reason for logging safety issues
                if chunk.candidates and chunk.candidates[0].finish_reason:
                    if str(chunk.candidates[0].finish_reason) == "SAFETY": # Enum comparison
                        logger.warning(f"LLM refused to generate content due to safety reasons. Prompt: {messages_for_api}")
                    elif str(chunk.candidates[0].finish_reason) not in ["STOP", "MAX_TOKENS", "UNSPECIFIED"]: # Other non-normal finish reasons
                        logger.warning(f"LLM generation stopped due to: {chunk.candidates[0].finish_reason}. Candidate: {chunk.candidates[0]}")


        except api_core_exceptions.DeadlineExceeded: # Keep this if other google libs might throw it
            logger.warning(f"{self} completion timeout (DeadlineExceeded)")
            await self._call_event_handler("on_completion_timeout")
        except genai.types.BlockedPromptException as e:
            logger.warning(f"{self} prompt was blocked: {e}")
            # Potentially push a specific frame or call an event handler
        except genai.types.StopCandidateException as e: # Candidate finished due to safety or other reasons
            logger.warning(f"{self} candidate generation stopped: {e}")
        except genai.types.BrokenResponseError as e: # If stream breaks
            logger.error(f"{self} stream broken: {e}")
        except GoogleAuthError as e:
            logger.error(f"{self} Google Authentication Error: {e}")
            # Potentially re-raise or handle as a critical error
            raise
        except genai.errors.GoogleAPIError as e: # Catch specific Google API errors
            logger.error(f"{self} Google API Error: {e}")
        except Exception as e:
            logger.exception(f"{self} encountered an unhandled exception: {e}")
        finally:
            if grounding_metadata_dict:
                llm_search_frame = LLMSearchResponseFrame(
                    search_result=search_result_text, # The aggregated text response
                    origins=grounding_metadata_dict.get("origins", []),
                    rendered_content=grounding_metadata_dict.get("rendered_content"),
                )
                await self.push_frame(llm_search_frame)

            if final_usage_metadata:
                prompt_tokens = final_usage_metadata.prompt_token_count if final_usage_metadata.prompt_token_count else 0
                # candidates_token_count in the new SDK's usage_metadata (when present, e.g. first chunk)
                # refers to "Tokens in the generated candidate(s)" for that response object.
                # If it's only in the first chunk, it's the total completion.
                completion_tokens = final_usage_metadata.candidates_token_count if final_usage_metadata.candidates_token_count else 0
                # total_tokens from metadata, or sum if more reliable
                # total_tokens_from_meta = final_usage_metadata.total_token_count if final_usage_metadata.total_token_count else 0
                # Calculated total_tokens is safer for pipecat's metrics
                calculated_total_tokens = prompt_tokens + completion_tokens
            else: # Fallback if no usage metadata was found
                prompt_tokens = 0
                completion_tokens = 0 # Cannot be determined without metadata
                calculated_total_tokens = 0


            await self.start_llm_usage_metrics(
                LLMTokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=calculated_total_tokens,
                )
            )
            await self.push_frame(LLMFullResponseEndFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        context = None

        if isinstance(frame, OpenAILLMContextFrame):
            # Ensure it's a GoogleLLMContext, upgrading if necessary
            if not isinstance(frame.context, GoogleLLMContext):
                 frame.context = GoogleLLMContext.upgrade_to_google(frame.context)
            context = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            # Create a new GoogleLLMContext from OpenAI standard messages
            # The LLMMessagesFrame contains messages in OpenAI dict format.
            google_context = GoogleLLMContext()
            # LLMMessagesFrame.messages are List[Dict]
            # System message needs to be extracted if present.
            raw_messages = frame.messages
            system_msg_text = None
            user_model_messages = []
            for msg_dict in raw_messages:
                if msg_dict.get("role") == "system" and isinstance(msg_dict.get("content"), str):
                    system_msg_text = msg_dict["content"]
                else:
                    user_model_messages.append(msg_dict)
            
            if system_msg_text:
                google_context.system_message_text = system_msg_text
            
            google_context.add_messages(user_model_messages) # Converts and adds
            context = google_context

        elif isinstance(frame, VisionImageRawFrame):
            context = GoogleLLMContext()
            # If there's a system instruction associated with this service instance, set it.
            if self._system_instruction_text:
                context.system_message_text = self._system_instruction_text
            context.add_image_frame_message(
                format=frame.format, size=frame.size, image=frame.image, text=frame.text
            )
        elif isinstance(frame, LLMUpdateSettingsFrame):
            # Ensure max_tokens is mapped to max_output_tokens
            if "max_tokens" in frame.settings:
                frame.settings["max_output_tokens"] = frame.settings.pop("max_tokens")
            await self._update_settings(frame.settings)
        else:
            await self.push_frame(frame, direction)

        if context:
            await self._process_context(context)

    def create_context_aggregator(
        self,
        context: OpenAILLMContext, # Input can be the base OpenAILLMContext
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> GoogleContextAggregatorPair:
        
        # Ensure we are working with a GoogleLLMContext instance
        if not isinstance(context, GoogleLLMContext):
            google_context = GoogleLLMContext.upgrade_to_google(context)
        else:
            google_context = context
        
        # If the original context was plain OpenAILLMContext, it might not have system_message_text set.
        # If this service has a default system instruction, apply it here.
        if not google_context.system_message_text and self._system_instruction_text:
            google_context.system_message_text = self._system_instruction_text

        google_context.set_llm_adapter(self.get_llm_adapter())

        user = GoogleUserContextAggregator(google_context, params=user_params)
        assistant = GoogleAssistantContextAggregator(google_context, params=assistant_params)
        return GoogleContextAggregatorPair(_user=user, _assistant=assistant)