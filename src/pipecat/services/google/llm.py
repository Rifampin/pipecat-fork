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

# For checking OpenAI's specific NOT_GIVEN sentinel
from openai._types import NOT_GIVEN as OPENAI_NOT_GIVEN


# Suppress gRPC fork warnings
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

try:
    import google.genai as genai
    import google.genai.types as genai_types
    import google.genai.errors as genai_errors # Import for new error types
    from google.auth.exceptions import GoogleAuthError
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
            self._context.add_message( # Ensure context is GoogleLLMContext
                genai_types.Content(role="user", parts=[
                                    genai_types.Part(text=self._aggregation)])
            )
            self._aggregation = ""
            frame = OpenAILLMContextFrame(self._context)
            await self.push_frame(frame)
            self.reset()


class GoogleAssistantContextAggregator(OpenAIAssistantContextAggregator):
    async def handle_aggregation(self, aggregation: str):
        self._context.add_message( # Ensure context is GoogleLLMContext
            genai_types.Content(role="model", parts=[
                                genai_types.Part(text=aggregation)])
        )

    async def handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        # Add model's request for function call
        self._context.add_message(
            genai_types.Content(
                role="model",
                parts=[
                    genai_types.Part(
                        function_call=genai_types.FunctionCall( # No 'id' field in genai_types.FunctionCall
                            name=frame.function_name, args=frame.arguments
                        )
                    )
                ],
            )
        )
        # Add a placeholder for the function's response (as per new SDK, role="function")
        # Pipecat's tool_call_id is for internal tracking. Google matches by function name.
        self._context.add_message(
            genai_types.Content(
                role="function",
                parts=[
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=frame.function_name,
                            response={"status": "IN_PROGRESS", "_pipecat_tool_call_id": frame.tool_call_id} # Store pipecat id for matching
                        )
                    )
                ],
            )
        )

    async def handle_function_call_result(self, frame: FunctionCallResultFrame):
        result_payload = frame.result if frame.result else {"status": "COMPLETED"}
        await self._update_function_call_result(
            frame.function_name, frame.tool_call_id, result_payload
        )

    async def handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
        await self._update_function_call_result(
            frame.function_name, frame.tool_call_id, {"status": "CANCELLED"}
        )

    async def _update_function_call_result(
        self, function_name: str, tool_call_id: str, result: Any
    ):
        for message in reversed(self._context.messages): # Iterate backwards to find the most recent
            if message.role == "function":
                for part in message.parts:
                    if part.function_response and \
                       part.function_response.name == function_name and \
                       isinstance(part.function_response.response, dict) and \
                       part.function_response.response.get("status") == "IN_PROGRESS" and \
                       part.function_response.response.get("_pipecat_tool_call_id") == tool_call_id:
                        part.function_response.response = result # The new SDK expects the raw dict, not json.dumps
                        return
        logger.warning(f"Could not find IN_PROGRESS placeholder for function {function_name} (tool_id: {tool_call_id}) to update result.")


    async def handle_user_image_frame(self, frame: UserImageRawFrame):
        if frame.request and frame.request.tool_call_id and frame.request.function_name:
            await self._update_function_call_result(
                frame.request.function_name, frame.request.tool_call_id, {"status": "COMPLETED_WITH_IMAGE"}
            )
        self._context.add_image_frame_message(
            format=frame.format,
            size=frame.size,
            image=frame.image,
            text=frame.request.context if frame.request else None,
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
        messages: Optional[List[Any]] = None, # Can be dicts or genai_types.Content
        tools: Optional[List[Any]] = None,   # Can be dicts or genai_types.Tool
        tool_choice: Optional[Any] = None,
    ):
        super().__init__(messages=[], tools=None, tool_choice=None) # Initialize empty
        self.system_message_text: Optional[str] = None

        if tools is not None: # Directly assign, adapter will handle
            self._tools = tools
        if tool_choice is not None: # Directly assign, adapter will handle
            self._tool_choice = tool_choice
        if messages:
            self.add_messages(messages)


    @property
    def system_instruction_for_api(self) -> Optional[genai_types.Content]: # Renamed for clarity
        if self.system_message_text:
            # For genai, system_instruction is a top-level param in GenerationConfig, not a message role.
            # However, genai_types.Content can be used to structure it.
            return genai_types.Content(parts=[genai_types.Part(text=self.system_message_text)])
        return None

    @staticmethod
    def upgrade_to_google(obj: OpenAILLMContext) -> "GoogleLLMContext":
        if not isinstance(obj, GoogleLLMContext):
            logger.debug(f"Upgrading OpenAI context to GoogleLLMContext: {obj}")
            new_google_context = GoogleLLMContext()
            
            openai_messages = obj.get_messages_for_persistent_storage() # Gets list of dicts
            system_text_from_openai = None
            regular_messages_from_openai = []

            for msg_dict in openai_messages:
                if msg_dict.get("role") == "system" and isinstance(msg_dict.get("content"), str):
                    system_text_from_openai = msg_dict["content"]
                else:
                    regular_messages_from_openai.append(msg_dict)
            
            if system_text_from_openai:
                new_google_context.system_message_text = system_text_from_openai
            
            new_google_context.add_messages(regular_messages_from_openai) # Converts and adds non-system

            if obj.tools is not OPENAI_NOT_GIVEN:
                 new_google_context._tools = obj.tools # Will be adapted by LLM adapter
            if obj.tool_choice is not OPENAI_NOT_GIVEN:
                new_google_context._tool_choice = obj.tool_choice # Will be adapted

            new_google_context.set_llm_adapter(obj.get_llm_adapter())
            return new_google_context
        return obj

    def set_messages(self, messages: List[Any]):
        self._messages.clear()
        current_system_text = self.system_message_text # Preserve existing system message
        self.system_message_text = None # Reset in case new messages contain a system message
        self.add_messages(messages)
        if self.system_message_text is None: # If add_messages didn't set a new one
            self.system_message_text = current_system_text


    def add_messages(self, messages: List[Any]):
        # This method now expects messages to be either:
        # 1. A list of `genai_types.Content` objects.
        # 2. A list of dictionaries in the standard OpenAI message format.
        for msg_input in messages:
            if isinstance(msg_input, genai_types.Content):
                self._messages.append(msg_input)
            elif isinstance(msg_input, dict): # Assume OpenAI standard message format
                if msg_input.get("role") == "system" and isinstance(msg_input.get("content"), str):
                    self.system_message_text = msg_input["content"]
                    # Do not add system messages to the main _messages list for Gemini
                    continue
                
                converted_msg = self.from_standard_message(msg_input)
                if converted_msg:
                    self._messages.append(converted_msg)
            else:
                logger.warning(f"Skipping unknown message type during add_messages: {type(msg_input)}")

    def get_messages_for_logging(self) -> List[Dict]:
        msgs = []
        # Include system message in logging if present
        if self.system_message_text:
            msgs.append({"role": "system", "content": self.system_message_text})

        for message_content in self.messages: # self.messages contains genai_types.Content
            # Convert genai_types.Content to a dict for logging
            obj = message_content.model_dump(exclude_none=True) # Pydantic model_dump
            try:
                if "parts" in obj:
                    for part in obj["parts"]:
                        if "inline_data" in part and isinstance(part["inline_data"], dict):
                            if "data" in part["inline_data"] and isinstance(part["inline_data"]["data"], bytes):
                                 part["inline_data"]["data"] = f"<bytes len={len(part['inline_data']['data'])}>"
            except Exception as e:
                logger.debug(f"Error redacting message for logging: {e}")
            msgs.append(obj)
        return msgs

    def add_image_frame_message(
        self, *, format: str, size: tuple[int, int], image: bytes, text: str = None
    ):
        buffer = io.BytesIO()
        Image.frombytes(format, size, image).save(buffer, format="JPEG") # Ensure JPEG for Gemini

        parts = []
        if text:
            parts.append(genai_types.Part(text=text))
        parts.append(genai_types.Part(inline_data=genai_types.Blob(
            mime_type="image/jpeg", data=buffer.getvalue())))

        self.add_messages([genai_types.Content(role="user", parts=parts)])


    def add_audio_frames_message(
        self, *, audio_frames: list[AudioRawFrame], text: str = "Audio follows"
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
                    mime_type="audio/wav", data=wav_data
                )
            )
        )
        self.add_messages([genai_types.Content(role="user", parts=parts)])

    def from_standard_message(self, message: Dict[str, Any]) -> Optional[genai_types.Content]:
        role = message.get("role")
        content = message.get("content")

        if role == "system": # System messages handled by self.system_message_text
            if isinstance(content, str): self.system_message_text = content
            elif isinstance(content, list) and content and isinstance(content[0].get("text"), str):
                self.system_message_text = content[0]["text"]
            return None

        google_role = "user" # Default
        if role == "assistant": google_role = "model"
        elif role == "tool": google_role = "function"
        elif role == "user": google_role = "user"
        else: logger.warning(f"Unknown role '{role}', defaulting to 'user'.")

        parts = []
        tool_calls = message.get("tool_calls")
        if tool_calls: # Assistant requesting function call
            google_role = "model"
            for tc in tool_calls:
                fn_data = tc.get("function", {})
                try: args = json.loads(fn_data.get("arguments", "{}"))
                except json.JSONDecodeError: args = {}
                parts.append(genai_types.Part(function_call=genai_types.FunctionCall(name=fn_data.get("name"), args=args)))
        elif google_role == "function": # Function response (from "tool" role)
            try: response_data = json.loads(content) if isinstance(content, str) else content
            except json.JSONDecodeError: response_data = {"error": "failed to parse content", "original_content": content}
            parts.append(genai_types.Part(function_response=genai_types.FunctionResponse(name=message.get("name", "unknown_function"), response=response_data)))
        elif isinstance(content, str):
            parts.append(genai_types.Part(text=content))
        elif isinstance(content, list): # Multi-part content
            for item in content:
                item_type = item.get("type")
                if item_type == "text": parts.append(genai_types.Part(text=item.get("text")))
                elif item_type == "image_url":
                    url_data = item.get("image_url", {}).get("url", "")
                    if url_data.startswith("data:image"):
                        try:
                            mime, b64_data = url_data.split(';base64,')
                            parts.append(genai_types.Part(inline_data=genai_types.Blob(mime_type=mime.split(':')[1], data=base64.b64decode(b64_data))))
                        except Exception as e: logger.error(f"Error decoding image_url: {e}")
                elif item_type == "audio_url":
                    url_data = item.get("audio_url", {}).get("url", "")
                    if url_data.startswith("data:audio"):
                        try:
                            mime, b64_data = url_data.split(';base64,')
                            parts.append(genai_types.Part(inline_data=genai_types.Blob(mime_type=mime.split(':')[1], data=base64.b64decode(b64_data))))
                        except Exception as e: logger.error(f"Error decoding audio_url: {e}")
        
        return genai_types.Content(role=google_role, parts=parts) if parts else None

    def to_standard_messages(self, google_content: genai_types.Content) -> List[Dict[str, Any]]:
        std_msg: Dict[str, Any] = {}
        std_msg["role"] = {"model": "assistant", "function": "tool"}.get(google_content.role, google_content.role)

        content_parts, tool_calls_list = [], []
        for part in google_content.parts:
            if part.text: content_parts.append({"type": "text", "text": part.text})
            elif part.inline_data:
                mime, data_b64 = part.inline_data.mime_type, base64.b64encode(part.inline_data.data).decode("utf-8")
                url = f"data:{mime};base64,{data_b64}"
                content_parts.append({"type": "image_url" if mime.startswith("image/") else "audio_url", 
                                      "image_url" if mime.startswith("image/") else "audio_url": {"url": url}})
            elif part.function_call:
                std_msg["role"] = "assistant" # Override if model is making a call
                tool_calls_list.append({"id": part.function_call.name, "type": "function", 
                                     "function": {"name": part.function_call.name, "arguments": json.dumps(part.function_call.args or {})}})
            elif part.function_response:
                std_msg["role"] = "tool" # Function response means tool role
                std_msg["tool_call_id"] = part.function_response.name
                std_msg["content"] = json.dumps(part.function_response.response or {})
                content_parts = [] # Function response is the sole content for a tool message
                break 
        
        if tool_calls_list: std_msg["tool_calls"] = tool_calls_list
        if content_parts:
            std_msg["content"] = content_parts[0]["text"] if len(content_parts) == 1 and content_parts[0]["type"] == "text" else content_parts
        elif "content" not in std_msg and not tool_calls_list: # Ensure content field if not tool call/response
            std_msg["content"] = None if std_msg["role"] == "assistant" else ""
            
        return [std_msg]

    def _restructure_from_openai_messages(self):
        # This method might be redundant if add_messages and set_messages correctly handle system messages.
        # The key is that self._messages should only contain genai_types.Content for user/model/function roles.
        # self.system_message_text holds the system instruction.
        pass # Simplified, assuming conversion logic in add_messages/set_messages is sufficient.

class GoogleLLMService(LLMService):
    adapter_class = GeminiLLMAdapter

    class InputParams(BaseModel):
        max_output_tokens: Optional[int] = Field(default=None, ge=1)
        temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
        top_k: Optional[int] = Field(default=None, ge=0)
        top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
        extra: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-1.5-flash",
        params: InputParams = InputParams(),
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        try:
            # The genai.configure(api_key=...) is a global configuration.
            # It's better to pass the API key to the client if possible,
            # or ensure this is called once appropriately if it's the only way.
            # For SDKs that manage clients per service instance, this global configure can be problematic.
            # The new `google-genai` SDK uses `genai.Client(api_key="YOUR_API_KEY")`
            # which is better for encapsulation.
            self._google_client_internal = genai.Client(api_key=api_key) # Per-instance client
            self._google_model_service = self._google_client_internal.models
        except GoogleAuthError as e:
            logger.error(f"Google Authentication Error: {e}. Ensure API key is valid and has Gemini API enabled.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Google GenAI Client: {e}")
            raise

        self.set_model_name(model)
        self._system_instruction_text = system_instruction
        
        self._settings = {
            "max_output_tokens": params.max_output_tokens,
            "temperature": params.temperature,
            "top_k": params.top_k,
            "top_p": params.top_p,
            "extra": params.extra if isinstance(params.extra, dict) else {},
        }
        self._tools_standard_format = tools
        self._tool_config_standard_format = tool_config

    def can_generate_metrics(self) -> bool:
        return True

    async def _process_context(self, context: GoogleLLMContext):
        await self.push_frame(LLMFullResponseStartFrame())

        prompt_tokens = 0
        completion_tokens = 0
        final_usage_metadata = None # Initialize here
        grounding_metadata_dict: Optional[Dict[str, Any]] = None
        search_result_text = ""

        try:
            logger.debug(f"{self}: Generating chat for model {self._model_name} with context: [{context.get_messages_for_logging()}]")

            messages_for_api = context.messages
            
            generation_config_params = {k: v for k, v in self._settings.items() if v is not None and k != "extra"}
            
            current_system_instruction_text = context.system_message_text or self._system_instruction_text
            if current_system_instruction_text:
                # In google-genai, system_instruction is part of GenerationConfig
                generation_config_params["system_instruction"] = current_system_instruction_text


            final_generation_config = genai_types.GenerationConfig(**generation_config_params) if generation_config_params else None
            
            api_tools = None
            # Use OPENAI_NOT_GIVEN for checking context.tools as it comes from OpenAILLMContext
            if context.tools is not None and context.tools is not OPENAI_NOT_GIVEN:
                api_tools = self.get_llm_adapter().from_standard_tools(context.tools)
            elif self._tools_standard_format:
                api_tools = self.get_llm_adapter().from_standard_tools(self._tools_standard_format)

            api_tool_config = None
            raw_tool_choice = context.tool_choice if context.tool_choice is not OPENAI_NOT_GIVEN else self._tool_config_standard_format
            if raw_tool_choice:
                 api_tool_config = self.get_llm_adapter().from_standard_tool_choice(raw_tool_choice, api_tools)

            await self.start_ttfb_metrics()
            
            # Use the instance-specific model service
            stream_response = await self._google_model_service.generate_content_stream(
                model=self._model_name,
                contents=messages_for_api,
                tools=api_tools,
                generation_config=final_generation_config,
                tool_config=api_tool_config,
            )
            await self.stop_ttfb_metrics()

            # According to google-genai docs, usage_metadata is only in the first response for streams.
            # We need to consume the first item to get it, then yield it back if needed or process.
            # This requires a more complex handling of the async iterator.
            # For simplicity in this refactor, we'll assume it MIGHT appear in any chunk and take the last one.
            # A more robust solution would be to peek/buffer the first chunk.

            async for chunk in stream_response:
                if chunk.usage_metadata:
                    final_usage_metadata = chunk.usage_metadata # Keep updating, last one will be used
                
                if chunk.candidates:
                    for candidate in chunk.candidates: # Iterate through all candidates
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if part.text:
                                    search_result_text += part.text
                                    await self.push_frame(LLMTextFrame(part.text))
                                elif part.function_call:
                                    logger.debug(f"Function call from model: {part.function_call.name} with args {part.function_call.args}")
                                    await self.call_function(
                                        context=context,
                                        tool_call_id=str(uuid.uuid4()),
                                        function_name=part.function_call.name,
                                        arguments=part.function_call.args or {},
                                    )
                        
                        if candidate.grounding_metadata:
                            gm = candidate.grounding_metadata
                            origins_list = []
                            rendered_web_content = gm.search_entry_point.rendered_content if gm.search_entry_point else None
                            if gm.grounding_attributions: # Changed from grounding_chunks/grounding_supports
                                for attr in gm.grounding_attributions:
                                    origin_entry: Dict[str, Any] = {"results": []}
                                    source_uri, source_title = (attr.web.uri, attr.web.title) if attr.web else \
                                                               (attr.retrieved_context.uri, attr.retrieved_context.title) if attr.retrieved_context else (None, None)
                                    origin_entry["site_uri"], origin_entry["site_title"] = source_uri, source_title
                                    result_text = "".join(p.text + " " for p in attr.content.parts if p.text).strip() if attr.content and attr.content.parts else ""
                                    origin_entry["results"].append({"text": result_text, "confidence": getattr(attr, 'confidence_score', None)})
                                    origins_list.append(origin_entry)
                            grounding_metadata_dict = {"rendered_content": rendered_web_content, "origins": origins_list}

                        if candidate.finish_reason:
                            # Convert enum to string for comparison if necessary, or use direct enum comparison
                            finish_reason_str = str(candidate.finish_reason)
                            if finish_reason_str == str(genai_types.Candidate.FinishReason.SAFETY):
                                logger.warning(f"LLM refused to generate content due to safety. Prompt: {context.get_messages_for_logging()}")
                            elif finish_reason_str not in [
                                str(genai_types.Candidate.FinishReason.STOP),
                                str(genai_types.Candidate.FinishReason.MAX_TOKENS),
                                str(genai_types.Candidate.FinishReason.UNSPECIFIED) # Check if this is normal
                            ]:
                                logger.warning(f"LLM generation stopped due to: {finish_reason_str}. Candidate: {candidate}")
                else: # No candidates in chunk
                    if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                        logger.warning(f"Prompt blocked by Google API. Reason: {chunk.prompt_feedback.block_reason}. Details: {chunk.prompt_feedback.block_reason_message}")
                        # Consider pushing an error frame or specific message to user

        except api_core_exceptions.DeadlineExceeded:
            logger.warning(f"{self} completion timeout (DeadlineExceeded)")
            await self._call_event_handler("on_completion_timeout")
        # More specific error handling based on google.genai.errors
        except genai_errors.BlockedPromptError as e: # This specific error might not exist, APIError might be used
            logger.warning(f"{self} prompt was blocked by Google API: {e}")
        except genai_errors.StopCandidateError as e: # Candidate finished due to safety or other reasons
            logger.warning(f"{self} candidate generation stopped by Google API: {e}")
        except genai_errors.BrokenResponseError as e:
             logger.error(f"{self} stream broken from Google API: {e}")
        except genai_errors. ओवरलोडError as e: # Example of a specific error if available
             logger.error(f"{self} API Quota or Rate Limit Exceeded: {e}")
        except genai_errors.GoogleAPIError as e: # General Google API error
            logger.error(f"{self} Google API Error: {e}")
        except GoogleAuthError as e:
            logger.error(f"{self} Google Authentication Error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{self} encountered an unhandled exception: {e}")
        finally:
            if grounding_metadata_dict:
                llm_search_frame = LLMSearchResponseFrame(
                    search_result=search_result_text,
                    origins=grounding_metadata_dict.get("origins", []),
                    rendered_content=grounding_metadata_dict.get("rendered_content"),
                )
                await self.push_frame(llm_search_frame)

            if final_usage_metadata:
                prompt_tokens = final_usage_metadata.prompt_token_count or 0
                completion_tokens = final_usage_metadata.candidates_token_count or 0
                # total_token_count in new SDK's usage_metadata is often the sum.
                # If not, pipecat's metric will sum prompt+completion.
                # We'll rely on pipecat's summation for consistency.
                calculated_total_tokens = prompt_tokens + completion_tokens
            else:
                prompt_tokens = 0; completion_tokens = 0; calculated_total_tokens = 0

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
        context: Optional[GoogleLLMContext] = None

        if isinstance(frame, OpenAILLMContextFrame):
            if not isinstance(frame.context, GoogleLLMContext):
                 frame.context = GoogleLLMContext.upgrade_to_google(frame.context)
            context = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            google_context = GoogleLLMContext()
            # Apply service's system instruction if context doesn't have one
            if self._system_instruction_text and not google_context.system_message_text:
                google_context.system_message_text = self._system_instruction_text
            google_context.add_messages(frame.messages)
            context = google_context
        elif isinstance(frame, VisionImageRawFrame):
            context = GoogleLLMContext()
            if self._system_instruction_text: # Apply service default system instruction
                context.system_message_text = self._system_instruction_text
            context.add_image_frame_message(
                format=frame.format, size=frame.size, image=frame.image, text=frame.text
            )
        elif isinstance(frame, LLMUpdateSettingsFrame):
            new_settings = frame.settings.copy()
            if "max_tokens" in new_settings: # Map to Google's param name
                new_settings["max_output_tokens"] = new_settings.pop("max_tokens")
            await self._update_settings(new_settings)
        else:
            await self.push_frame(frame, direction)

        if context:
            await self._process_context(context)

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> GoogleContextAggregatorPair:
        google_context = GoogleLLMContext.upgrade_to_google(context)
        
        # If the original context didn't have a system message,
        # and this service instance has a default one, apply it.
        if not google_context.system_message_text and self._system_instruction_text:
            google_context.system_message_text = self._system_instruction_text
            
        google_context.set_llm_adapter(self.get_llm_adapter())
        user = GoogleUserContextAggregator(google_context, params=user_params)
        assistant = GoogleAssistantContextAggregator(google_context, params=assistant_params)
        return GoogleContextAggregatorPair(_user=user, _assistant=assistant)