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
    import google.genai.errors as genai_errors
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
            # Ensure context is GoogleLLMContext for correct add_message behavior
            if not isinstance(self._context, GoogleLLMContext):
                logger.error("GoogleUserContextAggregator received a non-GoogleLLMContext.")
                # Handle error or attempt conversion, though ideally context is already correct type
                # For now, proceed hoping add_message can handle it or was overridden
            self._context.add_message(
                genai_types.Content(role="user", parts=[
                                    genai_types.Part(text=self._aggregation)])
            )
            self._aggregation = ""
            frame = OpenAILLMContextFrame(self._context)
            await self.push_frame(frame)
            self.reset()


class GoogleAssistantContextAggregator(OpenAIAssistantContextAggregator):
    async def handle_aggregation(self, aggregation: str):
        if not isinstance(self._context, GoogleLLMContext):
            logger.error("GoogleAssistantContextAggregator received a non-GoogleLLMContext.")
        self._context.add_message(
            genai_types.Content(role="model", parts=[
                                genai_types.Part(text=aggregation)])
        )

    async def handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        if not isinstance(self._context, GoogleLLMContext):
            logger.error("GoogleAssistantContextAggregator received a non-GoogleLLMContext for function call.")
            return

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
        self._context.add_message(
            genai_types.Content(
                role="function", # Correct role for function responses in new SDK
                parts=[
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=frame.function_name,
                            response={"status": "IN_PROGRESS", "_pipecat_tool_call_id": frame.tool_call_id}
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
        if not isinstance(self._context, GoogleLLMContext):
            logger.error("Cannot update function call result on a non-GoogleLLMContext.")
            return

        for message in reversed(self._context.messages):
            if message.role == "function":
                for part in message.parts:
                    if part.function_response and \
                       part.function_response.name == function_name and \
                       isinstance(part.function_response.response, dict) and \
                       part.function_response.response.get("status") == "IN_PROGRESS" and \
                       part.function_response.response.get("_pipecat_tool_call_id") == tool_call_id:
                        part.function_response.response = result # Assign the dict directly
                        return
        logger.warning(f"Could not find IN_PROGRESS placeholder for function {function_name} (tool_id: {tool_call_id}) to update result.")


    async def handle_user_image_frame(self, frame: UserImageRawFrame):
        if not isinstance(self._context, GoogleLLMContext):
            logger.error("Cannot handle user image frame on a non-GoogleLLMContext.")
            return

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
        messages: Optional[List[Any]] = None,
        tools: Optional[List[Any]] = None,
        tool_choice: Optional[Any] = None,
    ):
        super().__init__(messages=[], tools=None, tool_choice=None)
        self.system_message_text: Optional[str] = None

        if tools is not None:
            self._tools = tools # Adapter will handle conversion if needed
        if tool_choice is not None:
            self._tool_choice = tool_choice # Adapter will handle conversion

        if messages:
            self.add_messages(messages)


    @property
    def system_instruction_for_api(self) -> Optional[str]:
        """Returns the system instruction string for the API's GenerationConfig."""
        return self.system_message_text

    @staticmethod
    def upgrade_to_google(obj: OpenAILLMContext) -> "GoogleLLMContext":
        if not isinstance(obj, GoogleLLMContext):
            logger.debug(f"Upgrading OpenAI context to GoogleLLMContext: {obj}")
            new_google_context = GoogleLLMContext()
            
            openai_messages = obj.get_messages_for_persistent_storage()
            system_text_from_openai = None
            regular_messages_from_openai = []

            for msg_dict in openai_messages:
                if msg_dict.get("role") == "system" and isinstance(msg_dict.get("content"), str):
                    system_text_from_openai = msg_dict["content"]
                else:
                    regular_messages_from_openai.append(msg_dict)
            
            if system_text_from_openai:
                new_google_context.system_message_text = system_text_from_openai
            
            new_google_context.add_messages(regular_messages_from_openai)

            if obj.tools is not OPENAI_NOT_GIVEN: # Use imported OPENAI_NOT_GIVEN
                 new_google_context._tools = obj.tools
            if obj.tool_choice is not OPENAI_NOT_GIVEN: # Use imported OPENAI_NOT_GIVEN
                new_google_context._tool_choice = obj.tool_choice

            new_google_context.set_llm_adapter(obj.get_llm_adapter())
            return new_google_context
        return obj

    def set_messages(self, messages: List[Any]):
        self._messages.clear()
        current_system_text = self.system_message_text
        self.system_message_text = None # Reset, add_messages might set it
        self.add_messages(messages)
        if self.system_message_text is None: # If not set by new messages
            self.system_message_text = current_system_text

    def add_messages(self, messages: List[Any]):
        for msg_input in messages:
            if isinstance(msg_input, genai_types.Content):
                self._messages.append(msg_input)
            elif isinstance(msg_input, dict):
                if msg_input.get("role") == "system" and isinstance(msg_input.get("content"), str):
                    self.system_message_text = msg_input["content"]
                    continue
                converted_msg = self.from_standard_message(msg_input)
                if converted_msg:
                    self._messages.append(converted_msg)
            else:
                logger.warning(f"Skipping unknown message type during add_messages: {type(msg_input)}")

    def get_messages_for_logging(self) -> List[Dict]:
        msgs = []
        if self.system_message_text: # Log system message if present
            msgs.append({"role": "system (pipecat internal)", "content": self.system_message_text})

        for message_content in self._messages:
            obj = message_content.model_dump(exclude_none=True, by_alias=True) # Use by_alias for Pydantic
            try:
                if "parts" in obj:
                    for part in obj["parts"]:
                        if "inlineData" in part and isinstance(part["inlineData"], dict): # Check camelCase from by_alias
                            if "data" in part["inlineData"] and isinstance(part["inlineData"]["data"], str): # Data is base64 string
                                 part["inlineData"]["data"] = f"<base64_bytes len_approx={len(part['inlineData']['data']) * 3 // 4}>"
            except Exception as e:
                logger.debug(f"Error redacting message for logging: {e}")
            msgs.append(obj)
        return msgs

    def add_image_frame_message(
        self, *, format: str, size: tuple[int, int], image: bytes, text: str = None
    ):
        buffer = io.BytesIO()
        # Convert to JPEG as it's widely supported
        Image.frombytes(format, size, image).save(buffer, format="JPEG")

        parts = []
        if text:
            parts.append(genai_types.Part(text=text))
        parts.append(genai_types.Part(inline_data=genai_types.Blob(
            mime_type="image/jpeg", data=buffer.getvalue())))
        self.add_messages([genai_types.Content(role="user", parts=parts)])

    def add_audio_frames_message(
        self, *, audio_frames: list[AudioRawFrame], text: str = "Audio follows"
    ):
        if not audio_frames: return
        sample_rate, num_channels = audio_frames[0].sample_rate, audio_frames[0].num_channels
        audio_data = b"".join(frame.audio for frame in audio_frames)
        wav_data = self.create_wav_header(sample_rate, num_channels, 16, len(audio_data)) + audio_data
        parts = [genai_types.Part(text=text)] if text else []
        parts.append(genai_types.Part(inline_data=genai_types.Blob(mime_type="audio/wav", data=wav_data)))
        self.add_messages([genai_types.Content(role="user", parts=parts)])

    def from_standard_message(self, message: Dict[str, Any]) -> Optional[genai_types.Content]:
        role, content = message.get("role"), message.get("content")
        if role == "system":
            if isinstance(content, str): self.system_message_text = content
            elif isinstance(content, list) and content and isinstance(content[0].get("text"), str):
                self.system_message_text = content[0]["text"]
            return None

        google_role = {"assistant": "model", "tool": "function"}.get(role, "user")
        if role not in ["user", "assistant", "tool"]: logger.warning(f"Unknown role '{role}', defaulting to 'user'.")

        parts = []
        if message.get("tool_calls"):
            google_role = "model"
            for tc in message.get("tool_calls", []):
                fn_data = tc.get("function", {})
                try: args = json.loads(fn_data.get("arguments", "{}"))
                except json.JSONDecodeError: args = {}
                parts.append(genai_types.Part(function_call=genai_types.FunctionCall(name=fn_data.get("name"), args=args)))
        elif google_role == "function":
            try: response_data = json.loads(content) if isinstance(content, str) else content
            except json.JSONDecodeError: response_data = {"error": "failed to parse content", "original_content": content}
            parts.append(genai_types.Part(function_response=genai_types.FunctionResponse(name=message.get("name", message.get("tool_call_id", "unknown_function")), response=response_data)))
        elif isinstance(content, str):
            parts.append(genai_types.Part(text=content))
        elif isinstance(content, list):
            for item in content:
                item_type = item.get("type")
                if item_type == "text": parts.append(genai_types.Part(text=item.get("text")))
                elif item_type in ["image_url", "audio_url"]:
                    url_data = item.get(item_type, {}).get("url", "")
                    if url_data.startswith("data:"):
                        try:
                            header, b64_data = url_data.split(';base64,', 1)
                            mime_type = header.split(':', 1)[1]
                            data_bytes = base64.b64decode(b64_data)
                            parts.append(genai_types.Part(inline_data=genai_types.Blob(mime_type=mime_type, data=data_bytes)))
                        except Exception as e: logger.error(f"Error decoding {item_type} data: {e}")
        return genai_types.Content(role=google_role, parts=parts) if parts else None

    def to_standard_messages(self, google_content: genai_types.Content) -> List[Dict[str, Any]]:
        std_msg: Dict[str, Any] = {"role": {"model": "assistant", "function": "tool"}.get(google_content.role, google_content.role)}
        content_list_parts, tool_calls_list = [], []

        for part in google_content.parts:
            if part.text: content_list_parts.append({"type": "text", "text": part.text})
            elif part.inline_data:
                mime, data_b64 = part.inline_data.mime_type, base64.b64encode(part.inline_data.data).decode("utf-8")
                url = f"data:{mime};base64,{data_b64}"
                url_type_key = "image_url" if mime.startswith("image/") else "audio_url"
                content_list_parts.append({"type": url_type_key, url_type_key: {"url": url}})
            elif part.function_call:
                std_msg["role"] = "assistant"
                tool_calls_list.append({"id": part.function_call.name, "type": "function", "function": {"name": part.function_call.name, "arguments": json.dumps(part.function_call.args or {})}})
            elif part.function_response:
                std_msg.update({"role": "tool", "tool_call_id": part.function_response.name, "content": json.dumps(part.function_response.response or {})})
                content_list_parts = [] # Function response is the sole content
                break
        
        if tool_calls_list: std_msg["tool_calls"] = tool_calls_list
        if content_list_parts:
            std_msg["content"] = content_list_parts[0]["text"] if len(content_list_parts) == 1 and content_list_parts[0]["type"] == "text" else content_list_parts
        elif "content" not in std_msg and not tool_calls_list:
            std_msg["content"] = None if std_msg["role"] == "assistant" else ""
        return [std_msg]

    def _restructure_from_openai_messages(self):
        pass # Assuming conversion logic in add_messages/set_messages is now sufficient


class GoogleLLMService(LLMService):
    adapter_class = GeminiLLMAdapter

    class InputParams(BaseModel):
        max_output_tokens: Optional[int] = Field(default=None, ge=1)
        temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0) # Google's range is 0-1 for some models
        top_k: Optional[int] = Field(default=None, ge=0)
        top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
        extra: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def __init__(
        self,
        *,
        api_key: str, # Should be directly used by genai.Client
        model: str = "gemini-1.5-flash-latest", # Use "latest" for auto-updates
        params: InputParams = InputParams(),
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        try:
            self._google_client_internal = genai.Client(api_key=api_key)
            self._google_model_service = self._google_client_internal.models
        except GoogleAuthError as e:
            logger.error(f"Google Authentication Error: {e}. Ensure API key is valid and Gemini API enabled.")
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
        }
        if isinstance(params.extra, dict): # Merge extra params
            self._settings.update(params.extra)

        self._tools_standard_format = tools
        self._tool_config_standard_format = tool_config

    def can_generate_metrics(self) -> bool:
        return True

    async def _process_context(self, context: GoogleLLMContext):
        await self.push_frame(LLMFullResponseStartFrame())
        prompt_tokens, completion_tokens, final_usage_metadata = 0, 0, None
        grounding_metadata_dict: Optional[Dict[str, Any]] = None
        search_result_text = ""

        try:
            logger.debug(f"{self}: Generating chat for model {self._model_name} with context: [{context.get_messages_for_logging()}]")
            messages_for_api = context.messages
            
            generation_config_payload: Dict[str, Any] = {}
            for key, value in self._settings.items():
                if value is not None: generation_config_payload[key] = value
            
            current_system_instruction = context.system_instruction_for_api or \
                                         (genai_types.Content(parts=[genai_types.Part(text=self._system_instruction_text)]) if self._system_instruction_text else None)
            if current_system_instruction:
                # system_instruction for genai.types.GenerationConfig is a string or Content
                generation_config_payload["system_instruction"] = current_system_instruction.parts[0].text if current_system_instruction.parts else ""


            effective_tools_std = context.tools if context.tools not in [None, OPENAI_NOT_GIVEN] else self._tools_standard_format
            if effective_tools_std:
                api_tools = self.get_llm_adapter().from_standard_tools(effective_tools_std)
                if api_tools: generation_config_payload["tools"] = api_tools

            effective_tool_choice_std = context.tool_choice if context.tool_choice not in [None, OPENAI_NOT_GIVEN] else self._tool_config_standard_format
            if effective_tool_choice_std:
                current_api_tools = generation_config_payload.get("tools")
                api_tool_config = self.get_llm_adapter().from_standard_tool_choice(effective_tool_choice_std, current_api_tools)
                if api_tool_config: generation_config_payload["tool_config"] = api_tool_config
            
            final_gc_obj = genai_types.GenerationConfig(**{k: v for k, v in generation_config_payload.items() if k in genai_types.GenerateContentConfig.model_fields}) if generation_config_payload else None

            await self.start_ttfb_metrics()
            stream_response = await self._google_model_service.generate_content_stream(
                model=self._model_name, contents=messages_for_api, config=final_gc_obj
            )
            await self.stop_ttfb_metrics()

            async for chunk in stream_response:
                if chunk.usage_metadata: final_usage_metadata = chunk.usage_metadata
                if chunk.candidates:
                    for candidate in chunk.candidates:
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if part.text:
                                    search_result_text += part.text
                                    await self.push_frame(LLMTextFrame(part.text))
                                elif part.function_call:
                                    logger.debug(f"Function call: {part.function_call.name} args: {part.function_call.args}")
                                    await self.call_function(context=context, tool_call_id=str(uuid.uuid4()), function_name=part.function_call.name, arguments=part.function_call.args or {})
                        if candidate.grounding_metadata:
                            gm = candidate.grounding_metadata
                            origins_list = []
                            rendered_web_content = gm.search_entry_point.rendered_content if gm.search_entry_point else None
                            if gm.grounding_attributions:
                                for attr in gm.grounding_attributions:
                                    uri, title = (attr.web.uri, attr.web.title) if attr.web else (attr.retrieved_context.uri, attr.retrieved_context.title) if attr.retrieved_context else (None, None)
                                    text = "".join(p.text + " " for p in attr.content.parts if p.text).strip() if attr.content and attr.content.parts else ""
                                    origins_list.append({"site_uri": uri, "site_title": title, "results": [{"text": text, "confidence": getattr(attr, 'confidence_score', None)}]})
                            grounding_metadata_dict = {"rendered_content": rendered_web_content, "origins": origins_list}
                        if candidate.finish_reason:
                            fr = candidate.finish_reason
                            if fr == genai_types.FinishReason.SAFETY: logger.warning(f"LLM safety stop. Prompt: {context.get_messages_for_logging()}")
                            elif fr not in [genai_types.FinishReason.STOP, genai_types.FinishReason.MAX_TOKENS, genai_types.FinishReason.UNSPECIFIED, genai_types.FinishReason.OTHER]:
                                logger.warning(f"LLM stop: {fr}. Candidate: {candidate}")
                elif chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                    logger.warning(f"Prompt blocked: {chunk.prompt_feedback.block_reason}. Details: {chunk.prompt_feedback.block_reason_message}")

        except api_core_exceptions.DeadlineExceeded:
            logger.warning(f"{self} completion timeout")
            await self._call_event_handler("on_completion_timeout")
        except genai_errors.APIError as e:
            logger.error(f"{self} Google API Error: {e}. Details: {getattr(e, 'response', 'N/A')}")
            if "API_KEY_INVALID" in str(e): logger.error("API Key issue.")
            elif hasattr(e, 'response') and e.response and e.response.status_code == 429: logger.error("Rate limit/quota exceeded.")
        except GoogleAuthError as e: logger.error(f"{self} Google Auth Error: {e}"); raise
        except Exception as e: logger.exception(f"{self} unhandled exception: {e}")
        finally:
            if grounding_metadata_dict:
                await self.push_frame(LLMSearchResponseFrame(search_result=search_result_text, origins=grounding_metadata_dict.get("origins", []), rendered_content=grounding_metadata_dict.get("rendered_content")))
            if final_usage_metadata:
                prompt_tokens = final_usage_metadata.prompt_token_count or 0
                completion_tokens = final_usage_metadata.candidates_token_count or 0
            await self.start_llm_usage_metrics(LLMTokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens))
            await self.push_frame(LLMFullResponseEndFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        context: Optional[GoogleLLMContext] = None

        if isinstance(frame, OpenAILLMContextFrame):
            context = GoogleLLMContext.upgrade_to_google(frame.context)
        elif isinstance(frame, LLMMessagesFrame):
            google_context = GoogleLLMContext()
            if self._system_instruction_text and not any(m.get("role") == "system" for m in frame.messages):
                google_context.system_message_text = self._system_instruction_text
            google_context.add_messages(frame.messages)
            context = google_context
        elif isinstance(frame, VisionImageRawFrame):
            context = GoogleLLMContext()
            if self._system_instruction_text: context.system_message_text = self._system_instruction_text
            context.add_image_frame_message(format=frame.format, size=frame.size, image=frame.image, text=frame.text)
        elif isinstance(frame, LLMUpdateSettingsFrame):
            new_settings = frame.settings.copy()
            if "max_tokens" in new_settings: new_settings["max_output_tokens"] = new_settings.pop("max_tokens")
            await self._update_settings(new_settings)
        else:
            await self.push_frame(frame, direction)

        if context:
            # Ensure system instruction from service default is applied if context doesn't have one
            if not context.system_message_text and self._system_instruction_text:
                context.system_message_text = self._system_instruction_text
            await self._process_context(context)

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> GoogleContextAggregatorPair:
        google_context = GoogleLLMContext.upgrade_to_google(context)
        if not google_context.system_message_text and self._system_instruction_text:
            google_context.system_message_text = self._system_instruction_text
            
        google_context.set_llm_adapter(self.get_llm_adapter()) # Adapter should be set on the final context type
        user = GoogleUserContextAggregator(google_context, params=user_params)
        assistant = GoogleAssistantContextAggregator(google_context, params=assistant_params)
        return GoogleContextAggregatorPair(_user=user, _assistant=assistant)