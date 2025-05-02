# Create a new file: audio_aggregator.py

#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    EmulateUserStartedSpeakingFrame,
    EmulateUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AudioContextAggregator(FrameProcessor):
    """
    Aggregates audio frames between user speaking events and adds them to the LLM context.
    This processor works by capturing audio frames between UserStartedSpeakingFrame and 
    UserStoppedSpeakingFrame events and adding them to the context.
    """

    def __init__(self, context: OpenAILLMContext, *, user_id: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self._context = context
        self._user_id = user_id
        self._user_speaking = False
        self._emulating_vad = False
        self._audio_frames: List[AudioRawFrame] = []
        self._lock = asyncio.Lock()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Handle user started speaking events
        if isinstance(frame, UserStartedSpeakingFrame) or isinstance(frame, EmulateUserStartedSpeakingFrame):
            await self._handle_user_started_speaking(frame)
            await self.push_frame(frame, direction)
        
        # Handle user stopped speaking events
        elif isinstance(frame, UserStoppedSpeakingFrame) or isinstance(frame, EmulateUserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking(frame)
            await self.push_frame(frame, direction)
        
        # Collect audio frames when user is speaking
        elif isinstance(frame, AudioRawFrame) and self._user_speaking:
            async with self._lock:
                self._audio_frames.append(frame)
            await self.push_frame(frame, direction)
        
        # Pass through other frames
        else:
            await self.push_frame(frame, direction)

    async def _handle_user_started_speaking(self, frame):
        self._user_speaking = True
        
        # Reset audio frames collection
        async with self._lock:
            self._audio_frames = []
        
        # If we get a non-emulated UserStartedSpeakingFrame but we are in the
        # middle of emulating VAD, let's stop emulating VAD
        if isinstance(frame, UserStartedSpeakingFrame) and not frame.emulated and self._emulating_vad:
            self._emulating_vad = False

    async def _handle_user_stopped_speaking(self, frame):
        if not self._user_speaking:
            return
            
        self._user_speaking = False
        
        # Add collected audio frames to context
        async with self._lock:
            if self._audio_frames:
                logger.debug(f"Adding {len(self._audio_frames)} audio frames to context")
                await self._add_audio_to_context(self._audio_frames)
                self._audio_frames = []

    async def _add_audio_to_context(self, audio_frames: List[AudioRawFrame]):
        """Add the audio frames to the context and push the updated context."""
        if not audio_frames:
            return
            
        # Add audio frames to context
        self._context.add_audio_frames_message(audio_frames=audio_frames)
        
        # Push updated context
        frame = OpenAILLMContextFrame(self._context)
        await self.push_frame(frame)