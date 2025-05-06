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
    InputAudioRawFrame,
    EmulateUserStartedSpeakingFrame,
    EmulateUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame
from pipecat.services.google.llm import GoogleLLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class UserAudioAggregator(FrameProcessor):
    """Buffers audio based on VAD and pushes audio context when user stops speaking."""
    def __init__(self, context, user_context_aggregator):
        super().__init__()
        self._context = context
        self._user_context_aggregator = user_context_aggregator
        self._audio_frames = []
        self._start_secs = 0.2
        self._user_speaking = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
            await self.push_frame(frame, direction) # Pass VAD frames
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            if self._audio_frames:
                logger.debug(f"Adding {len(self._audio_frames)} audio frames to context.")
                GoogleLLMContext.upgrade_to_google(self._context)
                self._context.add_audio_frames_message(audio_frames=self._audio_frames)
                await self._user_context_aggregator.push_frame(
                    self._user_context_aggregator.get_context_frame()
                )
                self._audio_frames = []
            else:
                logger.warning("User stopped speaking but no audio frames collected.")
            await self.push_frame(frame, direction) # Pass VAD frames
        elif isinstance(frame, InputAudioRawFrame):
            if self._user_speaking:
                self._audio_frames.append(frame)
            else: # Pre-roll buffering
                self._audio_frames.append(frame)
                frame_duration = len(frame.audio) / 2 * frame.num_channels / frame.sample_rate
                buffer_duration = frame_duration * len(self._audio_frames)
                while buffer_duration > self._start_secs:
                    self._audio_frames.pop(0)
                    buffer_duration -= frame_duration
            # Do not push raw audio downstream here
        else:
            await self.push_frame(frame, direction)
