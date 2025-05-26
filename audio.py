from __future__ import annotations
import io
import base64
import asyncio
import threading
import numpy as np
import sounddevice as sd
from typing import Optional, Callable, Awaitable
from pydub import AudioSegment
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection

# Constants
CHUNK_LENGTH_S = 0.05  # 50ms chunks for smoother playback
SAMPLE_RATE = 24000
CHANNELS = 1
AUDIO_FORMAT = np.int16
SILENCE_THRESHOLD = 100  # Minimum amplitude to consider as silence

class AudioPlayerAsync:
    def __init__(self):
        """
        Asynchronous audio player with interrupt capabilities and smooth playback.
        Uses sounddevice for low-latency audio output.
        """
        self.queue = []
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.playing = False
        self._should_play = True
        self._frame_count = 0
        self._stream = None
        self._current_audio_id = None  # Tracks current audio session
        
        # Initialize audio stream with error handling
        try:
            self._stream = sd.OutputStream(
                callback=self._audio_callback,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=AUDIO_FORMAT,
                blocksize=int(CHUNK_LENGTH_S * SAMPLE_RATE),
            )
        except sd.PortAudioError as e:
            print(f"Failed to initialize audio stream: {e}")
            raise

    def _audio_callback(self, outdata: np.ndarray, frames: int, time, status) -> None:
        """
        Callback function for audio stream that handles playback and interruption.
        """
        if not self._should_play:
            outdata.fill(0)
            return

        with self.lock:
            data = np.zeros(frames, dtype=AUDIO_FORMAT)
            
            # Process audio queue
            while len(self.queue) > 0 and len(data) < frames:
                item = self.queue[0]
                frames_needed = frames - len(data)
                
                # Take what we need from the first item
                data_added = item[:frames_needed]
                data = np.concatenate((data, data_added))
                
                # Remove used frames or keep remainder
                if len(item) > frames_needed:
                    self.queue[0] = item[frames_needed:]
                else:
                    self.queue.pop(0)

            self._frame_count += len(data)

        # Ensure we don't underflow the buffer
        if len(data) < frames:
            data = np.concatenate((data, np.zeros(frames - len(data), dtype=AUDIO_FORMAT)))

        outdata[:] = data.reshape(-1, 1)

    def add_data(self, data: bytes, audio_id: Optional[str] = None) -> None:
        """
        Add audio data to the playback queue.
        
        Args:
            data: PCM16 audio data as bytes
            audio_id: Optional identifier for the audio session
        """
        if not data:
            return

        with self.lock:
            # If this is new audio, clear previous queue
            if audio_id and audio_id != self._current_audio_id:
                self.queue = []
                self._current_audio_id = audio_id
                self._frame_count = 0

            # Convert bytes to numpy array
            np_data = np.frombuffer(data, dtype=AUDIO_FORMAT)
            
            # Split into chunks for smoother playback
            chunk_size = int(SAMPLE_RATE * CHUNK_LENGTH_S)
            for i in range(0, len(np_data), chunk_size):
                self.queue.append(np_data[i:i + chunk_size])

            if not self.playing:
                self.start()

    def start(self) -> None:
        """Start audio playback if not already playing."""
        if not self.playing and self._stream:
            try:
                self._stream.start()
                self.playing = True
                self._should_play = True
            except sd.PortAudioError as e:
                print(f"Failed to start audio stream: {e}")

    def stop(self, clear_queue: bool = True) -> None:
        """
        Stop audio playback immediately.
        
        Args:
            clear_queue: Whether to clear the playback queue
        """
        self._should_play = False
        if clear_queue:
            with self.lock:
                self.queue = []
        
        # Let the callback handle the actual stopping to avoid glitches
        self.event.set()

    def pause(self) -> None:
        """Pause audio playback without clearing the queue."""
        self._should_play = False

    def resume(self) -> None:
        """Resume paused audio playback."""
        self._should_play = True
        if not self.playing and self._stream:
            self.start()

    def reset_frame_count(self) -> None:
        """Reset the frame counter."""
        with self.lock:
            self._frame_count = 0

    def get_frame_count(self) -> int:
        """Get current frame count."""
        with self.lock:
            return self._frame_count

    def terminate(self) -> None:
        """Clean up resources."""
        self.stop()
        if self._stream:
            self._stream.close()
            self._stream = None

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self.playing and self._should_play

    def get_queue_size(self) -> int:
        """Get number of frames in the playback queue."""
        with self.lock:
            return sum(len(chunk) for chunk in self.queue)


async def send_audio_worker_sounddevice(
    connection: AsyncRealtimeConnection,
    should_send: Optional[Callable[[], bool]] = None,
    start_send: Optional[Callable[[], Awaitable[None]]] = None,
    audio_player: Optional[AudioPlayerAsync] = None
) -> None:
    """
    Worker that captures audio from microphone and sends to realtime connection.
    
    Args:
        connection: OpenAI realtime connection
        should_send: Function to check if we should send audio
        start_send: Async function to call when starting to send
        audio_player: Optional audio player for echo cancellation
    """
    sent_audio = False
    read_size = int(SAMPLE_RATE * 0.02)  # 20ms chunks

    try:
        with sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype="int16",
            blocksize=read_size
        ) as stream:
            while True:
                if stream.read_available < read_size:
                    await asyncio.sleep(0)
                    continue

                # Read audio data
                data, _ = stream.read(read_size)
                
                # Optional echo cancellation
                if audio_player and audio_player.is_playing():
                    # Get currently playing audio and subtract from input
                    pass  # Implementation would go here

                # Check if we should send audio
                if should_send() if should_send else True:
                    if not sent_audio and start_send:
                        await start_send()
                    
                    # Send audio chunk
                    await connection.send({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(data).decode("utf-8")
                    })
                    sent_audio = True
                elif sent_audio:
                    # Commit audio and request response
                    await connection.send({"type": "input_audio_buffer.commit"})
                    await connection.send({"type": "response.create"})
                    sent_audio = False

                await asyncio.sleep(0)

    except Exception as e:
        print(f"Audio capture error: {e}")
        raise


def audio_to_pcm16_base64(audio_bytes: bytes) -> bytes:
    """
    Convert audio bytes to PCM16 format at 24kHz mono.
    
    Args:
        audio_bytes: Input audio in any format supported by pydub
        
    Returns:
        PCM16 audio data as bytes
    """
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        return (
            audio.set_frame_rate(SAMPLE_RATE)
            .set_channels(CHANNELS)
            .set_sample_width(2)  # 16-bit
            .raw_data
        )
    except Exception as e:
        print(f"Audio conversion error: {e}")
        raise