

### mic stream

import queue
import re
import sys
import pyaudio


class MicrophoneStream:
    def __init__(
        self,
        sample_rate: int = 16000,
    ):
        """
        Creates a stream of audio from the microphone.

        Args:
            chunk_size: The size of each chunk of audio to read from the microphone.
            channels: The number of channels to record audio from.
            sample_rate: The sample rate to record audio at.
        """
        try:
            import pyaudio
        except ImportError:
            raise Exception('py audio not installed')

        self._pyaudio = pyaudio.PyAudio()
        self.sample_rate = sample_rate

        self._chunk_size = int(self.sample_rate * 40  / 1000)
        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=self._chunk_size,
        )

        self._open = True

    def __iter__(self):
        """
        Returns the iterator object.
        """

        return self

    def __next__(self):
        """
        Reads a chunk of audio from the microphone.
        """
        if not self._open:
            raise StopIteration

        try:
            return self._stream.read(self._chunk_size)
        except KeyboardInterrupt:
            raise StopIteration

    def close(self):
        """
        Closes the stream.
        """

        self._open = False

        if self._stream.is_active():
            self._stream.stop_stream()

        self._stream.close()
        self._pyaudio.terminate()









