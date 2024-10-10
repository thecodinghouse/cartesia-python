import base64
import json
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import requests

from cartesia._constants import BACKOFF_FACTOR, MAX_RETRIES
from cartesia._logger import logger
from cartesia._types import OutputFormat
from cartesia.utils.retry import retry_on_connection_error


class _BYTES:
    """This class contains methods to generate audio using Server-Sent Events.

    Usage:
        >>> for audio_chunk in client.tts.bytes(
        ...     model_id="sonic-english", transcript="Hello world!", voice={"mode": "id", "id": "voice_id"},
        ...     output_format={"container": "mp3", "encoding": "mp3", "sample_rate": 44100}
        ... ):
        ...     audio = audio_chunk["audio"]
    """

    def __init__(
        self,
        http_url: str,
        headers: Dict[str, str],
        timeout: float,
    ):
        self.http_url = http_url
        self.headers = headers
        self.timeout = timeout

    def send(
        self,
        model_id: str,
        transcript: str,
        output_format: OutputFormat,
        voice: Dict[str, str],
        duration: Optional[int] = None,
        language: Optional[str] = None,
    ) -> Union[bytes, Generator[bytes, None, None]]:
        """Send a request to the server to generate audio using Bytes.

        Args:
            model_id: The ID of the model to use for generating audio.
            transcript: The text to converted to speech.
            voice: A dictionary containing mode and id or embedding of the voice to use for generating audio.
            output_format: A dictionary containing the details of the output format.
            duration: The duration of the audio in seconds.
            language: The language code for the audio request. This can only be used with `model_id = sonic-multilingual`
        Returns:
            The method returns a file.
        """
        request_body = {
            "model_id": model_id,
            "transcript": transcript,
            "voice": voice,
            "output_format": {
                "container": output_format["container"],
                "encoding": output_format["encoding"],
                "sample_rate": output_format["sample_rate"],
            },
            "language": language,
        }

        if duration is not None:
            request_body["duration"] = duration

        generator = self._bytes_generator_wrapper(request_body)
        chunks = []
        for chunk in generator:
            chunks.append(chunk)

        return {"audio": b"".join(chunks)}

    @retry_on_connection_error(
        max_retries=MAX_RETRIES, backoff_factor=BACKOFF_FACTOR, logger=logger
    )
    def _bytes_generator_wrapper(self, request_body: Dict[str, Any]):
        """Need to wrap the bytes generator in a function for the retry decorator to work."""
        try:
            for chunk in self._bytes_generator(request_body):
                yield chunk
        except Exception as e:
            raise RuntimeError(f"Error generating audio. {e}")

    def _bytes_generator(self, request_body: Dict[str, Any]):
        response = requests.post(
            f"{self.http_url}/tts/bytes",
            data=json.dumps(request_body),
            headers=self.headers,
            timeout=(self.timeout, self.timeout),
        )
        if not response.ok:
            raise ValueError(f"Failed to generate audio. {response.text}")

        for chunk in response.iter_content(chunk_size=None):
            yield chunk

        
