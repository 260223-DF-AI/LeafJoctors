import mimetypes
import os
from pathlib import Path

from sagemaker.deserializers import JSONDeserializer
from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer


def _build_predictor(endpoint_name):
    return Predictor(
        endpoint_name=endpoint_name,
        serializer=IdentitySerializer(content_type="application/octet-stream"),
        deserializer=JSONDeserializer(),
    )


def make_prediction(inp, endpoint_name=None, content_type=None):
    """Send raw image bytes to a SageMaker endpoint and return the JSON response."""
    resolved_endpoint = (
        endpoint_name or os.getenv("SAGEMAKER_ENDPOINT_NAME", "").strip()
    )
    if not resolved_endpoint:
        raise ValueError(
            "Set SAGEMAKER_ENDPOINT_NAME or pass endpoint_name explicitly."
        )

    if isinstance(inp, (str, os.PathLike, Path)):
        input_path = Path(inp)
        payload = input_path.read_bytes()
        guessed_content_type, _ = mimetypes.guess_type(input_path.name)
        content_type = (
            content_type or guessed_content_type or "application/octet-stream"
        )
    elif isinstance(inp, (bytes, bytearray)):
        payload = bytes(inp)
        content_type = content_type or "application/octet-stream"
    else:
        raise TypeError("inp must be raw bytes or a file path.")

    predictor = _build_predictor(resolved_endpoint)
    return predictor.predict(
        payload,
        initial_args={
            "ContentType": content_type,
            "Accept": "application/json",
        },
    )


if __name__ == "__main__":
    result = make_prediction("data\\cedar_apple_rust_leaf.jpg", endpoint_name="pytorch-inference-2026-04-16-15-51-52-235")
    print(result)