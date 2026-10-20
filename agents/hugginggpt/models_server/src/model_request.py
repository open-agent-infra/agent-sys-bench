from pydantic import BaseModel
from typing import Optional


class InferenceRequest(BaseModel):
    messages: Optional[list[dict]] = None
    text: Optional[str] = None
    img_url: Optional[str] = None
    audio_url: Optional[str] = None
