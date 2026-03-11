"""Voice.Me TTS - A deep learning toolkit for Text-to-Speech."""
from TTS.api import TTS
from TTS.utils.manage import ModelManager

# Read version
import os

with open(os.path.join(os.path.dirname(__file__), "VERSION"), "r", encoding="utf-8") as f:
    version = f.read().strip()

__version__ = version

__all__ = ["TTS", "ModelManager", "__version__"]
