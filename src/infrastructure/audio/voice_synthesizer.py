"""
MaBuy V5 — Voice AI Co-Pilot
MaIA Neural Voice Synthesizer using Edge-TTS
Supports dynamic persona tone modulation (Yandere, Tsundere, Quant, Zen) with local audio caching.
"""
import os
import hashlib
import logging
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)

PERSONA_VOICE_PROFILES = {
    "yandere": {
        "ru": {"voice": "ru-RU-SvetlanaNeural", "rate": "+4%", "pitch": "+6Hz"},
        "en": {"voice": "en-US-AriaNeural", "rate": "+5%", "pitch": "+5Hz"}
    },
    "tsundere": {
        "ru": {"voice": "ru-RU-DariyaNeural", "rate": "+8%", "pitch": "+4Hz"},
        "en": {"voice": "en-US-AnaNeural", "rate": "+8%", "pitch": "+4Hz"}
    },
    "quant": {
        "ru": {"voice": "ru-RU-SvetlanaNeural", "rate": "+0%", "pitch": "+0Hz"},
        "en": {"voice": "en-US-JennyNeural", "rate": "+0%", "pitch": "+0Hz"}
    },
    "zen": {
        "ru": {"voice": "ru-RU-SvetlanaNeural", "rate": "-6%", "pitch": "-3Hz"},
        "en": {"voice": "en-US-AriaNeural", "rate": "-6%", "pitch": "-3Hz"}
    }
}

class MaIAVoiceSynthesizer:
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            self.cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "cache", "voice_audio"))
        else:
            self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, text: str, persona: str, lang: str) -> str:
        cache_key = hashlib.sha256(f"{persona}_{lang}_{text}".encode('utf-8')).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"maia_{persona}_{lang}_{cache_key}.ogg")

    async def synthesize_speech(
        self,
        text: str,
        persona: str = "quant",
        lang: str = "ru",
        clean_markdown: bool = True
    ) -> Optional[str]:
        """
        Synthesizes text into high quality OGG/Opus audio suitable for Telegram Voice Notes.
        Returns the absolute filepath to the audio file.
        """
        if not text or not text.strip():
            return None

        # Clean text from HTML/Markdown tags and emoticons for clean natural speech
        spoken_text = text
        if clean_markdown:
            import re
            spoken_text = re.sub(r'<[^>]+>', '', spoken_text)
            spoken_text = re.sub(r'[*_`#~]', '', spoken_text)
            spoken_text = re.sub(r'\[.*?\]', '', spoken_text)
            spoken_text = re.sub(r'\(.*?\)', '', spoken_text)
            spoken_text = ' '.join(spoken_text.split())

        # Limit text length to ~400 chars for concise punchy voice clips (10-25 seconds)
        if len(spoken_text) > 400:
            spoken_text = spoken_text[:395] + "..."

        persona_key = persona.lower()
        if persona_key not in PERSONA_VOICE_PROFILES:
            persona_key = "quant"
            
        lang_key = "ru" if lang.startswith("ru") else "en"
        profile = PERSONA_VOICE_PROFILES[persona_key].get(lang_key, PERSONA_VOICE_PROFILES[persona_key]["ru"])

        output_path = self._get_cache_path(spoken_text, persona_key, lang_key)
        
        # Check cache hit
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.debug(f"Audio cache hit for MaIA voice: {output_path}")
            return output_path

        try:
            import edge_tts
            communicate = edge_tts.Communicate(
                text=spoken_text,
                voice=profile["voice"],
                rate=profile["rate"],
                pitch=profile["pitch"]
            )
            await communicate.save(output_path)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"✅ Synthesized MaIA voice [{persona_key.upper()}] ({len(spoken_text)} chars) -> {output_path}")
                return output_path
        except Exception as e:
            logger.error(f"Voice synthesis failed for MaIA ({persona_key}): {e}")
            
        return None
