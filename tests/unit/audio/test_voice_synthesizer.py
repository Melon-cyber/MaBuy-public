import os
import unittest
import asyncio
from unittest.mock import patch, AsyncMock
from src.infrastructure.audio.voice_synthesizer import MaIAVoiceSynthesizer

class TestMaIAVoiceSynthesizer(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.synthesizer = MaIAVoiceSynthesizer()

    async def test_empty_text_returns_none(self):
        res = await self.synthesizer.synthesize_speech("")
        self.assertIsNone(res)

    @patch("edge_tts.Communicate")
    async def test_synthesize_speech_mock(self, mock_comm_cls):
        mock_comm = AsyncMock()
        mock_comm_cls.return_value = mock_comm
        
        # Test synthesis for Yandere
        test_path = os.path.join(self.synthesizer.cache_dir, "test_sample.ogg")
        with open(test_path, "w") as f:
            f.write("audio_bytes")
            
        with patch.object(self.synthesizer, "_get_cache_path", return_value=test_path):
            path = await self.synthesizer.synthesize_speech(
                text="Привет, я Майя! Твой квантовый ко-пилот.",
                persona="yandere",
                lang="ru"
            )
            self.assertEqual(path, test_path)

        if os.path.exists(test_path):
            os.remove(test_path)

if __name__ == "__main__":
    unittest.main()
