import os
import logging
from cryptography.fernet import Fernet, InvalidToken
from typing import Optional

logger = logging.getLogger(__name__)

class EncryptionService:
    """
    Authenticated Symmetric Encryption via Fernet (AES-128-CBC + HMAC-SHA256).
    Uses a 256-bit master key (128-bit AES encryption + 128-bit HMAC-SHA256 authentication).
    Used to encrypt and decrypt sensitive API keys and remote database backup archives.
    Never stores the root encryption key in the DB.
    """
    
    def __init__(self, key: Optional[str] = None):
        self.key = key or os.environ.get("ENCRYPTION_KEY")
        if not self.key:
            import base64, hashlib
            fallback_seed = os.environ.get("MABUY_SECRET_KEY", "mabuy_production_default_safe_seed_2026")
            self.key = base64.urlsafe_b64encode(hashlib.sha256(fallback_seed.encode()).digest()).decode()
        
        try:
            self.cipher = Fernet(self.key.encode('utf-8'))
        except Exception as e:
            import base64, hashlib
            self.key = base64.urlsafe_b64encode(hashlib.sha256(str(self.key).encode()).digest()).decode()
            self.cipher = Fernet(self.key.encode('utf-8'))

    @staticmethod
    def generate_key() -> str:
        """Utility to generate a new secure Fernet key. Run this once and save to .env"""
        return Fernet.generate_key().decode('utf-8')

    def encrypt(self, plain_text: str) -> str:
        if not plain_text:
            return ""
        try:
            encrypted = self.cipher.encrypt(plain_text.encode('utf-8'))
            return encrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    def decrypt(self, cipher_text: str) -> str:
        if not cipher_text:
            return ""
        try:
            decrypted = self.cipher.decrypt(cipher_text.encode('utf-8'))
            return decrypted.decode('utf-8')
        except InvalidToken:
            logger.error("Decryption failed: Invalid Token or wrong ENCRYPTION_KEY.")
            raise
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

    def encrypt_bytes(self, raw_bytes: bytes) -> bytes:
        """Encrypts arbitrary binary bytes (e.g. database compressed archives)."""
        if not raw_bytes:
            return b""
        try:
            return self.cipher.encrypt(raw_bytes)
        except Exception as e:
            logger.error(f"Binary encryption failed: {e}")
            raise

    def decrypt_bytes(self, cipher_bytes: bytes) -> bytes:
        """Decrypts encrypted binary bytes."""
        if not cipher_bytes:
            return b""
        try:
            return self.cipher.decrypt(cipher_bytes)
        except Exception as e:
            logger.error(f"Binary decryption failed: {e}")
            raise
