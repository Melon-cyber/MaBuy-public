# 🚨 MaBuy Disaster Recovery (DR) Protocol

Этот документ описывает регламент действий при полном выходе из строя VPS-сервера (сгорание диска, блокировка хостинга, аппаратный сбой).

---

## 🔐 1. Хранение Главного Ключа Шифрования (`ENCRYPTION_KEY`)

> [!CAUTION]
> **КРИТИЧЕСКОЕ ПРАВИЛО:** Если единственная копия `ENCRYPTION_KEY` хранится только в файле `.env` на VPS, при гибели диска зашифрованные бэкапы в Telegram **расшифровать будет невозможно**!

### Регламент холодного хранения ключа:
1. Скопируйте значение `ENCRYPTION_KEY` из файла `.env` сервера.
2. Сохраните его в **защищенный менеджер паролей** (Bitwarden, 1Password, Apple Keychain или KeePassXC) в защищенную запись `MaBuy Production Encryption Key`.
3. *(Опционально)* Сохраните копию в распечатанном виде в физическом оффлайн-сейфе.

---

## 📦 2. Как восстановить проект и базу данных из Telegram Vault за 30 секунд

Если сервер погиб, а вам нужно запустить MaBuy на новом VPS:

### Шаг 1: Разверните чистый репозиторий на новом сервере
```bash
git clone <YOUR_REPO_URL> MaBuy
cd MaBuy
python -m venv venv
source venv/bin/activate  # или venv\Scripts\activate на Windows
pip install -r requirements.txt
```

### Шаг 2: Создайте `.env` и вставьте сохраненный `ENCRYPTION_KEY`
```bash
cp .env.example .env
nano .env
# Вставьте ENCRYPTION_KEY, TELEGRAM_BOT_TOKEN и другие ключи
```

### Шаг 3: Скачайте последний бэкап из Telegram
1. Откройте чат с ботом или сохраненные сообщения в Telegram.
2. Скачайте последний файл `mabuy_backup_YYYYMMDD_HHMMSS.db.enc.gz`.
3. Положите его в папку `data/backups/`.

### Шаг 4: Запустите скрипт расшифровки и восстановления
Запустите в Python:
```python
from src.infrastructure.persistence.backup_service import SQLiteBackupService
from src.infrastructure.security.encryption import EncryptionService
import gzip

# 1. Расшифровка
enc = EncryptionService()
with open("data/backups/mabuy_backup_LATEST.db.enc.gz", "rb") as f:
    decrypted_gz = enc.decrypt_bytes(f.read())

# 2. Распаковка Gzip
with open("data/mabuy_prod.db", "wb") as f_out:
    f_out.write(gzip.decompress(decrypted_gz))

print("✅ База данных успешно восстановлена и готова к работе!")
```

### Шаг 5: Запустите демона и бота
```bash
python scripts/auto_trader_daemon.py
python scripts/launch_aiogram_bot.py
```
Вся история сделок, подписки пользователей и статистика полностью восстановлены!
