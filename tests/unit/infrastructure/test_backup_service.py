import unittest
import os
import shutil
import sqlite3
import time
from src.infrastructure.persistence.backup_service import SQLiteBackupService

class TestSQLiteBackupService(unittest.TestCase):
    def setUp(self):
        self.test_dir = f"temp_test_backup_suite_{int(time.time() * 1000)}"
        os.makedirs(self.test_dir, exist_ok=True)
        self.db_path = os.path.join(self.test_dir, "test_prod.db")
        self.backup_dir = os.path.join(self.test_dir, "backups")

        # Create and populate test database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, name TEXT, balance REAL);")
            cursor.execute("INSERT INTO users VALUES ('1', 'Alice', 5000.0);")
            cursor.execute("INSERT INTO users VALUES ('2', 'Bob', 12000.0);")
            conn.commit()

        self.backup_service = SQLiteBackupService(
            db_path=self.db_path,
            backup_dir=self.backup_dir,
            retention_days=7
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir, ignore_errors=True)
            except Exception:
                pass

    def test_create_hot_backup_and_restore(self):
        # 1. Create hot backup
        gz_path = self.backup_service.create_hot_backup()
        self.assertIsNotNone(gz_path)
        self.assertTrue(os.path.exists(gz_path))
        self.assertTrue(gz_path.endswith(".db.gz"))
        self.assertGreater(os.path.getsize(gz_path), 50)

        # 2. Restore to a new location
        restored_db_path = os.path.join(self.test_dir, "restored.db")
        success = self.backup_service.restore_backup(gz_path, restored_db_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(restored_db_path))

        # 3. Verify data integrity in restored DB
        with sqlite3.connect(restored_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, balance FROM users ORDER BY id;")
            rows = cursor.fetchall()
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0], ("Alice", 5000.0))
            self.assertEqual(rows[1], ("Bob", 12000.0))

    def test_background_backup_thread(self):
        thread = self.backup_service.run_backup_in_background()
        thread.join(timeout=5.0)
        self.assertFalse(thread.is_alive())

        # Verify a backup was produced
        backups = [f for f in os.listdir(self.backup_dir) if f.endswith(".db.gz")]
        self.assertGreaterEqual(len(backups), 1)

    def test_backup_rotation_purges_old_archives(self):
        # Create an artificial old backup
        old_file = os.path.join(self.backup_dir, "mabuy_backup_20200101_000000.db.gz")
        with open(old_file, "wb") as f:
            f.write(b"dummy")
        
        # Set modification time to 10 days ago
        ten_days_ago = time.time() - (10 * 86400.0)
        os.utime(old_file, (ten_days_ago, ten_days_ago))

        deleted = self.backup_service.rotate_backups()
        self.assertEqual(deleted, 1)
        self.assertFalse(os.path.exists(old_file))

    def test_remote_vault_handles_missing_credentials_gracefully(self):
        # When token/chat_id is None, it safely returns False without raising exceptions
        result = self.backup_service.send_to_remote_vault("non_existent.db.gz")
        self.assertFalse(result)

if __name__ == "__main__":
    unittest.main()
