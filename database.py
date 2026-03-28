import sqlite3
import datetime
import os
from cryptography.fernet import Fernet

DB_NAME = "medical_bot.db"
KEY_FILE = "secret.key"


def load_or_generate_key():
    # Load existing key or create new
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            return f.read()
    else:
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
        return key


# Initialize encryption tool
cipher_suite = Fernet(load_or_generate_key())


def init_db():
    # Create table if not exist
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                username TEXT,
                timestamp TEXT,
                result_class BLOB, -- Храним как байты после шифрования
                confidence REAL
            )
        """
        )
        conn.commit()
    print("Database with encryption initialized")


def save_to_db(user_id, username, result_class, confidence):
    # Encrypt diagnosis
    encrypted_class = cipher_suite.encrypt(result_class.encode("utf-8"))

    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO predictions (user_id, username, timestamp, result_class, confidence) VALUES (?, ?, ?, ?, ?)",
            (
                user_id,
                username,
                datetime.datetime.now().isoformat(),
                encrypted_class,
                confidence,
            ),
        )
        conn.commit()


def get_history():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username, result_class, confidence FROM predictions")
        rows = cursor.fetchall()

        results = []
        for user, enc_data, conf in rows:
            decrypted_data = cipher_suite.decrypt(enc_data).decode("utf-8")
            results.append((user, decrypted_data, conf))
        return results
