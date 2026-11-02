"""
ChurnGuard AI - Simple Authentication Module
Demo-level credential check (replace with a real auth provider in production).
"""

import hashlib
from typing import Tuple

# ── Demo Credentials (hash stored, never plaintext) ─────────────────────────
_USERS = {
    "admin": hashlib.sha256("admin123".encode()).hexdigest(),
    "analyst": hashlib.sha256("analyst@cg".encode()).hexdigest(),
    "demo": hashlib.sha256("demo2024".encode()).hexdigest(),
}


def verify_credentials(username: str, password: str) -> Tuple[bool, str]:
    """
    Verify login credentials.

    Returns:
        (success, role_or_error_message)
    """
    username = username.strip().lower()
    pw_hash = hashlib.sha256(password.encode()).hexdigest()

    if username in _USERS and _USERS[username] == pw_hash:
        role = "Administrator" if username == "admin" else "Analyst"
        return True, role

    return False, "Invalid username or password."


def get_demo_hint() -> str:
    return "Demo credentials → username: demo  |  password: demo2024"
