"""
ChurnGuard AI - Authentication Module
======================================
Secure file-based auth using PBKDF2-HMAC-SHA256 with per-user salt.
User store: users.json (never committed — added to .gitignore).
Roles: Administrator | Analyst | Viewer
"""

from __future__ import annotations
import hashlib, os, json, secrets
from typing import Tuple, Dict, List, Optional

# ── Paths ────────────────────────────────────────────────────────────────────
_ROOT     = os.path.dirname(os.path.dirname(__file__))
_USERS_DB = os.path.join(_ROOT, "users.json")

# ── Default bootstrap users (hashed at runtime, never stored as plaintext) ──
_BOOTSTRAP = [
    {"username": "admin",   "password": "admin123",   "role": "Administrator"},
    {"username": "analyst", "password": "analyst@cg", "role": "Analyst"},
    {"username": "demo",    "password": "demo2024",   "role": "Viewer"},
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Return (hex_hash, hex_salt) using PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
    return dk.hex(), salt


def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    dk, _ = _hash_password(password, salt)
    return secrets.compare_digest(dk, stored_hash)


# ── User Store ────────────────────────────────────────────────────────────────
def _bootstrap_db() -> List[Dict]:
    """Create the initial user database with hashed bootstrap passwords."""
    users = []
    for entry in _BOOTSTRAP:
        pw_hash, salt = _hash_password(entry["password"])
        users.append({
            "username": entry["username"],
            "pw_hash":  pw_hash,
            "salt":     salt,
            "role":     entry["role"],
        })
    return users


def _load_users() -> List[Dict]:
    """Load users from JSON store; bootstrap if missing."""
    if not os.path.exists(_USERS_DB):
        users = _bootstrap_db()
        _save_users(users)
        return users
    try:
        with open(_USERS_DB, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        users = _bootstrap_db()
        _save_users(users)
        return users


def _save_users(users: List[Dict]) -> None:
    try:
        with open(_USERS_DB, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)
    except OSError:
        pass  # read-only filesystem (Streamlit Cloud) — graceful degradation


def _find_user(users: List[Dict], username: str) -> Optional[Dict]:
    username = username.strip().lower()
    return next((u for u in users if u["username"] == username), None)


# ── Public API ────────────────────────────────────────────────────────────────
def verify_credentials(username: str, password: str) -> Tuple[bool, str]:
    """
    Verify login credentials.

    Returns:
        (success, role_or_error_message)
    """
    if not username or not password:
        return False, "Username and password are required."
    users  = _load_users()
    record = _find_user(users, username.strip().lower())
    if record and _verify_password(password, record["pw_hash"], record["salt"]):
        return True, record["role"]
    return False, "Invalid username or password."


def register_user(username: str, password: str, role: str = "Analyst") -> Tuple[bool, str]:
    """
    Register a new user.

    Returns:
        (success, message)
    """
    username = username.strip().lower()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if role not in ("Administrator", "Analyst", "Viewer"):
        return False, "Invalid role."

    users = _load_users()
    if _find_user(users, username):
        return False, f"Username '{username}' is already taken."

    pw_hash, salt = _hash_password(password)
    users.append({"username": username, "pw_hash": pw_hash, "salt": salt, "role": role})
    _save_users(users)
    return True, f"Account created for '{username}'."


def delete_user(username: str) -> Tuple[bool, str]:
    """Delete a user by username (admin-only action enforced in UI)."""
    username = username.strip().lower()
    if username in ("admin", "demo"):
        return False, "Cannot delete built-in accounts."
    users = _load_users()
    before = len(users)
    users = [u for u in users if u["username"] != username]
    if len(users) == before:
        return False, f"User '{username}' not found."
    _save_users(users)
    return True, f"User '{username}' deleted."


def list_users() -> List[Dict]:
    """Return a sanitised user list (no hashes or salts)."""
    return [
        {"username": u["username"], "role": u["role"]}
        for u in _load_users()
    ]


def change_password(username: str, old_password: str, new_password: str) -> Tuple[bool, str]:
    """Change a user's password."""
    username = username.strip().lower()
    if len(new_password) < 6:
        return False, "New password must be at least 6 characters."
    users  = _load_users()
    record = _find_user(users, username)
    if not record:
        return False, "User not found."
    if not _verify_password(old_password, record["pw_hash"], record["salt"]):
        return False, "Current password is incorrect."
    pw_hash, salt = _hash_password(new_password)
    record["pw_hash"] = pw_hash
    record["salt"]    = salt
    _save_users(users)
    return True, "Password updated successfully."


def get_demo_hint() -> str:
    return "Demo credentials → username: demo  |  password: demo2024"
