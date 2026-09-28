"""Password hashing and session tokens."""
import hashlib
import hmac
import secrets

ITERATIONS = 200_000


def hash_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    """Hash a password with PBKDF2-HMAC-SHA256 and a random 16-byte salt."""
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return salt, digest


def verify_password(password: str, salt: bytes, expected: bytes) -> bool:
    """Recompute the hash with the stored salt and compare in constant time."""
    _, digest = hash_password(password, salt)
    return hmac.compare_digest(digest, expected)


def generate_token(nbytes: int = 32) -> str:
    """Create a URL-safe random session token."""
    return secrets.token_urlsafe(nbytes)
