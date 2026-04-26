"""Gateway identity and authorization model.

Normalizes platform-specific identities (Telegram user ID, CLI user, Web UI user)
into internal identities with roles and permissions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

IDENTITY_STORE_PATH = Path(os.environ.get(
    "ML_INTERN_IDENTITY_STORE",
    str(Path.home() / ".cache" / "ml-intern" / "identities.json"),
))

# Default permissions by role
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "owner": {
        "run_plan", "approve", "reject", "kill_job", "view_logs",
        "create_cron", "cancel_cron", "model_select", "session_new",
        "session_interrupt", "run_bash", "run_training", "gateway_admin",
    },
    "admin": {
        "run_plan", "approve", "reject", "kill_job", "view_logs",
        "create_cron", "cancel_cron", "model_select", "session_new",
        "session_interrupt", "run_bash", "run_training",
    },
    "user": {
        "run_plan", "view_logs", "create_cron", "cancel_cron",
        "model_select", "session_new", "session_interrupt",
    },
    "viewer": {
        "view_logs", "view_status",
    },
}

# Commands that require specific permissions
COMMAND_PERMISSIONS: dict[str, str] = {
    "new": "session_new",
    "model": "model_select",
    "cron": "create_cron",
    "cancelcron": "cancel_cron",
    "interrupt": "session_interrupt",
    "kill": "kill_job",
    "approve": "approve",
    "reject": "reject",
    "run": "run_bash",
    "jobs": "view_logs",
    "logs": "view_logs",
    "jobs": "view_logs",
    "logs": "view_logs",
    "kill": "kill_job",
    "gateway": "view_status",
    "status": "view_status",
    "sessions": "view_status",
    "crons": "view_status",
    "models": "view_status",
    "approvals": "view_status",
    "events": "gateway_admin",
}


def _read_identity_store() -> dict[str, Any]:
    try:
        if IDENTITY_STORE_PATH.exists():
            return json.loads(IDENTITY_STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"identities": []}


def _write_identity_store(data: dict[str, Any]) -> None:
    IDENTITY_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = IDENTITY_STORE_PATH.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, IDENTITY_STORE_PATH)


class GatewayIdentity:
    """Represents a normalized internal identity."""

    def __init__(
        self,
        identity_id: str,
        platform: str,
        platform_user_id: str,
        display_name: str = "",
        roles: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.identity_id = identity_id
        self.platform = platform
        self.platform_user_id = str(platform_user_id)
        self.display_name = display_name
        self.roles = roles or ["user"]
        self.extra = extra or {}

    @property
    def permissions(self) -> set[str]:
        perms: set[str] = set()
        for role in self.roles:
            perms.update(ROLE_PERMISSIONS.get(role, set()))
        return perms

    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "platform": self.platform,
            "platform_user_id": self.platform_user_id,
            "display_name": self.display_name,
            "roles": self.roles,
            "permissions": sorted(self.permissions),
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GatewayIdentity:
        return cls(
            identity_id=data["identity_id"],
            platform=data["platform"],
            platform_user_id=data["platform_user_id"],
            display_name=data.get("display_name", ""),
            roles=data.get("roles", ["user"]),
            extra=data.get("extra", {}),
        )


class IdentityManager:
    """Manages internal identities and authorization checks."""

    def __init__(self) -> None:
        self._cache: dict[str, GatewayIdentity] = {}

    def _key(self, platform: str, platform_user_id: str) -> str:
        return f"{platform}:{platform_user_id}"

    def resolve_or_create(
        self,
        platform: str,
        platform_user_id: str | int,
        display_name: str = "",
        default_roles: list[str] | None = None,
    ) -> GatewayIdentity:
        """Resolve an existing identity or create a new one."""
        key = self._key(platform, str(platform_user_id))
        if key in self._cache:
            return self._cache[key]

        store = _read_identity_store()
        for ident_data in store.get("identities", []):
            if ident_data.get("platform") == platform and str(ident_data.get("platform_user_id")) == str(platform_user_id):
                identity = GatewayIdentity.from_dict(ident_data)
                self._cache[key] = identity
                return identity

        # Auto-create with owner role for first identity, user for others
        if not default_roles:
            has_any = bool(store.get("identities"))
            default_roles = ["owner"] if not has_any else ["user"]

        identity = GatewayIdentity(
            identity_id=f"id_{os.urandom(6).hex()}",
            platform=platform,
            platform_user_id=str(platform_user_id),
            display_name=display_name,
            roles=default_roles,
        )

        store.setdefault("identities", []).append(identity.to_dict())
        _write_identity_store(store)
        self._cache[key] = identity
        return identity

    def get(self, platform: str, platform_user_id: str | int) -> GatewayIdentity | None:
        key = self._key(platform, str(platform_user_id))
        if key in self._cache:
            return self._cache[key]
        store = _read_identity_store()
        for ident_data in store.get("identities", []):
            if ident_data.get("platform") == platform and str(ident_data.get("platform_user_id")) == str(platform_user_id):
                identity = GatewayIdentity.from_dict(ident_data)
                self._cache[key] = identity
                return identity
        return None

    def check_command_permission(
        self, platform: str, platform_user_id: str | int, command: str,
    ) -> tuple[bool, GatewayIdentity | None]:
        """Check if a user has permission to run a command.

        Returns (allowed, identity).
        """
        identity = self.get(platform, str(platform_user_id))
        if not identity:
            return False, None

        required_perm = COMMAND_PERMISSIONS.get(command)
        if not required_perm:
            # Unknown commands just need any identity
            return True, identity

        return identity.has_permission(required_perm), identity


identity_manager = IdentityManager()
