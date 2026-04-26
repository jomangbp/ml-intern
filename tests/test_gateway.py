"""Tests for gateway identity and authorization."""

import os
import tempfile
import pytest
from pathlib import Path


# Setup path before imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from gateway.identity import (
    IdentityManager, GatewayIdentity, ROLE_PERMISSIONS,
    COMMAND_PERMISSIONS, _read_identity_store, _write_identity_store,
)


class TestGatewayIdentity:
    def test_create_identity(self):
        mgr = IdentityManager()
        identity = mgr.resolve_or_create("telegram", "12345", "TestUser")
        assert identity.identity_id.startswith("id_")
        assert identity.platform == "telegram"
        assert identity.platform_user_id == "12345"
        assert identity.display_name == "TestUser"

    def test_first_user_is_owner(self):
        mgr = IdentityManager()
        mgr._cache.clear()
        identity = mgr.resolve_or_create("test_platform", "999888777", "First", default_roles=["owner"])
        assert "owner" in identity.roles
        assert identity.has_permission("approve")
        assert identity.has_permission("kill_job")
        assert identity.has_permission("run_plan")

    def test_second_user_is_not_owner(self):
        mgr = IdentityManager()
        mgr._cache.clear()
        mgr.resolve_or_create("test_platform", "999888777", "First", default_roles=["owner"])
        second = mgr.resolve_or_create("test_platform", "888999666", "Second", default_roles=["user"])
        assert "owner" not in second.roles
        assert "user" in second.roles
        assert not second.has_permission("approve")
        assert second.has_permission("view_logs")

    def test_command_permission_check(self):
        mgr = IdentityManager()
        mgr._cache.clear()
        mgr.resolve_or_create("test_platform", "999888777", "Admin", default_roles=["owner"])
        allowed, identity = mgr.check_command_permission("test_platform", "999888777", "kill")
        assert allowed
        assert identity is not None

    def test_command_permission_denied(self):
        mgr = IdentityManager()
        mgr._cache.clear()
        mgr.resolve_or_create("telegram", "222", "Viewer", default_roles=["viewer"])
        allowed, identity = mgr.check_command_permission("telegram", "222", "kill")
        assert not allowed

    def test_unknown_command_allowed_for_any_identity(self):
        mgr = IdentityManager()
        mgr._cache.clear()
        mgr.resolve_or_create("telegram", "333", "User", default_roles=["viewer"])
        allowed, _ = mgr.check_command_permission("telegram", "333", "unknown_command")
        assert allowed

    def test_no_identity_denied(self):
        mgr = IdentityManager()
        mgr._cache.clear()
        allowed, identity = mgr.check_command_permission("telegram", "999", "kill")
        assert not allowed
        assert identity is None


class TestEventStore:
    def test_log_and_tail(self):
        from events.event_store import EventStore
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            store = EventStore(path=path)
            store.log("test.event", source="test", payload={"key": "value"})
            store.log("test.event2", source="test", payload={"key2": "value2"})
            events = store.tail(limit=10)
            assert len(events) == 2
            assert events[0]["type"] == "test.event"
            assert events[1]["type"] == "test.event2"
        finally:
            os.unlink(path)

    def test_filter_by_type(self):
        from events.event_store import EventStore
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            store = EventStore(path=path)
            store.log("type_a", source="test")
            store.log("type_b", source="test")
            store.log("type_a", source="test")
            events = store.tail(limit=10, event_type="type_a")
            assert len(events) == 2
        finally:
            os.unlink(path)

    def test_never_crashes(self):
        from events.event_store import EventStore
        store = EventStore(path="/nonexistent/dir/file.jsonl")
        result = store.log("test", source="test")  # Should not raise
        assert result is not None
        events = store.tail()
        assert events == []

    def test_stats(self):
        from events.event_store import EventStore
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            store = EventStore(path=path)
            store.log("a", source="test")
            store.log("b", source="test")
            store.log("a", source="test")
            stats = store.stats()
            assert stats["total_events"] == 3
            assert stats["event_types"]["a"] == 2
            assert stats["event_types"]["b"] == 1
        finally:
            os.unlink(path)


class TestApprovalStore:
    def test_create_and_get(self):
        from approvals.approval_store import ApprovalStore
        store = ApprovalStore()
        record = store.create(
            session_id="sess_123",
            tools=[{"tool": "bash", "arguments": {"command": "echo hello"}, "tool_call_id": "tc_1"}],
            platform="telegram",
            chat_id=12345,
        )
        assert record.approval_id.startswith("appr_")
        assert record.status == "pending"
        assert store.get(record.approval_id) is record

    def test_list_pending(self):
        from approvals.approval_store import ApprovalStore
        store = ApprovalStore()
        store.create(session_id="s1", tools=[{"tool": "bash", "arguments": {}, "tool_call_id": "tc1"}])
        store.create(session_id="s2", tools=[{"tool": "bash", "arguments": {}, "tool_call_id": "tc2"}])
        pending = store.list_pending()
        assert len(pending) == 2

    def test_list_pending_filtered(self):
        from approvals.approval_store import ApprovalStore
        store = ApprovalStore()
        store.create(session_id="s1", tools=[{"tool": "bash", "arguments": {}, "tool_call_id": "tc1"}],
                      platform="telegram", chat_id=111)
        store.create(session_id="s2", tools=[{"tool": "bash", "arguments": {}, "tool_call_id": "tc2"}],
                      platform="telegram", chat_id=222)
        pending = store.list_pending(platform="telegram", chat_id=111)
        assert len(pending) == 1


class TestCommandRouter:
    @pytest.mark.asyncio
    async def test_dispatch_unknown_command(self):
        from gateway.command_router import CommandRouter
        from gateway.adapter_base import GatewayCommand
        router = CommandRouter()
        cmd = GatewayCommand(source="telegram", command="test_cmd", user_id="123", platform="telegram")
        result = await router.dispatch(cmd)
        assert result is not None

    @pytest.mark.asyncio
    async def test_dispatch_registered_handler(self):
        from gateway.command_router import CommandRouter
        from gateway.adapter_base import GatewayCommand
        router = CommandRouter()

        async def handler(cmd):
            return {"response": f"Handled {cmd.command}"}

        router.register("test_cmd", handler)

        # Need an identity for auth
        from gateway.identity import identity_manager
        identity_manager._cache.clear()
        identity_manager.resolve_or_create("telegram", "123", "Test", default_roles=["owner"])

        cmd = GatewayCommand(source="telegram", command="test_cmd", user_id="123", platform="telegram")
        result = await router.dispatch(cmd)
        assert result["response"] == "Handled test_cmd"


class TestLocalJobManager:
    @pytest.mark.asyncio
    async def test_start_and_list_job(self):
        from jobs.local_job_manager import LocalJobManager, JOB_DIR, JOB_LOG_DIR
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = LocalJobManager()
            record = await mgr.start_job("echo hello", cwd=tmpdir, kind="test")
            assert record.job_id.startswith("job_")
            assert record.status == "running"
            assert record.pid is not None

            # Wait for completion
            import asyncio
            await asyncio.sleep(2)

            jobs = mgr.list_jobs()
            assert len(jobs) >= 1
            job = mgr.get_job(record.job_id)
            assert job is not None
            assert job.status in ("completed", "running")

    @pytest.mark.asyncio
    async def test_tail_logs(self):
        from jobs.local_job_manager import LocalJobManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = LocalJobManager()
            record = await mgr.start_job("echo test_log_output_123", cwd=tmpdir)
            import asyncio
            await asyncio.sleep(2)
            logs = mgr.tail_logs(record.job_id)
            assert "test_log_output_123" in logs
