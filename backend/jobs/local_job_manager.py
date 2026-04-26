"""Local Job Manager — unified abstraction for local training, scripts, and processes.

Provides stable job IDs, per-job logs, status tracking, and process control.
Persisted to disk so jobs survive gateway restarts.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from events.event_store import event_store

JOB_DIR = Path(os.environ.get(
    "ML_INTERN_JOB_DIR",
    str(Path.home() / ".cache" / "ml-intern" / "jobs"),
))
JOB_LOG_DIR = JOB_DIR / "logs"


def _new_job_id() -> str:
    return f"job_{uuid.uuid4().hex[:10]}"


def _persist_job(record: dict[str, Any]) -> None:
    try:
        JOB_DIR.mkdir(parents=True, exist_ok=True)
        path = JOB_DIR / f"{record['job_id']}.json"
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        pass


def _delete_job_file(job_id: str) -> None:
    try:
        (JOB_DIR / f"{job_id}.json").unlink(missing_ok=True)
    except Exception:
        pass


@dataclass
class JobRecord:
    """Represents a tracked local job."""
    job_id: str
    kind: str  # training, eval, script, watchdog
    command: str
    cwd: str = ""
    pid: int | None = None
    status: str = "queued"  # queued, running, completed, failed, cancelled, killed
    exit_code: int | None = None
    started_at: float | None = None
    ended_at: float | None = None
    log_path: str = ""
    created_at: float = field(default_factory=time.time)
    created_by: dict[str, Any] = field(default_factory=dict)  # source, user_id, chat_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "command": self.command,
            "cwd": self.cwd,
            "pid": self.pid,
            "status": self.status,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "log_path": self.log_path,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobRecord:
        return cls(
            job_id=data["job_id"],
            kind=data.get("kind", "script"),
            command=data.get("command", ""),
            cwd=data.get("cwd", ""),
            pid=data.get("pid"),
            status=data.get("status", "queued"),
            exit_code=data.get("exit_code"),
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
            log_path=data.get("log_path", ""),
            created_at=data.get("created_at", time.time()),
            created_by=data.get("created_by", {}),
        )

    @property
    def elapsed(self) -> str:
        if not self.started_at:
            return "not started"
        end = self.ended_at or time.time()
        secs = int(end - self.started_at)
        if secs < 60:
            return f"{secs}s"
        m, s = divmod(secs, 60)
        if m < 60:
            return f"{m}m{s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m"


class LocalJobManager:
    """Manages local job lifecycle: start, stop, kill, list, tail logs."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._monitor_tasks: dict[str, asyncio.Task] = {}

    async def start_job(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        kind: str = "script",
        created_by: dict[str, Any] | None = None,
    ) -> JobRecord:
        """Start a local job and return its record."""
        JOB_DIR.mkdir(parents=True, exist_ok=True)
        JOB_LOG_DIR.mkdir(parents=True, exist_ok=True)

        job_id = _new_job_id()
        log_path = str(JOB_LOG_DIR / f"{job_id}.log")

        record = JobRecord(
            job_id=job_id,
            kind=kind,
            command=command,
            cwd=cwd or os.getcwd(),
            log_path=log_path,
            status="queued",
            created_by=created_by or {},
        )

        # Start process
        try:
            log_file = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                cwd=cwd,
                env={**os.environ, **(env or {})},
                start_new_session=True,
            )
            record.pid = proc.pid
            record.status = "running"
            record.started_at = time.time()
        except Exception as e:
            record.status = "failed"
            record.ended_at = time.time()
            record.exit_code = -1
            _persist_job(record.to_dict())
            event_store.log("job.failed", source="job_manager", job_id=job_id,
                            payload={"command": command, "error": str(e)})
            raise

        self._jobs[job_id] = record
        self._procs[job_id] = proc
        _persist_job(record.to_dict())

        event_store.log("job.started", source="job_manager", job_id=job_id,
                        payload={"command": command, "pid": proc.pid, "kind": kind})

        # Start monitor task
        self._monitor_tasks[job_id] = asyncio.create_task(
            self._monitor_job(job_id), name=f"job-monitor-{job_id}"
        )

        return record

    async def _monitor_job(self, job_id: str) -> None:
        """Monitor a running process until it exits."""
        proc = self._procs.get(job_id)
        record = self._jobs.get(job_id)
        if not proc or not record:
            return

        # Run in thread to avoid blocking event loop
        exit_code = await asyncio.to_thread(proc.wait)

        record.exit_code = exit_code
        record.ended_at = time.time()
        record.status = "completed" if exit_code == 0 else "failed"

        self._procs.pop(job_id, None)
        _persist_job(record.to_dict())

        event_type = "job.completed" if exit_code == 0 else "job.failed"
        event_store.log(event_type, source="job_manager", job_id=job_id,
                        payload={"exit_code": exit_code, "elapsed": record.elapsed})

    async def stop_job(self, job_id: str, sig: str = "TERM") -> JobRecord | None:
        """Stop a running job with SIGTERM (default) or custom signal."""
        record = self._jobs.get(job_id)
        if not record:
            return None
        if record.status != "running":
            return record

        pid = record.pid
        if not pid:
            return record

        try:
            sig_num = getattr(signal, f"SIG{sig}", signal.SIGTERM)
            os.kill(pid, sig_num)
            record.status = "cancelled"
            record.ended_at = time.time()
        except ProcessLookupError:
            record.status = "completed"
            record.ended_at = time.time()
        except Exception as e:
            record.status = "failed"
            record.ended_at = time.time()

        self._procs.pop(job_id, None)
        _persist_job(record.to_dict())
        event_store.log("job.cancelled", source="job_manager", job_id=job_id,
                        payload={"signal": sig, "pid": pid})
        return record

    async def kill_job(self, job_id: str) -> JobRecord | None:
        """Force kill a running job with SIGKILL."""
        record = self._jobs.get(job_id)
        if not record:
            return None
        if record.status != "running":
            return record

        pid = record.pid
        if pid:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        record.status = "killed"
        record.ended_at = time.time()
        self._procs.pop(job_id, None)
        _persist_job(record.to_dict())
        event_store.log("job.killed", source="job_manager", job_id=job_id,
                        payload={"pid": pid})
        return record

    def get_job(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def list_jobs(self, status: str | None = None, kind: str | None = None, limit: int = 50) -> list[JobRecord]:
        """List jobs, optionally filtered by status and kind."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        if kind:
            jobs = [j for j in jobs if j.kind == kind]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def tail_logs(self, job_id: str, lines: int = 100) -> str:
        """Return the last N lines of a job's log."""
        record = self._jobs.get(job_id)
        if not record or not record.log_path:
            return "Job not found or no logs."
        try:
            path = Path(record.log_path)
            if not path.exists():
                return "Log file not found."
            all_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(all_lines[-lines:])
        except Exception as e:
            return f"Error reading logs: {e}"

    def running_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == "running")

    def restore(self) -> int:
        """Restore jobs from disk. Marks orphaned running jobs as unknown."""
        restored = 0
        if not JOB_DIR.exists():
            return restored
        for path in JOB_DIR.glob("job_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                record = JobRecord.from_dict(data)
                # Running jobs from previous session are orphaned
                if record.status == "running":
                    record.status = "unknown"  # process may or may not be alive
                    _persist_job(record.to_dict())
                self._jobs[record.job_id] = record
                restored += 1
            except Exception:
                pass
        return restored


# Singleton
job_manager = LocalJobManager()
