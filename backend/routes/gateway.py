"""Gateway API routes — jobs, approvals, events, health.

These routes complement the existing agent routes and provide
the Web UI with access to the gateway subsystems.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from typing import Any

router = APIRouter(prefix="/api", tags=["gateway"])


@router.get("/jobs")
async def list_jobs(status: str | None = None, kind: str | None = None, limit: int = 50):
    """List local jobs."""
    from jobs.local_job_manager import job_manager
    jobs = job_manager.list_jobs(status=status, kind=kind, limit=limit)
    return [j.to_dict() for j in jobs]


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get a single job."""
    from jobs.local_job_manager import job_manager
    record = job_manager.get_job(job_id)
    if not record:
        # Try partial match
        for j in job_manager.list_jobs():
            if j.job_id.startswith(job_id):
                record = j
                break
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    return record.to_dict()


@router.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str, lines: int = 100):
    """Get job log tail."""
    from jobs.local_job_manager import job_manager
    record = job_manager.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": record.job_id, "logs": job_manager.tail_logs(job_id, lines)}


@router.post("/jobs/{job_id}/stop")
async def stop_job(job_id: str):
    """Stop a running job."""
    from jobs.local_job_manager import job_manager
    record = await job_manager.stop_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    return record.to_dict()


@router.post("/jobs/{job_id}/kill")
async def kill_job(job_id: str):
    """Force kill a running job."""
    from jobs.local_job_manager import job_manager
    record = await job_manager.kill_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    return record.to_dict()


@router.get("/approvals")
async def list_approvals(status: str | None = None):
    """List approvals."""
    from approvals.approval_store import approval_store
    if status == "pending":
        records = approval_store.list_pending()
    else:
        # Return all from memory
        records = list(approval_store._pending.values())
        if status:
            records = [r for r in records if r.status == status]
    return [r.to_dict() for r in records]


@router.post("/approvals/{approval_id}/approve")
async def approve_approval(approval_id: str):
    """Approve a pending approval."""
    from approvals.approval_store import approval_store
    record = await approval_store.approve(approval_id)
    if not record:
        raise HTTPException(status_code=404, detail="Approval not found")
    return record.to_dict()


@router.post("/approvals/{approval_id}/reject")
async def reject_approval(approval_id: str):
    """Reject a pending approval."""
    from approvals.approval_store import approval_store
    record = await approval_store.reject(approval_id)
    if not record:
        raise HTTPException(status_code=404, detail="Approval not found")
    return record.to_dict()


@router.get("/gateway/events")
async def list_events(limit: int = 100, event_type: str | None = None, source: str | None = None):
    """Query gateway events."""
    from events.event_store import event_store
    return event_store.query(limit=limit, event_type=event_type, source=source)


@router.get("/gateway/events/stats")
async def event_stats():
    """Event store statistics."""
    from events.event_store import event_store
    return event_store.stats()


@router.get("/crons")
async def list_crons():
    """List all cron tasks."""
    from prompt_cron import prompt_cron_manager
    return await prompt_cron_manager.list()
