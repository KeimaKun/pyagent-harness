"""
harness/plan_manager.py
=======================
File-backed execution state machine for plan.json.

Task lifecycle:
    pending → in_progress → completed
                         ↘ failed → REPLAN → pending (rewritten/split)

State is always persisted to disk immediately after any transition.
"""
from __future__ import annotations

import json
import logging
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task status constants
# ---------------------------------------------------------------------------
STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_REPLAN = "replan"

VALID_ROLES = {"Planner", "Architect", "Executor", "Critic"}


# ---------------------------------------------------------------------------
# Task schema helper
# ---------------------------------------------------------------------------
def make_task(
    description: str,
    role: str = "Executor",
    output_files: list[str] | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Create a new task dict with defaults."""
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role '{role}'. Must be one of {VALID_ROLES}")
    return {
        "id": task_id or str(uuid.uuid4()),
        "description": description,
        "role": role,
        "status": STATUS_PENDING,
        "output_files": output_files or [],
        "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# PlanManager
# ---------------------------------------------------------------------------
class PlanManager:
    """
    Manages the plan.json file-backed state machine.

    plan.json schema: a JSON array of task objects.
    """

    def __init__(self, plan_path: Path | str | None = None) -> None:
        self._path = Path(plan_path) if plan_path else Path(__file__).parent.parent / "plan.json"
        self._tasks: list[dict[str, Any]] = []
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load plan from disk. Creates an empty plan if file doesn't exist."""
        if self._path.exists():
            raw = self._path.read_text(encoding="utf-8").strip()
            self._tasks = json.loads(raw) if raw else []
        else:
            self._tasks = []
            self._save()
        logger.info("Plan loaded: %d tasks from %s", len(self._tasks), self._path)

    def _save(self) -> None:
        """Persist current state to disk atomically."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._tasks, indent=2), encoding="utf-8")
        tmp.replace(self._path)
        logger.debug("Plan saved (%d tasks)", len(self._tasks))

    def reload_from_disk(self) -> None:
        """
        Re-read plan.json from disk.
        Called before each task turn to allow human steering edits.
        """
        logger.debug("Reloading plan from disk (human steering check)...")
        self.load()

    # ------------------------------------------------------------------
    # Task queries
    # ------------------------------------------------------------------
    @property
    def tasks(self) -> list[dict[str, Any]]:
        return list(self._tasks)

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        return next((t for t in self._tasks if t["id"] == task_id), None)

    def get_next_task(self) -> dict[str, Any] | None:
        """Return the first pending task, or None if all are done/failed."""
        return next((t for t in self._tasks if t["status"] == STATUS_PENDING), None)

    def pending_count(self) -> int:
        return sum(1 for t in self._tasks if t["status"] == STATUS_PENDING)

    def completed_count(self) -> int:
        return sum(1 for t in self._tasks if t["status"] == STATUS_COMPLETED)

    def failed_count(self) -> int:
        return sum(1 for t in self._tasks if t["status"] == STATUS_FAILED)

    def is_complete(self) -> bool:
        """True when all tasks are either completed or failed (no pending/in_progress)."""
        return all(
            t["status"] in (STATUS_COMPLETED, STATUS_FAILED)
            for t in self._tasks
        )

    # ------------------------------------------------------------------
    # Plan creation
    # ------------------------------------------------------------------
    def create_plan(self, tasks: list[dict[str, Any]]) -> None:
        """
        Replace the current plan with a new list of tasks.
        Each task must have at least 'description'. Other fields are defaulted.
        """
        self._tasks = []
        for raw in tasks:
            self._tasks.append(make_task(
                description=raw.get("description", ""),
                role=raw.get("role", "Executor"),
                output_files=raw.get("output_files", []),
                task_id=raw.get("id"),
            ))
        self._save()
        logger.info("New plan created: %d tasks", len(self._tasks))

    def append_tasks(self, tasks: list[dict[str, Any]]) -> None:
        """Append additional tasks to the existing plan (used during REPLAN)."""
        for raw in tasks:
            self._tasks.append(make_task(
                description=raw.get("description", ""),
                role=raw.get("role", "Executor"),
                output_files=raw.get("output_files", []),
                task_id=raw.get("id"),
            ))
        self._save()

    # ------------------------------------------------------------------
    # State transitions (only these methods may write status)
    # ------------------------------------------------------------------
    def _update_task(self, task_id: str, **kwargs: Any) -> None:
        for task in self._tasks:
            if task["id"] == task_id:
                task.update(kwargs)
                task["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._save()
                return
        raise KeyError(f"Task '{task_id}' not found in plan.")

    def mark_in_progress(self, task_id: str) -> None:
        logger.info("Task → in_progress: %s", task_id)
        self._update_task(task_id, status=STATUS_IN_PROGRESS)

    def mark_completed(self, task_id: str) -> None:
        """
        ONLY called by verifier.py after deterministic checks pass.
        The LLM cannot call this directly.
        """
        logger.info("Task → completed: %s", task_id)
        self._update_task(task_id, status=STATUS_COMPLETED, error=None)

    def mark_failed(self, task_id: str, error: str) -> None:
        logger.warning("Task → failed: %s | error: %.200s", task_id, error)
        self._update_task(task_id, status=STATUS_FAILED, error=error)

    # ------------------------------------------------------------------
    # REPLAN
    # ------------------------------------------------------------------
    def trigger_replan(
        self,
        failed_task_id: str,
        exc_info: str | None = None,
        replacement_tasks: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Enter REPLAN state for a failed task.

        Steps:
        1. Capture the traceback / error string.
        2. Mark the failed task with STATUS_REPLAN.
        3. Replace with replacement_tasks (if LLM provided decomposition),
           otherwise create a single retry task.
        4. Return the new pending tasks for the executor to process.

        Parameters
        ----------
        failed_task_id : str
        exc_info : str, optional
            Traceback or error description from the failed attempt.
        replacement_tasks : list[dict], optional
            LLM-generated replacement task dicts. If None, a simple retry is created.

        Returns
        -------
        list[dict]
            The newly inserted pending task(s).
        """
        failed_task = self.get_task(failed_task_id)
        if failed_task is None:
            raise KeyError(f"Task '{failed_task_id}' not found.")

        # Capture current traceback if not provided
        if exc_info is None:
            exc_info = traceback.format_exc()

        logger.warning(
            "REPLAN triggered for task %s\nTraceback preview:\n%s",
            failed_task_id,
            exc_info[:500],
        )

        # Mark original task as replan
        self._update_task(failed_task_id, status=STATUS_REPLAN, error=exc_info)

        # Build replacement tasks
        if replacement_tasks:
            new_tasks = [
                make_task(
                    description=rt.get("description", failed_task["description"]),
                    role=rt.get("role", failed_task["role"]),
                    output_files=rt.get("output_files", failed_task.get("output_files", [])),
                )
                for rt in replacement_tasks
            ]
        else:
            # Default: create a single retry task with annotated description
            new_tasks = [
                make_task(
                    description=f"[RETRY] {failed_task['description']}",
                    role=failed_task["role"],
                    output_files=failed_task.get("output_files", []),
                )
            ]

        # Insert new tasks right after the failed task
        idx = next(i for i, t in enumerate(self._tasks) if t["id"] == failed_task_id)
        for offset, nt in enumerate(new_tasks):
            self._tasks.insert(idx + 1 + offset, nt)

        self._save()
        logger.info(
            "REPLAN: inserted %d replacement task(s) after task %s",
            len(new_tasks),
            failed_task_id,
        )
        return new_tasks

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "total": len(self._tasks),
            "pending": self.pending_count(),
            "in_progress": sum(1 for t in self._tasks if t["status"] == STATUS_IN_PROGRESS),
            "completed": self.completed_count(),
            "failed": self.failed_count(),
            "replan": sum(1 for t in self._tasks if t["status"] == STATUS_REPLAN),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"<PlanManager path={self._path} "
            f"total={s['total']} completed={s['completed']} "
            f"pending={s['pending']} failed={s['failed']}>"
        )
