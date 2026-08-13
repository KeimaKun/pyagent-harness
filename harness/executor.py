"""
harness/executor.py
===================
ReAct loop executor with:
  - Sequential persona swapping based on task role
  - Context window flushing between tasks (reloads AGENT.md + plan state)
  - Live plan.json watcher before each task turn
  - Integration of all harness components
"""
from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any

from harness.config_loader import ConfigLoader
from harness.llm_engine import LLMEngine
from harness.plan_manager import PlanManager, STATUS_PENDING, STATUS_FAILED
from harness.skill_manager import SkillManager
from harness.verifier import check_ast, verify_task_completion, full_verification

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent role system prompts
# ---------------------------------------------------------------------------
_AGENT_MD_PATH = Path(__file__).parent.parent / ".agent" / "AGENT.md"

ROLE_PROMPTS: dict[str, str] = {
    "Planner": textwrap.dedent("""
        You are the PLANNER agent.
        Your ONLY job is to decompose the user's goal into an ordered list of atomic tasks.
        Each task must be independently executable, have a clear acceptance criterion, and
        declare its expected output files.
        Output ONLY a valid JSON array conforming to the Planner Task List schema.
        Do NOT write prose. Do NOT add markdown code fences around the JSON.
    """).strip(),

    "Architect": textwrap.dedent("""
        You are the ARCHITECT agent.
        Your ONLY job is to design file structures, module interfaces, and data flows.
        You do NOT generate executable code.
        Output your design as structured JSON tool calls only.
        Do NOT self-mark any task as completed.
    """).strip(),

    "Executor": textwrap.dedent("""
        You are the EXECUTOR agent.
        Your ONLY job is to generate correct, minimal Python code to complete the current task.
        Rules:
        - Output pure Python code in a JSON tool call: {"tool": "run_python", "args": {"code": "..."}}
        - Do NOT use os.system, eval, exec, or subprocess with shell=True
        - All file paths must be relative to the project working directory
        - If the task requires a skill, reference only the tool calls defined in the skill's SKILL.md
        - You CANNOT mark the task as completed — the verifier does this
    """).strip(),

    "Critic": textwrap.dedent("""
        You are the CRITIC agent.
        Your ONLY job is to review task outputs against acceptance criteria.
        Output ONLY a JSON verification result: {"status": "pass|fail", "reason": "..."}
        Do NOT modify files. Do NOT generate code.
        If the output fails, provide a specific, actionable reason.
    """).strip(),
}


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
class Executor:
    """
    Orchestrates the ReAct loop across all tasks in plan.json.

    Responsibilities:
    - Load AGENT.md before each task (context flush)
    - Swap persona system prompt based on task.role
    - Reload plan.json from disk before each task (human steering support)
    - Execute LLM-generated Python code in isolated subprocesses
    - Pass results through the verifier
    - Trigger REPLAN on failure
    """

    def __init__(
        self,
        config: ConfigLoader,
        llm: LLMEngine,
        plan: PlanManager,
        skills: SkillManager,
        working_dir: Path | str | None = None,
        dry_run: bool = False,
    ) -> None:
        self._config = config
        self._llm = llm
        self._plan = plan
        self._skills = skills
        self._cwd = Path(working_dir) if working_dir else Path(__file__).parent.parent
        self._dry_run = dry_run
        self._iteration_count = 0

    # ------------------------------------------------------------------
    # Context window management
    # ------------------------------------------------------------------
    def _load_agent_md(self) -> str:
        """Load AGENT.md global instructions."""
        if _AGENT_MD_PATH.exists():
            return _AGENT_MD_PATH.read_text(encoding="utf-8")
        logger.warning("AGENT.md not found at %s", _AGENT_MD_PATH)
        return "# AGENT.md\n(not found — proceeding without global constraints)"

    def _flush_and_reload_context(self, task: dict[str, Any]) -> None:
        """
        Flush LLM context, then reload:
        1. AGENT.md global constraints
        2. Active role system prompt
        3. Current plan.json summary
        4. Matched skill body (if any)
        """
        self._llm.flush_context()

        agent_md = self._load_agent_md()
        role = task.get("role", "Executor")
        role_prompt = ROLE_PROMPTS.get(role, ROLE_PROMPTS["Executor"])

        plan_summary = (
            f"Current plan state:\n"
            f"{self._plan.summary()}\n\n"
            f"Active task:\n"
            f"  ID: {task['id']}\n"
            f"  Role: {role}\n"
            f"  Description: {task['description']}\n"
            f"  Output files expected: {task.get('output_files', [])}"
        )

        # Load relevant skills
        skill_context = ""
        matched = self._skills.match_skills_for_task(task["description"])
        if matched:
            skill_names = [s.name for s in matched]
            logger.info("Matched skills for task: %s", skill_names)
            skill_bodies = []
            for s in matched:
                try:
                    body = self._skills.load_skill_body(s.name)
                    skill_bodies.append(f"### Skill: {s.name}\n{body}")
                except Exception as exc:
                    logger.warning("Failed to load skill body '%s': %s", s.name, exc)
            skill_context = "\n\n".join(skill_bodies)

        system_prompt = "\n\n---\n\n".join(filter(None, [
            agent_md,
            role_prompt,
            plan_summary,
            skill_context if skill_context else None,
        ]))

        self._llm.set_system_prompt(system_prompt)
        logger.info(
            "Context flushed and reloaded for task %s | role=%s | skills=%s",
            task["id"],
            role,
            [s.name for s in matched],
        )

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------
    def _execute_python_code(
        self,
        code: str,
        task: dict[str, Any],
    ) -> tuple[int, str, str]:
        """
        Write code to a temp file and execute it in a subprocess.

        Returns (exit_code, stdout, stderr).
        """
        timeout = self._config.execution_timeout

        if self._dry_run:
            logger.info("[DRY RUN] Would execute:\n%s", code)
            return 0, "[DRY RUN] skipped", ""

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
            dir=self._cwd,
        ) as tmp:
            tmp.write(code)
            tmp_path = Path(tmp.name)

        try:
            result = subprocess.run(
                [sys.executable, str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self._cwd),
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", f"Execution timed out after {timeout}s"
        except Exception as exc:
            return 1, "", str(exc)
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # ReAct single-task loop
    # ------------------------------------------------------------------
    def _run_task(self, task: dict[str, Any]) -> bool:
        """
        Execute one task through the ReAct loop.
        Returns True if task passed verification.
        """
        task_id = task["id"]
        role = task.get("role", "Executor")
        max_iters = self._config.max_react_iterations
        ast_enabled = self._config.ast_safety_check

        self._plan.mark_in_progress(task_id)

        # --- Context flush & reload ---
        self._flush_and_reload_context(task)

        logger.info("Starting ReAct loop | task=%s | role=%s", task_id, role)

        for iteration in range(1, max_iters + 1):
            self._iteration_count += 1
            logger.info("[Iter %d/%d] task=%s", iteration, max_iters, task_id)

            try:
                # === THOUGHT ===
                thought_prompt = (
                    f"Task: {task['description']}\n"
                    f"Expected output files: {task.get('output_files', [])}\n\n"
                    f"Think step-by-step, then emit a tool call JSON."
                )
                response = self._llm.generate(thought_prompt)
                logger.debug("LLM response (iter %d):\n%s", iteration, response[:500])

                # === ACTION: parse tool call ===
                import json
                try:
                    parsed = json.loads(response)
                except json.JSONDecodeError:
                    # Try to extract JSON from markdown code block
                    import re
                    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(1))
                    else:
                        logger.warning("Could not parse tool call from response, retrying...")
                        continue

                tool = parsed.get("tool", "")
                args = parsed.get("args", {})

                # === OBSERVATION: handle tool ===
                if tool == "run_python":
                    code = args.get("code", "")

                    # AST safety gate
                    if ast_enabled:
                        safe, ast_msg = check_ast(code)
                        if not safe:
                            obs = f"[AST BLOCKED] {ast_msg}"
                            logger.warning(obs)
                            self._llm.add_user_message(
                                f"Your code was blocked by the AST safety checker:\n{ast_msg}\n"
                                "Rewrite the code without the blocked patterns."
                            )
                            continue

                    exit_code, stdout, stderr = self._execute_python_code(code, task)
                    obs = f"Exit code: {exit_code}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                    logger.info("Execution result:\n%s", obs[:1000])

                    # === Deterministic completion gate ===
                    passed, gate_msg = verify_task_completion(
                        exit_code,
                        task.get("output_files", []),
                        cwd=self._cwd,
                    )

                    if passed:
                        self._plan.mark_completed(task_id)
                        print(f"\n✅ Task COMPLETED: {task['description'][:80]}")
                        print(f"   Verification:\n{textwrap.indent(gate_msg, '   ')}")
                        return True
                    else:
                        # Feed observation back for retry
                        self._llm.add_user_message(
                            f"Code executed but verification failed:\n{gate_msg}\n"
                            f"STDERR: {stderr[:300]}\n"
                            "Fix the code and retry."
                        )
                        self._llm.add_assistant_message(obs)
                        continue

                elif tool == "done":
                    # LLM claims completion — BLOCKED, must go through verifier
                    logger.warning(
                        "LLM attempted self-completion (tool='done') — blocked by verifier gate."
                    )
                    self._llm.add_user_message(
                        "You cannot mark a task as done yourself. "
                        "Emit a 'run_python' tool call to produce the output files, "
                        "and the verifier will confirm completion."
                    )
                    continue

                else:
                    self._llm.add_user_message(
                        f"Unknown tool '{tool}'. Use 'run_python' with a 'code' argument."
                    )
                    continue

            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                logger.error("Unhandled error in ReAct iteration %d:\n%s", iteration, tb)
                self._llm.add_user_message(
                    f"An error occurred:\n{tb[:500]}\nContinue from where you left off."
                )

        # Max iterations exhausted
        logger.error("Max ReAct iterations reached for task %s", task_id)
        return False

    # ------------------------------------------------------------------
    # Main execution loop
    # ------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        """
        Execute all pending tasks in plan.json sequentially.
        Handles REPLAN on failure.

        Returns a summary dict.
        """
        print(f"\n{'='*60}")
        print(f"  🤖 AGENT HARNESS — Execution Starting")
        print(f"{'='*60}")
        print(f"  Working dir : {self._cwd}")
        print(f"  Plan path   : {self._plan._path}")
        print(f"  Dry run     : {self._dry_run}")
        print(f"{'='*60}\n")

        max_replan_cycles = 3
        replan_cycles = 0

        while not self._plan.is_complete():
            # === Live plan watcher: reload before each task ===
            self._plan.reload_from_disk()

            task = self._plan.get_next_task()
            if task is None:
                break

            print(f"\n📋 [{task['role']}] {task['description'][:100]}")
            print(f"   Task ID: {task['id']}")

            start_time = time.monotonic()
            success = self._run_task(task)
            elapsed = time.monotonic() - start_time

            if not success and replan_cycles < max_replan_cycles:
                replan_cycles += 1
                print(f"\n♻️  REPLAN cycle {replan_cycles}/{max_replan_cycles} for task {task['id']}")
                try:
                    # Ask LLM to decompose the failed task
                    self._flush_and_reload_context(task)
                    replan_prompt = (
                        f"The following task failed after {self._config.max_react_iterations} attempts:\n"
                        f"  Description: {task['description']}\n"
                        f"  Error: {task.get('error', 'unknown')}\n\n"
                        "Decompose it into 2 simpler sub-tasks. "
                        "Output ONLY a JSON array of task objects."
                    )
                    replan_response = self._llm.generate(replan_prompt)
                    import json
                    replacement = json.loads(replan_response)
                    if not isinstance(replacement, list):
                        replacement = None
                except Exception as exc:
                    logger.warning("REPLAN decomposition failed: %s", exc)
                    replacement = None

                error_info = traceback.format_exc()
                new_tasks = self._plan.trigger_replan(
                    task["id"],
                    exc_info=error_info,
                    replacement_tasks=replacement,
                )
                print(f"   Inserted {len(new_tasks)} replacement task(s)")
            elif not success:
                self._plan.mark_failed(task["id"], "Max REPLAN cycles exceeded")
                print(f"\n❌ Task permanently FAILED (REPLAN limit reached): {task['id']}")

            logger.info("Task %s finished in %.1fs | success=%s", task["id"], elapsed, success)

        summary = self._plan.summary()
        print(f"\n{'='*60}")
        print(f"  📊 Execution Complete")
        print(f"{'='*60}")
        print(f"  Total tasks : {summary['total']}")
        print(f"  Completed   : {summary['completed']}")
        print(f"  Failed      : {summary['failed']}")
        print(f"  REPLAN      : {summary['replan']}")
        print(f"  Pending     : {summary['pending']}")
        print(f"{'='*60}\n")
        return summary
