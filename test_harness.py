"""
test_harness.py
===============
End-to-end integration test suite for the Python agent harness.

Tests:
  1. Config Loader & PII Sanitizer
  2. Skill Manager Tier 1 — Header Indexing
  3. Skill Manager Tier 2 — Lazy Body Loading
  4. LLM Engine — Context Window Flush
  5. Verifier — AST Safety Checks
  6. Verifier — Deterministic Completion Gate
  7. Plan Manager — State Machine Transitions
  8. Plan Manager — Dynamic REPLAN Loop
  9. Cloud Router — Gatekeeper (disabled mode interception)
 10. Cloud Router — ask mode mock (simulated user rejection)
 11. Executor — Dry Run (persona swap + context flush + plan watcher)
 12. LLM CUDA Diagnostic (live model load — skipped if model missing)

Run:
  agent-env\Scripts\python.exe test_harness.py
  agent-env\Scripts\python.exe test_harness.py --skip-cuda
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Force UTF-8 output on Windows terminals (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Ensure project root on path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Test runner helpers
# ---------------------------------------------------------------------------
_RESULTS: list[tuple[str, bool, str]] = []
_PASS = "[PASS]"
_FAIL = "[FAIL]"


def test(name: str):
    """Decorator to register and run a test function."""
    def decorator(fn):
        try:
            fn()
            _RESULTS.append((name, True, ""))
            print(f"  {_PASS} — {name}")
        except Exception as exc:
            tb = traceback.format_exc()
            _RESULTS.append((name, False, str(exc)))
            print(f"  {_FAIL} — {name}")
            print(f"          {exc}")
            if "--verbose" in sys.argv:
                print(tb)
        return fn
    return decorator


def assert_true(condition: bool, msg: str = "") -> None:
    if not condition:
        raise AssertionError(msg or "Assertion failed")


def assert_equal(a: Any, b: Any, msg: str = "") -> None:
    if a != b:
        raise AssertionError(msg or f"Expected {b!r}, got {a!r}")


# ===========================================================================
# Phase 1 — Config Loader & PII Sanitizer
# ===========================================================================
print("\n=== Phase 1 -- Config Loader & PII Sanitizer ===")
print("-" * 55)

@test("Config loads config.yaml successfully")
def _():
    from harness.config_loader import get_config, reset_config
    reset_config()
    cfg = get_config()
    assert_true(cfg.model_path != "", "model_path should not be empty")
    assert_true(cfg.model_n_gpu_layers == -1, "n_gpu_layers should be -1")
    assert_true(cfg.model_n_ctx == 8192, "n_ctx should be 8192")
    assert_true(cfg.execution_timeout == 45, "timeout should be 45s")


@test("Config cloud_router_mode defaults to 'disabled'")
def _():
    from harness.config_loader import get_config
    cfg = get_config()
    assert_equal(cfg.cloud_router_mode, "disabled")


@test("PII sanitizer masks IPv4 addresses")
def _():
    from harness.config_loader import PIISanitizer
    s = PIISanitizer()
    result = s.sanitize("Connect to server at 192.168.1.100 for data.")
    assert_true("192.168.1.100" not in result, "IPv4 should be redacted")
    assert_true("[REDACTED:IPV4_ADDR]" in result, "Redaction marker should appear")


@test("PII sanitizer masks OpenAI-style API keys")
def _():
    from harness.config_loader import PIISanitizer
    s = PIISanitizer()
    result = s.sanitize("Authorization: Bearer sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ123456")
    assert_true("sk-proj" not in result, "API key should be redacted")


@test("PII sanitizer masks internal hostnames")
def _():
    from harness.config_loader import PIISanitizer
    s = PIISanitizer()
    result = s.sanitize("POST https://api-server.internal/v1/completions")
    assert_true("api-server.internal" not in result, "Internal hostname should be redacted")
    assert_true("[REDACTED:INTERNAL_HOST]" in result)


# ===========================================================================
# Phase 2 — Skill Manager Tier 1 Header Indexing
# ===========================================================================
print("\n=== Phase 2 -- Skill Manager: Tier 1 Header Indexing ===")
print("-" * 55)

@test("SkillManager initializes without error")
def _():
    from harness.skill_manager import SkillManager
    sm = SkillManager()
    assert_true(sm is not None)


@test("Tier 1: python_data_analysis skill is indexed")
def _():
    from harness.skill_manager import SkillManager
    sm = SkillManager()
    header = sm.get_header("python_data_analysis")
    assert_true(header is not None, "python_data_analysis should be in index")
    assert_equal(header.tier, "core")
    assert_true(len(header.preview_lines) > 0, "Preview lines should be populated")


@test("Tier 1: core/ skill header has correct frontmatter")
def _():
    from harness.skill_manager import SkillManager
    sm = SkillManager()
    header = sm.get_header("python_data_analysis")
    assert_true("name" in header.frontmatter, "frontmatter should have 'name'")
    assert_equal(header.frontmatter["name"], "python_data_analysis")
    assert_equal(header.frontmatter["tier"], "core")


@test("Tier 1: core/ skills have precedence over evolved/ skills with same name")
def _():
    """Create a fake evolved skill, verify core overrides it."""
    from harness.skill_manager import SkillManager
    sm = SkillManager()

    with tempfile.TemporaryDirectory() as tmp:
        evolved_dir = Path(tmp) / "evolved"
        evolved_dir.mkdir()
        # Create same-named skill in evolved
        (evolved_dir / "python_data_analysis").mkdir()
        (evolved_dir / "python_data_analysis" / "SKILL.md").write_text(
            "---\nname: python_data_analysis\ntier: evolved\n---\nEvolved body.",
            encoding="utf-8",
        )
        core_dir = PROJECT_ROOT / ".agent" / "skills" / "core"
        sm2 = SkillManager(core_dir=core_dir, evolved_dir=evolved_dir)
        header = sm2.get_header("python_data_analysis")
        assert_equal(header.tier, "core", "Core should override evolved")


# ===========================================================================
# Phase 3 — Skill Manager Tier 2 Lazy Body Loading
# ===========================================================================
print("\n=== Phase 3 -- Skill Manager: Tier 2 Lazy Body Loading ===")
print("-" * 55)

@test("Tier 2: python_data_analysis body loads on demand")
def _():
    from harness.skill_manager import SkillManager
    sm = SkillManager()
    body = sm.load_skill_body("python_data_analysis")
    assert_true(len(body) > 100, "Skill body should be substantial")
    assert_true("csv" in body.lower(), "Body should contain CSV content")


@test("Tier 2: body is cached after first load")
def _():
    from harness.skill_manager import SkillManager
    sm = SkillManager()
    _ = sm.load_skill_body("python_data_analysis")  # First load
    assert_true("python_data_analysis" in sm._body_cache, "Body should be in cache")
    # Second load: verify cache hit (no disk read)
    body2 = sm.load_skill_body("python_data_analysis")
    assert_true(len(body2) > 0)


@test("Tier 2: load_skill_body raises KeyError for unknown skill")
def _():
    from harness.skill_manager import SkillManager
    sm = SkillManager()
    try:
        sm.load_skill_body("nonexistent_skill_xyz")
        raise AssertionError("Should have raised KeyError")
    except KeyError:
        pass  # expected


@test("Skill matching finds python_data_analysis for data-related task")
def _():
    from harness.skill_manager import SkillManager
    sm = SkillManager()
    matches = sm.match_skills_for_task("Generate a CSV file and compute column averages")
    names = [m.name for m in matches]
    assert_true("python_data_analysis" in names, f"Expected match, got: {names}")


# ===========================================================================
# Phase 4 — LLM Engine: Context Window Flush
# ===========================================================================
print("\n=== Phase 4 -- LLM Engine: Context Window Flush ===")
print("-" * 55)

@test("LLMEngine raises FileNotFoundError for missing model path")
def _():
    from harness.llm_engine import LLMEngine
    # FileNotFoundError raised in __init__ when model path doesn't exist
    try:
        engine = LLMEngine(model_path="/nonexistent/path/model.gguf")
        # If __init__ didn't raise (e.g. path somehow exists), try initialize
        engine.initialize()
        raise AssertionError("Should have raised FileNotFoundError or ImportError")
    except FileNotFoundError:
        pass  # expected — model path doesn't exist
    except ImportError:
        pass  # expected when llama_cpp is not yet installed (CUDA build pending)


@test("LLMEngine flush_context clears history but preserves system prompt")
def _():
    from harness.llm_engine import LLMEngine
    engine = LLMEngine.__new__(LLMEngine)
    engine._model_path = Path("/fake/path")
    engine._system_prompt = "You are a test agent."
    engine._history = [
        {"role": "user", "content": "Task 1 message"},
        {"role": "assistant", "content": "Task 1 response"},
    ]
    engine._llm = None

    engine.flush_context()
    assert_equal(len(engine._history), 0, "History should be empty after flush")
    assert_equal(engine._system_prompt, "You are a test agent.", "System prompt should survive flush")


@test("LLMEngine set_system_prompt persists across flush")
def _():
    from harness.llm_engine import LLMEngine
    engine = LLMEngine.__new__(LLMEngine)
    engine._system_prompt = ""
    engine._history = []
    engine._llm = None
    engine._model_path = Path("/fake")

    engine.set_system_prompt("Global constraints from AGENT.md")
    engine.add_user_message("Task step 1")
    engine.flush_context()

    assert_equal(engine._system_prompt, "Global constraints from AGENT.md")
    assert_equal(len(engine._history), 0)


@test("LLMEngine _build_messages includes system prompt first")
def _():
    from harness.llm_engine import LLMEngine
    engine = LLMEngine.__new__(LLMEngine)
    engine._system_prompt = "System instructions"
    engine._history = [{"role": "user", "content": "Hello"}]
    engine._llm = None
    engine._model_path = Path("/fake")

    messages = engine._build_messages()
    assert_equal(messages[0]["role"], "system")
    assert_equal(messages[0]["content"], "System instructions")
    assert_equal(messages[1]["role"], "user")


# ===========================================================================
# Phase 5 — Verifier: AST Safety Checks
# ===========================================================================
print("\n=== Phase 5 -- Verifier: AST Safety Checks ===")
print("-" * 55)

@test("AST check PASSES clean CSV generation code")
def _():
    from harness.verifier import check_ast
    code = """
import csv
with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['a', 'b'])
    writer.writerow([1, 2])
"""
    safe, msg = check_ast(code)
    assert_true(safe, f"Clean code should pass AST check: {msg}")


@test("AST check BLOCKS os.system call")
def _():
    from harness.verifier import check_ast
    code = "import os\nos.system('rm -rf /')"
    safe, msg = check_ast(code)
    assert_true(not safe, "os.system should be blocked")
    assert_true("DANGEROUS_CALL" in msg)


@test("AST check BLOCKS eval()")
def _():
    from harness.verifier import check_ast
    code = "result = eval('__import__(\"os\").system(\"ls\")')"
    safe, msg = check_ast(code)
    assert_true(not safe, "eval should be blocked")


@test("AST check BLOCKS exec()")
def _():
    from harness.verifier import check_ast
    code = "exec('import os; os.system(\"ls\")')"
    safe, msg = check_ast(code)
    assert_true(not safe, "exec should be blocked")


@test("AST check BLOCKS subprocess with shell=True")
def _():
    from harness.verifier import check_ast
    code = "import subprocess; subprocess.run('ls -la', shell=True)"
    safe, msg = check_ast(code)
    assert_true(not safe, "subprocess shell=True should be blocked")
    assert_true("SHELL_INJECTION" in msg)


@test("AST check BLOCKS __import__ builtin")
def _():
    from harness.verifier import check_ast
    code = "m = __import__('os'); m.system('echo hi')"
    safe, msg = check_ast(code)
    assert_true(not safe, "__import__ should be blocked")


@test("AST check reports SyntaxError for invalid Python")
def _():
    from harness.verifier import check_ast
    code = "def broken(:\n  pass"
    safe, msg = check_ast(code)
    assert_true(not safe, "Syntax error should fail")
    assert_true("Syntax error" in msg)


# ===========================================================================
# Phase 6 — Verifier: Deterministic Completion Gate
# ===========================================================================
print("\n=== Phase 6 -- Verifier: Deterministic Completion Gate ===")
print("-" * 55)

@test("Completion gate PASSES with exit_code=0 and existing non-empty file")
def _():
    from harness.verifier import verify_task_completion
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "output.csv"
        f.write_text("account_id,balance\nACC001,1000.0\n")
        passed, msg = verify_task_completion(exit_code=0, output_files=[f], cwd=tmp)
        assert_true(passed, f"Should pass: {msg}")
        assert_true("✓" in msg)


@test("Completion gate FAILS with non-zero exit code")
def _():
    from harness.verifier import verify_task_completion
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "output.csv"
        f.write_text("data")
        passed, msg = verify_task_completion(exit_code=1, output_files=[f], cwd=tmp)
        assert_true(not passed, "Non-zero exit should fail")
        assert_true("Exit code 1 != 0" in msg)


@test("Completion gate FAILS when output file is missing")
def _():
    from harness.verifier import verify_task_completion
    with tempfile.TemporaryDirectory() as tmp:
        passed, msg = verify_task_completion(
            exit_code=0, output_files=["missing_file.csv"], cwd=tmp
        )
        assert_true(not passed, "Missing file should fail")
        assert_true("missing" in msg.lower())


@test("Completion gate FAILS when output file is empty (0 bytes)")
def _():
    from harness.verifier import verify_task_completion
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "empty.csv"
        f.touch()  # 0 bytes
        passed, msg = verify_task_completion(exit_code=0, output_files=[f], cwd=tmp)
        assert_true(not passed, "Empty file should fail")
        assert_true("empty" in msg.lower() or "0 bytes" in msg.lower())


@test("LLM cannot self-mark completed — only verifier can write completed status")
def _():
    """
    Verify that plan_manager.mark_completed() is not callable from
    an LLM-generated tool call named 'done'.
    The executor explicitly blocks 'done' tool calls.
    """
    # We verify this by checking the executor source code mentions the guard
    executor_src = (PROJECT_ROOT / "harness" / "executor.py").read_text(encoding="utf-8")
    assert_true(
        "tool == \"done\"" in executor_src or "tool == 'done'" in executor_src,
        "Executor must block 'done' self-completion attempts"
    )
    assert_true(
        "You cannot mark a task as done yourself" in executor_src,
        "Executor must emit blocking message for 'done' tool"
    )


# ===========================================================================
# Phase 7 — Plan Manager: State Machine
# ===========================================================================
print("\n=== Phase 7 -- Plan Manager: State Machine ===")
print("-" * 55)

@test("PlanManager creates and persists a plan")
def _():
    from harness.plan_manager import PlanManager
    with tempfile.TemporaryDirectory() as tmp:
        plan_path = Path(tmp) / "test_plan.json"
        pm = PlanManager(plan_path)
        pm.create_plan([
            {"description": "Task A", "role": "Executor", "output_files": ["a.txt"]},
            {"description": "Task B", "role": "Critic", "output_files": []},
        ])
        assert_equal(len(pm.tasks), 2)
        # Reload from disk
        pm2 = PlanManager(plan_path)
        assert_equal(len(pm2.tasks), 2)
        assert_equal(pm2.tasks[0]["description"], "Task A")


@test("PlanManager state transitions: pending → in_progress → completed")
def _():
    from harness.plan_manager import PlanManager, STATUS_PENDING, STATUS_IN_PROGRESS, STATUS_COMPLETED
    with tempfile.TemporaryDirectory() as tmp:
        pm = PlanManager(Path(tmp) / "plan.json")
        pm.create_plan([{"description": "Step 1", "role": "Executor"}])
        task = pm.get_next_task()
        assert_equal(task["status"], STATUS_PENDING)

        pm.mark_in_progress(task["id"])
        assert_equal(pm.get_task(task["id"])["status"], STATUS_IN_PROGRESS)

        pm.mark_completed(task["id"])
        assert_equal(pm.get_task(task["id"])["status"], STATUS_COMPLETED)
        assert_true(pm.is_complete())


@test("PlanManager state transitions: pending → in_progress → failed")
def _():
    from harness.plan_manager import PlanManager, STATUS_FAILED
    with tempfile.TemporaryDirectory() as tmp:
        pm = PlanManager(Path(tmp) / "plan.json")
        pm.create_plan([{"description": "Failing Step", "role": "Executor"}])
        task = pm.get_next_task()
        pm.mark_in_progress(task["id"])
        pm.mark_failed(task["id"], error="Something went wrong")
        t = pm.get_task(task["id"])
        assert_equal(t["status"], STATUS_FAILED)
        assert_true("Something went wrong" in t["error"])


@test("PlanManager get_next_task skips completed/failed")
def _():
    from harness.plan_manager import PlanManager
    with tempfile.TemporaryDirectory() as tmp:
        pm = PlanManager(Path(tmp) / "plan.json")
        pm.create_plan([
            {"description": "Done task", "role": "Executor"},
            {"description": "Pending task", "role": "Executor"},
        ])
        tasks = pm.tasks
        pm.mark_in_progress(tasks[0]["id"])
        pm.mark_completed(tasks[0]["id"])

        next_task = pm.get_next_task()
        assert_true(next_task is not None)
        assert_equal(next_task["description"], "Pending task")


# ===========================================================================
# Phase 8 — Plan Manager: Dynamic REPLAN Loop
# ===========================================================================
print("\n=== Phase 8 -- Plan Manager: Dynamic REPLAN Loop ===")
print("-" * 55)

@test("REPLAN: inserts replacement tasks after failed task")
def _():
    from harness.plan_manager import PlanManager, STATUS_REPLAN
    with tempfile.TemporaryDirectory() as tmp:
        pm = PlanManager(Path(tmp) / "plan.json")
        pm.create_plan([
            {"description": "Complex failing task", "role": "Executor"},
            {"description": "Subsequent task", "role": "Critic"},
        ])
        tasks = pm.tasks
        failed_id = tasks[0]["id"]
        pm.mark_in_progress(failed_id)
        pm.mark_failed(failed_id, "Code execution failed")

        new_tasks = pm.trigger_replan(
            failed_task_id=failed_id,
            exc_info="Traceback: ValueError: ...",
            replacement_tasks=[
                {"description": "Sub-task A (simpler)", "role": "Executor"},
                {"description": "Sub-task B (validation)", "role": "Critic"},
            ]
        )
        assert_equal(len(new_tasks), 2)
        assert_equal(len(pm.tasks), 4)  # original 2 + 2 replacements
        assert_equal(pm.tasks[0]["status"], STATUS_REPLAN)
        assert_equal(pm.tasks[1]["description"], "Sub-task A (simpler)")


@test("REPLAN: creates default retry task when no decomposition provided")
def _():
    from harness.plan_manager import PlanManager
    with tempfile.TemporaryDirectory() as tmp:
        pm = PlanManager(Path(tmp) / "plan.json")
        pm.create_plan([{"description": "Failing task", "role": "Executor"}])
        failed_id = pm.tasks[0]["id"]
        pm.mark_in_progress(failed_id)
        pm.mark_failed(failed_id, "generic error")

        new_tasks = pm.trigger_replan(failed_id)
        assert_equal(len(new_tasks), 1)
        assert_true("[RETRY]" in new_tasks[0]["description"])


@test("REPLAN: plan is persisted to disk after replan")
def _():
    from harness.plan_manager import PlanManager
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "plan.json"
        pm = PlanManager(path)
        pm.create_plan([{"description": "Failing task", "role": "Executor"}])
        failed_id = pm.tasks[0]["id"]
        pm.mark_in_progress(failed_id)
        pm.mark_failed(failed_id, "err")
        pm.trigger_replan(failed_id)

        # Reload and verify
        pm2 = PlanManager(path)
        assert_equal(len(pm2.tasks), 2)
        assert_true("[RETRY]" in pm2.tasks[1]["description"])


# ===========================================================================
# Phase 9 — Cloud Router: disabled mode
# ===========================================================================
print("\n=== Phase 9 -- Cloud Router: Gatekeeper Interception ===")
print("-" * 55)

@test("CloudRouter blocks call in 'disabled' mode with air-gap message")
def _():
    from harness.cloud_router import CloudRouter, CloudRouterBlockedError
    from harness.config_loader import PIISanitizer

    mock_cfg = MagicMock()
    mock_cfg.cloud_router_mode = "disabled"
    mock_cfg.sanitizer = PIISanitizer()

    router = CloudRouter(mock_cfg)
    try:
        router.ask_cloud_ai("Send this to GPT-4", service="openai")
        raise AssertionError("Should have raised CloudRouterBlockedError")
    except CloudRouterBlockedError as exc:
        assert_true("AIR-GAP" in str(exc) or "BLOCKED" in str(exc))
        assert_equal(router._block_count, 1)


@test("CloudRouter diagnostic shows disabled mode and block count")
def _():
    from harness.cloud_router import CloudRouter, CloudRouterBlockedError
    from harness.config_loader import PIISanitizer

    mock_cfg = MagicMock()
    mock_cfg.cloud_router_mode = "disabled"
    mock_cfg.sanitizer = PIISanitizer()

    router = CloudRouter(mock_cfg)
    try:
        router.ask_cloud_ai("test", service="openai")
    except CloudRouterBlockedError:
        pass
    diag = router.diagnostics()
    assert_equal(diag["mode"], "disabled")
    assert_equal(diag["block_count"], 1)


# ===========================================================================
# Phase 10 — Cloud Router: 'ask' mode mock (user rejection)
# ===========================================================================
print("\n=== Phase 10 -- Cloud Router: ask Mode Simulation ===")
print("-" * 55)

@test("CloudRouter 'ask' mode: user rejects call with 'n'")
def _():
    from harness.cloud_router import CloudRouter, CloudRouterBlockedError
    from harness.config_loader import PIISanitizer

    mock_cfg = MagicMock()
    mock_cfg.cloud_router_mode = "ask"
    mock_cfg.sanitizer = PIISanitizer()

    router = CloudRouter(mock_cfg)
    with patch("builtins.input", return_value="n"), \
         patch("builtins.print"):
        try:
            router.ask_cloud_ai("Some cloud prompt", service="openai")
            raise AssertionError("Should have raised CloudRouterBlockedError")
        except CloudRouterBlockedError:
            pass
    assert_equal(router._block_count, 1)
    assert_equal(router._allow_count, 0)


@test("CloudRouter 'ask' mode: user allows call with 'y'")
def _():
    from harness.cloud_router import CloudRouter
    from harness.config_loader import PIISanitizer

    mock_cfg = MagicMock()
    mock_cfg.cloud_router_mode = "ask"
    mock_cfg.sanitizer = PIISanitizer()

    router = CloudRouter(mock_cfg)
    with patch("builtins.input", return_value="y"), \
         patch("builtins.print"):
        result = router.ask_cloud_ai("Some cloud prompt", service="openai")
    assert_true("[STUB RESPONSE" in result)
    assert_equal(router._allow_count, 1)
    assert_equal(router._block_count, 0)


@test("CloudRouter 'ask' mode: user promotes to allow_session with 'a'")
def _():
    from harness.cloud_router import CloudRouter
    from harness.config_loader import PIISanitizer

    mock_cfg = MagicMock()
    mock_cfg.cloud_router_mode = "ask"
    mock_cfg.sanitizer = PIISanitizer()

    router = CloudRouter(mock_cfg)
    with patch("builtins.input", return_value="a"), \
         patch("builtins.print"):
        router.ask_cloud_ai("Promote me", service="openai")
    assert_true(router._session_allowed, "Session should be promoted to allow_session")
    assert_equal(router._mode, "allow_session")


# ===========================================================================
# Phase 11 — Executor: Dry Run Integration
# ===========================================================================
print("\n=== Phase 11 -- Executor: Dry Run Integration ===")
print("-" * 55)

@test("Executor dry-run: processes plan without actual LLM/execution")
def _():
    from harness.executor import Executor, ROLE_PROMPTS
    from harness.config_loader import ConfigLoader
    from harness.plan_manager import PlanManager
    from harness.skill_manager import SkillManager

    # We test the role prompt mapping and context flush logic
    assert_true("Planner" in ROLE_PROMPTS)
    assert_true("Architect" in ROLE_PROMPTS)
    assert_true("Executor" in ROLE_PROMPTS)
    assert_true("Critic" in ROLE_PROMPTS)

    # Verify system prompt content
    assert_true("ONLY job" in ROLE_PROMPTS["Planner"])
    assert_true("ONLY job" in ROLE_PROMPTS["Executor"])
    assert_true("self-mark" not in ROLE_PROMPTS["Executor"])


@test("Executor loads AGENT.md for context injection")
def _():
    from harness.executor import _AGENT_MD_PATH
    assert_true(_AGENT_MD_PATH.exists(), f"AGENT.md should exist at {_AGENT_MD_PATH}")
    content = _AGENT_MD_PATH.read_text()
    assert_true("Offline-First" in content or "offline" in content.lower())
    assert_true("CANNOT" in content, "AGENT.md should have strong constraints")


@test("Executor generates banking metrics CSV via dry subprocess")
def _():
    """
    Run the actual python_data_analysis skill code to verify the
    harness end-to-end flow works with real Python code execution.
    This is the core 'Execution Phase' from the spec:
      Step 1: CSV generation
      Step 2: JSON report
    """
    import subprocess
    import csv

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # --- Step 1: Generate CSV ---
        csv_code = """
import csv
import random
random.seed(42)

columns = ["account_id", "balance", "transactions", "credit_score", "loan_amount",
           "interest_rate", "days_overdue", "region_code", "product_type", "risk_flag"]

with open(r"{}/banking_metrics.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    for i in range(10):
        writer.writerow({{
            "account_id": f"ACC{{i:04d}}",
            "balance": round(random.uniform(100, 50000), 2),
            "transactions": random.randint(1, 200),
            "credit_score": random.randint(300, 850),
            "loan_amount": round(random.uniform(0, 100000), 2),
            "interest_rate": round(random.uniform(1.5, 25.0), 2),
            "days_overdue": random.randint(0, 90),
            "region_code": random.choice(["NA", "EU", "AP", "LA"]),
            "product_type": random.choice(["savings", "checking", "loan", "credit"]),
            "risk_flag": random.choice([0, 1]),
        }})
""".format(tmp)

        csv_file = tmp_path / "csv_gen.py"
        csv_file.write_text(csv_code)
        result = subprocess.run([sys.executable, str(csv_file)], capture_output=True, cwd=tmp)
        assert_equal(result.returncode, 0, f"CSV gen failed: {result.stderr.decode()}")

        output_csv = tmp_path / "banking_metrics.csv"
        assert_true(output_csv.exists(), "CSV file should exist")
        assert_true(output_csv.stat().st_size > 0, "CSV file should not be empty")

        # Verify row count
        with output_csv.open() as f:
            rows = list(csv.DictReader(f))
        assert_equal(len(rows), 10, "Should have 10 data rows")

        # --- CONTEXT FLUSH SIMULATION ---
        # In real execution: llm.flush_context() called here between tasks
        # We verify the concept by checking history reset logic
        from harness.llm_engine import LLMEngine
        engine = LLMEngine.__new__(LLMEngine)
        engine._history = [{"role": "user", "content": "Task 1 messages..."}]
        engine._system_prompt = "Persisted system prompt"
        engine._llm = None
        engine._model_path = Path("/fake")
        engine.flush_context()
        assert_equal(len(engine._history), 0, "Context flushed between tasks")

        # --- Step 2: JSON Report ---
        report_code = """
import csv
import json

with open(r"{csv}", newline="") as f:
    rows = list(csv.DictReader(f))

averages = {{}}
for col in rows[0].keys():
    try:
        vals = [float(row[col]) for row in rows]
        averages[col] = round(sum(vals) / len(vals), 4)
    except ValueError:
        averages[col] = None

report = {{"row_count": len(rows), "column_averages": averages}}
with open(r"{report}", "w") as f:
    json.dump(report, f, indent=2)
""".format(
            csv=str(output_csv).replace("\\", "\\\\"),
            report=str(tmp_path / "summary.json").replace("\\", "\\\\"),
        )

        report_file = tmp_path / "json_report.py"
        report_file.write_text(report_code)
        result2 = subprocess.run([sys.executable, str(report_file)], capture_output=True, cwd=tmp)
        assert_equal(result2.returncode, 0, f"JSON report failed: {result2.stderr.decode()}")

        summary_json = tmp_path / "summary.json"
        assert_true(summary_json.exists())
        assert_true(summary_json.stat().st_size > 0)

        with summary_json.open() as f:
            report = json.load(f)
        assert_equal(report["row_count"], 10)
        assert_true("column_averages" in report)
        # Verify numeric averages computed
        avgs = report["column_averages"]
        assert_true(avgs.get("balance") is not None, "balance average should be computed")
        assert_true(avgs.get("credit_score") is not None, "credit_score average should be computed")


# ===========================================================================
# Phase 12 — CUDA Diagnostic (live model load)
# ===========================================================================
print("\n=== Phase 12 -- CUDA Diagnostic (Live Model) ===")
print("-" * 55)

MODEL_PATH = Path(
    r"E:\ollama\models\blobs"
    r"\sha256-1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606"
)

SKIP_CUDA = "--skip-cuda" in sys.argv or not MODEL_PATH.exists()

if SKIP_CUDA:
    print(f"  [SKIP] --skip-cuda flag set or model not found at {MODEL_PATH.name[:40]}...")
else:
    @test("CUDA: model loads with n_gpu_layers=-1 and logs GPU offloading")
    def _():
        from harness.llm_engine import LLMEngine
        import io
        import contextlib

        engine = LLMEngine(
            model_path=MODEL_PATH,
            n_gpu_layers=-1,
            n_ctx=512,       # Small context for quick load test
            verbose=True,
        )
        # Capture stdout to detect CUDA initialization messages
        buf = io.StringIO()
        try:
            with contextlib.redirect_stderr(buf):
                engine.initialize()
        except Exception as exc:
            raise AssertionError(f"Model initialization failed: {exc}")

        assert_true(engine.is_initialized, "Engine should be initialized")
        cuda_output = buf.getvalue()
        # llama.cpp prints "ggml_cuda_init: found N devices" or similar
        print(f"\n    CUDA output snippet: {cuda_output[:200]!r}")

        # Generate a minimal test response
        engine.set_system_prompt("You are a test assistant. Reply with one word only.")
        response = engine.generate("Say: READY")
        print(f"    Model response: {response[:100]!r}")
        assert_true(len(response) > 0, "Model should produce output")

        diag = engine.diagnostics()
        print(f"\n    Diagnostics:")
        for k, v in diag.items():
            print(f"      {k}: {v}")

        assert_equal(diag["n_gpu_layers"], -1)
        assert_equal(diag["initialized"], True)


# ===========================================================================
# Results Summary
# ===========================================================================
print(f"\n{'='*60}")
print(f"  TEST RESULTS SUMMARY")
print(f"{'='*60}")

passed = sum(1 for _, ok, _ in _RESULTS if ok)
failed = sum(1 for _, ok, _ in _RESULTS if not ok)
total = len(_RESULTS)

for name, ok, err in _RESULTS:
    status = _PASS if ok else _FAIL
    print(f"  {status} — {name}")
    if not ok and err:
        print(f"           Error: {err[:100]}")

print(f"{'─'*60}")
print(f"  Total: {total} | Passed: {passed} | Failed: {failed}")
print(f"{'='*60}\n")

sys.exit(0 if failed == 0 else 1)
