"""
cli.py — Interactive CLI (REPL) entrypoint for the Agent Harness.

Usage:
    python cli.py                          # interactive mode, uses config.yaml
    python cli.py --config my_config.yaml  # custom config
    python cli.py --mode ask               # override cloud router mode
    python cli.py --verbose                # debug logging

Inside the REPL, type a goal and press Enter.
Type /help for a list of commands.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

# Force UTF-8 output on Windows (prevents UnicodeEncodeError with emoji/box chars)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# ANSI colour helpers (works on Windows 10+ with ENABLE_VIRTUAL_TERMINAL)
# ---------------------------------------------------------------------------
try:
    import colorama
    colorama.init(autoreset=True)
    _HAS_COLOUR = True
except ImportError:
    _HAS_COLOUR = False


def _c(text: str, code: str) -> str:
    """Wrap *text* in an ANSI escape if colour is available."""
    if not _HAS_COLOUR:
        return text
    return f"\033[{code}m{text}\033[0m"


def cyan(t: str) -> str:   return _c(t, "96")
def green(t: str) -> str:  return _c(t, "92")
def yellow(t: str) -> str: return _c(t, "93")
def red(t: str) -> str:    return _c(t, "91")
def bold(t: str) -> str:   return _c(t, "1")
def dim(t: str) -> str:    return _c(t, "2")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,          # quiet by default in interactive mode
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("harness.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
BANNER = f"""
{cyan('╔══════════════════════════════════════════════════════════╗')}
{cyan('║')}   {bold('🤖  AGENT HARNESS')}  —  Interactive CLI  v1.0.0            {cyan('║')}
{cyan('╚══════════════════════════════════════════════════════════╝')}
  {dim('Type a goal and press Enter to run the harness.')}
  {dim('Type')} {yellow('/help')} {dim('for available commands, or')} {yellow('/exit')} {dim('to quit.')}
"""

HELP_TEXT = f"""
{bold('Available commands:')}
  {yellow('/help')}                   Show this help message
  {yellow('/status')}                 Show current plan status
  {yellow('/plan')}                   Print the current plan (plan.json)
  {yellow('/reset')}                  Clear plan.json and start fresh
  {yellow('/resume')}                 Re-run executor on the current plan (skip planning)
  {yellow('/config')}                 Print active configuration
  {yellow('/mode <mode>')}            Change cloud router mode (disabled | ask | allow_session)
  {yellow('/dryrun')}                 Toggle dry-run mode on/off
  {yellow('/verbose')}                Toggle verbose (DEBUG) logging on/off
  {yellow('/exit')} or {yellow('/quit')}          Exit the CLI

  {dim('Anything else is treated as a new goal.')}
"""


# ---------------------------------------------------------------------------
# Helper: pretty-print plan tasks
# ---------------------------------------------------------------------------
def _print_plan(plan_path: Path) -> None:
    if not plan_path.exists() or plan_path.stat().st_size == 0:
        print(yellow("  plan.json is empty."))
        return
    try:
        tasks: list[dict] = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(red(f"  Could not parse plan.json: {exc}"))
        return

    if not tasks:
        print(yellow("  No tasks in plan.json."))
        return

    status_colour = {
        "pending":     dim,
        "in_progress": cyan,
        "completed":   green,
        "failed":      red,
        "replan":      yellow,
    }
    print(bold(f"\n  Plan — {len(tasks)} task(s):"))
    for i, t in enumerate(tasks, 1):
        colour = status_colour.get(t.get("status", ""), dim)
        badge  = colour(f"[{t.get('status', '?'):11s}]")
        role   = dim(f"({t.get('role', '?'):9s})")
        desc   = t.get("description", "")[:72]
        print(f"  {i:2d}. {badge} {role} {desc}")
    print()


def _print_status(plan_path: Path) -> None:
    if not plan_path.exists() or plan_path.stat().st_size == 0:
        print(yellow("  No active plan."))
        return
    try:
        tasks: list[dict] = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception:
        print(red("  plan.json is corrupt."))
        return

    counts: dict[str, int] = {}
    for t in tasks:
        counts[t.get("status", "unknown")] = counts.get(t.get("status", "unknown"), 0) + 1

    parts = [f"{green('✔')} completed={counts.get('completed', 0)}",
             f"{red('✘')} failed={counts.get('failed', 0)}",
             f"{cyan('↻')} in_progress={counts.get('in_progress', 0)}",
             f"{dim('○')} pending={counts.get('pending', 0)}"]
    print("  Status: " + "  ".join(parts))


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
class CLISession:
    """Holds all warm-started harness objects across REPL iterations."""

    def __init__(self, cfg, llm, skills, plan_path: Path, dry_run: bool) -> None:
        self.cfg       = cfg
        self.llm       = llm
        self.skills    = skills
        self.plan_path = plan_path
        self.dry_run   = dry_run
        self.verbose   = False

    # ------------------------------------------------------------------
    # Run a goal (planning + execution)
    # ------------------------------------------------------------------
    def run_goal(self, goal: str) -> None:
        from harness.plan_manager import PlanManager
        from harness.llm_engine import PLANNER_GRAMMAR, _safe_json_loads
        from harness.executor import Executor

        plan = PlanManager(self.plan_path)

        # --- Planning phase ---
        print(cyan(f"\n  🧠  Planning: {goal[:70]}…\n"))
        if self.dry_run:
            tasks = [
                {
                    "id": str(uuid.uuid4()),
                    "description": f"[DRY RUN] Synthetic task 1 for: {goal[:40]}",
                    "role": "Executor",
                    "status": "pending",
                    "output_files": ["output_task1.py"],
                    "error": None,
                },
                {
                    "id": str(uuid.uuid4()),
                    "description": f"[DRY RUN] Synthetic task 2 for: {goal[:40]}",
                    "role": "Critic",
                    "status": "pending",
                    "output_files": [],
                    "error": None,
                },
            ]
        else:
            planner_prompt = (
                f"Decompose this goal into an ordered list of tasks:\n\n{goal}\n\n"
                "Output ONLY a valid JSON array of task objects. "
                "Each task must include: id (uuid), description, role (Executor), "
                "status (pending), output_files (list of expected filenames), error (null)."
            )
            agent_md = Path(__file__).parent / ".agent" / "AGENT.md"
            self.llm.set_system_prompt(agent_md.read_text(encoding="utf-8"))
            raw_plan = self.llm.generate(planner_prompt, grammar_str=PLANNER_GRAMMAR)
            tasks = _safe_json_loads(raw_plan)

        plan.create_plan(tasks)

        print(bold(f"  📋  {len(tasks)} task(s) planned:"))
        for t in tasks:
            print(f"     [{t['role']:9s}] {t['description'][:72]}")
        print()

        self._run_executor(plan)

    # ------------------------------------------------------------------
    # Resume (execution only, no re-planning)
    # ------------------------------------------------------------------
    def resume(self) -> None:
        from harness.plan_manager import PlanManager
        plan = PlanManager(self.plan_path)
        if not plan.tasks:
            print(yellow("  No tasks to resume. Run a goal first."))
            return
        print(cyan("  ↻  Resuming execution on existing plan…\n"))
        self._run_executor(plan)

    # ------------------------------------------------------------------
    def _run_executor(self, plan) -> None:
        from harness.executor import Executor
        executor = Executor(
            config=self.cfg,
            llm=self.llm,
            plan=plan,
            skills=self.skills,
            working_dir=Path(__file__).parent,
            dry_run=self.dry_run,
        )
        summary = executor.run()
        ok  = summary.get("completed", 0)
        bad = summary.get("failed", 0)
        pnd = summary.get("pending", 0)

        result_line = (
            f"  {green('✔')} {ok} completed  "
            f"{red('✘') if bad else dim('✘')} {bad} failed  "
            f"{dim('○')} {pnd} pending"
        )
        print(bold("\n  ── Run complete ──"))
        print(result_line)
        print()


# ---------------------------------------------------------------------------
# Main REPL loop
# ---------------------------------------------------------------------------
def repl(session: CLISession) -> None:
    print(BANNER)
    cfg_path = getattr(session.cfg, "_path", "config.yaml")
    print(f"  {dim('Config:')} {dim(str(Path(cfg_path).resolve()))}")
    print(f"  {dim('Model: ')} {dim(Path(session.cfg.model_path).name[:55])}")
    print(f"  {dim('Mode:  ')} {dim(session.cfg.cloud_router_mode)}")
    print(f"  {dim('Dry run:')} {dim(str(session.dry_run))}")
    print()

    while True:
        try:
            raw = input(f"{cyan('agent')} {bold('›')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{yellow('  Interrupted. Type /exit to quit.')}")
            continue

        if not raw:
            continue

        # ── built-in commands ────────────────────────────────────────
        if raw.lower() in ("/exit", "/quit", "exit", "quit"):
            print(green("  Goodbye! 👋"))
            break

        elif raw.lower() == "/help":
            print(HELP_TEXT)

        elif raw.lower() == "/status":
            _print_status(session.plan_path)

        elif raw.lower() == "/plan":
            _print_plan(session.plan_path)

        elif raw.lower() == "/reset":
            session.plan_path.write_text("[]", encoding="utf-8")
            print(green("  plan.json cleared."))

        elif raw.lower() == "/resume":
            session.resume()

        elif raw.lower() == "/config":
            cfg = session.cfg
            print(bold("\n  Active configuration:"))
            print(f"    model_path       : {cfg.model_path}")
            print(f"    n_gpu_layers     : {cfg.model_n_gpu_layers}")
            print(f"    n_ctx            : {cfg.model_n_ctx}")
            print(f"    temperature      : {cfg.model_temperature}")
            print(f"    max_tokens       : {cfg.model_max_tokens}")
            print(f"    cloud_router mode: {cfg.cloud_router_mode}")
            print(f"    ast_safety_check : {cfg.ast_safety_check}")
            print(f"    timeout_seconds  : {cfg.execution_timeout}")
            print(f"    max_react_iters  : {cfg.max_react_iterations}")
            print(f"    dry_run          : {session.dry_run}")
            print()

        elif raw.lower().startswith("/mode "):
            mode = raw.split(None, 1)[1].strip()
            valid = {"disabled", "ask", "allow_session"}
            if mode not in valid:
                print(red(f"  Invalid mode '{mode}'. Choose from: {', '.join(sorted(valid))}"))
            else:
                session.cfg._data.setdefault("cloud_router", {})["mode"] = mode
                print(green(f"  Cloud router mode set to: {mode}"))

        elif raw.lower() == "/dryrun":
            session.dry_run = not session.dry_run
            state = green("ON") if session.dry_run else red("OFF")
            print(f"  Dry-run mode: {state}")

        elif raw.lower() == "/verbose":
            session.verbose = not session.verbose
            level = logging.DEBUG if session.verbose else logging.WARNING
            logging.getLogger().setLevel(level)
            state = green("ON") if session.verbose else red("OFF")
            print(f"  Verbose logging: {state}")

        elif raw.startswith("/"):
            print(yellow(f"  Unknown command: {raw}  (type /help for help)"))

        # ── treat as a goal ─────────────────────────────────────────
        else:
            try:
                session.run_goal(raw)
            except KeyboardInterrupt:
                print(f"\n{yellow('  Execution interrupted. Plan state preserved.')}")
            except Exception as exc:
                logger.exception("Unexpected error during goal execution")
                print(red(f"\n  ✘ Error: {exc}"))
                print(dim("  (See harness.log for details. Type /status to check plan state.)"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-harness-cli",
        description="Interactive REPL for the offline-first Python Agent Harness.",
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml (default: config.yaml in project root).")
    parser.add_argument("--mode", choices=["disabled", "ask", "allow_session"], default=None,
                        help="Override cloud_router.mode from config.yaml.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan without executing any code.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging from the start.")
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # -- Config -----------------------------------------------------------
    from harness.config_loader import get_config, reset_config
    reset_config()
    cfg = get_config(args.config)

    if args.mode:
        cfg._data.setdefault("cloud_router", {})["mode"] = args.mode

    # -- LLM (warm-start once, reused across goals) -----------------------
    print(f"\n{cyan('  ⚙  Loading LLM…')} {dim('(this may take a moment)')}")
    from harness.llm_engine import create_engine_from_config
    llm = create_engine_from_config(cfg)

    if not args.dry_run:
        llm.initialize()
        diag = llm.diagnostics()
        logger.info("LLM diagnostics: %s", diag)
        print(green("  ✔  LLM ready."))
    else:
        print(yellow("  [DRY RUN] LLM initialization skipped."))

    # -- Skills -----------------------------------------------------------
    from harness.skill_manager import SkillManager
    skills = SkillManager()
    indexed = skills.list_skills()
    print(f"  📚 {len(indexed)} skill(s) indexed.")

    # -- Plan path --------------------------------------------------------
    plan_path = Path(__file__).parent / "plan.json"
    if not plan_path.exists():
        plan_path.write_text("[]", encoding="utf-8")

    # -- Session & REPL ---------------------------------------------------
    session = CLISession(
        cfg=cfg,
        llm=llm,
        skills=skills,
        plan_path=plan_path,
        dry_run=args.dry_run,
    )
    session.verbose = args.verbose

    repl(session)
    return 0


if __name__ == "__main__":
    sys.exit(main())
