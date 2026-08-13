"""
harness/main.py — CLI entrypoint & live execution orchestration.
(Also importable as main.py at project root.)

Usage:
  python main.py --goal "Your goal here"
  python main.py --goal "..." --plan plan.json --mode ask
  python main.py --plan plan.json          # resume from existing plan
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Force UTF-8 output on Windows (prevents UnicodeEncodeError with emoji/box chars)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Logging setup (must happen before any harness imports)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("harness.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-harness",
        description="Offline-first Python agent harness (llama-cpp-python + CUDA)",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Natural language goal. The Planner will decompose this into tasks.",
    )
    parser.add_argument(
        "--plan",
        type=str,
        default=None,
        help="Path to plan.json (default: plan.json in project root). "
             "If provided without --goal, resumes an existing plan.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: config.yaml in project root).",
    )
    parser.add_argument(
        "--mode",
        choices=["disabled", "ask", "allow_session"],
        default=None,
        help="Override cloud_router.mode from config.yaml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and plan without executing any code.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    from harness.config_loader import get_config, reset_config
    reset_config()
    cfg = get_config(args.config)

    if args.mode:
        # Runtime override of cloud router mode
        cfg._data.setdefault("cloud_router", {})["mode"] = args.mode
        logger.info("Cloud router mode overridden via CLI: %s", args.mode)

    print(f"\n{'═'*60}")
    print(f"  🤖  AGENT HARNESS v1.0.0")
    print(f"{'═'*60}")
    print(f"  Model         : {Path(cfg.model_path).name[:50]}")
    print(f"  n_gpu_layers  : {cfg.model_n_gpu_layers}")
    print(f"  n_ctx         : {cfg.model_n_ctx}")
    print(f"  Cloud router  : {cfg.cloud_router_mode}")
    print(f"  AST safety    : {cfg.ast_safety_check}")
    print(f"  Dry run       : {args.dry_run}")
    print(f"{'═'*60}\n")

    # ------------------------------------------------------------------
    # 2. Initialize LLM Engine
    # ------------------------------------------------------------------
    from harness.llm_engine import create_engine_from_config
    llm = create_engine_from_config(cfg)

    if not args.dry_run:
        logger.info("Initializing LLM (loading GGUF into GPU)...")
        llm.initialize()
        diag = llm.diagnostics()
        logger.info("LLM diagnostics: %s", diag)
    else:
        logger.info("[DRY RUN] Skipping LLM initialization.")

    # ------------------------------------------------------------------
    # 3. Initialize Skill Manager (Tier 1 indexing)
    # ------------------------------------------------------------------
    from harness.skill_manager import SkillManager
    skills = SkillManager()
    indexed = skills.list_skills()
    print(f"  📚 Skills indexed: {len(indexed)}")
    for s in indexed:
        print(f"     [{s.tier:7s}] {s.name} — {s.frontmatter.get('description', '')[:60]}")
    print()

    # ------------------------------------------------------------------
    # 4. Load or create plan
    # ------------------------------------------------------------------
    from harness.plan_manager import PlanManager
    plan_path = Path(args.plan) if args.plan else None
    plan = PlanManager(plan_path)

    if args.goal and not plan.tasks:
        # Use Planner persona to decompose the goal
        logger.info("Planner phase: decomposing goal...")
        planner_prompt = (
            f"Decompose this goal into an ordered list of tasks:\n\n{args.goal}\n\n"
            "Output ONLY a valid JSON array of task objects. "
            "Each task must include: id (uuid), description, role (Executor), "
            "status (pending), output_files (list of expected filenames), error (null)."
        )

        if not args.dry_run:
            from harness.llm_engine import PLANNER_GRAMMAR, _safe_json_loads
            llm.set_system_prompt(
                (Path(__file__).parent / ".agent" / "AGENT.md").read_text()
            )
            raw_plan = llm.generate(planner_prompt, grammar_str=PLANNER_GRAMMAR)
            tasks = _safe_json_loads(raw_plan)
        else:
            # Dry-run: synthetic plan for demonstration
            import uuid
            tasks = [
                {
                    "id": str(uuid.uuid4()),
                    "description": f"[DRY RUN] Synthetic task 1 for: {args.goal[:40]}",
                    "role": "Executor",
                    "status": "pending",
                    "output_files": ["output_task1.py"],
                    "error": None,
                },
                {
                    "id": str(uuid.uuid4()),
                    "description": f"[DRY RUN] Synthetic task 2 for: {args.goal[:40]}",
                    "role": "Critic",
                    "status": "pending",
                    "output_files": [],
                    "error": None,
                },
            ]

        plan.create_plan(tasks)
        print(f"  📋 Plan created: {len(tasks)} tasks")
        for t in tasks:
            print(f"     [{t['role']:9s}] {t['description'][:70]}")
        print()

    elif not plan.tasks:
        print("  No goal provided and plan.json is empty. Use --goal to start.")
        return 1

    # ------------------------------------------------------------------
    # 5. Execute plan
    # ------------------------------------------------------------------
    from harness.executor import Executor
    executor = Executor(
        config=cfg,
        llm=llm,
        plan=plan,
        skills=skills,
        working_dir=Path(__file__).parent,
        dry_run=args.dry_run,
    )

    summary = executor.run()

    exit_code = 0 if summary["failed"] == 0 and summary["pending"] == 0 else 1
    logger.info("Harness exiting with code %d | summary=%s", exit_code, summary)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
