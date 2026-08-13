"""
harness/skill_manager.py
========================
Two-tier skill discovery and management system.

Tier 1 — Header Indexing:
    On initialization, reads the first 5 lines / YAML frontmatter from every
    SKILL.md found under .agent/skills/core/ and .agent/skills/evolved/.
    core/ skills strictly override evolved/ skills with the same name.

Tier 2 — Lazy Body Loading:
    Full SKILL.md content is read on demand via load_skill_body().

CRUD helpers (evolved/ only):
    create_skill(), update_skill(), delete_skill()

Proposal creation:
    create_proposal() writes to .agent/proposals/ when a core/ skill blocks.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SKILLS_ROOT = Path(__file__).parent.parent / ".agent" / "skills"
_CORE_DIR = _SKILLS_ROOT / "core"
_EVOLVED_DIR = _SKILLS_ROOT / "evolved"
_PROPOSALS_DIR = Path(__file__).parent.parent / ".agent" / "proposals"

HEADER_LINES = 5  # Number of lines read for Tier 1 indexing


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class SkillHeader:
    """Lightweight Tier-1 representation of a skill."""
    name: str
    tier: str                        # "core" or "evolved"
    path: Path
    frontmatter: dict[str, Any]      # Parsed YAML frontmatter (if present)
    preview_lines: list[str]         # First HEADER_LINES raw lines


@dataclass
class SkillIndex:
    """Complete skill index with precedence resolution."""
    skills: dict[str, SkillHeader] = field(default_factory=dict)

    def get(self, name: str) -> SkillHeader | None:
        return self.skills.get(name)

    def all_names(self) -> list[str]:
        return sorted(self.skills.keys())


# ---------------------------------------------------------------------------
# Frontmatter parser (lightweight — no external deps)
# ---------------------------------------------------------------------------
def _parse_frontmatter(lines: list[str]) -> tuple[dict[str, Any], list[str]]:
    """
    Extract YAML frontmatter between --- delimiters.
    Returns (frontmatter_dict, remaining_lines).
    """
    if not lines or lines[0].strip() != "---":
        return {}, lines

    end_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        return {}, lines

    fm_text = "".join(lines[1:end_idx])
    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        fm = {}
    return fm, lines[end_idx + 1:]


# ---------------------------------------------------------------------------
# Core scanner
# ---------------------------------------------------------------------------
def _scan_skill_dir(directory: Path, tier: str) -> dict[str, SkillHeader]:
    """Walk a skill directory and build headers for each SKILL.md found."""
    headers: dict[str, SkillHeader] = {}
    if not directory.exists():
        logger.debug("Skill directory does not exist, skipping: %s", directory)
        return headers

    for skill_dir in sorted(directory.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue

        try:
            with skill_md.open("r", encoding="utf-8") as fh:
                all_lines = fh.readlines()

            preview = all_lines[:HEADER_LINES]
            frontmatter, _ = _parse_frontmatter(all_lines)

            # Resolve skill name: prefer frontmatter > directory name
            name = frontmatter.get("name", skill_dir.name)

            headers[name] = SkillHeader(
                name=name,
                tier=tier,
                path=skill_md,
                frontmatter=frontmatter,
                preview_lines=[ln.rstrip("\n") for ln in preview],
            )
            logger.debug("Indexed %s skill: %s", tier, name)
        except Exception as exc:
            logger.warning("Failed to index skill at %s: %s", skill_dir, exc)

    return headers


# ---------------------------------------------------------------------------
# SkillManager
# ---------------------------------------------------------------------------
class SkillManager:
    """
    Manages Tier-1 header indexing and Tier-2 lazy body loading for skills.
    Provides CRUD helpers for the evolved/ sandbox and proposal creation.
    """

    def __init__(
        self,
        core_dir: Path | None = None,
        evolved_dir: Path | None = None,
        proposals_dir: Path | None = None,
    ) -> None:
        self._core_dir = core_dir or _CORE_DIR
        self._evolved_dir = evolved_dir or _EVOLVED_DIR
        self._proposals_dir = proposals_dir or _PROPOSALS_DIR
        self._index = SkillIndex()
        self._body_cache: dict[str, str] = {}

        self._proposals_dir.mkdir(parents=True, exist_ok=True)
        self._evolved_dir.mkdir(parents=True, exist_ok=True)

        self.refresh_index()

    # ------------------------------------------------------------------
    # Tier 1 — Header Indexing
    # ------------------------------------------------------------------
    def refresh_index(self) -> None:
        """Re-scan both skill directories and rebuild the index."""
        evolved = _scan_skill_dir(self._evolved_dir, "evolved")
        core = _scan_skill_dir(self._core_dir, "core")

        # Build merged index: start with evolved, then let core override
        merged: dict[str, SkillHeader] = {**evolved, **core}
        self._index = SkillIndex(skills=merged)
        self._body_cache.clear()  # invalidate body cache on re-index
        logger.info(
            "Skill index refreshed: %d core, %d evolved, %d total (after precedence)",
            len(core),
            len(evolved),
            len(merged),
        )

    @property
    def index(self) -> SkillIndex:
        return self._index

    def list_skills(self) -> list[SkillHeader]:
        return [self._index.skills[n] for n in self._index.all_names()]

    def get_header(self, name: str) -> SkillHeader | None:
        return self._index.get(name)

    # ------------------------------------------------------------------
    # Tier 2 — Lazy Body Loading
    # ------------------------------------------------------------------
    def load_skill_body(self, name: str) -> str:
        """
        Load the full SKILL.md body for the given skill name.
        Result is cached in memory until the next refresh_index() call.
        """
        if name in self._body_cache:
            logger.debug("Skill body cache hit: %s", name)
            return self._body_cache[name]

        header = self._index.get(name)
        if header is None:
            raise KeyError(f"Skill '{name}' not found in index.")

        body = header.path.read_text(encoding="utf-8")
        self._body_cache[name] = body
        logger.info("Skill body lazy-loaded: %s (tier=%s, %d bytes)", name, header.tier, len(body))
        return body

    def match_skills_for_task(self, task_description: str) -> list[SkillHeader]:
        """
        Return skills whose name or frontmatter tags appear in the task description.
        Simple keyword matching — no embedding required.
        """
        task_lower = task_description.lower()
        matches: list[SkillHeader] = []
        for header in self.list_skills():
            name_match = header.name.lower() in task_lower
            tag_match = any(
                tag.lower() in task_lower
                for tag in header.frontmatter.get("tags", [])
            )
            if name_match or tag_match:
                matches.append(header)
        return matches

    # ------------------------------------------------------------------
    # CRUD — Evolved Skills only
    # ------------------------------------------------------------------
    def _assert_evolved(self, name: str) -> None:
        header = self._index.get(name)
        if header and header.tier == "core":
            raise PermissionError(
                f"Skill '{name}' is a core/ skill and is read-only. "
                "Use create_proposal() to suggest modifications."
            )

    def create_skill(
        self,
        name: str,
        description: str,
        tags: list[str],
        body: str,
        version: str = "1.0",
    ) -> Path:
        """Create a new evolved skill. Raises if a core/ skill with same name exists."""
        self._assert_evolved(name)
        skill_dir = self._evolved_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md = skill_dir / "SKILL.md"

        frontmatter = {
            "name": name,
            "description": description,
            "tier": "evolved",
            "version": version,
            "tags": tags,
        }
        fm_text = "---\n" + yaml.dump(frontmatter, default_flow_style=False) + "---\n\n"
        skill_md.write_text(fm_text + body, encoding="utf-8")
        logger.info("Created evolved skill: %s", name)
        self.refresh_index()
        return skill_md

    def update_skill(self, name: str, body: str, version: str | None = None) -> None:
        """Update the body of an existing evolved skill."""
        self._assert_evolved(name)
        header = self._index.get(name)
        if header is None:
            raise KeyError(f"Evolved skill '{name}' not found.")

        fm = dict(header.frontmatter)
        if version:
            fm["version"] = version
        fm_text = "---\n" + yaml.dump(fm, default_flow_style=False) + "---\n\n"
        header.path.write_text(fm_text + body, encoding="utf-8")
        logger.info("Updated evolved skill: %s", name)
        self.refresh_index()

    def delete_skill(self, name: str) -> None:
        """Delete an evolved skill and its directory."""
        self._assert_evolved(name)
        header = self._index.get(name)
        if header is None:
            raise KeyError(f"Evolved skill '{name}' not found.")
        shutil.rmtree(header.path.parent)
        logger.info("Deleted evolved skill: %s", name)
        self.refresh_index()

    # ------------------------------------------------------------------
    # Proposals — for core/ skill modifications
    # ------------------------------------------------------------------
    def create_proposal(
        self,
        skill_name: str,
        reason: str,
        proposed_diff: str,
    ) -> Path:
        """
        Write a proposal file when a core/ skill blocks execution.
        Proposal is saved to .agent/proposals/<skill_name>_<timestamp>_proposal.md
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        proposal_path = self._proposals_dir / f"{skill_name}_{ts}_proposal.md"

        content = f"""# Proposal: Modify Core Skill `{skill_name}`

**Generated:** {ts}

## Reason

{reason}

## Proposed Diff

```diff
{proposed_diff}
```

---
*This proposal was automatically generated by the agent harness.*
*A human must review and apply this change to the core/ skill manually.*
"""
        proposal_path.write_text(content, encoding="utf-8")
        logger.info("Proposal created: %s", proposal_path)
        return proposal_path
