"""
harness/cloud_router.py
=======================
Hybrid Cloud Router with interactive terminal gatekeeper.

Modes (from config.yaml > cloud_router > mode):
  disabled      — Block all calls immediately with an air-gap message.
  ask           — Pause terminal, show sanitized payload preview,
                  prompt user [y/a/n].  y=allow once, a=allow_session, n=block.
  allow_session — Pass through without prompting.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class CloudRouterBlockedError(Exception):
    """Raised when a cloud call is blocked by the router."""


class CloudRouterError(Exception):
    """Raised on internal router errors."""


# ---------------------------------------------------------------------------
# CloudRouter
# ---------------------------------------------------------------------------
class CloudRouter:
    """
    Intercepts calls to external cloud AI services and enforces the
    configured gatekeeper policy.

    Parameters
    ----------
    config : ConfigLoader
        Loaded configuration. Reads cloud_router.mode on each call
        (reload-aware: re-checks mode dynamically).
    sanitizer : PIISanitizer, optional
        Sanitizer to scrub payload previews shown to the user.
    """

    def __init__(self, config: Any, sanitizer: Any | None = None) -> None:
        self._config = config
        self._sanitizer = sanitizer or config.sanitizer
        self._session_allowed: bool = False   # set to True if user chooses 'a'
        self._block_count: int = 0
        self._allow_count: int = 0

    # ------------------------------------------------------------------
    @property
    def _mode(self) -> str:
        """Dynamic mode check — re-reads config each time."""
        if self._session_allowed:
            return "allow_session"
        return self._config.cloud_router_mode

    # ------------------------------------------------------------------
    def ask_cloud_ai(
        self,
        prompt: str,
        service: str = "generic",
        extra_context: dict[str, Any] | None = None,
    ) -> str:
        """
        Main tool handler for cloud AI calls.

        Parameters
        ----------
        prompt : str
            The prompt/query to send to the cloud service.
        service : str
            Identifier for the target service (e.g. 'openai', 'gemini').
        extra_context : dict, optional
            Additional metadata to display in `ask` mode preview.

        Returns
        -------
        str
            Response string, or raises CloudRouterBlockedError.
        """
        mode = self._mode

        if mode == "disabled":
            return self._handle_disabled(prompt, service)

        if mode == "ask":
            return self._handle_ask(prompt, service, extra_context or {})

        if mode == "allow_session":
            return self._handle_allow(prompt, service)

        raise CloudRouterError(f"Unknown cloud_router mode: '{mode}'")

    # ------------------------------------------------------------------
    # Mode handlers
    # ------------------------------------------------------------------
    def _handle_disabled(self, prompt: str, service: str) -> str:
        self._block_count += 1
        msg = (
            f"\n{'='*60}\n"
            f"  ✈️  AIR-GAP ENFORCED — Cloud call BLOCKED\n"
            f"{'='*60}\n"
            f"  Service  : {service}\n"
            f"  Mode     : disabled (config.yaml > cloud_router > mode)\n"
            f"  Action   : Call blocked. No data left this machine.\n"
            f"{'='*60}\n"
            f"  To enable cloud routing, change config.yaml:\n"
            f"    cloud_router:\n"
            f"      mode: ask        # interactive approval\n"
            f"      # mode: allow_session  # permit all calls this session\n"
            f"{'='*60}\n"
        )
        logger.warning("Cloud call BLOCKED (disabled mode) | service=%s", service)
        raise CloudRouterBlockedError(msg)

    def _handle_ask(
        self,
        prompt: str,
        service: str,
        extra_context: dict[str, Any],
    ) -> str:
        """Pause terminal, show sanitized preview, request user approval."""
        sanitized_prompt = self._sanitizer.sanitize(prompt)
        preview = sanitized_prompt[:500] + ("..." if len(sanitized_prompt) > 500 else "")

        print(f"\n{'='*60}")
        print(f"  ⚠️  CLOUD ROUTER GATEKEEPER — Approval Required")
        print(f"{'='*60}")
        print(f"  Service      : {service}")
        print(f"  Payload (sanitized preview):")
        print(f"  {'─'*50}")
        for line in preview.splitlines():
            print(f"    {line}")
        print(f"  {'─'*50}")
        if extra_context:
            print(f"  Extra context: {extra_context}")
        print(f"\n  [y] Allow this call once")
        print(f"  [a] Allow all calls this session (promote to allow_session)")
        print(f"  [n] Block this call")
        print(f"{'='*60}")

        try:
            choice = input("  Your choice [y/a/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = "n"

        if choice == "y":
            self._allow_count += 1
            logger.info("Cloud call APPROVED (once) | service=%s", service)
            return self._execute_cloud_call(prompt, service)

        if choice == "a":
            self._session_allowed = True
            self._allow_count += 1
            logger.info("Cloud call APPROVED (session) | service=%s", service)
            print("  Session-level approval granted. Subsequent calls will pass automatically.")
            return self._execute_cloud_call(prompt, service)

        # 'n' or anything else → block
        self._block_count += 1
        logger.info("Cloud call REJECTED by user | service=%s", service)
        msg = f"[CLOUD ROUTER] Call to '{service}' was rejected by the user."
        raise CloudRouterBlockedError(msg)

    def _handle_allow(self, prompt: str, service: str) -> str:
        self._allow_count += 1
        logger.info("Cloud call ALLOWED (allow_session) | service=%s", service)
        return self._execute_cloud_call(prompt, service)

    # ------------------------------------------------------------------
    # Actual HTTP execution (stub — extend for real services)
    # ------------------------------------------------------------------
    def _execute_cloud_call(self, prompt: str, service: str) -> str:
        """
        Execute the cloud API call.

        Currently a stub that logs and returns a placeholder.
        Extend this method to integrate real services (OpenAI, Gemini, etc.)
        while ensuring the sanitizer is applied to ALL outgoing payloads.
        """
        # Sanitize before transmission
        safe_prompt = self._sanitizer.sanitize(prompt)
        logger.info(
            "Cloud call executing | service=%s | payload_len=%d",
            service,
            len(safe_prompt),
        )
        # TODO: implement real HTTP client here
        # e.g.: return openai_client.chat(safe_prompt)
        return f"[STUB RESPONSE from {service}] Received {len(safe_prompt)} chars."

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, Any]:
        return {
            "mode": self._mode,
            "session_allowed": self._session_allowed,
            "block_count": self._block_count,
            "allow_count": self._allow_count,
        }
