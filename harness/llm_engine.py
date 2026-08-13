"""
harness/llm_engine.py
=====================
llama-cpp-python CUDA wrapper with GBNF grammar support.

Provides:
- LLMEngine: main class wrapping llama_cpp.Llama
- Built-in GBNF grammars for structured output
- flush_context(): clears chat history while preserving system prompt
- generate(): chat completion entry point
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON safety helper
# ---------------------------------------------------------------------------
def _safe_json_loads(text: str) -> Any:
    """
    Parse JSON from LLM output robustly.

    LLMs occasionally embed literal control characters (\n, \t, etc.) inside
    JSON string values, which violates RFC 8259 strict mode.  We try three
    strategies in order:

    1. Standard ``json.loads`` — fastest path, works when output is clean.
    2. ``json.loads(..., strict=False)`` — allows literal control chars in
       strings; sufficient for most real-world LLM quirks.
    3. Regex scrub + retry — replaces bare control chars inside strings as a
       last resort before giving up.
    """
    # Strategy 1: clean JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: lenient (allows literal newlines / tabs in strings)
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass

    # Strategy 3: strip raw control characters from the entire payload and retry
    _CTRL_RE = re.compile(r'[\x00-\x1f\x7f]')
    scrubbed = _CTRL_RE.sub(' ', text)
    try:
        return json.loads(scrubbed)
    except json.JSONDecodeError as exc:
        logger.error("_safe_json_loads failed all strategies | raw=%r", text[:200])
        raise exc

# ---------------------------------------------------------------------------
# GBNF Grammars
# ---------------------------------------------------------------------------
#  GBNF is a BNF variant used by llama.cpp to constrain model output.
#  Reference: https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md

TOOL_CALL_GRAMMAR = r"""
root   ::= object
object ::= "{" ws "\"tool\"" ws ":" ws string ws "," ws "\"args\"" ws ":" ws args-object ws "}"
args-object ::= "{" ws (pair (ws "," ws pair)*)? ws "}"
pair   ::= string ws ":" ws value
value  ::= string | number | "true" | "false" | "null" | object | array
array  ::= "[" ws (value (ws "," ws value)*)? ws "]"
string ::= "\"" ([^"\\] | "\\" .)* "\""
number ::= "-"? ([0-9]+ ("." [0-9]+)?)
ws     ::= ([ \t\n] ws)?
"""

VERIFICATION_GRAMMAR = r"""
root   ::= object
object ::= "{" ws "\"status\"" ws ":" ws status ws "," ws "\"reason\"" ws ":" ws string ws "}"
status ::= "\"pass\"" | "\"fail\""
string ::= "\"" ([^"\\] | "\\" .)* "\""
ws     ::= ([ \t\n] ws)?
"""

PLANNER_GRAMMAR = r"""
root        ::= array
array       ::= "[" ws (task-object (ws "," ws task-object)*)? ws "]"
task-object ::= "{" ws "\"id\"" ws ":" ws string ws "," ws "\"description\"" ws ":" ws string ws "," ws "\"role\"" ws ":" ws role ws "," ws "\"status\"" ws ":" ws "\"pending\"" ws "," ws "\"output_files\"" ws ":" ws file-array ws "," ws "\"error\"" ws ":" ws "null" ws "}"
role        ::= "\"Planner\"" | "\"Architect\"" | "\"Executor\"" | "\"Critic\""
file-array  ::= "[" ws (string (ws "," ws string)*)? ws "]"
string      ::= "\"" ([^"\\] | "\\" .)* "\""
ws          ::= ([ \t\n] ws)?
"""


# ---------------------------------------------------------------------------
# LLMEngine
# ---------------------------------------------------------------------------

class LLMEngine:
    """
    Wraps llama_cpp.Llama for structured, CUDA-accelerated inference.

    Parameters
    ----------
    model_path : str or Path
        Absolute path to the GGUF model file.
    n_gpu_layers : int
        Number of layers to offload to GPU (-1 = all).
    n_ctx : int
        Context window size in tokens.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum tokens to generate per call.
    verbose : bool
        Whether to print llama.cpp internal logs.
    """

    def __init__(
        self,
        model_path: str | Path,
        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        verbose: bool = True,
        enable_thinking: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        enable_thinking : bool
            If False (default), thinking tokens are suppressed at the llama.cpp
            level when the model supports it (Qwen3 series), and any
            ``<think>…</think>`` blocks that do appear are stripped from the
            raw output before it is returned.  Set to True only if you want
            the model's chain-of-thought reasoning to be preserved.
        """
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"GGUF model not found: {self._model_path}\n"
                "Ensure the model blob is present at the configured path."
            )

        self._n_gpu_layers = n_gpu_layers
        self._n_ctx = n_ctx
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._verbose = verbose
        self._enable_thinking = enable_thinking

        self._system_prompt: str = ""
        self._history: list[dict[str, str]] = []
        self._llm: Any = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Think-block stripper (Qwen3-Coder / DeepSeek-R1 compatibility)
    # ------------------------------------------------------------------
    _THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

    def _strip_think_blocks(self, text: str) -> str:
        """Remove <think>…</think> reasoning blocks from model output."""
        stripped = self._THINK_RE.sub("", text)
        return stripped.strip()

    # ------------------------------------------------------------------
    # Initialization (deferred to avoid import errors when llama_cpp missing)
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Load the model into GPU memory. Call once before first generate()."""
        try:
            from llama_cpp import Llama  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "llama_cpp is not installed. Install with:\n"
                "  $env:CMAKE_ARGS='-DLLAMA_CUDA=on'; pip install llama-cpp-python"
            ) from exc

        logger.info(
            "Loading GGUF model: %s  (n_gpu_layers=%d, n_ctx=%d)",
            self._model_path,
            self._n_gpu_layers,
            self._n_ctx,
        )
        self._llm = Llama(
            model_path=str(self._model_path),
            n_gpu_layers=self._n_gpu_layers,
            n_ctx=self._n_ctx,
            verbose=self._verbose,
            chat_format="chatml",   # Qwen2.5 uses ChatML format
        )
        logger.info("Model loaded successfully.")

    @property
    def is_initialized(self) -> bool:
        return self._llm is not None

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------
    def set_system_prompt(self, prompt: str) -> None:
        """Set the persistent system prompt (survives flush_context)."""
        self._system_prompt = prompt

    def flush_context(self) -> None:
        """
        Clear all chat history while preserving the system prompt.
        Call between tasks to enforce context window hygiene.
        """
        self._history.clear()
        logger.debug("Context window flushed — history cleared.")

    def add_user_message(self, content: str) -> None:
        self._history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self._history.append({"role": "assistant", "content": content})

    def _build_messages(self) -> list[dict[str, str]]:
        """Assemble the full message list for a chat completion call."""
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend(self._history)
        return messages

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------
    def generate(
        self,
        user_message: str | None = None,
        grammar_str: str | None = None,
        add_to_history: bool = True,
    ) -> str:
        """
        Run a chat completion.

        Parameters
        ----------
        user_message : str, optional
            If provided, appended to history before inference.
        grammar_str : str, optional
            A GBNF grammar string to constrain model output.
        add_to_history : bool
            Whether to store assistant response in history.

        Returns
        -------
        str
            The model's text response.
        """
        if not self.is_initialized:
            raise RuntimeError("LLMEngine not initialized. Call initialize() first.")

        if user_message is not None:
            self.add_user_message(user_message)

        messages = self._build_messages()

        # Build grammar object if provided
        grammar = None
        if grammar_str:
            try:
                from llama_cpp import LlamaGrammar  # type: ignore[import]
                grammar = LlamaGrammar.from_string(grammar_str)
            except Exception as exc:
                logger.warning("Failed to compile GBNF grammar: %s", exc)

        # Qwen3-series models suppress thinking via chat_template_kwargs in newer
        # llama-cpp-python builds; for broader compatibility we rely solely on
        # _strip_think_blocks() which runs unconditionally when enable_thinking=False.
        response = self._llm.create_chat_completion(
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            grammar=grammar,
        )

        content: str = response["choices"][0]["message"]["content"] or ""
        # Always strip residual think blocks (safety net for models that ignore
        # the enable_thinking=False flag or emit them unconditionally).
        if not self._enable_thinking:
            content = self._strip_think_blocks(content)

        if add_to_history:
            self.add_assistant_message(content)

        return content

    # ------------------------------------------------------------------
    # Structured generation helpers
    # ------------------------------------------------------------------
    def generate_tool_call(self, user_message: str) -> dict[str, Any]:
        """Generate a structured tool call JSON using TOOL_CALL_GRAMMAR."""
        raw = self.generate(user_message, grammar_str=TOOL_CALL_GRAMMAR)
        try:
            return _safe_json_loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("Tool call JSON parse failed: %s | raw=%r", exc, raw)
            raise

    def generate_verification(self, user_message: str) -> dict[str, str]:
        """Generate a pass/fail verification JSON using VERIFICATION_GRAMMAR."""
        raw = self.generate(user_message, grammar_str=VERIFICATION_GRAMMAR)
        try:
            return _safe_json_loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("Verification JSON parse failed: %s | raw=%r", exc, raw)
            raise

    def generate_plan(self, user_message: str) -> list[dict[str, Any]]:
        """Generate a structured task plan using PLANNER_GRAMMAR."""
        raw = self.generate(user_message, grammar_str=PLANNER_GRAMMAR)
        try:
            return _safe_json_loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("Planner JSON parse failed: %s | raw=%r", exc, raw)
            raise

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, Any]:
        """Return runtime diagnostics for logging/testing."""
        return {
            "model_path": str(self._model_path),
            "n_gpu_layers": self._n_gpu_layers,
            "n_ctx": self._n_ctx,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "enable_thinking": self._enable_thinking,
            "history_length": len(self._history),
            "initialized": self.is_initialized,
        }


# ---------------------------------------------------------------------------
# Factory from config
# ---------------------------------------------------------------------------
def create_engine_from_config(cfg: Any) -> LLMEngine:
    """Create an LLMEngine from a ConfigLoader instance."""
    return LLMEngine(
        model_path=cfg.model_path,
        n_gpu_layers=cfg.model_n_gpu_layers,
        n_ctx=cfg.model_n_ctx,
        temperature=cfg.model_temperature,
        max_tokens=cfg.model_max_tokens,
        verbose=cfg.model_verbose,
        enable_thinking=cfg.model_enable_thinking,
    )
