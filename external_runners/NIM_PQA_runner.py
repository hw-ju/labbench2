"""
External NIM-based PaperQA agent runner for LABBench2.

Runs the litqa2/litqa3-style literature QA benchmark using:
- Nemotron-Parse NIM (PDF parsing)
- Embedding NIM (e.g. llama-3.2-nv-embedqa)
- VLM NIM (e.g. nemotron-nano-12b-v2-vl) for answer generation

Requires paper-qa, paperqa-nemotron, and ldp installed (see setup.md).
The harness downloads question PDFs on-demand; this runner indexes them per-question
and runs the PaperQA LDP agent.

Trajectory logging (optional):
    Set LABBENCH2_PRINT_TRAJECTORIES=1 to print per-step trajectory details after each
    question and to save a Jupyter notebook (.ipynb) per question under
    LABBENCH2_TRAJECTORY_DIR (default: labbench2_trajectories). Open the notebook in
    Jupyter Lab to visualize context pairs (raw chunk + summary) and embedded images/media.

LDP and environment:
    When agent_type is ldp.agent.SimpleAgent, paper-qa uses LDP's RolloutManager
    and PaperQAEnvironment. In run_ldp_agent (paperqa.agents.main):
    - A PaperQAEnvironment instance is created with (query, settings, docs).
    - RolloutManager(agent, callbacks=[...]) is created; sample_trajectories(
        environments=[env], max_steps=...) is called.
    - The RolloutManager drives the loop: env.reset() → (obs, tools); then
      the agent chooses actions and the manager calls env.step(action) each time.
    So: LDP's RolloutManager is used; PaperQAEnvironment supplies all tools
    (paper_search, gather_evidence, gen_answer, complete, etc. via make_tools())
    and env.step() (exec_tool_calls, reward, done, truncated).

Grading:
    The labbench2 harness performs grading after the runner returns the answer.
    For litqa3 (open-ended), HybridEvaluator routes to LLMJudgeEvaluator, which
    uses an LLM (default: anthropic/claude-sonnet-4-5) to compare the submitted
    answer to the expected answer (question.ideal) and returns correct/incorrect/unsure.
    This runner does not implement grading; it only returns the answer text.

    To use the same VLM as this runner for the judge (e.g. no Anthropic key):
    set OPENAI_API_BASE and OPENAI_API_KEY to your VLM endpoint, then run evals
    with --judge-model "openai:nvidia/nemotron-nano-12b-v2-vl" or
    LABBENCH2_JUDGE_MODEL=openai:nvidia/nemotron-nano-12b-v2-vl (see setup.md).

Usage:
    uv run python -m evals.run_evals --agent external:./external_runners/NIM_PQA_runner.py:NIMPQARunner --tag litqa3 --limit 2
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from pathlib import Path

from evals.runners import AgentResponse

# Optional: set LABBENCH2_PRINT_TRAJECTORIES=1 to print step-by-step trajectory (like litqa2_LDP_nv_debug_rawchunks.ipynb)
_PRINT_TRAJECTORIES = os.environ.get("LABBENCH2_PRINT_TRAJECTORIES", "").strip().lower() in ("1", "true", "yes")

# LiteLLM / PaperQA env: set before importing litellm or paperqa
os.environ.setdefault("LITELLM_LOG", "INFO")
os.environ.setdefault("LITELLM_MAX_CALLBACKS", "500")

# PaperQA + NIM imports (require paper-qa, paperqa-nemotron in same env)
from paperqa.agents import agent_query
from paperqa.settings import (
    AgentSettings,
    AnswerSettings,
    IndexSettings,
    ParsingSettings,
    Settings,
)
from paperqa_nemotron import parse_pdf_to_pages

logger = logging.getLogger(__name__)

# NIM endpoint defaults (override with env)
DEFAULT_PARSE_API_BASE = "http://localhost:8002/v1"
DEFAULT_EMBEDDING_API_BASE = "http://localhost:8003/v1"
DEFAULT_VLM_API_BASE = "http://localhost:8004/v1"
PARSE_MODEL_NAME = "nvidia/nemotron-parse"
EMBEDDING_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
VLM_MODEL = "nvidia/nemotron-nano-12b-v2-vl"
CUSTOM_VLM_NAME = "selfhost-nemotron-vlm"
LITELLM_VLM_MODEL = f"openai/{VLM_MODEL}"
SELFHOST_API_KEY = "dummy"


def _extract_contexts_from_state(state) -> list[dict] | None:
    """Extract (raw_text, summary, score, raw_media) from state.session.contexts for trajectory log."""
    try:
        session = getattr(state, "session", None)
        contexts = getattr(session, "contexts", None) if session else None
        if not contexts:
            return None
        out = []
        for c in contexts:
            text_obj = getattr(c, "text", None)
            raw_text = getattr(text_obj, "text", "") or "" if text_obj else ""
            raw_media = []
            for m in getattr(text_obj, "media", None) or []:
                info = getattr(m, "info", {}) or {}
                info_safe = {k: v for k, v in info.items() if isinstance(v, (str, int, float, bool, type(None)))}
                try:
                    data_url = m.to_image_url() if hasattr(m, "to_image_url") else str(m)
                except Exception:
                    data_url = ""
                raw_media.append({"data_url": data_url, "info": info_safe})
            out.append({
                "summary": getattr(c, "context", ""),
                "score": getattr(c, "score", "?"),
                "raw_text": raw_text[:50000] + ("..." if len(raw_text) > 50000 else ""),
                "raw_media": raw_media,
            })
        return out
    except Exception:
        return None


class TrajectoryRecorder:
    """Records LDP rollout steps via paper-qa's on_* callbacks for same-detail output as litqa2_LDP_nv_debug_rawchunks.ipynb."""

    def __init__(self) -> None:
        self.steps: list[dict] = []
        self.last_contexts: list[dict] | None = None

    def clear(self) -> None:
        self.steps.clear()
        self.last_contexts = None

    async def on_env_reset(self, state) -> None:
        self.clear()

    async def on_agent_action(self, action, agent_state, _reward_or_placeholder: float) -> None:
        action_val = getattr(action, "value", action)
        if hasattr(action_val, "tool_calls"):
            action_repr = [str(getattr(tc, "function", tc)) for tc in (action_val.tool_calls or [])]
        else:
            action_repr = str(action_val)
        messages_repr = []
        if hasattr(agent_state, "messages"):
            for m in agent_state.messages:
                messages_repr.append(str(m)[:8000] + ("..." if len(str(m)) > 8000 else ""))
        else:
            messages_repr.append(repr(agent_state)[:1000])
        self.steps.append({
            "agent_state_messages": messages_repr,
            "action": action_repr,
            "observation": None,
            "reward": None,
            "contexts": None,
        })

    async def on_env_step(self, obs: list, reward: float, done: bool, truncated: bool) -> None:
        if not self.steps:
            return
        obs_repr = [str(m)[:8000] + ("..." if len(str(m)) > 8000 else "") for m in (obs or [])]
        self.steps[-1]["observation"] = obs_repr
        self.steps[-1]["reward"] = reward
        self.steps[-1]["contexts"] = self.last_contexts
        self.last_contexts = None

    async def capture_contexts_cb(self, state) -> None:
        self.last_contexts = _extract_contexts_from_state(state)

    def print_trajectory(self) -> None:
        if not self.steps:
            return
        print("=== Trajectory (same detail as litqa2_LDP_nv_debug_rawchunks) ===")
        print(f"Number of steps: {len(self.steps)}")
        for i, step in enumerate(self.steps):
            print(f"\n----------------- Step {i} ----------------")
            # agent_state.messages (match notebook order)
            msgs = step.get("agent_state_messages") or []
            print(f"** agent_state.messages ({len(msgs)}):")
            for j, m in enumerate(msgs):
                print(f"[{j}] {m}")
            # observation (input for this step = previous step's next_observation)
            obs_in = self.steps[i - 1].get("observation") if i > 0 else []
            if obs_in:
                print()
                print(f"** observation ({len(obs_in)}):")
                for j, m in enumerate(obs_in):
                    print(f"[{j}] {m}")
            # action (tool choice)
            print()
            print(f"** action (tool choice): {step.get('action')}")
            # next_observation (tool call results from env.step)
            obs = step.get("observation") or []
            if obs:
                print()
                print(f"** next_observation ({len(obs)}):")
                for j, m in enumerate(obs):
                    print(f"[{j}] {m}")
            # reward
            print()
            print(f"** reward: {step.get('reward')}")
            # gather_evidence context pairs (Raw chunk text, Media, Summary score, Summary)
            contexts = step.get("contexts") or []
            for idx, ctx in enumerate(contexts):
                print()
                print(f"** gather_evidence context pair [{idx}] (score={ctx.get('score', '?')}):")
                raw = (ctx.get("raw_text") or "").replace("\n", "\n  ")
                cap = 2000
                print("  [Raw chunk text]")
                print("  " + (raw[:cap] + ("..." if len(raw) > cap else "")))
                for m_idx, m in enumerate(ctx.get("raw_media") or []):
                    data_url = m.get("data_url") or ""
                    if data_url:
                        print(f"  [Media {m_idx}] (data URL, len={len(data_url)})")
                print(f"  [Summary score] {ctx.get('score', '?')}")
                print("  [Summary]")
                print("  " + (ctx.get("summary") or "").replace("\n", "\n  "))
        print("=== End trajectory ===")

    def save_notebook(self, question: str, path: Path) -> None:
        """Save trajectory to a Jupyter notebook (.ipynb) for viewing in Jupyter Lab with context pairs and embedded images."""
        if not self.steps:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cells = _trajectory_to_notebook_cells(question, self.steps)
        nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.11.0"},
            },
            "cells": cells,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        logger.info("Trajectory notebook saved to %s (open in Jupyter Lab to view context pairs and media)", path)


def _trajectory_to_notebook_cells(question: str, steps: list[dict]) -> list[dict]:
    """Build a list of Jupyter notebook cell dicts for trajectory visualization (markdown + embedded images)."""
    cells = []
    # Title and question
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# LabBench2 trajectory\n", "\n", "**Question:**\n", "\n", (question[:10000] + ("..." if len(question) > 10000 else "")) + "\n"],
    })
    # Media display width in notebook (match notebook's smaller size; notebook uses width=280, we use 200)
    MEDIA_WIDTH = 200
    for i, step in enumerate(steps):
        msgs = step.get("agent_state_messages") or []
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## Step {i}\n", "\n", f"### agent_state.messages ({len(msgs)})\n"],
        })
        for j, m in enumerate(msgs):
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"[{j}] {_escape_md(str(m))}\n"],
            })
        # observation (input for this step = previous step's next_observation)
        obs_in = steps[i - 1].get("observation") if i > 0 else []
        if obs_in:
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"### observation ({len(obs_in)})\n"] + [_escape_md(str(m)) + "\n" for m in obs_in],
            })
        # action (tool choice)
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### action (tool choice)\n", "\n", _escape_md(str(step.get("action", ""))) + "\n"],
        })
        # next_observation (tool call results from env.step)
        obs = step.get("observation") or []
        if obs:
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"### next_observation ({len(obs)})\n"] + [_escape_md(str(m)) + "\n" for m in obs],
            })
        # reward
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["### reward\n", "\n", str(step.get("reward", "")) + "\n"],
        })
        # gather_evidence context pairs: Raw chunk text, Media (smaller), Summary score, Summary
        contexts = step.get("contexts") or []
        for idx, ctx in enumerate(contexts):
            score = ctx.get("score", "?")
            raw_text = ctx.get("raw_text") or ""
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"### gather_evidence context pair [{idx}] (score={score})\n", "\n", "**Raw chunk text:**\n", "\n", "```\n", raw_text + "\n", "```\n"],
            })
            for m_idx, media in enumerate(ctx.get("raw_media") or []):
                data_url = media.get("data_url") or ""
                if data_url.startswith("data:"):
                    # Smaller embedded image via HTML (Jupyter Lab renders it)
                    cells.append({
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [f"**Media {m_idx}:**\n", "\n", f'<img src="{data_url}" width="{MEDIA_WIDTH}" />\n'],
                    })
                else:
                    cells.append({
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [f"**Media {m_idx}:** (non-embed URL or binary)\n", "\n", _escape_md(data_url[:2000]) + "\n"],
                    })
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"**Summary score:** {score}\n", "\n", "**Summary:**\n", "\n", (ctx.get("summary") or "") + "\n"],
            })
    return cells


def _escape_md(text: str) -> str:
    """Escape backticks and long code blocks that could break markdown."""
    if not text:
        return ""
    return text.replace("\\", "\\\\").replace("`", "\\`")


def _get_nim_settings() -> tuple[str, str, str]:
    parse_base = os.environ.get("PQA_PARSE_API_BASE", DEFAULT_PARSE_API_BASE)
    embed_base = os.environ.get("PQA_EMBEDDING_API_BASE", DEFAULT_EMBEDDING_API_BASE)
    vlm_base = os.environ.get("PQA_VLM_API_BASE", DEFAULT_VLM_API_BASE)
    return parse_base, embed_base, vlm_base


def _build_base_settings(
    parse_api_base: str,
    embedding_api_base: str,
    vlm_api_base: str,
) -> Settings:
    """Build PaperQA Settings for NIMs (parse, embedding, VLM). paper_directory is set per-question."""
    nvidia_vlm_config = {
        "model_list": [
            {
                "model_name": CUSTOM_VLM_NAME,
                "litellm_params": {
                    "model": LITELLM_VLM_MODEL,
                    "api_base": vlm_api_base,
                    "api_key": SELFHOST_API_KEY,
                    "temperature": 0,
                    "max_tokens": 2048,
                },
            }
        ]
    }
    nvidia_embedding_config = {
        "kwargs": {
            "api_base": embedding_api_base,
            "api_key": SELFHOST_API_KEY,
            "encoding_format": "float",
            "input_type": "passage",
        }
    }
    parsing_settings = ParsingSettings(
        use_doc_details=False,
        parse_pdf=parse_pdf_to_pages,
        reader_config={
            "chunk_chars": 3000,
            "overlap": 250,
            "dpi": 300,
            "api_params": {
                "api_base": parse_api_base,
                "api_key": SELFHOST_API_KEY,
                "model_name": PARSE_MODEL_NAME,
                "temperature": 0,
                "max_tokens": 8995,
            },
        },
        enrichment_llm=CUSTOM_VLM_NAME,
        enrichment_llm_config=nvidia_vlm_config,
        multimodal=True,
    )
    # paper_directory and index_directory will be overridden per question
    index_settings = IndexSettings(
        paper_directory=Path.cwd(),
        index_directory=os.path.join(os.path.expanduser("~"), ".cache", "labbench2", "pqa_indexes"),
    )
    return Settings(
        llm=CUSTOM_VLM_NAME,
        llm_config=nvidia_vlm_config,
        summary_llm=CUSTOM_VLM_NAME,
        summary_llm_config=nvidia_vlm_config,
        embedding=f"openai/{EMBEDDING_MODEL}",
        embedding_config=nvidia_embedding_config,
        temperature=0,
        verbosity=0,
        answer=AnswerSettings(
            evidence_k=5,
            answer_max_sources=3,
        ),
        parsing=parsing_settings,
        agent=AgentSettings(
            agent_type="ldp.agent.SimpleAgent",
            agent_llm=LITELLM_VLM_MODEL,
            agent_llm_config={
                "model_list": [
                    {
                        "model_name": LITELLM_VLM_MODEL,
                        "litellm_params": {
                            "model": LITELLM_VLM_MODEL,
                            "api_base": vlm_api_base,
                            "api_key": SELFHOST_API_KEY,
                            "temperature": 0,
                            "max_tokens": 2048,
                        },
                    }
                ]
            },
            index=index_settings,
        ),
    )


class NIMPQARunner:
    """Runner that uses NIM-based PaperQA (Nemotron-Parse + Embedding + VLM) for litqa3."""

    _trajectory_file_index: int = 0

    def __init__(self) -> None:
        parse_base, embed_base, vlm_base = _get_nim_settings()
        self._base_settings = _build_base_settings(parse_base, embed_base, vlm_base)
        # LDP/SimpleAgent may use OPENAI_* for the VLM
        os.environ.setdefault("OPENAI_API_BASE", vlm_base)
        os.environ.setdefault("OPENAI_API_KEY", SELFHOST_API_KEY)
        logging.getLogger("LiteLLM").setLevel(logging.INFO)

    async def upload_files(
        self, files: list[Path], gcs_prefix: str | None = None
    ) -> dict[str, str]:
        """Return local path -> path mapping; harness already downloaded files to disk."""
        return {str(f): str(f) for f in files}

    async def execute(
        self, question: str, file_refs: dict[str, str] | None = None
    ) -> AgentResponse:
        """Run PaperQA agent on the question using the provided file paths as the paper set."""
        if not file_refs:
            return AgentResponse(
                text="[No files provided for this question.]",
                metadata={"error": "no_files"},
            )
        # Use the directory of the first file as paper_directory for this question
        first_path = Path(next(iter(file_refs.values())))
        files_dir = first_path.parent.resolve()
        settings = copy.deepcopy(self._base_settings)
        settings.agent.index.paper_directory = files_dir
        # Unique index subdir per question to avoid cross-question index reuse
        question_index_key = hashlib.sha256(str(files_dir).encode()).hexdigest()[:16]
        settings.agent.index.index_directory = str(
            Path(settings.agent.index.index_directory) / question_index_key
        )
        recorder = TrajectoryRecorder() if _PRINT_TRAJECTORIES else None
        if recorder:
            callbacks = dict(getattr(settings.agent, "callbacks", None) or {})
            callbacks.setdefault("gather_evidence_completed", []).append(recorder.capture_contexts_cb)
            settings.agent.callbacks = callbacks
        runner_kwargs = {}
        if recorder:
            runner_kwargs["on_env_reset_callback"] = recorder.on_env_reset
            runner_kwargs["on_agent_action_callback"] = recorder.on_agent_action
            runner_kwargs["on_env_step_callback"] = recorder.on_env_step
        try:
            response = await agent_query(
                question, settings, agent_type=settings.agent.agent_type, **runner_kwargs
            )
            if recorder:
                recorder.print_trajectory()
                out_dir = Path(os.environ.get("LABBENCH2_TRAJECTORY_DIR", "labbench2_trajectories"))
                out_path = out_dir / f"trajectory_{NIMPQARunner._trajectory_file_index}.ipynb"
                recorder.save_notebook(question, out_path)
                NIMPQARunner._trajectory_file_index += 1
            answer = response.session.answer or ""
            return AgentResponse(
                text=answer,
                raw_output=response.model_dump() if hasattr(response, "model_dump") else None,
                metadata={"status": getattr(response, "status", None)},
            )
        except Exception as e:
            logger.exception("PaperQA agent_query failed: %s", e)
            return AgentResponse(
                text=f"[Error: {e!s}]",
                metadata={"error": str(e)},
            )

    def extract_answer(self, response: AgentResponse) -> str:
        return response.text

    async def download_outputs(self, dest_dir: Path) -> Path | None:
        """This runner does not produce output files."""
        return None

    async def cleanup(self) -> None:
        pass
