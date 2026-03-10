"""
External NIM-based PaperQA agent runner for LABBench2.

Runs the litqa2/litqa3-style literature QA benchmark using:
- Nemotron-Parse NIM (PDF parsing)
- Embedding NIM (e.g. llama-3.2-nv-embedqa)
- VLM NIM (e.g. nemotron-nano-12b-v2-vl) for answer generation

Requires paper-qa, paperqa-nemotron, and ldp installed (see setup.md).
The harness downloads question PDFs on-demand; this runner indexes them per-question
and runs the PaperQA LDP agent.

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
import logging
import os
from pathlib import Path

from evals.runners import AgentResponse

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
        try:
            response = await agent_query(question, settings, agent_type=settings.agent.agent_type)
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
