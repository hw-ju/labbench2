# NIM-based PaperQA runner (NIM_PQA_runner) for LABBench2 (litqa3)

## 1. Environment setup
## Install Python 3.11 via deadsnakes (venv-based)
```bash
# Install Python 3.11 from deadsnakes PPA (no pyenv)
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Verify
python3.11 --version
```

```bash
# Clone repos
mkdir -p ~/forks
cd ~/forks

# paper-qa (fork with paper-qa-nemotron; replace with your fork URL if needed)
git clone https://github.com/hw-ju/paper-qa.git
cd paper-qa
git checkout azure_inference_debug   
# or your branch with nemotron parse + PyMuPDF failover. The latest original paper-qa repo might have already implemented PyMuPDF failover.
cd ..

# labbench2
git clone https://github.com/hw-ju/labbench2.git
cd labbench2
# git fetch origin
git checkout nim_pqa_runner   
# git log --oneline -n 10
cd ..

# Create venv and install
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Step 1: paper-qa (no .[pymupdf] here—Step 2 installs fork's paper-qa-pymupdf; .[pymupdf] would
# pull PyPI paper-qa-pymupdf which expects resolve_page_range in paperqa.readers and fails)
cd paper-qa
pip install .
cd ..

# Step 2: paper-qa-pymupdf from paper-qa fork
cd paper-qa/packages/paper-qa-pymupdf
pip install .
cd ../../..

# Step 3: paper-qa-nemotron from paper-qa fork (Nemotron-Parse + PyMuPDF fallback)
cd paper-qa/packages/paper-qa-nemotron
pip install .
cd ../../..

# Step 4: ldp (for SimpleAgent, RolloutManager)
pip install datasets aiohttp ldp

# Step 5: labbench2 harness + this runner
cd labbench2
pip install -e .
cd ..
```

### Sanity checks (after installing all packages)
```bash
cd ~/forks
source .venv/bin/activate

# Python version
python --version
# Expected: Python 3.11.x

# 1. PAPER-QA: location and dev version (fork)
python -c "import paperqa; print(f'paperqa: {paperqa.__file__}')"
# Expected: path under ~/forks/.venv/.../site-packages/paperqa/
pip show paper-qa | grep -E "^(Version|Location)"
# Fork: Version has dev/git hash; Location under your venv

# 2. PAPERQA-NEMOTRON: PyMuPDF fallback
python -c "from paperqa_nemotron.reader import PYMUPDF_AVAILABLE; print(f'PyMuPDF fallback: {PYMUPDF_AVAILABLE}')"

# Expected: True
python -c "import paperqa_nemotron; print(f'paperqa_nemotron: {paperqa_nemotron.__file__}')"
# Expected: path under ~/forks/.venv/.../site-packages/paperqa_nemotron/

# 3. LDP (SimpleAgent, evaluator)
python -c "from ldp.agent import SimpleAgent; from ldp.alg import Evaluator; print('ldp OK')"
# Expected: ldp OK

# 4. LABBENCH2 harness (editable install from your clone)
python -c "import evals.run_evals; print(f'evals: {evals.run_evals.__file__}')"
# Expected: path under ~/forks/labbench2/evals/ (editable install = your fork)
# Runner script path (from labbench2 repo root):
ls ~/forks/labbench2/external_runners/NIM_PQA_runner.py
# Expected: file exists
```

**Quick all-import check** (run from `~/forks` with venv active):

```bash
cd ~/forks
source .venv/bin/activate
python -c "
import paperqa
from paperqa_nemotron.reader import PYMUPDF_AVAILABLE
from ldp.agent import SimpleAgent
from ldp.alg import Evaluator
# Harness (from labbench2 editable install)
import evals.run_evals
print('✅ All imports successful!')
print(f'  PyMuPDF fallback: {PYMUPDF_AVAILABLE}')
"
```

## 2. NIMs

The three NIMs  must be running and reachable:

| NIM            | Default URL                 
|----------------|-----------------------------|
| Parse          | `http://localhost:8002/v1`  |
| Embedding      | `http://localhost:8003/v1` |
| VLM            | `http://localhost:8004/v1` |

```bash
# Move Docker to big disk AND keep nvidia runtime
sudo mkdir -p /ephemeral/docker
sudo tee /etc/docker/daemon.json << 'EOF'
{
  "data-root": "/ephemeral/docker",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF
sudo systemctl restart docker
which nvidia-container-runtime


# Login NGC
docker login -u \$oauthtoken -p YOUR_API_KEY nvcr.io

export NGC_API_KEY=YOUR_API_KEY
export LOCAL_NIM_CACHE=/ephemeral/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE"
chmod -R 777 "$LOCAL_NIM_CACHE"

#------- launch Parse -------------------
# Pin to one GPU (set to 0, 1, 2, ...; use nvidia-smi -L to list GPUs)
export NIM_PARSE_GPU=0
nohup docker run --rm --name parse \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=${NIM_PARSE_GPU} \
  --shm-size=16GB \
  -e NGC_API_KEY \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -e TMPDIR=/opt/nim/.cache \
  -u 0:0 \
  -p 8002:8000 \
  nvcr.io/nim/nvidia/nemotron-parse:latest > parse.log 2>&1 &

tail -f parse.log

#------- launch embedding -------------------
export NIM_EMBEDDING_GPU=1
# llama-3.2-nv-embedqa-1b-v2 (text QA retrieval, long context) https://build.nvidia.com/nvidia/llama-3_2-nv-embedqa-1b-v2/deploy
nohup docker run --rm --name embedding \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=${NIM_EMBEDDING_GPU} \
  --shm-size=16GB \
  -e NGC_API_KEY \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -e TMPDIR=/opt/nim/.cache \
  -u 0:0 \
  -p 8003:8000 \
  nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:latest > embedding.log 2>&1 &

tail -f embedding.log

#-------- launch VLM -------------------
# nemotron-nano-12b-v2-vl (multi-image, video, document VLM) https://build.nvidia.com/nvidia/nemotron-nano-12b-v2-vl/deploy
export NIM_VLM_GPU=2

nohup docker run --rm --name vlm \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=${NIM_VLM_GPU} \
  --shm-size=32GB \
  -e NGC_API_KEY \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -e TMPDIR=/opt/nim/.cache \
  -u 0:0 \
  -p 8004:8000 \
  nvcr.io/nim/nvidia/nemotron-nano-12b-v2-vl:latest > vlm.log 2>&1 &

tail -f vlm.log
```

## 3. Run litqa3 (first two questions)

From the **labbench2** repo root, with the **shared** venv activated (`~/forks/.venv`) and NIMs up. Use `uv run --active` so uv uses that env (where paper-qa etc. are installed); otherwise uv uses `labbench2/.venv` and you get "No module named 'paperqa'".

```bash
cd ~/forks/labbench2
source ../.venv/bin/activate
```

For litqa3 the dataset has no ``files`` column; provide PDFs via **--files-dir** (see below) or the harness will have no files for those questions.

**Grading:** The labbench2 harness grades answers after the runner returns. For litqa3, it uses an **LLM judge** (`LLMJudgeEvaluator`): the judge compares the submitted answer to the expected answer (question ideal) and returns correct/incorrect/unsure. The default judge model is `anthropic:claude-sonnet-4-5`. This runner does not perform grading; the harness does.

**Using your VLM as the judge (no Anthropic key needed):** Set the judge model and OpenAI-compatible env vars, then run the standard harness command:

```bash
cd ~/forks/labbench2
export HF_TOKEN=YOUR_HF_TOKEN

export OPENAI_API_BASE="http://localhost:8004/v1"
export OPENAI_API_KEY="dummy"
# Judge model must be provider:model (colon). Slash-only form causes pydantic_ai "Unknown model" / unpack error.
export LABBENCH2_JUDGE_MODEL="openai:nvidia/nemotron-nano-12b-v2-vl"
```

Or pass the judge model on the command line:

```bash
export OPENAI_API_BASE="http://localhost:8004/v1"
export OPENAI_API_KEY="dummy"
uv run --active python -m evals.run_evals --agent "external:./external_runners/NIM_PQA_runner.py:NIMPQARunner" --tag litqa3 --limit 2 --judge-model "openai:nvidia/nemotron-nano-12b-v2-vl"
```

**Using your own PDF directory:** By default the harness uses files from the dataset (GCS) when the question has a ``files`` column. For litqa3 (no ``files`` column) or to override with your own PDFs, pass `--files-dir` with a directory containing the PDFs (requires `--mode file`, which is the default). I downloaded two PDFs using the DOIs listed in the rows of the first two questions in [litqa3 HuggingFace Dataset](https://huggingface.co/datasets/futurehouse/labbench2/viewer/litqa3).

```bash
uv run --active python -m evals.run_evals --agent "external:./external_runners/NIM_PQA_runner.py:NIMPQARunner" --tag litqa3 --limit 2 --parallel 1 --files-dir ~/pdfs
```

The harness passes that directory as `inputs["files_path"]` to the runner; `create_agent_runner_task` lists files in that dir and passes them to the agent. So put your PDFs in a folder and pass that folder’s path to `--files-dir`.

**Trajectory logging:** Set `LABBENCH2_PRINT_TRAJECTORIES=1` to print step-by-step trajectory details and to **save a Jupyter notebook per question**. Notebooks are written under `LABBENCH2_TRAJECTORY_DIR` (default: `labbench2_trajectories/`) as `trajectory_0.ipynb`, `trajectory_1.ipynb`, etc. Open them in Jupyter Lab (e.g. on a Brev instance) to visualize context pairs (raw chunk + summary) and embedded images/media. Example:

```bash
export LABBENCH2_PRINT_TRAJECTORIES=1
export LABBENCH2_TRAJECTORY_DIR=./litqa3_trajectories   # optional; default is this
uv run --active python -m evals.run_evals --agent "external:./external_runners/NIM_PQA_runner.py:NIMPQARunner" --tag litqa3 --limit 2 --parallel 1 --files-dir ~/pdfs
# Then open labbench2_trajectories/trajectory_0.ipynb (and trajectory_1.ipynb) in Jupyter Lab.
```


**Using Azure endpoints instead of locally-hosted NIMs for all LLM roles in PaperQA:**
Use `NIM_PQA_runner_Azure.py` instead of `NIM_PQA_runner.py`.
```bash
export AZURE_BEARER_TOKEN=YOUR_API_KEY
export LABBENCH2_JUDGE_MODEL="openai:gpt-4o"

uv run --active python -m evals.run_evals --agent "external:./external_runners/NIM_PQA_runner_Azure.py:NIMAzurePQARunner" --tag litqa3 --limit 2 --parallel 1 --files-dir ~/pdfs
```
