import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urljoin, urlparse

import httpx
from filelock import FileLock
from pydantic_ai import BinaryContent

CACHE_DIR = Path.home() / ".cache" / "labbench2"
LOCKS_DIR = CACHE_DIR / ".locks"

GCS_BUCKET = "labbench2-data-public"
GCS_VALIDATOR_FILES_PREFIX = "validation"
GCS_API_URL = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
GCS_DOWNLOAD_URL = "https://storage.googleapis.com/{bucket}/{path}"

MEDIA_TYPES = {
    # Sequence formats
    ".gbff": "text/plain",
    ".gbk": "text/plain",
    ".gb": "text/plain",
    ".fasta": "text/plain",
    ".fa": "text/plain",
    ".fna": "text/plain",
    ".ffn": "text/plain",
    ".faa": "text/plain",
    ".txt": "text/plain",
    # Structured data
    ".json": "text/plain",  # application/json not supported by Vertex AI
    ".xml": "application/xml",
    ".csv": "text/plain",  # text/csv not supported by Anthropic document API
    # Documents
    ".pdf": "application/pdf",
    # Images
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

TEXT_EXTENSIONS = {
    ext
    for ext, mime in MEDIA_TYPES.items()
    if mime.startswith("text/") or mime == "application/xml"
}


def extract_question_from_inputs(inputs: Any) -> str:
    """Extract question text from various input formats.

    Handles three input formats:
    - dict: External agent mode with "question" key
    - list: File mode with question as first element
    - str: Simple text mode with question as string
    """
    if isinstance(inputs, dict):
        return inputs.get("question", "")
    elif isinstance(inputs, list):
        return inputs[0] if inputs else ""
    return str(inputs)


def get_media_type(extension: str) -> str:
    """Get MIME type for file extension."""
    return MEDIA_TYPES.get(extension.lower(), "application/octet-stream")


def load_file_as_binary_content(file_path: Path | str) -> BinaryContent:
    """Load a file as BinaryContent for Pydantic AI."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return BinaryContent(data=file_path.read_bytes(), media_type=get_media_type(file_path.suffix))


def is_text_injectable_format(file_path: Path) -> bool:
    """Check if file is a text-based format (FASTA, GenBank, etc.)."""
    return file_path.suffix.lower() in TEXT_EXTENSIONS


def _list_gcs_objects(bucket_name: str, prefix: str) -> list[str]:
    """List objects in a public GCS bucket."""
    objects = []
    page_token = None

    while True:
        params = {"prefix": prefix}
        if page_token:
            params["pageToken"] = page_token

        url = GCS_API_URL.format(bucket=bucket_name)
        response = httpx.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        for item in data.get("items", []):
            objects.append(item["name"])

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return objects


def _download_blobs(bucket_name: str, gcs_prefix: str, dest_dir: Path) -> None:
    """Download blobs from a public GCS bucket to a local directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    prefix = gcs_prefix.strip("/") + "/" if gcs_prefix.strip("/") else ""

    for blob_name in _list_gcs_objects(bucket_name, prefix):
        # skip empty directories
        if blob_name.endswith("/"):
            continue

        # get relative path, skip if empty
        relative_path = blob_name[len(prefix) :]
        if not relative_path:
            continue

        # skip if destination path already exists
        dest_path = dest_dir / relative_path
        if dest_path.exists():
            continue

        # Atomic download: temp file + rename
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=dest_path.parent,
            prefix=f".{dest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            temp_path = Path(tmp.name)

        try:
            url = GCS_DOWNLOAD_URL.format(bucket=bucket_name, path=blob_name)
            with httpx.stream("GET", url, timeout=60) as response:
                response.raise_for_status()
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
            temp_path.replace(dest_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()


def download_question_files(bucket_name: str, gcs_prefix: str) -> Path:
    """Download files from GCS with caching."""
    dest_dir = CACHE_DIR / bucket_name / gcs_prefix.strip("/")

    # Use a lock file to ensure only one process downloads at a time
    LOCKS_DIR.mkdir(parents=True, exist_ok=True)
    lock_name = gcs_prefix.strip("/").replace("/", "_") + ".lock"
    lock_path = LOCKS_DIR / lock_name

    with FileLock(lock_path, timeout=300):
        _download_blobs(bucket_name, gcs_prefix, dest_dir)

    return dest_dir


SOURCES_CACHE_SUBDIR = "sources"
PDF_LINK_PATTERN = re.compile(
    r'href\s*=\s*["\']([^"\']*\.pdf[^"\']*)["\']',
    re.IGNORECASE,
)
PDF_PATH_PATTERN = re.compile(r'["\']([^"\']*(?:/pdf/|\.pdf)[^"\']*)["\']', re.IGNORECASE)


def _resolve_pdf_url_from_source(source_url: str) -> str | None:
    """Resolve a DOI or landing-page URL to a direct PDF URL. Returns None if not found."""
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        # Try content negotiation: some servers return PDF with Accept: application/pdf
        try:
            r = client.get(
                source_url,
                headers={"Accept": "application/pdf"},
            )
            r.raise_for_status()
            ct = r.headers.get("content-type", "").split(";")[0].strip().lower()
            if "application/pdf" in ct:
                return str(r.url)
            # If we got a redirect to a URL that looks like a PDF, use it
            if r.url and (".pdf" in str(r.url).lower() or "/pdf/" in str(r.url)):
                return str(r.url)
        except Exception:
            pass

        # Heuristic: many journal sites use /doi/pdf/... for the same DOI path.
        # Resolve redirects first (doi.org -> journals.asm.org etc.)
        try:
            r = client.get(source_url)
            r.raise_for_status()
            landing_url = str(r.url)
            parsed = urlparse(landing_url)
            path = parsed.path.rstrip("/")
            if "/doi/" in path and "/pdf/" not in path:
                pdf_path = path.replace("/doi/", "/doi/pdf/", 1)
                pdf_url = f"{parsed.scheme}://{parsed.netloc}{pdf_path}"
                head = client.head(pdf_url, follow_redirects=True)
                if head.status_code == 200:
                    ct = head.headers.get("content-type", "").split(";")[0].strip().lower()
                    if "application/pdf" in ct:
                        return pdf_url
        except Exception:
            pass

        # Fetch as HTML and look for PDF links
        try:
            r = client.get(source_url)
            r.raise_for_status()
            base = str(r.url)
            text = r.text
        except Exception:
            return None

        # Prefer explicit .pdf hrefs, then any link containing /pdf/ or .pdf
        for pattern in (PDF_LINK_PATTERN, PDF_PATH_PATTERN):
            for match in pattern.finditer(text):
                raw = match.group(1).strip()
                if not raw or raw.startswith("#"):
                    continue
                url = urljoin(base, raw)
                if urlparse(url).scheme not in ("http", "https"):
                    continue
                # Quick check: HEAD request to see if it's a PDF
                try:
                    head = client.head(url, follow_redirects=True)
                    if head.status_code == 200:
                        ct = head.headers.get("content-type", "").split(";")[0].strip().lower()
                        if "application/pdf" in ct:
                            return url
                except Exception:
                    continue
    return None


def _download_pdf_to_path(client: httpx.Client, pdf_url: str, dest_path: Path) -> None:
    """Download a PDF from pdf_url to dest_path (overwrites if exists)."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with client.stream("GET", pdf_url, follow_redirects=True) as response:
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)


def download_pdfs_from_sources(sources: list[str], cache_key: str) -> Path:
    """Download PDF(s) from source URLs (e.g. DOIs) with caching.

    When a question has no ``files`` but has ``sources`` (e.g. DOI links), this
    resolves each source to a PDF URL, downloads to the cache, and returns the
    directory path. Used as a fallback for litqa3 and similar tags where the
    dataset does not populate the ``files`` column.

    Args:
        sources: List of URLs (e.g. https://doi.org/10.1234/xyz).
        cache_key: Subpath under the sources cache (e.g. ``litqa3/<question_id>``).

    Returns:
        Path to a directory containing the downloaded PDF(s). File names are
        derived from the URL (e.g. paper.pdf, paper_1.pdf for multiple).

    Raises:
        RuntimeError: If no PDF could be resolved or downloaded for any source.
    """
    safe_key = cache_key.strip("/").replace("/", "_")
    dest_dir = CACHE_DIR / SOURCES_CACHE_SUBDIR / safe_key
    if dest_dir.exists() and any(dest_dir.glob("*.pdf")):
        return dest_dir

    LOCKS_DIR.mkdir(parents=True, exist_ok=True)
    lock_name = f"sources_{safe_key}.lock"
    lock_path = LOCKS_DIR / lock_name
    with FileLock(lock_path, timeout=120):
        dest_dir.mkdir(parents=True, exist_ok=True)
        if any(dest_dir.glob("*.pdf")):
            return dest_dir
        with httpx.Client(follow_redirects=True, timeout=60) as client:
            downloaded = 0
            for i, url in enumerate(sources):
                if not url or not url.strip():
                    continue
                url = url.strip()
                pdf_url = _resolve_pdf_url_from_source(url)
                if not pdf_url:
                    continue
                stem = hashlib.sha256(pdf_url.encode()).hexdigest()[:12]
                name = f"paper_{i}.pdf" if len(sources) > 1 else "paper.pdf"
                dest_path = dest_dir / name
                try:
                    _download_pdf_to_path(client, pdf_url, dest_path)
                    if dest_path.exists() and dest_path.stat().st_size > 0:
                        downloaded += 1
                except Exception:
                    if dest_path.exists():
                        dest_path.unlink(missing_ok=True)
            if downloaded == 0:
                raise RuntimeError(
                    f"Could not resolve or download any PDF from sources: {sources}"
                )
    return dest_dir


def resolve_file_path(filename: str, question_files_path: Path | None) -> Path | None:
    """Resolve a file path by checking question directory first, then validators directory."""
    # First, check in the question's files directory
    if question_files_path:
        question_path = question_files_path / filename
        if question_path.exists():
            return question_path

    # Fall back to validator files directory
    validator_path = CACHE_DIR / GCS_BUCKET / GCS_VALIDATOR_FILES_PREFIX / filename
    if validator_path.exists():
        return validator_path

    # If not cached, try downloading validator files
    validator_dir = download_question_files(GCS_BUCKET, GCS_VALIDATOR_FILES_PREFIX)
    validator_path = validator_dir / filename
    if validator_path.exists():
        return validator_path

    return None


class GoogleVertexConfig(NamedTuple):
    project: str
    location: str


def setup_google_vertex_env(require_location: bool = True) -> GoogleVertexConfig | None:
    """Setup environment for Google Vertex AI authentication."""
    # Remove API keys so Google SDK uses OAuth2/ADC instead
    os.environ.pop("GOOGLE_API_KEY", None)
    os.environ.pop("GEMINI_API_KEY", None)

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION")

    if not project:
        if require_location:
            raise ValueError(
                "Vertex AI requires GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables"
            )
        return None

    if require_location and not location:
        raise ValueError(
            "Vertex AI requires GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables"
        )

    return GoogleVertexConfig(project=project, location=location or "")
