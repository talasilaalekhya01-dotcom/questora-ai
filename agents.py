import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from pypdf import PdfReader

from crewai import Agent, Crew, LLM, Process, Task


# ============================================================
# QUESTORA AI - COMPLETE RESILIENT AGENTS BACKEND
# ============================================================
#
# This file preserves the complete Questora research workflow:
#
# 1. Research topic input
# 2. OpenAlex paper retrieval
# 3. Crossref fallback when OpenAlex is unavailable
# 4. Topic-specific verified local cache fallback
# 5. Open-access PDF download attempts
# 6. PyPDF full-text extraction
# 7. Four CrewAI agents
# 8. Gemini primary model + optional fallback model
# 9. 429 / 503 / timeout retry handling
# 10. Deterministic verified fallback report when Gemini is unavailable
# 11. Strict no-fabrication source rules
# 12. Same dictionary contract expected by app.py
#
# IMPORTANT:
# This backend NEVER invents paper titles, authors, DOI numbers,
# citation counts, publication years, journals, or full-text findings.
# If live services fail and no verified cache exists, Questora returns
# a transparent report saying that verified papers were unavailable.
# ============================================================


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "").strip()
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "").strip()

# Keep the model already used in the Questora project.
PRIMARY_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini/gemini-3.6-flash",
).strip()

# Optional. Leave empty unless you have confirmed another Gemini model.
FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL",
    "",
).strip()

OPENALEX_URL = "https://api.openalex.org/works"
CROSSREF_URL = "https://api.crossref.org/works"

USER_AGENT = "QuestoraAI/1.0 Academic Research Assistant"
if CROSSREF_MAILTO:
    USER_AGENT += f" mailto:{CROSSREF_MAILTO}"

OPENALEX_TIMEOUT = 60
CROSSREF_TIMEOUT = 60
PDF_TIMEOUT = 45

OPENALEX_MAX_ATTEMPTS = 4
CROSSREF_MAX_ATTEMPTS = 3
PDF_MAX_ATTEMPTS = 2

MAX_PAPERS = 10
MAX_FULL_TEXT_PDFS = 3
MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 30
MAX_CHARS_PER_PDF = 22000

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / ".questora_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ============================================================
# GENERAL HELPERS
# ============================================================


def _safe_text(value: Any, default: str = "") -> str:
    """Return a stripped string without throwing on None."""

    if value is None:
        return default

    return str(value).strip()



def _clean_doi(value: Any) -> str:
    """Normalize DOI values from URLs to plain DOI strings."""

    doi = _safe_text(value)

    if not doi:
        return ""

    doi = re.sub(
        r"^https?://(?:dx\.)?doi\.org/",
        "",
        doi,
        flags=re.I,
    )

    return doi.strip()



def _normalize_title(value: Any) -> str:
    """Normalize a paper title for duplicate detection."""

    text = _safe_text(value).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text.strip()



def _retry_after_seconds(
    response: requests.Response,
    fallback: int = 40,
) -> int:
    """Read Retry-After from headers or a JSON body."""

    header_value = response.headers.get("Retry-After")

    if header_value:
        try:
            return max(
                1,
                min(
                    120,
                    int(float(header_value)),
                ),
            )
        except (TypeError, ValueError):
            pass

    try:
        body = response.json()
        body_value = body.get("retry_after")

        if body_value is not None:
            return max(
                1,
                min(
                    120,
                    int(float(body_value)),
                ),
            )
    except Exception:
        pass

    return fallback



def _is_daily_quota_error(error: Exception) -> bool:
    """Identify Gemini daily quota errors that should not be rapidly retried."""

    text = str(error).lower()

    if not any(
        marker in text
        for marker in [
            "429",
            "quota",
            "resource_exhausted",
            "resource exhausted",
        ]
    ):
        return False

    daily_markers = [
        "perday",
        "per day",
        "requestsperday",
        "free_tier_requests",
        "free tier requests",
        "daily quota",
        "generate requests per day",
    ]

    return any(
        marker in text
        for marker in daily_markers
    )



def _is_temporary_error(error: Exception) -> bool:
    """Identify transient network/provider errors that are safe to retry."""

    if _is_daily_quota_error(error):
        return False

    text = str(error).lower()

    markers = [
        "429",
        "500",
        "502",
        "503",
        "504",
        "rate limit",
        "resource exhausted",
        "resource_exhausted",
        "service unavailable",
        "temporarily unavailable",
        "unavailable",
        "high demand",
        "overloaded",
        "timeout",
        "timed out",
        "connection reset",
        "connection error",
        "server error",
        "name resolution",
        "getaddrinfo",
        "temporary failure in name resolution",
    ]

    return any(
        marker in text
        for marker in markers
    )



def _retry_seconds_from_error(
    error: Exception,
    fallback: int,
) -> int:
    """Try to read a provider-specified retry delay from an exception."""

    text = str(error)

    patterns = [
        r"retry(?:ing)?\s+(?:in|after)\s+([0-9]+(?:\.[0-9]+)?)\s*s",
        r"retry(?:ing)?\s+(?:in|after)\s+([0-9]+(?:\.[0-9]+)?)\s*seconds?",
        r"retryDelay[^0-9]*([0-9]+(?:\.[0-9]+)?)s",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.I,
        )

        if match:
            try:
                return max(
                    5,
                    min(
                        120,
                        int(float(match.group(1))),
                    ),
                )
            except (TypeError, ValueError):
                pass

    return fallback


# ============================================================
# LOCAL VERIFIED CACHE
# ============================================================


def _topic_cache_path(topic: str) -> Path:
    """Build a stable cache filename for a topic."""

    normalized = re.sub(
        r"\s+",
        " ",
        topic.strip().lower(),
    )

    digest = hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()[:20]

    return CACHE_DIR / f"{digest}.json"



def _save_verified_cache(
    topic: str,
    papers: List[Dict[str, Any]],
    provider: str,
) -> None:
    """Save only metadata that was returned by a real scholarly API."""

    if not papers:
        return

    safe_papers = []

    for paper in papers:
        copy = dict(paper)

        # Do not persist extracted PDF text because it can make the cache
        # very large and stale. The PDF can be re-read on a new run.
        copy["full_text_read"] = False
        copy["full_text"] = ""
        copy["full_text_source_url"] = None
        copy["pages_extracted"] = 0
        copy["characters_extracted"] = 0
        copy["pdf_error"] = None

        safe_papers.append(copy)

    payload = {
        "topic": topic,
        "provider": provider,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "papers": safe_papers,
    }

    path = _topic_cache_path(topic)

    try:
        path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as error:
        print(
            "[Questora] Could not write metadata cache:",
            repr(error),
        )



def _load_verified_cache(
    topic: str,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Load previously verified metadata for the exact same topic."""

    path = _topic_cache_path(topic)

    if not path.exists():
        return [], None

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )

        cached_topic = _safe_text(
            payload.get("topic")
        )

        if cached_topic.lower() != topic.strip().lower():
            return [], None

        papers = payload.get("papers") or []

        if not isinstance(papers, list):
            return [], None

        provider = _safe_text(
            payload.get("provider"),
            "cache",
        )

        output = []

        for paper in papers:
            if not isinstance(paper, dict):
                continue

            copy = dict(paper)
            copy["full_text_read"] = False
            copy["full_text"] = ""
            copy["full_text_source_url"] = None
            copy["pages_extracted"] = 0
            copy["characters_extracted"] = 0
            copy["pdf_error"] = None
            output.append(copy)

        return output, provider

    except Exception as error:
        print(
            "[Questora] Could not read metadata cache:",
            repr(error),
        )
        return [], None


# ============================================================
# PAPER STRUCTURE + DEDUPLICATION
# ============================================================


def _blank_paper() -> Dict[str, Any]:
    """Create the exact metadata structure expected by app.py."""

    return {
        "title": "Untitled",
        "authors": [],
        "year": None,
        "doi": "",
        "citations": 0,
        "source": "Unknown",
        "openalex_id": None,
        "openalex_url": None,
        "landing_page_url": None,
        "pdf_url": None,
        "oa_url": None,
        "is_open_access": False,
        "full_text_read": False,
        "full_text": "",
        "full_text_source_url": None,
        "pages_extracted": 0,
        "characters_extracted": 0,
        "pdf_error": None,
        "metadata_provider": "Unknown",
    }



def _deduplicate_papers(
    papers: List[Dict[str, Any]],
    limit: int = MAX_PAPERS,
) -> List[Dict[str, Any]]:
    """Remove duplicate results using DOI first and normalized title second."""

    seen_dois = set()
    seen_titles = set()
    output = []

    for paper in papers:
        doi = _clean_doi(
            paper.get("doi")
        ).lower()

        title_key = _normalize_title(
            paper.get("title")
        )

        if doi and doi in seen_dois:
            continue

        if title_key and title_key in seen_titles:
            continue

        if doi:
            seen_dois.add(doi)

        if title_key:
            seen_titles.add(title_key)

        output.append(paper)

        if len(output) >= limit:
            break

    return output


# ============================================================
# OPENALEX REQUEST WITH API KEY + ANONYMOUS MODE + RETRIES
# ============================================================


def _openalex_get(
    params: Dict[str, Any],
) -> Dict[str, Any]:
    request_params = dict(params)

    if OPENALEX_API_KEY:
        request_params["api_key"] = OPENALEX_API_KEY
    else:
        print(
            "\n[Questora] OPENALEX_API_KEY is not set. "
            "Using anonymous OpenAlex access."
        )

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }

    last_error: Optional[Exception] = None

    for attempt in range(
        1,
        OPENALEX_MAX_ATTEMPTS + 1,
    ):
        try:
            print(
                f"\n[Questora] OpenAlex request "
                f"{attempt}/{OPENALEX_MAX_ATTEMPTS}..."
            )

            response = requests.get(
                OPENALEX_URL,
                params=request_params,
                headers=headers,
                timeout=OPENALEX_TIMEOUT,
            )

            if response.status_code == 200:
                return response.json()

            if response.status_code == 429:
                wait_seconds = _retry_after_seconds(
                    response,
                    fallback=min(
                        40,
                        8 * attempt,
                    ),
                )

                print(
                    "[Questora] OpenAlex rate limit reached. "
                    f"Retrying in {wait_seconds} seconds..."
                )

                if attempt < OPENALEX_MAX_ATTEMPTS:
                    time.sleep(wait_seconds)
                    continue

                raise RuntimeError(
                    "OpenAlex rate limit was reached even after retries."
                )

            if response.status_code in {
                500,
                502,
                503,
                504,
            }:
                wait_seconds = min(
                    30,
                    4 * attempt,
                )

                print(
                    "[Questora] OpenAlex temporary server error "
                    f"({response.status_code}). "
                    f"Retrying in {wait_seconds} seconds..."
                )

                if attempt < OPENALEX_MAX_ATTEMPTS:
                    time.sleep(wait_seconds)
                    continue

            if response.status_code in {
                401,
                403,
            }:
                raise RuntimeError(
                    "OpenAlex rejected the configured API key."
                )

            body_preview = response.text[:500]

            raise RuntimeError(
                f"OpenAlex returned HTTP "
                f"{response.status_code}: {body_preview}"
            )

        except requests.Timeout as error:
            last_error = error

            if attempt < OPENALEX_MAX_ATTEMPTS:
                wait_seconds = min(
                    20,
                    3 * attempt,
                )

                print(
                    "[Questora] OpenAlex timed out. "
                    f"Retrying in {wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)
                continue

        except requests.ConnectionError as error:
            last_error = error

            if attempt < OPENALEX_MAX_ATTEMPTS:
                wait_seconds = min(
                    20,
                    3 * attempt,
                )

                print(
                    "[Questora] OpenAlex connection error. "
                    f"Retrying in {wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)
                continue

        except requests.RequestException as error:
            last_error = error
            break

    if last_error is not None:
        print(
            "[Questora] OpenAlex technical error:",
            repr(last_error),
        )

        raise RuntimeError(
            "Could not connect to OpenAlex after multiple attempts."
        ) from last_error

    raise RuntimeError(
        "OpenAlex request failed."
    )


# ============================================================
# OPENALEX SEARCH
# ============================================================


def search_openalex(
    topic: str,
    limit: int = MAX_PAPERS,
) -> List[Dict[str, Any]]:
    topic = _safe_text(topic)

    if not topic:
        raise ValueError(
            "Research topic cannot be empty."
        )

    data = _openalex_get(
        {
            "search": topic,
            "per-page": max(
                1,
                min(
                    int(limit),
                    25,
                ),
            ),
            "sort": "relevance_score:desc",
        }
    )

    papers: List[Dict[str, Any]] = []

    for item in data.get("results", []) or []:
        if not isinstance(item, dict):
            continue

        paper = _blank_paper()

        authors = []

        for authorship in item.get(
            "authorships",
            [],
        ) or []:
            if not isinstance(authorship, dict):
                continue

            author_info = authorship.get(
                "author",
                {},
            ) or {}

            name = _safe_text(
                author_info.get("display_name")
            )

            if name:
                authors.append(name)

        primary_location = (
            item.get("primary_location")
            or {}
        )

        primary_source = (
            primary_location.get("source")
            or {}
        )

        best_oa_location = (
            item.get("best_oa_location")
            or {}
        )

        open_access = (
            item.get("open_access")
            or {}
        )

        pdf_url = (
            best_oa_location.get("pdf_url")
            or primary_location.get("pdf_url")
            or None
        )

        oa_url = (
            best_oa_location.get(
                "landing_page_url"
            )
            or open_access.get("oa_url")
            or None
        )

        landing_page_url = (
            primary_location.get(
                "landing_page_url"
            )
            or oa_url
            or item.get("doi")
            or item.get("id")
        )

        source_name = (
            primary_source.get("display_name")
            or "Unknown"
        )

        raw_doi = item.get("doi")
        doi = _clean_doi(raw_doi)

        paper.update(
            {
                "title": (
                    item.get("display_name")
                    or "Untitled"
                ),
                "authors": authors,
                "year": item.get(
                    "publication_year"
                ),
                "doi": doi,
                "citations": (
                    item.get(
                        "cited_by_count",
                        0,
                    )
                    or 0
                ),
                "source": source_name,
                "openalex_id": item.get("id"),
                "openalex_url": item.get("id"),
                "landing_page_url": landing_page_url,
                "pdf_url": pdf_url,
                "oa_url": oa_url,
                "is_open_access": bool(
                    open_access.get(
                        "is_oa",
                        False,
                    )
                ),
                "metadata_provider": "OpenAlex",
            }
        )

        papers.append(paper)

    return _deduplicate_papers(
        papers,
        limit=limit,
    )


# ============================================================
# CROSSREF FALLBACK - NO API KEY REQUIRED
# ============================================================


def _crossref_year(
    item: Dict[str, Any],
) -> Optional[int]:
    """Extract the most useful publication year from Crossref metadata."""

    date_fields = [
        "published-print",
        "published-online",
        "published",
        "issued",
        "created",
    ]

    for field in date_fields:
        value = item.get(field)

        if not isinstance(value, dict):
            continue

        date_parts = value.get("date-parts")

        if (
            isinstance(date_parts, list)
            and date_parts
            and isinstance(date_parts[0], list)
            and date_parts[0]
        ):
            try:
                return int(date_parts[0][0])
            except (TypeError, ValueError):
                pass

    return None



def _crossref_pdf_url(
    item: Dict[str, Any],
) -> Optional[str]:
    """Return a direct PDF URL only when Crossref explicitly labels it PDF."""

    links = item.get("link") or []

    for link in links:
        if not isinstance(link, dict):
            continue

        content_type = _safe_text(
            link.get("content-type")
        ).lower()

        url = _safe_text(
            link.get("URL")
        )

        if url and "pdf" in content_type:
            return url

    return None



def search_crossref(
    topic: str,
    limit: int = MAX_PAPERS,
) -> List[Dict[str, Any]]:
    """Search Crossref as a keyless scholarly metadata fallback."""

    topic = _safe_text(topic)

    if not topic:
        raise ValueError(
            "Research topic cannot be empty."
        )

    params = {
        "query.bibliographic": topic,
        "rows": max(
            1,
            min(
                int(limit),
                20,
            ),
        ),
    }

    if CROSSREF_MAILTO:
        params["mailto"] = CROSSREF_MAILTO

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }

    last_error: Optional[Exception] = None

    for attempt in range(
        1,
        CROSSREF_MAX_ATTEMPTS + 1,
    ):
        try:
            print(
                f"\n[Questora] Crossref request "
                f"{attempt}/{CROSSREF_MAX_ATTEMPTS}..."
            )

            response = requests.get(
                CROSSREF_URL,
                params=params,
                headers=headers,
                timeout=CROSSREF_TIMEOUT,
            )

            if response.status_code == 200:
                body = response.json()

                items = (
                    body.get("message", {})
                    .get("items", [])
                    or []
                )

                papers: List[Dict[str, Any]] = []

                for item in items:
                    if not isinstance(item, dict):
                        continue

                    paper = _blank_paper()

                    raw_titles = (
                        item.get("title")
                        or []
                    )

                    title = (
                        _safe_text(raw_titles[0])
                        if raw_titles
                        else "Untitled"
                    )

                    authors = []

                    for author in (
                        item.get("author")
                        or []
                    ):
                        if not isinstance(author, dict):
                            continue

                        given = _safe_text(
                            author.get("given")
                        )

                        family = _safe_text(
                            author.get("family")
                        )

                        full_name = " ".join(
                            part
                            for part in [
                                given,
                                family,
                            ]
                            if part
                        ).strip()

                        if full_name:
                            authors.append(full_name)

                    raw_container = (
                        item.get("container-title")
                        or []
                    )

                    source = (
                        _safe_text(raw_container[0])
                        if raw_container
                        else "Unknown"
                    )

                    doi = _clean_doi(
                        item.get("DOI")
                    )

                    landing_page_url = (
                        _safe_text(item.get("URL"))
                        or (
                            f"https://doi.org/{doi}"
                            if doi
                            else None
                        )
                    )

                    pdf_url = _crossref_pdf_url(
                        item
                    )

                    paper.update(
                        {
                            "title": title,
                            "authors": authors,
                            "year": _crossref_year(
                                item
                            ),
                            "doi": doi,
                            "citations": (
                                item.get(
                                    "is-referenced-by-count",
                                    0,
                                )
                                or 0
                            ),
                            "source": source,
                            "openalex_id": None,
                            "openalex_url": None,
                            "landing_page_url": landing_page_url,
                            "pdf_url": pdf_url,
                            "oa_url": landing_page_url,
                            "is_open_access": bool(
                                pdf_url
                            ),
                            "metadata_provider": "Crossref",
                        }
                    )

                    papers.append(paper)

                return _deduplicate_papers(
                    papers,
                    limit=limit,
                )

            if response.status_code == 429:
                wait_seconds = _retry_after_seconds(
                    response,
                    fallback=min(
                        20,
                        4 * attempt,
                    ),
                )

                if attempt < CROSSREF_MAX_ATTEMPTS:
                    print(
                        "[Questora] Crossref rate limited. "
                        f"Retrying in {wait_seconds} seconds..."
                    )
                    time.sleep(wait_seconds)
                    continue

            if response.status_code in {
                500,
                502,
                503,
                504,
            }:
                if attempt < CROSSREF_MAX_ATTEMPTS:
                    wait_seconds = min(
                        15,
                        3 * attempt,
                    )

                    print(
                        "[Questora] Crossref temporary server error. "
                        f"Retrying in {wait_seconds} seconds..."
                    )

                    time.sleep(wait_seconds)
                    continue

            raise RuntimeError(
                f"Crossref returned HTTP "
                f"{response.status_code}: "
                f"{response.text[:400]}"
            )

        except requests.Timeout as error:
            last_error = error

            if attempt < CROSSREF_MAX_ATTEMPTS:
                wait_seconds = min(
                    15,
                    3 * attempt,
                )
                print(
                    "[Questora] Crossref timed out. "
                    f"Retrying in {wait_seconds} seconds..."
                )
                time.sleep(wait_seconds)
                continue

        except requests.ConnectionError as error:
            last_error = error

            if attempt < CROSSREF_MAX_ATTEMPTS:
                wait_seconds = min(
                    15,
                    3 * attempt,
                )
                print(
                    "[Questora] Crossref connection error. "
                    f"Retrying in {wait_seconds} seconds..."
                )
                time.sleep(wait_seconds)
                continue

        except requests.RequestException as error:
            last_error = error
            break

    if last_error is not None:
        raise RuntimeError(
            "Could not connect to Crossref after multiple attempts."
        ) from last_error

    raise RuntimeError(
        "Crossref search failed."
    )


# ============================================================
# RESILIENT METADATA RETRIEVAL
# ============================================================


def retrieve_verified_papers(
    topic: str,
    limit: int = MAX_PAPERS,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Retrieve paper metadata using real scholarly sources only.

    Priority:
        1. OpenAlex
        2. Crossref
        3. Topic-specific local cache of previously verified data

    Returns:
        (papers, provider_label)
    """

    openalex_error: Optional[Exception] = None
    crossref_error: Optional[Exception] = None

    try:
        print(
            "\n[Questora] Trying OpenAlex first..."
        )

        papers = search_openalex(
            topic=topic,
            limit=limit,
        )

        if papers:
            _save_verified_cache(
                topic=topic,
                papers=papers,
                provider="OpenAlex",
            )

            return papers, "OpenAlex"

    except Exception as error:
        openalex_error = error

        print(
            "[Questora] OpenAlex unavailable:",
            repr(error),
        )

    try:
        print(
            "\n[Questora] Falling back to Crossref..."
        )

        papers = search_crossref(
            topic=topic,
            limit=limit,
        )

        if papers:
            _save_verified_cache(
                topic=topic,
                papers=papers,
                provider="Crossref",
            )

            return papers, "Crossref"

    except Exception as error:
        crossref_error = error

        print(
            "[Questora] Crossref unavailable:",
            repr(error),
        )

    cached_papers, cached_provider = (
        _load_verified_cache(topic)
    )

    if cached_papers:
        print(
            "\n[Questora] Using previously verified "
            "topic-specific metadata cache."
        )

        return (
            cached_papers[:limit],
            f"Local cache ({cached_provider})",
        )

    print(
        "\n[Questora] No verified scholarly source or cache "
        "is currently available."
    )

    if openalex_error is not None:
        print(
            "[Questora] Last OpenAlex error:",
            repr(openalex_error),
        )

    if crossref_error is not None:
        print(
            "[Questora] Last Crossref error:",
            repr(crossref_error),
        )

    return [], "No verified paper source available"


# ============================================================
# PDF DOWNLOAD + EXTRACTION
# ============================================================


def _download_pdf_bytes(
    url: str,
) -> bytes:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": (
            "application/pdf,"
            "text/html;q=0.8,"
            "*/*;q=0.5"
        ),
    }

    last_error: Optional[Exception] = None

    for attempt in range(
        1,
        PDF_MAX_ATTEMPTS + 1,
    ):
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=PDF_TIMEOUT,
                allow_redirects=True,
            )

            if response.status_code == 429:
                wait_seconds = _retry_after_seconds(
                    response,
                    fallback=12,
                )

                if attempt < PDF_MAX_ATTEMPTS:
                    time.sleep(wait_seconds)
                    continue

            response.raise_for_status()

            content = response.content

            if not content:
                raise RuntimeError(
                    "Downloaded PDF was empty."
                )

            if len(content) > MAX_PDF_BYTES:
                raise RuntimeError(
                    "PDF is larger than the "
                    f"{MAX_PDF_BYTES // (1024 * 1024)} MB limit."
                )

            content_type = response.headers.get(
                "Content-Type",
                "",
            ).lower()

            looks_like_pdf = (
                content[:5] == b"%PDF-"
            )

            if (
                not looks_like_pdf
                and "pdf" not in content_type
            ):
                raise RuntimeError(
                    "The supplied full-text URL did not return a PDF file."
                )

            return content

        except Exception as error:
            last_error = error

            if attempt < PDF_MAX_ATTEMPTS:
                time.sleep(2 * attempt)
                continue

    raise RuntimeError(
        str(
            last_error
            or "PDF download failed."
        )
    )



def _extract_pdf_text(
    pdf_bytes: bytes,
) -> Tuple[str, int]:
    reader = PdfReader(
        BytesIO(pdf_bytes)
    )

    chunks: List[str] = []
    pages_used = 0
    total_chars = 0

    for page in reader.pages[
        :MAX_PDF_PAGES
    ]:
        try:
            page_text = (
                page.extract_text()
                or ""
            )
        except Exception:
            page_text = ""

        page_text = page_text.strip()

        if not page_text:
            continue

        remaining = (
            MAX_CHARS_PER_PDF
            - total_chars
        )

        if remaining <= 0:
            break

        page_text = page_text[
            :remaining
        ]

        chunks.append(page_text)
        total_chars += len(page_text)
        pages_used += 1

        if total_chars >= MAX_CHARS_PER_PDF:
            break

    text = "\n\n".join(
        chunks
    ).strip()

    if not text:
        raise RuntimeError(
            "No readable text could be extracted from this PDF."
        )

    return text, pages_used



def extract_full_text_from_papers(
    papers: List[Dict[str, Any]],
    max_pdfs: int = MAX_FULL_TEXT_PDFS,
) -> List[Dict[str, Any]]:
    """Extract up to max_pdfs real PDF documents."""

    full_text_papers: List[Dict[str, Any]] = []

    for paper in papers:
        if len(full_text_papers) >= max_pdfs:
            break

        pdf_url = _safe_text(
            paper.get("pdf_url")
        )

        if not pdf_url:
            continue

        print(
            "\n[Questora] Trying full-text PDF:",
            paper.get("title"),
        )

        try:
            pdf_bytes = _download_pdf_bytes(
                pdf_url
            )

            text, pages_extracted = (
                _extract_pdf_text(
                    pdf_bytes
                )
            )

            paper["full_text_read"] = True
            paper["full_text"] = text
            paper["full_text_source_url"] = pdf_url
            paper["pages_extracted"] = pages_extracted
            paper["characters_extracted"] = len(text)
            paper["pdf_error"] = None

            full_text_papers.append(
                paper
            )

            print(
                "[Questora] Full text extracted: "
                f"{pages_extracted} page(s), "
                f"{len(text)} characters."
            )

        except Exception as error:
            paper["full_text_read"] = False
            paper["pdf_error"] = str(error)

            print(
                "[Questora] PDF skipped:",
                error,
            )

    return full_text_papers


# ============================================================
# PROMPT FORMATTING
# ============================================================


def format_papers_for_agent(
    papers: List[Dict[str, Any]],
) -> str:
    if not papers:
        return (
            "NO VERIFIED ACADEMIC PAPER METADATA WAS AVAILABLE. "
            "Do not invent references."
        )

    blocks = []

    for index, paper in enumerate(
        papers,
        start=1,
    ):
        authors = (
            ", ".join(
                paper.get(
                    "authors",
                    [],
                )[:8]
            )
            or "Unknown"
        )

        if len(
            paper.get(
                "authors",
                [],
            )
        ) > 8:
            authors += ", et al."

        block = f"""
PAPER {index}

Title:
{paper.get('title') or 'Untitled'}

Authors:
{authors}

Publication Year:
{paper.get('year') or 'Unknown'}

Source / Journal:
{paper.get('source') or 'Unknown'}

DOI:
{paper.get('doi') or 'Not available'}

Citation Count:
{paper.get('citations', 0)}

Metadata Provider:
{paper.get('metadata_provider') or 'Unknown'}

OpenAlex ID:
{paper.get('openalex_id') or 'Not available'}

Landing Page:
{paper.get('landing_page_url') or 'Not available'}

Open Access:
{'Yes' if paper.get('is_open_access') else 'No'}

PDF URL:
{paper.get('pdf_url') or 'Not available'}

Full Text Read by Questora:
{'Yes' if paper.get('full_text_read') else 'No'}

------------------------------------------------------------
""".strip()

        blocks.append(block)

    return "\n\n".join(blocks)



def format_full_text_for_agent(
    full_text_papers: List[Dict[str, Any]],
) -> str:
    if not full_text_papers:
        return (
            "NO FULL-TEXT PDF WAS SUCCESSFULLY EXTRACTED. "
            "Do not claim that any paper was read in full."
        )

    blocks = []

    for index, paper in enumerate(
        full_text_papers,
        start=1,
    ):
        authors = (
            ", ".join(
                paper.get(
                    "authors",
                    [],
                )[:8]
            )
            or "Unknown"
        )

        text = (
            paper.get(
                "full_text",
                "",
            )
            or ""
        )

        block = f"""
FULL-TEXT PAPER {index}

Title:
{paper.get('title') or 'Untitled'}

Authors:
{authors}

Year:
{paper.get('year') or 'Unknown'}

DOI:
{paper.get('doi') or 'Not available'}

Metadata Provider:
{paper.get('metadata_provider') or 'Unknown'}

Source URL:
{paper.get('full_text_source_url') or paper.get('pdf_url') or 'Not available'}

Pages Extracted:
{paper.get('pages_extracted', 0)}

Characters Extracted:
{paper.get('characters_extracted', 0)}

EXTRACTED TEXT:

{text}

============================================================
""".strip()

        blocks.append(block)

    return "\n\n".join(blocks)


# ============================================================
# LLM
# ============================================================


def create_llm(
    model_name: Optional[str] = None,
) -> LLM:
    """Create the Gemini LLM only when a key is actually available."""

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    return LLM(
        model=(
            model_name
            or PRIMARY_MODEL
        ),
        api_key=GEMINI_API_KEY,
        temperature=0.20,
        timeout=180,
    )


# ============================================================
# AGENTS
# ============================================================


def create_agents(
    llm: LLM,
):
    # --------------------------------------------------------
    # AGENT 01 - RESEARCH AGENT
    # --------------------------------------------------------

    research_agent = Agent(
        role="Research Agent",
        goal=(
            "Build an accurate, structured understanding of the user's "
            "research topic without inventing sources, statistics, or "
            "experimental results."
        ),
        backstory=(
            "You are Questora AI's research strategist. You map concepts, "
            "context, methods, applications, limitations, research gaps, "
            "and future directions. When a factual claim depends on "
            "literature, you rely on the supplied academic evidence and "
            "clearly identify uncertainty."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # --------------------------------------------------------
    # AGENT 02 - PDF READER AGENT
    # --------------------------------------------------------

    pdf_reader_agent = Agent(
        role="PDF Reader Agent",
        goal=(
            "Read only the full-text academic PDF content actually extracted "
            "by Questora and identify useful evidence, methods, findings, "
            "limitations, and research details."
        ),
        backstory=(
            "You are Questora AI's full-text reading specialist. You never "
            "claim to have read a paper unless extracted PDF text is explicitly "
            "supplied. You do not fill missing sections from memory or invent "
            "quotations."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # --------------------------------------------------------
    # AGENT 03 - ACADEMIC PAPER ANALYSIS AGENT
    # --------------------------------------------------------

    paper_analysis_agent = Agent(
        role="Academic Paper Analysis Agent",
        goal=(
            "Analyze the real retrieved academic papers and verified full-text "
            "evidence while clearly separating metadata-only observations "
            "from full-text findings."
        ),
        backstory=(
            "You are Questora AI's evidence analyst. You compare papers, "
            "identify themes, methods, strengths, limitations, gaps, and "
            "disagreements. Every title, author, DOI, citation count, "
            "publication year, and source must come from the supplied verified "
            "academic metadata."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # --------------------------------------------------------
    # AGENT 04 - REPORT AGENT
    # --------------------------------------------------------

    report_agent = Agent(
        role="Report Agent",
        goal=(
            "Create a clear, professional, evidence-grounded research report "
            "using only the supplied topic analysis, verified metadata, and "
            "verified PDF evidence."
        ),
        backstory=(
            "You are Questora AI's final research writer. Your job is to turn "
            "the research pipeline into a useful report while preserving "
            "source integrity. You never invent references and never describe "
            "metadata-only papers as full-text analyzed papers."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    return (
        research_agent,
        pdf_reader_agent,
        paper_analysis_agent,
        report_agent,
    )


# ============================================================
# TASKS
# ============================================================


def create_tasks(
    topic: str,
    language: str,
    papers_text: str,
    full_text_text: str,
    research_agent: Agent,
    pdf_reader_agent: Agent,
    paper_analysis_agent: Agent,
    report_agent: Agent,
):
    # --------------------------------------------------------
    # TASK 01 - TOPIC FOUNDATION
    # --------------------------------------------------------

    research_task = Task(
        description=f"""
RESEARCH TOPIC:

{topic}

OUTPUT LANGUAGE:

{language}

Develop a structured research foundation covering:

1. Topic overview
2. Background
3. Core concepts
4. Technologies / architecture / methodology
5. How it works
6. Current approaches
7. Applications
8. Advantages
9. Challenges and limitations
10. Research gaps
11. Future directions
12. Suggested research questions

STRICT RULES:

- Do not invent academic papers.
- Do not invent authors, DOI numbers, citation counts, statistics, or experiments.
- Do not present uncertain information as verified evidence.
- Use professional academic language.
- Write in {language}.
""",
        expected_output=(
            "A structured research foundation for the topic."
        ),
        agent=research_agent,
    )

    # --------------------------------------------------------
    # TASK 02 - VERIFIED FULL-TEXT READING
    # --------------------------------------------------------

    pdf_reader_task = Task(
        description=f"""
You are given ONLY the academic PDF text that Questora successfully extracted.

RESEARCH TOPIC:

{topic}

VERIFIED FULL-TEXT INPUT:

============================================================
{full_text_text}
============================================================

For each supplied full-text paper, identify:

1. Research objective
2. Method / approach
3. Important findings
4. Useful evidence for the topic
5. Limitations stated or visible in the supplied text
6. Relevant concepts / terminology
7. What this paper contributes to the user's research question

STRICT RULES:

- Analyze ONLY text supplied above.
- If no full-text PDF is supplied, explicitly say no full-text analysis was possible.
- Never invent missing sections, results, tables, quotations, statistics, or conclusions.
- Do not claim a paper was read in full if it is not included above.
- Write in {language}.
""",
        expected_output=(
            "A verified full-text evidence analysis, or a clear statement "
            "that no full-text PDF was available."
        ),
        agent=pdf_reader_agent,
    )

    # --------------------------------------------------------
    # TASK 03 - ACADEMIC PAPER ANALYSIS
    # --------------------------------------------------------

    paper_analysis_task = Task(
        description=f"""
RESEARCH TOPIC:

{topic}

VERIFIED ACADEMIC PAPER METADATA:

============================================================
{papers_text}
============================================================

Analyze the supplied papers and the PDF Reader Agent's verified findings.

Create these sections:

1. Most relevant retrieved papers
2. Common themes
3. Methods / approaches represented
4. Highly cited retrieved papers
5. Open-access papers
6. Full-text-analyzed papers
7. Metadata-only papers
8. Agreements and differences across the evidence
9. Research gaps
10. Papers that may deserve deeper reading

STRICT RULES:

- Every paper title, author, DOI, year, source, and citation count must come from the supplied metadata.
- Do not invent papers or publication details.
- Metadata-only papers may be discussed only at the metadata level.
- Only treat papers marked Full Text Read by Questora: Yes as full-text analyzed.
- Do not create fake quotations.
- Write in {language}.
""",
        expected_output=(
            "An evidence analysis that explicitly separates metadata-only "
            "and verified full-text evidence."
        ),
        agent=paper_analysis_agent,
        context=[
            research_task,
            pdf_reader_task,
        ],
    )

    # --------------------------------------------------------
    # TASK 04 - FINAL REPORT
    # --------------------------------------------------------

    report_task = Task(
        description=f"""
Create the final Questora AI research report for:

{topic}

Use the Research Agent, PDF Reader Agent, and Academic Paper Analysis Agent outputs.
Also use the supplied verified academic metadata below for the reference section.

VERIFIED ACADEMIC PAPER METADATA:

============================================================
{papers_text}
============================================================

Use this structure where appropriate:

# {topic}

## 1. Executive Summary
## 2. Introduction
## 3. Background
## 4. Core Concepts
## 5. Technologies / Architecture / Methodology
## 6. How It Works
## 7. Current Research Landscape
## 8. Applications
## 9. Advantages
## 10. Challenges and Limitations
## 11. Retrieved Academic Papers
## 12. Verified Full-Text Evidence
## 13. Literature Analysis
## 14. Highly Cited Retrieved Papers
## 15. Recommended Papers to Read
## 16. Research Gaps
## 17. Future Directions
## 18. Suggested Research Questions
## 19. Conclusion
## 20. Retrieved References

IMPORTANT EVIDENCE RULES:

- List only papers actually supplied through verified scholarly metadata providers.
- Do not invent titles, authors, journals, publication years, DOI numbers, or citation counts.
- Clearly distinguish full-text-analyzed papers from metadata-only papers.
- Never claim full-text analysis for a paper unless Questora actually extracted its PDF text.
- Do not invent quotations, experiments, sample sizes, numerical results, or statistics.
- If evidence is insufficient for a claim, state the limitation.
- Do not add external references that were not supplied by Questora.

At the end add a short methodology note explaining:
- which metadata provider was used,
- that Questora performed full-text analysis only for PDFs successfully extracted,
- that metadata-only papers were not represented as fully read.

Write everything in {language}.
""",
        expected_output=(
            "A professional, evidence-grounded research report using only "
            "verified Questora inputs and real academic paper metadata."
        ),
        agent=report_agent,
        context=[
            research_task,
            pdf_reader_task,
            paper_analysis_task,
        ],
    )

    return (
        research_task,
        pdf_reader_task,
        paper_analysis_task,
        report_task,
    )


# ============================================================
# CREW
# ============================================================


def create_crew(
    topic: str,
    language: str,
    papers_text: str,
    full_text_text: str,
    model_name: Optional[str] = None,
) -> Crew:
    llm = create_llm(
        model_name=model_name
    )

    (
        research_agent,
        pdf_reader_agent,
        paper_analysis_agent,
        report_agent,
    ) = create_agents(llm)

    (
        research_task,
        pdf_reader_task,
        paper_analysis_task,
        report_task,
    ) = create_tasks(
        topic=topic,
        language=language,
        papers_text=papers_text,
        full_text_text=full_text_text,
        research_agent=research_agent,
        pdf_reader_agent=pdf_reader_agent,
        paper_analysis_agent=paper_analysis_agent,
        report_agent=report_agent,
    )

    return Crew(
        agents=[
            research_agent,
            pdf_reader_agent,
            paper_analysis_agent,
            report_agent,
        ],
        tasks=[
            research_task,
            pdf_reader_task,
            paper_analysis_task,
            report_task,
        ],
        process=Process.sequential,
        verbose=True,
    )


# ============================================================
# RESILIENT GEMINI / CREWAI EXECUTION
# ============================================================


def _build_model_plan() -> List[str]:
    """Create a retry plan without inventing an unsupported fallback model."""

    if (
        FALLBACK_MODEL
        and FALLBACK_MODEL != PRIMARY_MODEL
    ):
        return [
            PRIMARY_MODEL,
            PRIMARY_MODEL,
            FALLBACK_MODEL,
            FALLBACK_MODEL,
        ]

    return [
        PRIMARY_MODEL,
        PRIMARY_MODEL,
        PRIMARY_MODEL,
        PRIMARY_MODEL,
    ]



def run_ai_report(
    topic: str,
    language: str,
    papers_text: str,
    full_text_text: str,
) -> str:
    """Run the four-agent CrewAI workflow with 429/503 resilience."""

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    model_plan = _build_model_plan()
    retry_delays = [10, 25, 50]
    last_error: Optional[Exception] = None

    for attempt, model_name in enumerate(
        model_plan,
        start=1,
    ):
        try:
            print(
                f"\n[Questora] Starting four-agent workflow "
                f"(attempt {attempt}/{len(model_plan)}) "
                f"using {model_name}..."
            )

            # Rebuild Crew every retry so failed task state is not reused.
            crew = create_crew(
                topic=topic,
                language=language,
                papers_text=papers_text,
                full_text_text=full_text_text,
                model_name=model_name,
            )

            result = crew.kickoff()

            final_report = (
                getattr(
                    result,
                    "raw",
                    None,
                )
                or str(result)
            ).strip()

            if not final_report:
                raise RuntimeError(
                    "Questora returned an empty report."
                )

            return final_report

        except Exception as error:
            last_error = error

            print(
                "\n[Questora] Agent workflow error:",
                repr(error),
            )

            # Daily quota will not recover from a short retry.
            if _is_daily_quota_error(error):
                break

            if (
                _is_temporary_error(error)
                and attempt < len(model_plan)
            ):
                fallback_delay = retry_delays[
                    min(
                        attempt - 1,
                        len(retry_delays) - 1,
                    )
                ]

                wait_seconds = _retry_seconds_from_error(
                    error,
                    fallback_delay,
                )

                print(
                    "[Questora] Temporary Gemini/API issue. "
                    f"Retrying in {wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)
                continue

            break

    raise RuntimeError(
        str(
            last_error
            or "Gemini workflow unavailable."
        )
    )


# ============================================================
# DETERMINISTIC VERIFIED FALLBACK REPORT
# ============================================================


def _paper_markdown(
    paper: Dict[str, Any],
    index: int,
) -> str:
    """Render one verified paper without adding unsupported interpretation."""

    authors = (
        ", ".join(
            paper.get(
                "authors",
                [],
            )[:8]
        )
        or "Unknown"
    )

    if len(
        paper.get(
            "authors",
            [],
        )
    ) > 8:
        authors += ", et al."

    full_text_status = (
        "Full-text analyzed"
        if paper.get("full_text_read")
        else "Metadata only"
    )

    access_status = (
        "Open access / PDF link available"
        if paper.get("is_open_access")
        else "Open-access status not confirmed"
    )

    return (
        f"### {index}. {paper.get('title') or 'Untitled'}\n"
        f"- **Authors:** {authors}\n"
        f"- **Year:** {paper.get('year') or 'Unknown'}\n"
        f"- **Source:** {paper.get('source') or 'Unknown'}\n"
        f"- **DOI:** {paper.get('doi') or 'Not available'}\n"
        f"- **Citation count:** {paper.get('citations', 0)}\n"
        f"- **Metadata provider:** {paper.get('metadata_provider') or 'Unknown'}\n"
        f"- **Access:** {access_status}\n"
        f"- **Evidence status:** {full_text_status}\n"
    )



def build_deterministic_report(
    topic: str,
    language: str,
    papers: List[Dict[str, Any]],
    full_text_papers: List[Dict[str, Any]],
    provider: str,
    ai_error: Optional[Exception] = None,
) -> str:
    """
    Build a report that is always safe to show.

    This report intentionally avoids generating scientific conclusions when
    Gemini is unavailable. It presents verified metadata, full-text extraction
    status, evidence transparency, and next-step guidance only.
    """

    total = len(papers)
    full_count = len(full_text_papers)
    metadata_only = max(
        0,
        total - full_count,
    )

    lines = [
        f"# {topic}",
        "",
        "## 1. Questora Research Status",
        "",
        (
            "Questora completed this run using its resilient verified-output mode. "
            "The information below is limited to academic metadata and PDF text "
            "that Questora could actually verify."
        ),
        "",
        f"- **Metadata source used:** {provider}",
        f"- **Verified papers available:** {total}",
        f"- **Full-text PDFs successfully extracted:** {full_count}",
        f"- **Metadata-only papers:** {metadata_only}",
        f"- **Requested output language:** {language}",
        "",
    ]

    if ai_error is not None:
        lines.extend(
            [
                "### AI generation status",
                "",
                (
                    "The generative AI layer was unavailable for this run. "
                    "This can happen because of a missing Gemini key, daily quota, "
                    "temporary high demand, or another provider error. Questora "
                    "therefore returned verified source information instead of "
                    "inventing an AI synthesis."
                ),
                "",
            ]
        )

    if not papers:
        lines.extend(
            [
                "## 2. Verified Academic Papers",
                "",
                (
                    "No verified paper metadata was available from OpenAlex, "
                    "Crossref, or the topic-specific local cache during this run."
                ),
                "",
                (
                    "Questora intentionally returned no paper titles rather than "
                    "fabricating references."
                ),
                "",
                "## 3. Recommended Action",
                "",
                "- Check the network connection and try the research again.",
                "- Configure an OpenAlex API key for more reliable OpenAlex access.",
                "- Crossref is attempted automatically when OpenAlex is unavailable.",
                "- If the same topic was successfully searched before, verified cached metadata is reused automatically.",
                "",
                "## 4. Methodology Note",
                "",
                (
                    "Questora never creates paper titles, DOI values, authors, "
                    "citation counts, publication years, or journals when no "
                    "verified scholarly source is available."
                ),
            ]
        )

        return "\n".join(lines)

    lines.extend(
        [
            "## 2. Verified Academic Papers",
            "",
        ]
    )

    for index, paper in enumerate(
        papers,
        start=1,
    ):
        lines.append(
            _paper_markdown(
                paper,
                index,
            )
        )
        lines.append("")

    lines.extend(
        [
            "## 3. Full-Text Evidence Availability",
            "",
        ]
    )

    if full_text_papers:
        for index, paper in enumerate(
            full_text_papers,
            start=1,
        ):
            lines.extend(
                [
                    (
                        f"### Full-text source {index}: "
                        f"{paper.get('title') or 'Untitled'}"
                    ),
                    (
                        f"- **Pages extracted:** "
                        f"{paper.get('pages_extracted', 0)}"
                    ),
                    (
                        f"- **Characters extracted:** "
                        f"{paper.get('characters_extracted', 0)}"
                    ),
                    (
                        f"- **Source URL:** "
                        f"{paper.get('full_text_source_url') or 'Not available'}"
                    ),
                    "",
                ]
            )
    else:
        lines.extend(
            [
                (
                    "No full-text PDF could be verified and extracted during "
                    "this run. All listed papers should therefore be treated "
                    "as metadata-only records."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## 4. Research Interpretation",
            "",
            (
                "Questora has not generated scientific conclusions in fallback "
                "mode because doing so without a working reasoning model could "
                "introduce unsupported claims. The verified papers above can "
                "still be opened through their DOI or landing-page links for "
                "manual review."
            ),
            "",
            "## 5. Evidence Rules Used",
            "",
            "- Only records returned by OpenAlex, Crossref, or a previously saved verified topic cache are listed.",
            "- A paper is marked full-text analyzed only when Questora successfully extracted text from its PDF.",
            "- Metadata-only records are not represented as if their full papers were read.",
            "- Missing authors, DOI values, publication years, sources, or links are shown as unavailable rather than guessed.",
            "- Citation counts are presented only when supplied by the metadata provider.",
            "",
            "## 6. Methodology Note",
            "",
            (
                f"Questora retrieved or recovered verified metadata using "
                f"**{provider}**. The generative AI layer was unavailable or "
                f"was intentionally bypassed for this fallback report. This "
                f"preserves output availability without sacrificing reference integrity."
            ),
        ]
    )

    return "\n".join(lines)


# ============================================================
# MAIN FUNCTION USED BY app.py
# ============================================================


def run_research(
    topic: str,
    language: str = "English",
) -> Dict[str, Any]:
    """
    Main Questora entry point.

    This function ALWAYS tries to return the same result structure expected by
    the current Streamlit app:

        {
            "report": str,
            "papers": list,
            "full_text_papers": list,
            "full_text_count": int,
        }

    Extra diagnostic fields are added but do not break app.py.
    """

    topic = _safe_text(topic)

    language = (
        _safe_text(
            language,
            "English",
        )
        or "English"
    )

    if not topic:
        raise ValueError(
            "Research topic cannot be empty."
        )

    print("\n" + "=" * 72)
    print("QUESTORA AI - RESILIENT RESEARCH RUN")
    print("=" * 72)
    print("Topic:", topic)
    print("Language:", language)
    print(
        "OpenAlex API key:",
        (
            "configured"
            if OPENALEX_API_KEY
            else "NOT configured"
        ),
    )
    print(
        "Gemini API key:",
        (
            "configured"
            if GEMINI_API_KEY
            else "NOT configured"
        ),
    )

    # --------------------------------------------------------
    # STEP 1 - RETRIEVE REAL VERIFIED PAPER METADATA
    # --------------------------------------------------------

    papers, metadata_provider = (
        retrieve_verified_papers(
            topic=topic,
            limit=MAX_PAPERS,
        )
    )

    print(
        f"\n[Questora] Verified papers available: "
        f"{len(papers)}"
    )

    print(
        f"[Questora] Metadata provider: "
        f"{metadata_provider}"
    )

    # --------------------------------------------------------
    # STEP 2 - READ AVAILABLE FULL-TEXT PDFS
    # --------------------------------------------------------

    if papers:
        full_text_papers = (
            extract_full_text_from_papers(
                papers,
                max_pdfs=MAX_FULL_TEXT_PDFS,
            )
        )
    else:
        full_text_papers = []

    full_text_count = len(
        full_text_papers
    )

    print(
        f"\n[Questora] Full-text PDF analysis available for "
        f"{full_text_count} paper(s)."
    )

    # --------------------------------------------------------
    # STEP 3 - FORMAT VERIFIED EVIDENCE FOR AGENTS
    # --------------------------------------------------------

    papers_text = format_papers_for_agent(
        papers
    )

    full_text_text = format_full_text_for_agent(
        full_text_papers
    )

    # --------------------------------------------------------
    # STEP 4 - TRY FOUR-AGENT GEMINI WORKFLOW
    # --------------------------------------------------------

    ai_error: Optional[Exception] = None
    final_report = ""

    if GEMINI_API_KEY:
        try:
            final_report = run_ai_report(
                topic=topic,
                language=language,
                papers_text=papers_text,
                full_text_text=full_text_text,
            )

        except Exception as error:
            ai_error = error

            print(
                "\n[Questora] Gemini/CrewAI unavailable. "
                "Switching to deterministic verified fallback report."
            )

            print(
                "[Questora] AI error:",
                repr(error),
            )

    else:
        ai_error = RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

        print(
            "\n[Questora] Gemini key is not configured. "
            "Using deterministic verified fallback report."
        )

    # --------------------------------------------------------
    # STEP 5 - ALWAYS PRODUCE A REPORT
    # --------------------------------------------------------

    if not final_report:
        final_report = build_deterministic_report(
            topic=topic,
            language=language,
            papers=papers,
            full_text_papers=full_text_papers,
            provider=metadata_provider,
            ai_error=ai_error,
        )

    # --------------------------------------------------------
    # STEP 6 - FINAL LOGGING
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("QUESTORA AI - RUN COMPLETE")
    print("=" * 72)
    print(
        "Retrieved papers:",
        len(papers),
    )
    print(
        "Full-text PDFs read:",
        full_text_count,
    )
    print(
        "Metadata provider:",
        metadata_provider,
    )
    print(
        "AI fallback used:",
        bool(ai_error is not None),
    )

    # --------------------------------------------------------
    # STEP 7 - RETURN EXACT APP.PY CONTRACT + OPTIONAL DIAGNOSTICS
    # --------------------------------------------------------

    return {
        "report": final_report,
        "papers": papers,
        "full_text_papers": full_text_papers,
        "full_text_count": full_text_count,
        "metadata_provider": metadata_provider,
        "used_ai_fallback": bool(
            ai_error is not None
        ),
        "ai_error": (
            str(ai_error)
            if ai_error is not None
            else None
        ),
    }


# ============================================================
# DIRECT TERMINAL TEST
# ============================================================


if __name__ == "__main__":
    print(
        "\nQUESTORA AI - DIRECT RESILIENCE TEST\n"
    )

    test_topic = input(
        "Enter research topic: "
    ).strip()

    if not test_topic:
        test_topic = (
            "Artificial Intelligence in Healthcare"
        )

    try:
        result = run_research(
            topic=test_topic,
            language="English",
        )

        print("\n" + "=" * 72)
        print("QUESTORA AI - TEST COMPLETE")
        print("=" * 72)

        print(
            "\nMetadata provider:",
            result.get(
                "metadata_provider"
            ),
        )

        print(
            "Papers:",
            len(
                result.get(
                    "papers",
                    [],
                )
            ),
        )

        print(
            "Full-text PDFs:",
            result.get(
                "full_text_count",
                0,
            ),
        )

        print(
            "Fallback report used:",
            result.get(
                "used_ai_fallback"
            ),
        )

        print("\n\nFINAL REPORT\n")
        print(
            result.get(
                "report",
                "",
            )
        )

    except Exception as error:
        print("\nQUESTORA FAILED:\n")
        print(error)
