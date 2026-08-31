#!/usr/bin/env python3
"""Crawl NCI Cancer.gov PDQ summaries and save one JSON file per cancer type.

The crawler discovers current PDQ URLs from NCI-maintained index pages instead of
hard-coding individual cancer pages. It downloads both patient and health-
professional versions where available, preserves heading hierarchy, and groups
site-specific treatment/screening/prevention pages into per-cancer JSON files.

General or multi-cancer PDQ collections (genetics, supportive/palliative care,
integrative/complementary therapies) are retained under general_topics/ so they
are not lost or incorrectly assigned to a single cancer type.

Example:
    pip install requests beautifulsoup4
    python nci_pdq_crawler.py --output data/nci_pdq

Useful options:
    python nci_pdq_crawler.py --collections adult_treatment pediatric_treatment
    python nci_pdq_crawler.py --output data/nci_pdq --delay 1.0
    python nci_pdq_crawler.py --use-cache
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://www.cancer.gov"

PDQ_INDEXES = {
    "adult_treatment": (
        "https://www.cancer.gov/publications/pdq/information-summaries/adult-treatment"
    ),
    "pediatric_treatment": (
        "https://www.cancer.gov/publications/pdq/information-summaries/pediatric-treatment"
    ),
    "screening": (
        "https://www.cancer.gov/publications/pdq/information-summaries/screening"
    ),
    "prevention": (
        "https://www.cancer.gov/publications/pdq/information-summaries/prevention"
    ),
    "genetics": (
        "https://www.cancer.gov/publications/pdq/information-summaries/genetics"
    ),
    "supportive_care": (
        "https://www.cancer.gov/publications/pdq/information-summaries/supportive-care"
    ),
    "integrative_therapies": (
        "https://www.cancer.gov/publications/pdq/information-summaries/cam"
    ),
}

# These collections can be assigned to a cancer/site reasonably safely by
# removing the topic suffix from the PDQ title. Other collections are often
# general or multi-cancer and are therefore saved separately.
CANCER_SPECIFIC_COLLECTIONS = {
    "adult_treatment",
    "pediatric_treatment",
    "screening",
    "prevention",
}

# Optional canonical aliases. Add more only when you are certain two NCI names
# refer to the same retrieval entity for your project.
CANONICAL_ALIASES = {
    "Gastric Cancer": "Stomach (Gastric) Cancer",
    "Stomach Cancer": "Stomach (Gastric) Cancer",
    "Renal Cell Cancer": "Kidney (Renal Cell) Cancer",
    "Liver Cancer (Primary)": "Hepatocellular (Liver) Cancer",
    "Liver (Hepatocellular) Cancer": "Hepatocellular (Liver) Cancer",
}

BOILERPLATE_HEADINGS = {
    "about this pdq summary",
    "changes to this summary",
    "reviewers and updates",
    "permissions to use this summary",
    "disclaimer",
    "contact us",
}

MONTH_PATTERN = (
    r"January|February|March|April|May|June|July|August|September|October|"
    r"November|December"
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_ws(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def slugify(text: str, max_length: int = 120) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return (text[:max_length].rstrip("_") or "unnamed")


def canonical_url(url: str) -> str:
    """Remove query and fragment so duplicate index aliases collapse."""
    parts = urlsplit(url)
    scheme = parts.scheme or "https"
    netloc = parts.netloc.lower()
    if netloc == "cancer.gov":
        netloc = "www.cancer.gov"
    path = re.sub(r"//+", "/", parts.path)
    return urlunsplit((scheme, netloc, path, "", ""))


def is_nci_pdq_content_url(url: str) -> bool:
    parts = urlsplit(url)
    if parts.netloc.lower() not in {"www.cancer.gov", "cancer.gov"}:
        return False

    path = parts.path.lower()
    if path.startswith("/publications/pdq/information-summaries"):
        return False

    # Current PDQ pages normally contain "pdq" in their path. This deliberately
    # avoids grabbing arbitrary NCI navigation links from index pages.
    return "pdq" in path


def build_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_html(session: requests.Session, url: str, timeout: int = 45) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def load_robots(session: requests.Session, user_agent: str) -> RobotFileParser:
    robots_url = urljoin(BASE_URL, "/robots.txt")
    response = session.get(robots_url, timeout=30)
    response.raise_for_status()

    parser = RobotFileParser()
    parser.set_url(robots_url)
    parser.parse(response.text.splitlines())

    # Touch can_fetch once so a malformed robots file fails early.
    parser.can_fetch(user_agent, BASE_URL)
    return parser


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_pdq_pages(
    session: requests.Session,
    collections: Iterable[str],
    robots: RobotFileParser | None,
    robots_user_agent: str,
) -> list[dict[str, Any]]:
    """Discover live PDQ pages from NCI-maintained index pages.

    We intentionally do not depend on the exact HTML structure of the index.
    Any Cancer.gov link containing "pdq" in the path is treated as a candidate,
    then deduplicated by canonical URL.
    """
    discovered: dict[str, dict[str, Any]] = {}

    for collection in collections:
        index_url = PDQ_INDEXES[collection]
        print(f"[INDEX] {collection}: {index_url}")

        if robots is not None and not robots.can_fetch(robots_user_agent, index_url):
            print(f"  [SKIP robots.txt] {index_url}")
            continue

        html = fetch_html(session, index_url)
        soup = BeautifulSoup(html, "html.parser")
        main = soup.find("main") or soup

        count_before = len(discovered)
        for anchor in main.find_all("a", href=True):
            target = canonical_url(urljoin(index_url, anchor["href"]))
            if not is_nci_pdq_content_url(target):
                continue

            label = normalize_ws(anchor.get_text(" ", strip=True))
            record = discovered.setdefault(
                target,
                {
                    "url": target,
                    "collections": set(),
                    "index_link_labels": set(),
                    "discovered_from": set(),
                },
            )
            record["collections"].add(collection)
            if label:
                record["index_link_labels"].add(label)
            record["discovered_from"].add(index_url)

        print(f"  discovered +{len(discovered) - count_before} unique PDQ URLs")

    output = []
    for record in discovered.values():
        output.append(
            {
                "url": record["url"],
                "collections": sorted(record["collections"]),
                "index_link_labels": sorted(record["index_link_labels"]),
                "discovered_from": sorted(record["discovered_from"]),
            }
        )

    return sorted(output, key=lambda x: x["url"])


# ---------------------------------------------------------------------------
# Page parsing
# ---------------------------------------------------------------------------


def infer_audience(url: str, title: str) -> str:
    path = urlsplit(url).path.lower()
    lower_title = title.lower()

    if "/hp/" in path or "health professional version" in lower_title:
        return "health_professional"
    if "/patient/" in path or "patient version" in lower_title:
        return "patient"
    return "unspecified"


def strip_pdq_suffix(title: str) -> str:
    """Convert an H1 such as
    'Bladder Cancer Treatment (PDQ®)–Health Professional Version'
    to 'Bladder Cancer Treatment'.
    """
    title = normalize_ws(title)
    title = re.sub(r"\s*\(PDQ[^)]*\).*?$", "", title, flags=re.IGNORECASE)
    title = re.sub(
        r"\s*[–—-]\s*(Health Professional|Patient) Version\s*$",
        "",
        title,
        flags=re.IGNORECASE,
    )
    return normalize_ws(title)


def derive_topic(collections: list[str]) -> str:
    # A URL should usually come from one index. If aliases put it in several,
    # choose a stable display value while retaining the full collections list.
    priority = [
        "adult_treatment",
        "pediatric_treatment",
        "screening",
        "prevention",
        "genetics",
        "supportive_care",
        "integrative_therapies",
    ]
    for item in priority:
        if item in collections:
            return item
    return collections[0] if collections else "unknown"


def derive_cancer_type(summary_name: str, collections: list[str]) -> str | None:
    """Derive a grouping key for cancer-specific collections.

    Genetics/supportive/integrative pages are intentionally NOT forced into one
    cancer file because many cover multiple cancers or cross-cutting symptoms.
    """
    if not any(c in CANCER_SPECIFIC_COLLECTIONS for c in collections):
        return None

    name = summary_name

    # Remove collection topic words only when they are a suffix. This preserves
    # names such as "Breast Cancer Treatment During Pregnancy" as a distinct
    # entity rather than guessing an equivalence.
    name = re.sub(r"\s+Treatment$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+Screening$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+Prevention$", "", name, flags=re.IGNORECASE)
    name = normalize_ws(name)

    # Explicit overview pages are not cancer types.
    if name.lower() in {
        "cancer screening overview",
        "cancer prevention overview",
    }:
        return None

    return CANONICAL_ALIASES.get(name, name)


def extract_last_updated(text: str) -> str | None:
    # Most PDQ pages expose an "Updated: Month D, YYYY" line near the end.
    match = re.search(
        rf"\bUpdated:\s*({MONTH_PATTERN})\s+(\d{{1,2}}),\s+(\d{{4}})",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return f"{match.group(1)} {match.group(2)}, {match.group(3)}"

    match = re.search(
        rf"\bReviewed:\s*({MONTH_PATTERN})\s+(\d{{1,2}}),\s+(\d{{4}})",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return f"{match.group(1)} {match.group(2)}, {match.group(3)}"

    return None


def element_text_without_nested_lists(li: Tag) -> str:
    """Get one <li>'s own text while avoiding duplication from nested lists."""
    clone = BeautifulSoup(str(li), "html.parser")
    root = clone.find("li")
    if root is None:
        return normalize_ws(li.get_text(" ", strip=True))
    for nested in root.find_all(["ul", "ol"]):
        nested.decompose()
    return normalize_ws(root.get_text(" ", strip=True))


def table_row_text(tr: Tag) -> str:
    cells = [normalize_ws(cell.get_text(" ", strip=True)) for cell in tr.find_all(["th", "td"])]
    cells = [cell for cell in cells if cell]
    return " | ".join(cells)


def is_boilerplate_path(path: list[str]) -> bool:
    lowered = {normalize_ws(item).lower() for item in path}
    return any(h in lowered for h in BOILERPLATE_HEADINGS)


def extract_sections(main: Tag) -> list[dict[str, Any]]:
    """Extract hierarchical sections while preserving headings and lists/tables."""
    # Work on a copy because we remove UI elements.
    working = BeautifulSoup(str(main), "html.parser")
    root = working.find("main") or working

    # Remove clear UI/non-content elements. We intentionally keep references,
    # disclaimers, and PDQ metadata sections and mark them as boilerplate later.
    for selector in ["script", "style", "noscript", "form", "nav", "footer", "aside"]:
        for node in root.find_all(selector):
            node.decompose()

    sections: list[dict[str, Any]] = []
    heading_path: list[str] = []
    blocks: list[str] = []

    def flush() -> None:
        nonlocal blocks
        cleaned: list[str] = []
        for block in blocks:
            block = normalize_ws(block)
            if not block:
                continue
            if cleaned and cleaned[-1] == block:
                continue
            cleaned.append(block)

        if cleaned:
            heading = heading_path[-1] if heading_path else "Document"
            sections.append(
                {
                    "section": heading,
                    "section_path": list(heading_path),
                    "is_boilerplate": is_boilerplate_path(heading_path),
                    "block_count": len(cleaned),
                    "text": "\n".join(cleaned),
                }
            )
        blocks = []

    tags = root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr", "blockquote"])

    for element in tags:
        # Skip paragraph/list items inside tables because each row is handled as
        # a single pipe-delimited record below.
        if element.name in {"p", "li", "blockquote"} and element.find_parent("table"):
            continue

        # Skip <p> inside <li>; the parent list item will capture it.
        if element.name == "p" and element.find_parent("li"):
            continue

        if re.fullmatch(r"h[1-6]", element.name or ""):
            flush()
            level = int(element.name[1])
            heading = normalize_ws(element.get_text(" ", strip=True))
            if not heading:
                continue
            heading_path = heading_path[: level - 1]
            heading_path.append(heading)
            continue

        if element.name == "li":
            text = element_text_without_nested_lists(element)
            if text:
                blocks.append(f"- {text}")
        elif element.name == "tr":
            text = table_row_text(element)
            if text:
                blocks.append(text)
        else:
            text = normalize_ws(element.get_text(" ", strip=True))
            if text:
                blocks.append(text)

    flush()
    return sections


def parse_pdq_page(html: str, discovery: dict[str, Any]) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main") or soup

    h1 = main.find("h1")
    if h1 is not None:
        title = normalize_ws(h1.get_text(" ", strip=True))
    elif soup.title is not None:
        title = normalize_ws(soup.title.get_text(" ", strip=True))
    else:
        title = "Untitled PDQ Summary"

    summary_name = strip_pdq_suffix(title)
    audience = infer_audience(discovery["url"], title)
    topic = derive_topic(discovery["collections"])

    full_main_text = normalize_ws(main.get_text(" ", strip=True))
    last_updated = extract_last_updated(full_main_text)
    sections = extract_sections(main)

    content_text = "\n\n".join(section["text"] for section in sections)
    sha256 = hashlib.sha256(content_text.encode("utf-8")).hexdigest()

    cancer_type = derive_cancer_type(summary_name, discovery["collections"])

    return {
        "source": "National Cancer Institute (NCI)",
        "source_domain": "cancer.gov",
        "collection": "PDQ",
        "collections": discovery["collections"],
        "topic": topic,
        "audience": audience,
        "language": "en",
        "cancer_type": cancer_type,
        "title": title,
        "summary_name": summary_name,
        "url": discovery["url"],
        "discovered_from": discovery["discovered_from"],
        "index_link_labels": discovery["index_link_labels"],
        "last_updated": last_updated,
        "retrieved_at": utc_now(),
        "content_sha256": sha256,
        "section_count": len(sections),
        "sections": sections,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_raw_html(output_dir: Path, url: str, html: str) -> str:
    raw_dir = output_dir / "raw_html"
    raw_dir.mkdir(parents=True, exist_ok=True)
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    path = raw_dir / f"{url_hash}.html"
    path.write_text(html, encoding="utf-8")
    return str(path.relative_to(output_dir))


def raw_cache_path(output_dir: Path, url: str) -> Path:
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    return output_dir / "raw_html" / f"{url_hash}.html"


def save_grouped_json(
    output_dir: Path,
    pages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cancers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    general: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for page in pages:
        if page["cancer_type"]:
            cancers[page["cancer_type"]].append(page)
        else:
            general[page["summary_name"]].append(page)

    cancer_dir = output_dir / "cancers"
    general_dir = output_dir / "general_topics"
    cancer_dir.mkdir(parents=True, exist_ok=True)
    general_dir.mkdir(parents=True, exist_ok=True)

    cancer_manifest: list[dict[str, Any]] = []
    general_manifest: list[dict[str, Any]] = []

    for cancer_type, cancer_pages in sorted(cancers.items()):
        cancer_pages.sort(key=lambda p: (p["topic"], p["audience"], p["title"]))
        filename = f"{slugify(cancer_type)}.json"
        path = cancer_dir / filename

        payload = {
            "schema_version": "1.0",
            "source": "National Cancer Institute (NCI)",
            "collection": "PDQ",
            "cancer_type": cancer_type,
            "generated_at": utc_now(),
            "page_count": len(cancer_pages),
            "pages": cancer_pages,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        cancer_manifest.append(
            {
                "cancer_type": cancer_type,
                "file": str(path.relative_to(output_dir)),
                "page_count": len(cancer_pages),
                "topics": sorted({p["topic"] for p in cancer_pages}),
                "audiences": sorted({p["audience"] for p in cancer_pages}),
            }
        )

    for summary_name, topic_pages in sorted(general.items()):
        topic_pages.sort(key=lambda p: (p["topic"], p["audience"], p["title"]))
        filename = f"{slugify(summary_name)}.json"
        path = general_dir / filename

        payload = {
            "schema_version": "1.0",
            "source": "National Cancer Institute (NCI)",
            "collection": "PDQ",
            "scope": "general_or_multi_cancer",
            "summary_name": summary_name,
            "generated_at": utc_now(),
            "page_count": len(topic_pages),
            "pages": topic_pages,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        general_manifest.append(
            {
                "summary_name": summary_name,
                "file": str(path.relative_to(output_dir)),
                "page_count": len(topic_pages),
                "topics": sorted({p["topic"] for p in topic_pages}),
                "audiences": sorted({p["audience"] for p in topic_pages}),
            }
        )

    return cancer_manifest, general_manifest


# ---------------------------------------------------------------------------
# Main crawl
# ---------------------------------------------------------------------------


def crawl(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_collections = (
        list(PDQ_INDEXES.keys()) if args.collections == ["all"] else args.collections
    )

    unknown = [x for x in selected_collections if x not in PDQ_INDEXES]
    if unknown:
        raise SystemExit(f"Unknown collection(s): {', '.join(unknown)}")

    session = build_session(args.user_agent)

    robots: RobotFileParser | None = None
    if not args.ignore_robots:
        try:
            robots = load_robots(session, args.user_agent)
            print("[OK] Loaded Cancer.gov robots.txt")
        except Exception as exc:
            raise SystemExit(
                "Could not load Cancer.gov robots.txt. For a reproducible and polite "
                "crawl, fix the network problem and retry. If you have independently "
                f"verified permission, you may use --ignore-robots. Error: {exc}"
            )

    discovered = discover_pdq_pages(
        session,
        selected_collections,
        robots,
        args.user_agent,
    )

    (output_dir / "discovered_pages.json").write_text(
        json.dumps(discovered, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[DISCOVERY] {len(discovered)} unique PDQ URLs")

    parsed_pages: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for idx, record in enumerate(discovered, start=1):
        url = record["url"]
        print(f"[{idx}/{len(discovered)}] {url}")

        if robots is not None and not robots.can_fetch(args.user_agent, url):
            print("  [SKIP robots.txt]")
            failures.append({"url": url, "reason": "disallowed_by_robots_txt"})
            continue

        try:
            cache_path = raw_cache_path(output_dir, url)
            if args.use_cache and cache_path.exists():
                html = cache_path.read_text(encoding="utf-8")
                print("  [CACHE]")
            else:
                html = fetch_html(session, url, timeout=args.timeout)

            raw_file = None
            if not args.no_raw_html:
                raw_file = save_raw_html(output_dir, url, html)

            page = parse_pdq_page(html, record)
            if raw_file:
                page["raw_html_file"] = raw_file

            parsed_pages.append(page)
            print(
                f"  -> {page['summary_name']} | {page['audience']} | "
                f"{page['section_count']} sections"
            )
        except Exception as exc:
            print(f"  [ERROR] {type(exc).__name__}: {exc}")
            failures.append(
                {
                    "url": url,
                    "collections": record.get("collections", []),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

        if args.delay > 0 and idx < len(discovered):
            time.sleep(args.delay)

    # Keep a complete parsed-page file in addition to the per-cancer outputs.
    (output_dir / "all_pages.json").write_text(
        json.dumps(parsed_pages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cancer_manifest, general_manifest = save_grouped_json(output_dir, parsed_pages)

    manifest = {
        "schema_version": "1.0",
        "source": "National Cancer Institute (NCI)",
        "collection": "PDQ",
        "base_url": BASE_URL,
        "generated_at": utc_now(),
        "collections_requested": selected_collections,
        "index_urls": {key: PDQ_INDEXES[key] for key in selected_collections},
        "discovered_url_count": len(discovered),
        "success_count": len(parsed_pages),
        "failure_count": len(failures),
        "cancer_file_count": len(cancer_manifest),
        "general_topic_file_count": len(general_manifest),
        "cancer_files": cancer_manifest,
        "general_topic_files": general_manifest,
        "failures": failures,
        "notes": [
            "Patient and health-professional pages are kept as separate page records.",
            "Treatment, screening, and prevention pages are grouped into per-cancer files when a safe title-based grouping is possible.",
            "Genetics, supportive/palliative care, and integrative/complementary therapy summaries are stored under general_topics to avoid incorrect single-cancer assignment.",
            "Section hierarchy is preserved for later RAG chunking.",
        ],
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[DONE]")
    print(f"  parsed pages:        {len(parsed_pages)}")
    print(f"  failed pages:        {len(failures)}")
    print(f"  cancer JSON files:   {len(cancer_manifest)}")
    print(f"  general topic files: {len(general_manifest)}")
    print(f"  output:              {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl NCI Cancer.gov PDQ summaries into per-cancer JSON files."
    )
    parser.add_argument(
        "--output",
        default="data/nci_pdq",
        help="Output directory (default: data/nci_pdq)",
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        default=["all"],
        choices=["all", *PDQ_INDEXES.keys()],
        help="PDQ collections to crawl (default: all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.75,
        help="Seconds to wait between PDQ page requests (default: 0.75)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
        help="Per-request timeout in seconds (default: 45)",
    )
    parser.add_argument(
        "--user-agent",
        default="CancerMythRAG/1.0 academic-research",
        help=(
            "HTTP User-Agent. For sustained crawling, include a project/contact URL "
            "or email so the site operator can identify you."
        ),
    )
    parser.add_argument(
        "--no-raw-html",
        action="store_true",
        help="Do not preserve raw downloaded HTML snapshots.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse raw_html snapshots when present instead of re-downloading pages.",
    )
    parser.add_argument(
        "--ignore-robots",
        action="store_true",
        help="Skip robots.txt enforcement. Use only after independently verifying permission.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    crawl(parse_args())
