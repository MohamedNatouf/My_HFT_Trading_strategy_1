import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from datetime import datetime

# Configure logging
logger = logging.getLogger("extract_alpaca_docs")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Base URLs to crawl
BASE_URLS = [
    "https://docs.alpaca.markets/docs/",
    "https://docs.alpaca.markets/reference/"
]

ALLOWED_HOST = "docs.alpaca.markets"
ALLOWED_PREFIXES = ("/docs/", "/reference/")

logger.info("Starting Alpaca docs crawl with %d base URLs", len(BASE_URLS))

# Helpers

def is_allowed_url(url: str) -> bool:
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False
        if p.netloc != ALLOWED_HOST:
            return False
        # Only allow paths strictly under /docs/ or /reference/
        return any(p.path.startswith(pref) for pref in ALLOWED_PREFIXES)
    except Exception:
        return False


def normalize_url(url: str) -> str:
    # Normalize simple trailing slash rules to reduce duplicates
    p = urlparse(url)
    path = p.path or "/"
    if not path.endswith("/") and (path.startswith("/docs/") or path.startswith("/reference/")):
        path = path + "/"
    return p._replace(path=path, fragment="").geturl()


# Output text file (incremental saves)
OUTPUT_PATH = "Alpaca_Full_Docs.txt"
# Ensure file header once
with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
    f.write(f"===== START CRAWL {datetime.utcnow().isoformat()}Z =====\n")

# Crawl state
visited = set()
queue = []

for base in BASE_URLS:
    if is_allowed_url(base):
        nb = normalize_url(base)
        queue.append(nb)
        visited.add(nb)
        logger.debug("Seed URL queued: %s", nb)
    else:
        logger.warning("Seed URL not allowed and skipped: %s", base)

processed_pages = 0

while queue:
    url = queue.pop(0)
    logger.info(
        "Processing URL: %s (processed=%d, queue=%d, visited=%d)",
        url, processed_pages, len(queue), len(visited)
    )

    try:
        r = requests.get(url, timeout=15, allow_redirects=True)
        final_url = r.url
        logger.debug(
            "HTTP GET %s -> final=%s status=%s, bytes=%d",
            url, final_url, getattr(r, "status_code", "?"), len(getattr(r, "content", b""))
        )
        if r.status_code != 200:
            logger.warning("Non-200 status for %s: %s", url, r.status_code)
    except Exception as e:
        logger.warning("Skip %s, error: %s", url, e, exc_info=True)
        continue

    # If redirected outside allowed scope, skip extraction entirely
    if not is_allowed_url(final_url):
        logger.info("Final URL outside allowed scope, skip extraction: %s", final_url)
        continue

    soup = BeautifulSoup(r.text, "html.parser")

    # Add section title
    title = soup.find("h1") or soup.find("title")
    title_text = title.get_text(strip=True) if title else final_url

    # Extract all paragraphs and code blocks
    added_count = 0
    extracted_blocks = []  # tuples of (type, text)
    for tag in soup.find_all(["p", "pre", "code"]):
        text = tag.get_text()
        if not text:
            continue
        text = text.strip()
        if not text:
            continue
        block_type = tag.name.upper()
        extracted_blocks.append((block_type, text))
        added_count += 1

    logger.info("Extracted %d text/code blocks from %s", added_count, final_url)

    # Find all links on this page to follow
    discovered = 0
    new_links = 0
    new_links_list = []
    for link in soup.find_all("a", href=True):
        href = link["href"].strip()
        if not href:
            continue
        # Only crawl within docs.alpaca.markets
        if href.startswith("/"):
            full = urljoin("https://docs.alpaca.markets", href)
        else:
            full = href

        discovered += 1

        if not is_allowed_url(full):
            continue

        full_norm = normalize_url(full)
        if full_norm not in visited:
            visited.add(full_norm)
            queue.append(full_norm)
            new_links += 1
            new_links_list.append(full_norm)

    logger.info(
        "Discovered %d links; queued %d new URLs; queue size now %d",
        discovered, new_links, len(queue)
    )

    # Incremental save: only when we actually have new data (content or links)
    if added_count > 0 or new_links > 0:
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            f.write("\n--- PAGE START ---\n")
            f.write(f"Title: {title_text}\n")
            f.write(f"URL: {final_url}\n")
            f.write(f"Blocks: {added_count}\n")
            if new_links_list:
                f.write("NewLinks:\n")
                for ln in new_links_list:
                    f.write(f" - {ln}\n")
            if extracted_blocks:
                f.write("Content:\n")
                for t, txt in extracted_blocks:
                    if t == "P":
                        f.write(f"[TEXT] {txt}\n")
                    elif t in ("PRE", "CODE"):
                        f.write("[CODE]\n")
                        f.write(txt + "\n")
                        f.write("[/CODE]\n")
                    else:
                        f.write(txt + "\n")
            f.write("--- PAGE END ---\n")
        logger.info("Incrementally saved content for %s", final_url)

    processed_pages += 1
    if processed_pages % 10 == 0:
        logger.info(
            "Progress: processed %d pages, visited=%d, queue=%d",
            processed_pages, len(visited), len(queue)
        )

    time.sleep(0.1)

logger.info("Crawl complete. Output file: %s", OUTPUT_PATH)
