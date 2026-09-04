"""India Code (indiacode.gov.in) DSpace 9 REST client — the statute source.

India Code migrated in 2026 from a DSpace 6 site (indiacode.nic.in, PDF
bitstreams at fixed paths) to DSpace 9 at indiacode.gov.in. The new site
exposes two things this project needs:

  CENTRAL / Acts      one item per central act, with ORIGINAL bitstreams
                      (English PDF, often a Hindi PDF) and TEXT extractions;
  CENTRAL / Section   one item per SECTION, with the section number, title
                      and the section text (HTML) in metadata — a structured
                      source that replaces PDF splitting for any central act.

Verified 2026-09-04: the 358 BNS section records match the committed
data/canonical/bns_2023.jsonl section for section. The Hindi PDFs' TEXT
bitstreams contain 0% Devanagari (image scans), so Hindi statute text is not
obtainable here.

Network access only through `requests`; nothing here needs authentication.
"""
import html
import re

import requests

API = "https://indiacode.gov.in/server/api"
UI = "https://indiacode.gov.in"
CENTRAL_ACTS_COLLECTION = "69a0c1fb-7b22-4481-b16a-1dc59b5d02e6"
CENTRAL_SECTION_COLLECTION = "9a5b5f4a-4ccf-4cef-9820-0d33f8a6707f"
_HEADERS = {"Accept": "application/json", "User-Agent": "Mozilla/5.0 (nyaya-model statute builder)"}


def _get(url: str, **params):
    r = requests.get(url, headers=_HEADERS, params=params, timeout=90)
    r.raise_for_status()
    return r.json()


def _md(item: dict, key: str):
    values = (item.get("metadata") or {}).get(key)
    return values[0]["value"] if values else None


def strip_section_html(raw: str) -> str:
    """Section text arrives as HTML with layout spans and footnote rules."""
    text = re.sub(r"<sup>.*?</sup>", "", raw or "", flags=re.S)   # footnote markers
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<hr[^>]*>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\.\s*\.", ".", text)          # "fine.." from a stripped footnote marker
    return text.strip()


def tidy_section_text(text: str, section: str, title: str) -> str:
    """Remove two artefacts of India Code's section text: amendment footnote
    numbers before bracketed insertions ("1 [It shall...") and a repeated
    "106. Title. -- " heading at the start of the body."""
    text = re.sub(r"(?<![\w(])\d{1,2}\s\[", "[", text)
    text = re.sub(rf"^\[?\s*{re.escape(section)}\.\s*{re.escape(title)}\.?\s*(?:--|—|–)?\s*", "", text, count=1)
    return text.strip()


def clean_title(title: str) -> str:
    return (title or "").strip().rstrip(".").strip()


def find_central_act(title: str) -> dict | None:
    """Exact-title lookup in CENTRAL / Acts. Returns uuid, handle, act_id
    (the prefix that section records carry), enforcement date and bitstreams."""
    res = _get(f"{API}/discover/search/objects", query=f'"{title}"',
               scope=CENTRAL_ACTS_COLLECTION, dsoType="item", size=20)
    for o in res["_embedded"]["searchResult"]["_embedded"]["objects"]:
        item = o["_embedded"]["indexableObject"]
        if clean_title(_md(item, "dc.title")).lower() != clean_title(title).lower():
            continue
        act_id = _md(item, "dc.identifier.act_id") or ""
        files = []
        for bundle in _get(f"{API}/core/items/{item['uuid']}/bundles", size=50)["_embedded"]["bundles"]:
            if bundle["name"] != "ORIGINAL":
                continue
            for bs in _get(bundle["_links"]["bitstreams"]["href"], size=50)["_embedded"].get("bitstreams", []):
                files.append({"name": bs["name"], "bytes": bs.get("sizeBytes"), "url": bs["_links"]["content"]["href"]})
        return {"uuid": item["uuid"], "handle": item.get("handle"), "title": _md(item, "dc.title"),
                "act_id": act_id, "act_id_prefix": act_id.rsplit("_", 1)[0] if act_id else "",
                "act_number": _md(item, "dc.identifier.act_number"),
                "enforcement_date": _md(item, "dc.date.enforcement_date"),
                "enact_date": _md(item, "dc.date.enact_date"),
                "bitstreams": files, "url": f"{UI}/handle/{item.get('handle')}"}
    return None


def iter_sections(act_id_prefix: str):
    """Every Section record of a central act, as dicts with section, order,
    title, text (HTML stripped), repealed flag."""
    page = 0
    while True:
        res = _get(f"{API}/discover/search/objects", query=f'"{act_id_prefix}"',
                   scope=CENTRAL_SECTION_COLLECTION, dsoType="item", size=100, page=page)
        result = res["_embedded"]["searchResult"]
        for o in result["_embedded"]["objects"]:
            item = o["_embedded"]["indexableObject"]
            if not (_md(item, "dc.identifier.act_id") or "").startswith(act_id_prefix):
                continue
            title = clean_title(_md(item, "dc.title"))
            text = strip_section_html(_md(item, "dc.identifier.section_page_note") or "")
            yield {"section": (_md(item, "dc.identifier.section_number") or "").strip(),
                   "order": int(_md(item, "dc.identifier.order_number") or 0),
                   "title": title, "text": text,
                   "repealed": title.lower().strip("[]. ") in ("repealed", "omitted") or
                   (_md(item, "dc.identifier.repealed") or "false").lower() == "true"}
        if page >= result["page"]["totalPages"] - 1:
            break
        page += 1


def to_iso_date(raw: str | None) -> str | None:
    """India Code writes enforcement dates as '1-7-2024' or '2023-12-25'."""
    if not raw:
        return None
    m = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{4})$", raw.strip())
    if m:
        return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", raw.strip())
    return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else None


def section_rows(act_id: str, act_name: str, sections, effective_date: str | None,
                 source_url: str) -> list[dict]:
    """StatuteSection rows (data/canonical schema) from API section records,
    ordered as in the act, repealed/empty sections dropped."""
    rows = []
    for s in sorted(sections, key=lambda x: x["order"]):
        if s["repealed"] or not s["text"]:
            continue
        rows.append({"act_id": act_id, "act_name": act_name, "section": s["section"], "title": s["title"],
                     "text": tidy_section_text(s["text"], s["section"], s["title"]),
                     "chapter": None, "subsection": None, "effective_date": effective_date,
                     "replaces": None, "punishment_summary": None, "tags": [], "source_url": source_url})
    return rows
