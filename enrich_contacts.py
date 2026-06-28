from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import tldextract  # optional but recommended
except ImportError:
    tldextract = None


PUBLIC_PROVIDERS = {
    "gmail", "hotmail", "outlook", "live", "msn", "yahoo", "yandex",
    "mail", "inbox", "bk", "list", "icloud", "me", "aol", "gmx",
    "protonmail", "pm", "qq", "naver", "daum", "wp", "rambler"
}

GENERIC_LOCALPARTS = {
    "info", "sales", "office", "contact", "admin", "support", "hr",
    "career", "careers", "marketing", "export", "import", "accounts",
    "accounting", "billing", "enquiry", "enquiries", "corp", "press",
    "media", "hello", "team", "noreply", "no-reply"
}

SURNAME_PARTICLES = {
    "da", "de", "del", "della", "der", "di", "du", "el", "la", "le",
    "st", "st.", "van", "von", "bin", "binti", "ibn", "ten", "ter", "al"
}

COMMON_TWO_LEVEL_SUFFIXES = {
    ("co", "uk"), ("org", "uk"), ("gov", "uk"), ("ac", "uk"),
    ("com", "tr"), ("org", "tr"), ("gov", "tr"), ("edu", "tr"),
    ("co", "at"), ("co", "jp"), ("co", "za"),
    ("com", "au"), ("net", "au"), ("org", "au"),
    ("com", "ru")
}

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MULTISPACE_RE = re.compile(r"\s+")


def normalise_text(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = unicodedata.normalize("NFKC", str(value)).replace("\xa0", " ")
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text or None


def fallback_registered_domain(domain: str) -> tuple[str, str, str]:
    parts = domain.lower().split(".")
    if len(parts) < 2:
        return "", domain.lower(), ""
    if len(parts) >= 3 and tuple(parts[-2:]) in COMMON_TWO_LEVEL_SUFFIXES:
        subdomain = ".".join(parts[:-3])
        registered = parts[-3]
        suffix = ".".join(parts[-2:])
        return subdomain, registered, suffix
    subdomain = ".".join(parts[:-2])
    registered = parts[-2]
    suffix = parts[-1]
    return subdomain, registered, suffix


def parse_email(email: object) -> dict:
    text = normalise_text(email)
    if not text:
        return {"valid": False, "email": None}

    text = text.lower()
    if not EMAIL_RE.match(text):
        return {"valid": False, "email": text}

    local, domain = text.rsplit("@", 1)

    if tldextract is not None:
        ext = tldextract.extract(domain)
        subdomain = ext.subdomain
        registered = ext.domain
        suffix = ext.suffix
    else:
        subdomain, registered, suffix = fallback_registered_domain(domain)

    return {
        "valid": True,
        "email": text,
        "local": local,
        "domain": domain,
        "subdomain": subdomain,
        "registered_domain": registered,
        "suffix": suffix,
        "numeric_local": local.isdigit(),
        "public_provider": registered in PUBLIC_PROVIDERS,
        "generic_local": local in GENERIC_LOCALPARTS,
    }


def strip_annotation(full_name: Optional[str]) -> Optional[str]:
    if not full_name:
        return None
    # Split only when the dash acts as a separator, not inside Jean-Luc.
    return re.split(r"\s[-–—]\s*", full_name, maxsplit=1)[0].strip()


def candidate_person_string(full_name: object, email_meta: dict) -> tuple[Optional[str], Optional[str]]:
    full_name_text = strip_annotation(normalise_text(full_name))

    if full_name_text:
        # Drop trailing standalone counters such as "2"
        full_name_text = re.sub(r"\b\d+\b$", "", full_name_text).strip()

    if full_name_text and "@" not in full_name_text and any(ch.isalpha() for ch in full_name_text):
        return full_name_text, "full_name"

    # Fallback to email local-part if it looks person-like.
    local = email_meta.get("local")
    if email_meta.get("valid") and local and not email_meta.get("numeric_local"):
        if any(sep in local for sep in "._-") and any(ch.isalpha() for ch in local):
            candidate = re.sub(r"[._-]+", " ", local).strip()
            return candidate, "email_local"

    return None, None


def looks_organisation_like(candidate: str) -> bool:
    lower = candidate.casefold()

    org_keywords = {
        "holding", "group", "ltd", "llc", "inc", "corp", "company",
        "ajansı", "ajansi", "odası", "odasi", "birliği", "birligi",
        "bank", "bankası", "bankasi", "travel", "tourism",
        "tekstil", "export", "import", "engineering"
    }

    if any(keyword in lower for keyword in org_keywords):
        return True

    # All-uppercase multi-token strings are often organisations/departments.
    if candidate == candidate.upper() and len(candidate.split()) >= 3:
        # Keep obvious person names out of this trap where possible.
        alpha_tokens = [t for t in candidate.split() if any(ch.isalpha() for ch in t)]
        if len(alpha_tokens) < 2:
            return True

    return False


def proper_case_email_token(token: str) -> str:
    parts = re.split(r"([-'])", token)
    out = []
    for part in parts:
        if part in {"-", "'"}:
            out.append(part)
        else:
            out.append(part[:1].upper() + part[1:].lower() if part else part)
    return "".join(out)


def split_name(full_name: object, email: object) -> dict:
    email_meta = parse_email(email)

    if email_meta.get("valid") and email_meta.get("numeric_local"):
        return {
            "name": None,
            "surname": None,
            "name_source": "unchanged_numeric_local",
            "review_reason": ""
        }

    candidate, source = candidate_person_string(full_name, email_meta)
    if not candidate:
        return {
            "name": None,
            "surname": None,
            "name_source": "no_person_name",
            "review_reason": "No usable personal name"
        }

    if looks_organisation_like(candidate):
        return {
            "name": None,
            "surname": None,
            "name_source": "organisation_like",
            "review_reason": "Organisation-like Full Name"
        }

    tokens = [t for t in candidate.split() if t]

    if source == "email_local":
        tokens = [proper_case_email_token(t) for t in tokens]

    if len(tokens) == 1:
        return {
            "name": tokens[0],
            "surname": None,
            "name_source": source,
            "review_reason": "Single-token name"
        }

    surname_tokens = [tokens[-1]]
    i = len(tokens) - 2

    while i >= 1 and tokens[i].casefold() in SURNAME_PARTICLES:
        surname_tokens.insert(0, tokens[i])
        i -= 1

    given_tokens = tokens[:i + 1]

    return {
        "name": " ".join(given_tokens),
        "surname": " ".join(surname_tokens),
        "name_source": source,
        "review_reason": "Ambiguous multi-token name" if len(tokens) >= 4 else ""
    }


def infer_company(email: object) -> dict:
    email_meta = parse_email(email)

    if not email_meta.get("valid"):
        return {
            "company": None,
            "company_source": "malformed_or_missing_email",
            "review_reason": "Malformed or missing email"
        }

    if email_meta.get("numeric_local"):
        return {
            "company": None,
            "company_source": "unchanged_numeric_local",
            "review_reason": ""
        }

    if email_meta.get("public_provider"):
        return {
            "company": None,
            "company_source": "public_email_provider",
            "review_reason": "Public email domain; company not inferred"
        }

    base = (email_meta["registered_domain"] or "").replace("-", " ").replace("_", " ").strip()
    company = " ".join(word[:1].upper() + word[1:].lower() for word in base.split()) if base else None

    return {
        "company": company,
        "company_source": "email_domain",
        "review_reason": ""
    }


def process_row(row: pd.Series) -> pd.Series:
    name_result = split_name(row.get("Full Name"), row.get("email"))
    company_result = infer_company(row.get("email"))

    # Preserve row if local-part is numeric-only.
    if name_result["name_source"] == "unchanged_numeric_local":
        return pd.Series({
            "name": row.get("name"),
            "surname": row.get("surname"),
            "company": row.get("company"),
            "_name_source": name_result["name_source"],
            "_company_source": company_result["company_source"],
            "_review_reason": "",
        })

    review_reasons = [r for r in [name_result["review_reason"], company_result["review_reason"]] if r]

    return pd.Series({
        "name": name_result["name"],
        "surname": name_result["surname"],
        "company": company_result["company"],
        "_name_source": name_result["name_source"],
        "_company_source": company_result["company_source"],
        "_review_reason": " | ".join(review_reasons),
    })


def process_workbook(input_path: Path, output_path: Path, sheet_name: Optional[str] = None) -> None:
    logging.info("Reading workbook: %s", input_path)

    df = pd.read_excel(input_path, sheet_name=sheet_name).copy()
    df.columns = [str(c).strip() for c in df.columns]

    expected = {"email", "Full Name", "name", "surname", "company", "Job Title"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    # Duplicate flags
    email_key = df["email"].fillna("").astype(str).str.strip().str.lower()
    df["_duplicate_email"] = email_key.duplicated(keep=False) & email_key.ne("")
    df["_duplicate_row"] = df.duplicated(keep=False)

    # Transform rows
    transformed = df.apply(process_row, axis=1)
    for col in ["name", "surname", "company", "_name_source", "_company_source", "_review_reason"]:
        df[col] = transformed[col]

    # Any flagged condition gets routed to review
    df["_needs_review"] = (
        df["_duplicate_email"]
        | df["_duplicate_row"]
        | df["_review_reason"].fillna("").ne("")
    )

    review_df = df[df["_needs_review"]].copy()
    deduped_df = df.drop_duplicates(subset=["email", "Full Name"], keep="first").copy()

    logging.info("Writing output workbook: %s", output_path)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="processed", index=False)
        review_df.to_excel(writer, sheet_name="review", index=False)
        deduped_df.to_excel(writer, sheet_name="deduped_optional", index=False)

    logging.info("Done. Processed %d rows; review queue has %d rows.", len(df), len(review_df))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to the input Excel workbook")
    parser.add_argument("output_path", help="Path to the output Excel workbook")
    parser.add_argument("--sheet", default=None, help="Sheet name; default is the first sheet")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    process_workbook(Path(args.input_path), Path(args.output_path), args.sheet)


if __name__ == "__main__":
    main()
