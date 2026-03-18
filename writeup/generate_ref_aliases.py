#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


def parse_bib_entries(text: str) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    i = 0
    n = len(text)

    while i < n:
        start = text.find("@", i)
        if start == -1:
            break

        brace = text.find("{", start)
        if brace == -1:
            break

        kind = text[start + 1:brace].strip().lower()
        if kind in {"comment", "preamble", "string"}:
            i = brace + 1
            continue

        depth = 0
        end = None
        for j in range(brace, n):
            char = text[j]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break

        if end is None:
            raise ValueError("Unbalanced braces while parsing BibTeX input.")

        entry = text[start:end].strip()
        match = re.match(r"@\w+\{([^,]+),", entry, re.S)
        if match:
            entries[match.group(1).strip()] = entry

        i = end

    return entries


def rewrite_key(entry: str, alias_key: str) -> str:
    return re.sub(r"^(@\w+\{)\s*([^,]+)", rf"\1{alias_key}", entry, count=1, flags=re.S)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate legacy citekey aliases from a Zotero export.")
    parser.add_argument("--input", required=True, help="Path to the synced Zotero BibTeX file.")
    parser.add_argument("--mapping", required=True, help="JSON map from legacy keys to canonical Zotero keys.")
    parser.add_argument("--output", required=True, help="Path to write the alias bibliography.")
    args = parser.parse_args()

    source_path = Path(args.input)
    mapping_path = Path(args.mapping)
    output_path = Path(args.output)

    entries = parse_bib_entries(source_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    rendered_entries: List[str] = []
    missing: List[str] = []

    for alias_key, canonical_key in sorted(mapping.items()):
        entry = entries.get(canonical_key)
        if entry is None:
            missing.append(f"{alias_key} -> {canonical_key}")
            continue
        rendered_entries.append(rewrite_key(entry, alias_key))

    if missing:
        missing_text = "\n".join(missing)
        raise SystemExit(f"Missing canonical entries in synced bibliography:\n{missing_text}")

    output_path.write_text(
        "% Auto-generated legacy citekey aliases for carm_proposal.tex.\n"
        "% Regenerate with ./sync_zotero_bib.sh.\n\n"
        + "\n\n".join(rendered_entries)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rendered_entries)} citekey aliases to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
