from __future__ import annotations

import argparse
import io
import re
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARXIV_EPRINT_URL = "https://arxiv.org/e-print/1308.2257"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "vandenberg2013_gc_ages.csv"


def _clean_latex_cell(cell: str) -> str:
    cleaned = cell.strip()
    cleaned = cleaned.replace(r"\phantom{0}", "")
    cleaned = cleaned.replace(r"\phantom{1}", "")
    cleaned = cleaned.replace(r"\,", " ")
    cleaned = cleaned.replace(r"$", "")
    cleaned = cleaned.replace("{", "")
    cleaned = cleaned.replace("}", "")
    cleaned = cleaned.replace("\\", "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _catalog_key(ngc: str, name: str) -> str:
    if ngc:
        return f"NGC {int(ngc)}"
    normalized = name.upper().replace("M ", "M ")
    aliases = {
        "TER 8": "TERZAN 8",
    }
    return aliases.get(normalized, normalized)


def _fetch_source_tex() -> str:
    with urllib.request.urlopen(ARXIV_EPRINT_URL, timeout=60) as response:
        payload = response.read()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
        member = archive.extractfile("ms.tex")
        if member is None:
            raise FileNotFoundError("ms.tex was not found in the arXiv source archive.")
        return member.read().decode("utf-8")


def _load_source_tex(path: Path | None) -> str:
    if path is not None:
        return path.read_text()
    return _fetch_source_tex()


def _extract_table_rows(source: str) -> list[str]:
    label_index = source.find(r"\label{tab:tab2}")
    if label_index < 0:
        raise ValueError("Could not find VandenBerg et al. Table 2 label.")
    start = source.find(r"\startdata", label_index)
    end = source.find(r"\enddata", start)
    if start < 0 or end < 0:
        raise ValueError("Could not find Table 2 start/end markers.")
    table_text = source[start + len(r"\startdata") : end]
    rows = []
    current = ""
    for raw_line in table_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        current = f"{current} {line}".strip()
        if r"\\" in line:
            rows.append(current)
            current = ""
    return rows


def parse_vandenberg_ages(source: str) -> pd.DataFrame:
    rows = []
    for raw_row in _extract_table_rows(source):
        row = raw_row.split(r"\\", 1)[0].strip()
        cells = [_clean_latex_cell(cell) for cell in row.split("&")]
        if len(cells) != 12:
            raise ValueError(f"Expected 12 columns, got {len(cells)} in row: {raw_row}")
        ngc, name, feh, age, method, figures, age_range, hb_type, rg, mv, ve0, logsigma0 = cells
        match = re.match(r"(?P<age>[0-9.]+)\s*pm\s*(?P<err>[0-9.]+)", age)
        if match is None:
            raise ValueError(f"Could not parse age cell: {age!r}")
        ngc_value = int(ngc) if ngc else None
        cluster_name = name if name else (f"NGC {ngc_value}" if ngc_value is not None else "")
        rows.append(
            {
                "source": "VandenBerg et al. 2013",
                "ngc": ngc_value,
                "name": cluster_name,
                "catalog_match_key": _catalog_key(ngc, cluster_name),
                "vandenberg_feh": float(feh),
                "age_gyr": float(match.group("age")),
                "age_error_gyr": float(match.group("err")),
                "age_method": method,
                "figures": figures,
                "independent_age_range_gyr": age_range,
                "hb_type": float(hb_type),
                "galactocentric_radius_kpc": float(rg),
                "absolute_v_magnitude": float(mv),
                "central_escape_velocity_kms": float(ve0),
                "log10_central_velocity_dispersion": float(logsigma0),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-latex", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = _load_source_tex(args.source_latex)
    table = parse_vandenberg_ages(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote {len(table)} VandenBerg et al. age rows to {args.output}")


if __name__ == "__main__":
    main()
