from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_NAME = "Tennis-Data"
URL_TEMPLATE = "http://tennis-data.co.uk/{year}/{year}.xlsx"
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "manifest.json"


def sha256_file(path: Path) -> str | None:
	if not path.exists():
		return None

	digest = hashlib.sha256()

	with path.open("rb") as file:
		for chunk in iter(lambda: file.read(1024 * 1024), b""):
			digest.update(chunk)

	return digest.hexdigest()


def build_entry(year: int) -> dict[str, Any]:
	filename = f"{year}.xlsx"
	local_path = RAW_DIR / filename

	exists = local_path.exists()

	return {
		"dataset": "tennis_matches",
		"source_name": SOURCE_NAME,
		"year": year,
		"source_url": URL_TEMPLATE.format(year=year),
		"local_path": str(local_path.relative_to(REPO_ROOT)),
		"file_format": "xlsx",
		"exists_locally": exists,
		"file_size_bytes": local_path.stat().st_size if exists else None,
		"sha256": sha256_file(local_path),
		"created_at_utc": datetime.now(timezone.utc).isoformat(),
		"raw_data_policy": "immutable",
		"notes": "Yearly tennis match data used as raw input for winner prediction.",
	}


def create_manifest(start_year: int, end_year: int) -> Path:
	if start_year > end_year:
		raise ValueError("start-year must be <= end-year")

	RAW_DIR.mkdir(parents=True, exist_ok=True)

	manifest = {
		"manifest_version": 1,
		"project": "tennis-prediction",
		"description": "Raw data manifest for yearly Tennis-Data Excel files.",
		"generated_at_utc": datetime.now(timezone.utc).isoformat(),
		"sources": [
			build_entry(year)
			for year in range(start_year, end_year + 1)
		],
	}

	MANIFEST_PATH.write_text(
		json.dumps(manifest, indent=2, ensure_ascii=False),
		encoding="utf-8",
	)

	return MANIFEST_PATH


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--start-year", type=int, required=True)
	parser.add_argument("--end-year", type=int, required=True)
	args = parser.parse_args()

	manifest_path = create_manifest(args.start_year, args.end_year)
	print(f"Created {manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
	main()
