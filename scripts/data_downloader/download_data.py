from requests import request
import argparse
from pathlib import Path

from create_manifest import create_manifest

URL_TEMPLATE = "http://tennis-data.co.uk/{year}/{year}.xlsx"
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

def download_file(url: str, dest: Path) -> None:
	response = request("GET", url, stream=True)
	response.raise_for_status()

	dest.parent.mkdir(parents=True, exist_ok=True)

	with dest.open("wb") as file:
		for chunk in response.iter_content(chunk_size=8192):
			file.write(chunk)

def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--start-year", type=int, required=True)
	parser.add_argument("--end-year", type=int, required=True)
	args = parser.parse_args()

	if args.start_year > args.end_year:
		raise ValueError("start-year must be <= end-year")

	for year in range(args.start_year, args.end_year + 1):
		url = URL_TEMPLATE.format(year=year)
		dest = RAW_DIR / f"{year}.xlsx"
		download_file(url, dest)

	manifest_path = create_manifest(args.start_year, args.end_year)
	print(f"Created {manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
	main()
