from pathlib import Path
import pandas as pd


candidates = [
    Path("data/processed/normalized_matches.parquet"),
    Path("data/processed/normalized_matches.csv"),
    Path("data/interim/tennis_matches_raw.data"),
    Path("data/interim/tennis_matches_raw.csv"),
]

data_path = next((p for p in candidates if p.exists()), None)
if data_path is None:
    raise FileNotFoundError("Nessun dataset trovato nei path candidati")

print(f"Uso dataset: {data_path}")

if data_path.suffix == ".parquet":
    df = pd.read_parquet(data_path)
else:
    df = pd.read_csv(data_path)

print(df.shape)
print(df.columns.tolist())
print(df.head(3))