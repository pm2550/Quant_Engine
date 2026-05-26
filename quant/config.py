from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
PRICES_DIR = DATA_DIR / "prices"
RESULTS_DIR = ROOT / "results"
LOG_DIR = ROOT / "logs"

def load(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def all_symbols(portfolio: dict) -> list[str]:
    pos = list(portfolio.get("positions", {}).keys())
    watch = [w["symbol"] for w in portfolio.get("watchlist", [])]
    return list(dict.fromkeys(pos + watch))
