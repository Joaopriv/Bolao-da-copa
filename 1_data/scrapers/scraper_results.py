"""Scraper de resultados históricos de seleções — dataset martj42/international_results.

Fonte gratuita e confiável (1872–2026), atualizada com frequência. Baixa três CSVs:
  - results.csv      : todos os jogos (inclui os 72 fixtures da Copa 2026 com placar NaN)
  - shootouts.csv    : decisões por pênaltis (útil para mata-mata em iterações futuras)
  - goalscorers.csv  : gols individuais (reservado para análises futuras)

NÃO usa scraping frágil de HTML — é download direto de CSV cru via HTTPS.

Se SUPABASE_URL/SUPABASE_KEY estiverem configuradas, results.csv também é sincronizado
(UPSERT, sem duplicar) para a tabela `matches` (ver supabase/schema.sql).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

# Permite importar config_loader/db_client/preprocess independente de onde o script é
# chamado.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_loader import load_config, path  # noqa: E402
import db_client  # noqa: E402
import preprocess  # noqa: E402


def download() -> dict[str, Path]:
    """Baixa todos os CSVs do martj42 para data.raw_dir. Retorna os caminhos salvos."""
    cfg = load_config()["data"]
    raw_dir = path(cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}
    for fname in cfg["files"]:
        url = f"{cfg['source_url']}/{fname}"
        dest = raw_dir / fname
        print(f"  baixando {fname} ...", end=" ", flush=True)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        n_lines = resp.content.count(b"\n")
        print(f"ok ({n_lines:,} linhas) -> {dest}")
        saved[fname] = dest

    _sync_to_supabase(saved["results.csv"])
    return saved


def _sync_to_supabase(results_path: Path) -> None:
    """UPSERT de results.csv (normalizado) na tabela `matches`. No-op sem credenciais."""
    if db_client.get_client() is None:
        return

    cfg = load_config()
    raw = pd.read_csv(results_path)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = preprocess.normalize_names(raw, cfg.get("team_aliases", {}))
    raw["neutral"] = preprocess.to_neutral_int(raw["neutral"])

    # A chave única da tabela é (date, home_team, away_team, tournament); o dataset martj42
    # tem raríssimas colisões nessa chave (duplicatas exatas ou 2 jogos distintos no mesmo
    # dia/torneio). Mantém a primeira ocorrência para o upsert não falhar.
    key_cols = ["date", "home_team", "away_team", "tournament"]
    n_dups = raw.duplicated(subset=key_cols, keep="first").sum()
    if n_dups:
        print(f"\n  ({n_dups} linha(s) com chave (date,home,away,tournament) duplicada "
              f"no dataset martj42 — mantendo a primeira ocorrência)")
        raw = raw.drop_duplicates(subset=key_cols, keep="first")

    rows = []
    for _, r in raw.iterrows():
        rows.append({
            "date": r["date"].strftime("%Y-%m-%d"),
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "home_score": None if pd.isna(r["home_score"]) else int(r["home_score"]),
            "away_score": None if pd.isna(r["away_score"]) else int(r["away_score"]),
            "tournament": r["tournament"],
            "neutral": bool(r["neutral"]),
            "source": "martj42",
        })

    print(f"  sincronizando {len(rows):,} jogos com Supabase (tabela matches) ...",
          end=" ", flush=True)
    db_client.upsert("matches", rows, on_conflict="date,home_team,away_team,tournament")
    print("ok")


if __name__ == "__main__":
    download()
