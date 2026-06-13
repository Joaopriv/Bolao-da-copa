"""Pré-processamento: results.csv cru -> matches_weighted.csv limpo e ponderado.

Decisões de modelagem:
- Normaliza nomes de seleção via tabela de aliases do config.yaml.
- Filtra a partir de `data.min_date` (janela de relevância).
- Separa jogos JOGADOS (com placar) dos FIXTURES da Copa 2026 (placar NaN, data futura).
- Grava o peso de competição ESTÁTICO (w_comp). O peso TEMPORAL depende da data de
  referência (início do torneio no backtest, ou hoje na previsão), então é calculado
  sob demanda em `dataset.py` — NÃO é congelado aqui (evita vazamento e rigidez).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_loader import load_config, path  # noqa: E402


def normalize_names(df: pd.DataFrame, aliases: dict) -> pd.DataFrame:
    """Normaliza home_team/away_team via tabela de aliases (config.yaml: team_aliases)."""
    df = df.copy()
    df["home_team"] = df["home_team"].replace(aliases)
    df["away_team"] = df["away_team"].replace(aliases)
    return df


def to_neutral_int(series: pd.Series) -> pd.Series:
    """Normaliza a coluna `neutral` para 0/1, vinda de CSV (string) ou Supabase (bool)."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    return series.astype(str).str.upper().isin(["TRUE", "1", "T"]).astype(int)


def _result_label(row) -> str | float:
    if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
        return np.nan
    if row["home_score"] > row["away_score"]:
        return "H"
    if row["home_score"] < row["away_score"]:
        return "A"
    return "D"


def derive_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Adiciona played/result/w_comp/is_future/is_wc2026_fixture a partir das colunas cruas.

    Esperado em `df`: date, home_team, away_team, home_score, away_score, tournament,
    neutral (já em 0/1). Compartilhado entre o caminho CSV (build) e o caminho Supabase
    (dataset.load_matches), para que ambas as fontes produzam o mesmo schema.
    """
    pcfg = cfg["preprocess"]
    today = pd.Timestamp(cfg["data"]["today"])
    df = df.copy()

    df["played"] = df["home_score"].notna() & df["away_score"].notna()
    df["is_wc2026_fixture"] = (
        (df["tournament"] == cfg["predict_2026"]["tournament_label"])
        & (df["date"] >= pd.Timestamp(f"{cfg['predict_2026']['season_year']}-01-01"))
    )
    df["result"] = df.apply(_result_label, axis=1)

    # Peso de competição (estático). Torneios não listados usam o default.
    comp_w = pcfg["competition_weights"]
    default_w = pcfg["default_competition_weight"]
    df["w_comp"] = df["tournament"].map(comp_w).fillna(default_w)

    # Sanidade: previsões só fazem sentido para jogos até hoje (treino) + fixtures futuros.
    df["is_future"] = df["date"] > today
    return df


def build() -> Path:
    """Constrói matches_weighted.csv a partir do results.csv cru. Retorna o caminho."""
    cfg = load_config()
    dcfg = cfg["data"]

    raw = pd.read_csv(path(dcfg["raw_dir"], "results.csv"))
    raw["date"] = pd.to_datetime(raw["date"])
    raw = normalize_names(raw, cfg.get("team_aliases", {}))

    min_date = pd.Timestamp(dcfg["min_date"])
    df = raw[raw["date"] >= min_date].copy()
    df["neutral"] = to_neutral_int(df["neutral"])
    df = derive_columns(df, cfg)

    cols = [
        "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "neutral", "played", "result", "w_comp",
        "is_future", "is_wc2026_fixture",
    ]
    out = df[cols].sort_values("date").reset_index(drop=True)

    dest = path(dcfg["processed_dir"], "matches_weighted.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)

    n_played = int(out["played"].sum())
    n_fix = int(out["is_wc2026_fixture"].sum())
    print(f"  matches_weighted.csv: {len(out):,} jogos ({n_played:,} jogados, "
          f"{n_fix} fixtures Copa 2026) -> {dest}")
    return dest


if __name__ == "__main__":
    build()
