"""Acesso aos jogos (Supabase, fonte de verdade; CSV como fallback) + peso temporal sob
demanda.

Ponte entre os dados processados e os modelos/backtest. O peso temporal
w_t = exp(-lambda * dias_antes_da_referencia) é calculado em relação a uma data de
referência (início do torneio no backtest, ou hoje na previsão da Copa) para garantir
splits cronológicos SEM vazamento: ao prever um torneio, só entram jogos anteriores a ele.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "1_data"))

from config_loader import load_config, path  # noqa: E402
import db_client  # noqa: E402
import preprocess  # noqa: E402

_MATCH_COLS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "home_xg", "away_xg",
    "tournament", "neutral", "played", "result", "w_comp",
    "is_future", "is_wc2026_fixture",
]


@lru_cache(maxsize=1)
def load_matches() -> pd.DataFrame:
    """Carrega os jogos do Supabase (tabela `matches`) se configurado, senão do CSV
    matches_weighted.csv. Ambas as fontes passam por `preprocess.derive_columns` para
    produzir o mesmo schema.

    Cacheado: dentro de uma execução (--compare/--select/--predict-2026), os dados não
    mudam, e split_for_tournament/run_model_on_tournament chamam isto várias vezes por
    torneio — sem cache eram dezenas de fetches paginados ao Supabase + reprocessamento
    completo do dataset, dominando o tempo total.
    """
    cfg = load_config()
    rows = db_client.fetch_all("matches")
    if rows:
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df["neutral"] = preprocess.to_neutral_int(df["neutral"])
        df = preprocess.derive_columns(df, cfg)
        return df[_MATCH_COLS].sort_values("date").reset_index(drop=True)

    df = pd.read_csv(path(cfg["data"]["processed_dir"], "matches_weighted.csv"))
    df["date"] = pd.to_datetime(df["date"])
    if "home_xg" not in df.columns:
        df["home_xg"] = np.nan
    if "away_xg" not in df.columns:
        df["away_xg"] = np.nan
    return df


@lru_cache(maxsize=1)
def _w_overlap_map() -> dict[str, float]:
    """{team: w_overlap} via squad_strength (D6.1). {} se a tabela estiver vazia/ausente
    -- with_weights cai no default (preprocess.squad_overlap_weight), preservando o
    comportamento da Iteração 1."""
    rows = db_client.fetch_all("squad_strength")
    return {r["team"]: r["w_overlap"] for r in (rows or []) if r.get("w_overlap") is not None}


def with_weights(df: pd.DataFrame, ref_date, lam: float) -> pd.DataFrame:
    """Adiciona w_t e a coluna final `weight` = w_t * w_comp * w_overlap.

    w_overlap = média(squad_strength.w_overlap[home], squad_strength.w_overlap[away])
    (D6.1); times/seleções sem cobertura caem no default
    `preprocess.squad_overlap_weight` (1.0 na Iteração 1).
    """
    cfg = load_config()
    default_overlap = cfg["preprocess"]["squad_overlap_weight"]
    overlap_map = _w_overlap_map()
    ref = pd.Timestamp(ref_date)
    out = df.copy()
    days = (ref - out["date"]).dt.days.clip(lower=0)
    out["w_t"] = np.exp(-lam * days)
    h_ov = out["home_team"].map(overlap_map).fillna(default_overlap)
    a_ov = out["away_team"].map(overlap_map).fillna(default_overlap)
    out["weight"] = out["w_t"] * out["w_comp"] * (h_ov + a_ov) / 2
    return out


def training_frame(ref_date, lam: float, min_weight: float = 1e-6) -> pd.DataFrame:
    """Jogos JOGADOS estritamente antes de `ref_date`, com pesos calculados.

    Esta é a base de treino honesta para prever qualquer evento começando em ref_date.
    """
    df = load_matches()
    ref = pd.Timestamp(ref_date)
    train = df[(df["played"]) & (df["date"] < ref)].copy()
    train = with_weights(train, ref_date, lam)
    return train[train["weight"] > min_weight].reset_index(drop=True)


def get_test_tournament(spec: dict) -> pd.DataFrame:
    """Jogos JOGADOS de um torneio de teste: mesmo label + janela de datas do config."""
    df = load_matches()
    mask = (
        (df["tournament"] == spec["label"])
        & (df["date"] >= pd.Timestamp(spec["start"]))
        & (df["date"] <= pd.Timestamp(spec["end"]))
        & (df["played"])
    )
    return df[mask].sort_values("date").reset_index(drop=True)


def get_wc2026_fixtures() -> pd.DataFrame:
    """Os 72 jogos da fase de grupos da Copa 2026 (placar NaN, vindos do próprio dataset)."""
    df = load_matches()
    return df[df["is_wc2026_fixture"]].sort_values("date").reset_index(drop=True)
