"""[Iteração 2 / D3] RAPM-lite: impacto de jogador no CLUBE via Ridge em nível de temporada.

Adaptação "lite" do RAPM clássico (que opera em nível de stint/substituição): o
Understat (D1) só fornece agregados por TEMPORADA, então a unidade de observação aqui
é (time, liga, temporada) — `team_seasons` (fonte: y) — e a "presença em campo" de cada
jogador é sua fração de minutos disponíveis nessa temporada — `player_seasons` (fonte: X).

X[ts, p] = minutos_jogador / (games_time_temporada * 90)
y_attack[ts]  = team_seasons.xg90   (Δ esperado se p jogasse 100% dos minutos)
y_defense[ts] = team_seasons.xga90

Ridge separado para ataque/defesa (alpha por RidgeCV sobre cfg.rapm.alpha_grid).
defense_delta = -coef_defense (sinal: positivo = defesa BOA, consistente com o resto
do projeto). std_error = bootstrap (resample de linhas time-temporada, cfg.rapm.bootstrap_iters).

Linhas com `team` contendo vírgula (jogador transferido em meio à temporada — Understat
retorna `team_title="ClubeA,ClubeB"`, ~3% das linhas) são DESCARTADAS do design matrix:
não há como atribuir a fração de minutos a um único time-temporada sem inventar dados.
Esses jogadores continuam aparecendo via outras temporadas/times, se houver.

Saída: player_impact (source='rapm_club').
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_loader import load_config  # noqa: E402
import db_client  # noqa: E402


def build_design_matrix(player_seasons: list[dict], team_seasons: list[dict]):
    """Retorna (X csr esparsa [n_team_seasons x n_players], y_attack, y_defense, player_ids)."""
    ts_index: dict[tuple, dict] = {}
    for r in team_seasons:
        key = (r["team"], r["league"], r["season"])
        ts_index[key] = r

    ts_keys = sorted(ts_index.keys())
    ts_pos = {k: i for i, k in enumerate(ts_keys)}

    player_ids = sorted({r["player_id"] for r in player_seasons if "," not in r["team"]})
    p_pos = {pid: j for j, pid in enumerate(player_ids)}

    rows, cols, data = [], [], []
    for r in player_seasons:
        if "," in r["team"]:
            continue
        key = (r["team"], r["competition"], r["season"])
        ts = ts_index.get(key)
        if ts is None or not ts.get("games"):
            continue
        frac = (r["minutes"] or 0.0) / (ts["games"] * 90.0)
        if frac <= 0:
            continue
        rows.append(ts_pos[key])
        cols.append(p_pos[r["player_id"]])
        data.append(frac)

    n_ts, n_p = len(ts_keys), len(player_ids)
    X = sparse.csr_matrix((data, (rows, cols)), shape=(n_ts, n_p))
    y_attack = np.array([ts_index[k]["xg90"] for k in ts_keys], dtype=float)
    y_defense = np.array([ts_index[k]["xga90"] for k in ts_keys], dtype=float)
    return X, y_attack, y_defense, player_ids


def select_alpha(X, y, alpha_grid: list[float], cv_folds: int, seed: int = 42) -> float:
    model = RidgeCV(alphas=alpha_grid, cv=KFold(n_splits=cv_folds, shuffle=True, random_state=seed))
    model.fit(X, y)
    return float(model.alpha_)


def fit_rapm(X, y_attack, y_defense, player_ids, alpha_grid, cv_folds, seed=42) -> dict:
    alpha_attack = select_alpha(X, y_attack, alpha_grid, cv_folds, seed)
    alpha_defense = select_alpha(X, y_defense, alpha_grid, cv_folds, seed)
    m_attack = Ridge(alpha=alpha_attack).fit(X, y_attack)
    m_defense = Ridge(alpha=alpha_defense).fit(X, y_defense)
    return {
        "alpha_attack": alpha_attack, "alpha_defense": alpha_defense,
        "attack_delta": dict(zip(player_ids, m_attack.coef_)),
        "defense_delta": dict(zip(player_ids, -m_defense.coef_)),
    }


def bootstrap_std_error(X, y_attack, y_defense, alpha_attack, alpha_defense,
                         player_ids, iters: int, seed: int = 42) -> dict[str, float]:
    """Reamostra LINHAS (time-temporada) com reposição; std_error = max(se_attack, se_defense)."""
    rng = np.random.default_rng(seed)
    n, n_p = X.shape
    attack_samples = np.empty((iters, n_p))
    defense_samples = np.empty((iters, n_p))
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]
        ma = Ridge(alpha=alpha_attack).fit(Xb, y_attack[idx])
        md = Ridge(alpha=alpha_defense).fit(Xb, y_defense[idx])
        attack_samples[i] = ma.coef_
        defense_samples[i] = -md.coef_
    se_attack = attack_samples.std(axis=0, ddof=1)
    se_defense = defense_samples.std(axis=0, ddof=1)
    return {pid: float(max(se_attack[j], se_defense[j])) for j, pid in enumerate(player_ids)}


def compute_rapm(verbose: bool = True) -> list[dict]:
    cfg = load_config()
    player_seasons = [r for r in (db_client.fetch_all("player_seasons") or []) if r["source"] == "understat"]
    team_seasons = [r for r in (db_client.fetch_all("team_seasons") or []) if r["source"] == "understat"]
    if not player_seasons or not team_seasons:
        print("  player_seasons/team_seasons (source=understat) vazios — rode --scrape-understat antes.")
        return []

    X, y_attack, y_defense, player_ids = build_design_matrix(player_seasons, team_seasons)

    alpha_grid = cfg["rapm"]["alpha_grid"]
    cv_folds = cfg["rapm"]["cv_folds"]
    iters = cfg["rapm"]["bootstrap_iters"]
    seed = cfg["validation"]["random_seed"]

    if verbose:
        print(f"  design matrix: {X.shape[0]} time-temporadas x {X.shape[1]} jogadores "
              f"({X.nnz} entradas não-nulas)")

    fit = fit_rapm(X, y_attack, y_defense, player_ids, alpha_grid, cv_folds, seed)
    if verbose:
        print(f"  alpha (RidgeCV, grid={alpha_grid}): ataque={fit['alpha_attack']}, "
              f"defesa={fit['alpha_defense']}")

    std_errors = bootstrap_std_error(X, y_attack, y_defense, fit["alpha_attack"], fit["alpha_defense"],
                                      player_ids, iters, seed)

    rows = []
    for pid in player_ids:
        rows.append({
            "player_id": pid,
            "attack_delta": round(float(fit["attack_delta"][pid]), 4),
            "defense_delta": round(float(fit["defense_delta"][pid]), 4),
            "std_error": round(std_errors[pid], 4),
            "source": "rapm_club",
        })
    return rows


def run(verbose: bool = True) -> None:
    rows = compute_rapm(verbose=verbose)
    if rows:
        db_client.upsert("player_impact", rows, on_conflict="player_id,source")
    print(f"  Upsert: {len(rows)} player_impact (source=rapm_club).")


if __name__ == "__main__":
    run()
