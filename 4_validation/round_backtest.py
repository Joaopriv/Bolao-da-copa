"""[Iteração 2 / Prompt E5] Backtest WC2022 por rodada.

Compara, rodada a rodada, um modelo re-treinado a cada rodada com os resultados reais
já jogados (sequencial -- mesma lógica de `--update-round`) contra o mesmo modelo
treinado UMA ÚNICA VEZ antes do torneio (estático -- mesma lógica de `--compare`).

Objetivo: medir se o retreino por rodada (PASSO2b/3 de `--update-round`) reduz o RPS
antes de operar ao vivo na Copa 2026 (instrução permanente do E5 -- PAUSA OBRIGATÓRIA).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import load_config, path  # noqa: E402
import dataset  # noqa: E402
import models_registry as reg  # noqa: E402
import metrics as M  # noqa: E402
import significance  # noqa: E402

_RES_IDX = {"H": 0, "D": 1, "A": 2}

# Estrutura real do WC2022: 3 rodadas de grupo (16 jogos cada) + oitavas(8) + quartas(4)
# + semis(2) + final/3º lugar(2) = 64 jogos.
_BLOCK_SIZES = [16, 16, 16, 8, 4, 2, 2]


def _load_chosen_model_name() -> str:
    sel_path = path("5_outputs", "selected_model.json")
    if sel_path.exists():
        return json.loads(sel_path.read_text(encoding="utf-8")).get("chosen_model", "poisson")
    return "poisson"


def _predict_block(model, block) -> tuple[np.ndarray, np.ndarray]:
    P, y = [], []
    for _, r in block.iterrows():
        home, away = str(r["home_team"]), str(r["away_team"])
        P.append(model.predict_proba(home, away, bool(r["neutral"])))
        y.append(_RES_IDX[r["result"]])
    return np.asarray(P, dtype=float), np.asarray(y, dtype=int)


def run(tournament: str = "WC2022", model_name: str | None = None, verbose: bool = True) -> dict:
    """Sequencial (retreina a cada rodada com `dataset.training_frame(ref_date)`
    expandido pelas rodadas já jogadas) vs estático (treinado uma vez em `spec["start"]`).

    Retorna {"rows": [...], "trend_falling": bool, "model": str}.
    """
    cfg = load_config()
    lam = cfg["preprocess"]["temporal_decay_lambda"]
    vcfg = cfg["validation"]
    iters, seed = vcfg["bootstrap_iterations"], vcfg["random_seed"]

    spec = next(s for s in vcfg["test_tournaments"] if s["name"] == tournament)
    games = dataset.get_test_tournament(spec)
    if len(games) != sum(_BLOCK_SIZES):
        raise ValueError(f"{tournament}: esperado {sum(_BLOCK_SIZES)} jogos, "
                          f"encontrado {len(games)}")

    model_name = model_name or _load_chosen_model_name()
    if verbose:
        print(f"● Backtest {tournament} por rodada -- modelo '{model_name}' "
              f"(sequencial vs estático) ...")

    blocks, idx = [], 0
    for size in _BLOCK_SIZES:
        blocks.append(games.iloc[idx:idx + size])
        idx += size

    train_static = dataset.training_frame(spec["start"], lam=lam)
    model_static = reg.build_member(model_name, cfg).fit(train_static)

    rows = []
    rps_seq_means = []
    for i, block in enumerate(blocks[1:], start=2):
        ref_date = block.iloc[0]["date"]
        train_seq = dataset.training_frame(ref_date, lam=lam)
        model_seq = reg.build_member(model_name, cfg).fit(train_seq)

        P_seq, y = _predict_block(model_seq, block)
        P_static, _ = _predict_block(model_static, block)

        rps_seq = M.rps_vector(P_seq, y)
        rps_static = M.rps_vector(P_static, y)
        cmp = significance.compare_pair(rps_seq, rps_static, "sequencial", "estatico",
                                          iters=iters, seed=seed)

        rps_seq_means.append(float(rps_seq.mean()))
        rows.append({
            "round": i, "n_games": len(block),
            "rps_seq": float(rps_seq.mean()), "rps_static": float(rps_static.mean()),
            "diff_lo": cmp["diff_lo"], "diff_hi": cmp["diff_hi"],
            "crosses_zero": cmp["crosses_zero"],
        })
        if verbose:
            print(f"  rodada {i}: {len(block)} jogos | RPS seq={rps_seq.mean():.4f} | "
                  f"RPS estático={rps_static.mean():.4f} | "
                  f"IC95% diff=[{cmp['diff_lo']:+.4f}, {cmp['diff_hi']:+.4f}]")

    trend_falling = rps_seq_means[-1] < rps_seq_means[0]
    return {"rows": rows, "trend_falling": trend_falling, "model": model_name}


def format_report(result: dict) -> str:
    """Monta o relatório da PAUSA OBRIGATÓRIA (instrução permanente do E5), em PT-BR."""
    lines = []
    lines.append("━━ BACKTEST WC2022 POR RODADA ━━")
    lines.append("")
    lines.append(f" Modelo: {result['model']}")
    lines.append("")
    lines.append(" Rodada | Jogos | RPS sequencial | RPS estático | IC95% diff             | Conclusão")
    lines.append(" " + "─" * 86)
    for row in result["rows"]:
        if row["crosses_zero"]:
            conclusao = "≈ empate estatístico"
        elif row["rps_seq"] < row["rps_static"]:
            conclusao = "✅ sequencial melhor"
        else:
            conclusao = "⚠ sequencial pior"
        lines.append(
            f" {row['round']:6d} | {row['n_games']:5d} | {row['rps_seq']:>15.4f} "
            f"| {row['rps_static']:>12.4f} "
            f"| [{row['diff_lo']:+.4f}, {row['diff_hi']:+.4f}] | {conclusao}"
        )
    lines.append("")
    tendencia = "cai" if result["trend_falling"] else "não cai"
    lines.append(f" Tendência agregada: RPS sequencial {tendencia} ao longo das rodadas.")
    lines.append("")
    lines.append(" Interpretação:")
    lines.append(" RPS seq cai a cada rodada → retreino por rodada ajuda → ✅ liberar")
    lines.append(" RPS seq não cai / piora  → retreino não está ajudando → ⚠ reportar")
    lines.append("")
    lines.append(" ⚠ IMPORTANTE: mesmo que RPS sequencial não melhore, o --update-round")
    lines.append(" ainda tem valor operacional (inserir resultados, atualizar odds,")
    lines.append(" regenerar JSON para o frontend). O retreino por rodada é que pode")
    lines.append(" não estar contribuindo — reportar honestamente ao usuário antes")
    lines.append(" de prosseguir para a Copa 2026 ao vivo.")
    return "\n".join(lines)


if __name__ == "__main__":
    res = run()
    print()
    print(format_report(res))
