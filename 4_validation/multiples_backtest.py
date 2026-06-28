"""[Backtest da ESTRATÉGIA de múltiplas] Responde "quantas múltiplas de alta
probabilidade teríamos acertado?" -- valida a múltipla inteira, não a perna isolada.

Para cada torneio de teste (walk-forward sem vazamento), monta uma perna por jogo
segundo o perfil (seguro/vitoria/gols), registra (prob_prevista, acertou?). Depois:

1. Calibração da PERNA por faixa de confiança (a base de tudo).
2. Calibração da MÚLTIPLA: amostra (bootstrap) muitas combinações de K pernas de jogos
   distintos, agrupa pela prob conjunta prevista (= produto) e compara com a taxa REAL
   de acerto (todas as K pernas certas). Se o modelo diz "múltipla de 70%" e na prática
   ~70% batem, a prob conjunta é confiável; se batem bem menos, as pernas não são tão
   independentes/calibradas quanto o produto assume.

Foca em ALTA probabilidade (o pedido do usuário): além do agrupamento por faixa, reporta
o headline "de K pernas entre as N de maior prob, quantas múltiplas bateriam".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "utils"))

from config_loader import load_config  # noqa: E402
import dataset  # noqa: E402
import models_registry as reg  # noqa: E402
import markets as MK  # noqa: E402


def _leg(home, away, neutral, hs, a, result, model, score_model, profile) -> dict | None:
    """(prob_prevista, acertou) da perna do jogo segundo o perfil."""
    p = model.predict_proba(home, away, neutral=neutral)
    pH, pD, pA = float(p[0]), float(p[1]), float(p[2])
    fav_home = pH >= pA

    if profile == "vitoria":
        prob = pH if fav_home else pA
        hit = (result == "H") if fav_home else (result == "A")
        return {"prob": prob, "hit": int(hit)}

    if profile == "gols":
        grid = score_model.predict_scoreline(home, away, neutral)
        if grid is None:
            return None
        side = "home" if fav_home else "away"
        prob = MK.team_over_under(grid, side, [0.5])[0.5]["over"]
        hit = (hs >= 1) if fav_home else (a >= 1)
        return {"prob": float(prob), "hit": int(hit)}

    # seguro: dupla chance de maior prob (cobre os dois resultados mais provaveis).
    probs = {"H": pH, "D": pD, "A": pA}
    drop = min(probs, key=probs.get)          # resultado descartado (menos provavel)
    prob = 1.0 - probs[drop]
    hit = (result != drop)
    return {"prob": float(prob), "hit": int(hit)}


def _bootstrap_multiples(legs: list[dict], k: int, iters: int, seed: int) -> list[dict]:
    """Amostra `iters` múltiplas de k pernas distintas; retorna [(pred_joint, hit)]."""
    probs = np.array([l["prob"] for l in legs])
    hits = np.array([l["hit"] for l in legs])
    n = len(legs)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(iters):
        idx = rng.choice(n, size=k, replace=False)
        out.append({"pred": float(np.prod(probs[idx])), "hit": int(hits[idx].all())})
    return out


def run(profile: str = "seguro", k: int = 3, iters: int = 20000,
        verbose: bool = True) -> dict:
    cfg = load_config()
    vcfg = cfg["validation"]
    lam = cfg["preprocess"]["temporal_decay_lambda"]
    seed = vcfg["random_seed"]

    sel_path = ROOT / "5_outputs" / "selected_model.json"
    model_name = (json.loads(sel_path.read_text(encoding="utf-8")).get("chosen_model", "poisson")
                  if sel_path.exists() else "poisson")

    legs = []
    for spec in vcfg["test_tournaments"]:
        train = dataset.training_frame(spec["start"], lam=lam)
        test = dataset.get_test_tournament(spec)
        model = reg.build_member(model_name, cfg) if model_name != "ensemble" \
            else reg.build_ensemble(cfg=cfg)
        model.fit(train)
        score_model = model if getattr(model, "supports_scoreline", False) \
            else reg.build_member("dixon_coles", cfg).fit(train)
        for _, r in test.iterrows():
            leg = _leg(str(r["home_team"]), str(r["away_team"]), bool(r["neutral"]),
                       int(r["home_score"]), int(r["away_score"]), r["result"],
                       model, score_model, profile)
            if leg is not None:
                legs.append(leg)

    legs.sort(key=lambda l: l["prob"], reverse=True)
    probs = np.array([l["prob"] for l in legs])
    hits = np.array([l["hit"] for l in legs])

    # 1) Calibração da PERNA por faixa.
    leg_bins = []
    for lo in (0.5, 0.6, 0.7, 0.8, 0.9):
        hi = lo + 0.1
        m = (probs >= lo) & (probs < hi if hi < 1.0 else probs <= hi)
        if m.sum():
            leg_bins.append({"faixa": f"{int(lo*100)}-{int(hi*100)}%", "n": int(m.sum()),
                             "pred": float(probs[m].mean()), "real": float(hits[m].mean())})

    # 2) Calibração da MÚLTIPLA (bootstrap) por faixa de prob conjunta.
    boot = _bootstrap_multiples(legs, k, iters, seed)
    bp = np.array([b["pred"] for b in boot])
    bh = np.array([b["hit"] for b in boot])
    mult_bins = []
    for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
        hi = lo + 0.2
        m = (bp >= lo) & (bp < hi if hi < 1.0 else bp <= hi)
        if m.sum():
            mult_bins.append({"faixa": f"{int(lo*100)}-{int(hi*100)}%", "n": int(m.sum()),
                              "pred": float(bp[m].mean()), "real": float(bh[m].mean())})

    # 3) Headline "alta probabilidade": múltiplas K-a-K, em blocos NÃO sobrepostos a
    # partir das pernas de maior prob (cada perna usada uma vez -> múltiplas reais).
    n_full = (len(legs) // k) * k
    blocks = [legs[i:i + k] for i in range(0, n_full, k)]
    realized = [{"pred": float(np.prod([l["prob"] for l in b])),
                 "hit": int(all(l["hit"] for l in b))} for b in blocks]
    top_half = realized[:max(1, len(realized) // 2)]  # metade de MAIOR prob
    top_hit = sum(r["hit"] for r in top_half)

    if verbose:
        nomes = {"seguro": "dupla chance do favorito", "vitoria": "vitória seca",
                 "gols": "favorito marca 1+"}
        print(f"  modelo: {model_name} | perfil: {profile} ({nomes[profile]}) | "
              f"{len(legs)} pernas (7 torneios, walk-forward)\n")
        print("  [1] Calibração da PERNA (prevista vs real por faixa de confiança):")
        print(f"      {'faixa':>9s} {'n':>4s} {'prevista':>9s} {'real':>8s}")
        for b in leg_bins:
            print(f"      {b['faixa']:>9s} {b['n']:>4d} {b['pred']:>8.1%} {b['real']:>8.1%}")

        print(f"\n  [2] Calibração da MÚLTIPLA de {k} pernas (bootstrap, {iters} amostras):")
        print(f"      {'prob conj.':>10s} {'n':>6s} {'prevista':>9s} {'real bate':>10s}")
        for b in mult_bins:
            print(f"      {b['faixa']:>10s} {b['n']:>6d} {b['pred']:>8.1%} {b['real']:>9.1%}")

        print(f"\n  [3] Estratégia real: {len(blocks)} múltiplas de {k} pernas (blocos "
              f"não sobrepostos, das pernas de maior prob).")
        print(f"      Metade de MAIOR probabilidade: {top_hit}/{len(top_half)} bateram "
              f"({top_hit/len(top_half):.1%})  |  prob média prevista "
              f"{np.mean([r['pred'] for r in top_half]):.1%}")
        all_hit = sum(r["hit"] for r in realized)
        print(f"      Todas as {len(realized)} múltiplas: {all_hit}/{len(realized)} "
              f"({all_hit/len(realized):.1%})")

    return {"model": model_name, "profile": profile, "k": k,
            "leg_bins": leg_bins, "mult_bins": mult_bins,
            "realized": realized, "top_half_hit": top_hit, "top_half_n": len(top_half)}


if __name__ == "__main__":
    run()
