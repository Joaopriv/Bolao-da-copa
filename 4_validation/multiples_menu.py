"""[Cardápio de múltiplas — EV real] Monta duplas e triplas (same-game, cross-game
e misto) sobre os jogos REAIS da rodada atual da Copa 2026, com probabilidade
conjunta honesta + retorno + EV de verdade (odds reais dos 3 mercados). Modo
terminal; passatempo. READ-ONLY produção.

MOTOR (reusado, não reimplementado) — de 4_validation/multiples_calibration.py:
- `_match_legs`: cardápio de máscaras booleanas sobre a grade de placar (h,a).
- `_gen_cross_combos`: combos de k itens distintos com teto + amostra por seed.
- A prob de uma perna isolada por máscara TEM de bater com utils/markets.py — cada
  perna roda esse assert ao ser construída (mesma garantia do _sanity_assert).
Probabilidade CONJUNTA:
- same-game (≥2 pernas no MESMO jogo): soma das células da grade que satisfazem o
  AND das máscaras (mercados correlacionados — a grade captura isso exato).
- cross-game (jogos distintos): produto das conjuntas por jogo (independência
  válida entre jogos — é o que a produção realmente assume em utils/multiples.py).

CARDÁPIO (só pernas com ODD REAL; nada sintético):
- 1X2 (odds_2026 h2h): vitória do favorito (odd direta) e dupla chance do favorito
  (odd DERIVADA de h2h: 1/(1/odd_fav + 1/odd_empate) — flag "odd derivada").
- totals (odds_2026_markets): over/under por linha que tem odd gravada.
- BTTS (odds_2026_markets): sim/não.
- team_totals fica de fora (a casa não oferece). Favorito = pH ≥ pA.

RANKING POR FAIXA DE RETORNO (2-3x, 3-5x, 5-10x, 10x+). Dentro da faixa, ordena por
prob conjunta desc e mostra o EV ao lado. NUNCA ranqueia por prob global (só devolveria
favoritão). EV é ETIQUETA, não veto: mostra +EV e −EV, marca qual é, não filtra as −EV.

HONESTIDADE:
- same-game/misto: o retorno por PRODUTO de odds não é o que a casa paga (SGP é
  precificado abaixo) → flag de EV otimista/inatingível.
- cross-game/misto: a calibração mostrou cross-game otimista (superestima acerto)
  → flag de prob possivelmente inflada.
- qualquer perna de totals/BTTS: markets_check.py/markets_compare já documentaram
  que esses mercados NÃO têm skill comprovado (viés vs climatologia, IC do BSS
  cruza zero) — só 1X2 tem skill validado. Múltipla com 1+ perna nesses mercados
  leva flag de EV não-confiável (mais forte quanto maior a fração de pernas
  sem-skill; máxima quando TODAS as pernas são totals/BTTS).
Imprime nº de pernas, tipo (same/cross/misto) e a prob de CADA perna.

FRONTEIRA: lê selected_model.json, odds_2026, odds_2026_markets e as previsões pelo
mesmo caminho do predict_2026. Não edita produção. Não realimenta produção.
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from math import ceil, comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # 4_validation (sibling engine)
sys.path.insert(0, str(ROOT / "utils"))

from config_loader import load_config, path  # noqa: E402
import dataset  # noqa: E402
import db_client  # noqa: E402
import models_registry as reg  # noqa: E402
import markets as MK  # noqa: E402
from multiples_calibration import _match_legs, _gen_cross_combos  # noqa: E402 (motor)

# Faixas de retorno (limite inferior incl., superior excl.). < 2x não entra (sem graça).
_RETURN_TIERS = [(2.0, 3.0, "2-3x"), (3.0, 5.0, "3-5x"),
                 (5.0, 10.0, "5-10x"), (10.0, float("inf"), "10x+")]
_COMBO_CAP = 6000   # teto de combos candidatos POR tamanho (k=2,3); acima disso, amostra

# Confiabilidade por família, documentada em markets_check.py / markets_compare:
# 1X2 tem skill validado; totals/BTTS NÃO (viés conhecido, IC do BSS cruza zero).
_UNRELIABLE_FAMILIES = {"totals", "btts"}


def _load_chosen(cfg) -> tuple[str, list | None]:
    """selected_model.json (mesmo caminho de predict_2026/multiples); senão dixon_coles."""
    sel_path = ROOT / "5_outputs" / "selected_model.json"
    if sel_path.exists():
        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        return sel.get("chosen_model", "dixon_coles"), sel.get("ensemble_weights")
    return "dixon_coles", None


def _build_game_legs(grid, fav_home, fav_disp, od, om, n) -> list[dict]:
    """Pernas de UM jogo que têm odd real. Cada perna carrega (família, seleção, máscara,
    prob do modelo, odd, EV). Roda o assert prob-máscara == markets.py em cada perna."""
    H, A = np.indices((n, n))
    base = _match_legs(grid, fav_home)
    g = MK._normalize(grid)

    def prob(mask):
        return float((g * mask).sum())

    def add(legs, family, sel, mask, p_model, odd, derived=False):
        assert np.isclose(prob(mask), p_model, atol=1e-9), f"máscara ≠ markets.py: {sel}"
        legs.append({"family": family, "sel": sel, "mask": mask, "prob": float(p_model),
                     "odd": float(odd), "ev": float(p_model) * float(odd) - 1.0,
                     "derived": derived})

    legs: list[dict] = []

    # 1X2 — odds_2026 (h2h). Vitória favorito = odd direta; DC favorito = odd derivada.
    if od is not None and od.get("odd_home") is not None and od.get("odd_away") is not None:
        p = MK.result_1x2(grid)
        fav_odd = float(od["odd_home"] if fav_home else od["odd_away"])
        fav_p = p["home"] if fav_home else p["away"]
        add(legs, "1X2", f"Vitória {fav_disp}", base["fav_win"][1], fav_p, fav_odd)
        draw_odd = od.get("odd_draw")
        if draw_odd:
            dc_odd = 1.0 / (1.0 / fav_odd + 1.0 / float(draw_odd))
            dc_p = (p["home"] + p["draw"]) if fav_home else (p["away"] + p["draw"])
            add(legs, "1X2", f"{fav_disp} ou empate", base["fav_dc"][1], dc_p, dc_odd,
                derived=True)

    # totals + BTTS — odds_2026_markets (já em long format, odd real por seleção).
    tot_odds: dict[float, dict] = defaultdict(dict)
    btts_odds: dict[str, float] = {}
    for r in om:
        if r["market"] == "totals" and r.get("line") is not None and r.get("odd"):
            tot_odds[float(r["line"])][r["selection"]] = float(r["odd"])
        elif r["market"] == "btts" and r.get("odd"):
            btts_odds[r["selection"]] = float(r["odd"])

    if tot_odds:
        mk_tot = MK.total_over_under(grid, sorted(tot_odds))
        for L in sorted(tot_odds):
            assert L % 1 == 0.5, f"linha inteira {L} seria PUSH, não over/under puro"
            ov = (H + A) >= ceil(L)
            if "over" in tot_odds[L]:
                add(legs, "totals", f"Over {L:g}", ov, mk_tot[L]["over"], tot_odds[L]["over"])
            if "under" in tot_odds[L]:
                add(legs, "totals", f"Under {L:g}", ~ov, mk_tot[L]["under"], tot_odds[L]["under"])

    if btts_odds:
        b = MK.btts(grid)
        if "yes" in btts_odds:
            add(legs, "btts", "Ambos marcam: sim", base["btts"][1], b["yes"], btts_odds["yes"])
        if "no" in btts_odds:
            add(legs, "btts", "Ambos marcam: não", ~base["btts"][1], b["no"], btts_odds["no"])

    return legs


def _valid(combo, pool) -> bool:
    """Combo válido = no máx. 1 perna por (jogo, família). Barra contradições (over×under
    no mesmo jogo) e redundâncias (over1.5×over2.5 com odd inflada)."""
    seen = set()
    for i in combo:
        key = (pool[i]["gi"], pool[i]["family"])
        if key in seen:
            return False
        seen.add(key)
    return True


def _eval_combo(combo, pool, g_by_gi) -> dict:
    """Métricas de uma múltipla: prob conjunta (grade same-game × produto cross-game),
    retorno (produto das odds), EV, tipo e flags."""
    by_game: dict[int, list] = defaultdict(list)
    for i in combo:
        by_game[pool[i]["gi"]].append(pool[i])

    prob = 1.0
    for gi, legs in by_game.items():
        if len(legs) == 1:
            prob *= legs[0]["prob"]
        else:
            mask = np.logical_and.reduce([l["mask"] for l in legs])
            prob *= float((g_by_gi[gi] * mask).sum())   # conjunta same-game pela grade

    ret = float(np.prod([pool[i]["odd"] for i in combo]))
    ev = prob * ret - 1.0
    same_game = any(len(v) >= 2 for v in by_game.values())
    n_games = len(by_game)
    tipo = "same" if n_games == 1 else ("misto" if same_game else "cross")

    flags = []
    if same_game:
        flags.append("retorno por produto de odds; casa paga SGP abaixo → EV otimista")
    if tipo in ("cross", "misto"):
        flags.append("cross-game: calibração indica prob possivelmente otimista")
    if any(pool[i]["derived"] for i in combo):
        flags.append("contém DC com odd derivada de h2h")
    n_unreliable = sum(1 for i in combo if pool[i]["family"] in _UNRELIABLE_FAMILIES)
    if n_unreliable == len(combo):
        flags.append("EV NÃO-CONFIÁVEL: todas as pernas são de mercados sem skill "
                      "comprovado (totals/BTTS) -- EV provavelmente reflete viés do "
                      "modelo, não edge real")
    elif n_unreliable > 0:
        flags.append(f"EV não-confiável: {n_unreliable}/{len(combo)} perna(s) de "
                      f"mercado sem skill comprovado (totals/BTTS) -- número "
                      f"possivelmente inflado")

    return {"legs": [pool[i] for i in combo], "prob": prob, "ret": ret, "ev": ev,
            "tipo": tipo, "n_legs": len(combo), "flags": flags}


def _build_pool(round_n: int | None) -> dict:
    """Carrega config/modelo, resolve a rodada e monta o pool de pernas com odd real
    (mesmo caminho de predict_2026). Reusado por `run` e `run_target`."""
    cfg = load_config()
    lam = cfg["preprocess"]["temporal_decay_lambda"]
    today = cfg["data"]["today"]
    disp = cfg.get("display_names", {})
    seed = cfg["validation"]["random_seed"]

    chosen_name, weights = _load_chosen(cfg)
    train = dataset.training_frame(today, lam=lam)
    model = (reg.build_ensemble(weights=weights, cfg=cfg, bayesian_mode="full")
             if chosen_name == "ensemble"
             else reg.build_member(chosen_name, cfg, mode="full")).fit(train)
    score_model = model if getattr(model, "supports_scoreline", False) \
        else reg.build_member("dixon_coles", cfg).fit(train)

    # round vem de copa_2026_results (fonte autoritativa, igual predict_2026); neutral vem
    # dos fixtures (vantagem de anfitrião por jogo). Rodada atual = menor round não jogado.
    stage_round = {r["game_id"]: r["round"]
                   for r in (db_client.fetch_all("copa_2026_results") or [])}
    fixtures = dataset.get_wc2026_fixtures()
    upcoming = fixtures[~fixtures["played"]].copy()

    def _gid(r):
        return f"{r['home_team']}-{r['away_team']}-{r['date'].strftime('%Y%m%d')}"

    rounds_seen = [stage_round.get(_gid(r)) for _, r in upcoming.iterrows()]
    known = sorted({x for x in rounds_seen if x is not None})
    if round_n is None:
        round_n = known[0] if known else None

    odds1x2 = {r["match_id"]: r for r in (db_client.fetch_all("odds_2026") or [])}
    mkt_by_game: dict[str, list] = defaultdict(list)
    for r in (db_client.fetch_all("odds_2026_markets") or []):
        if r["market"] in ("totals", "btts"):
            mkt_by_game[r["match_id"]].append(r)

    pool: list[dict] = []
    g_by_gi: dict[int, np.ndarray] = {}
    sem_odds: list[str] = []
    gi = 0
    for _, r in upcoming.iterrows():
        gid = _gid(r)
        if round_n is not None and stage_round.get(gid) != round_n:
            continue
        home, away, neutral = str(r["home_team"]), str(r["away_team"]), bool(r["neutral"])
        od = odds1x2.get(gid)
        om = mkt_by_game.get(gid, [])
        has_1x2 = od is not None and od.get("odd_home") is not None
        gname = f"{disp.get(home, home)} x {disp.get(away, away)}"
        if not has_1x2 and not om:
            sem_odds.append(gname)
            continue
        grid = score_model.predict_scoreline(home, away, neutral)
        if grid is None:
            continue
        hda = model.predict_proba(home, away, neutral=neutral)
        fav_home = float(hda[0]) >= float(hda[2])
        fav_disp = disp.get(home, home) if fav_home else disp.get(away, away)
        legs = _build_game_legs(grid, fav_home, fav_disp, od, om, grid.shape[0])
        if not legs:
            continue
        g_by_gi[gi] = MK._normalize(grid)
        for leg in legs:
            leg.update({"gi": gi, "game": gname})
            pool.append(leg)
        gi += 1

    return {"chosen_name": chosen_name, "round_n": round_n, "pool": pool,
            "g_by_gi": g_by_gi, "sem_odds": sem_odds, "seed": seed}


def _generate_combos(pool: list[dict], g_by_gi: dict[int, np.ndarray],
                      seed: int) -> tuple[list[dict], list[int]]:
    """Duplas e triplas do pool de pernas; teto + amostra por seed se explodir.
    Reusado por `run` e `run_target` — mesmo motor, mesmas regras (`_valid`/`_eval_combo`)."""
    rng = np.random.default_rng(seed)
    sampled = []
    evaluated: list[dict] = []
    for k in (2, 3):
        if comb(len(pool), k) > _COMBO_CAP:
            sampled.append(k)
        for combo in _gen_cross_combos(len(pool), k, _COMBO_CAP, rng):
            if _valid(combo, pool):
                m = _eval_combo(combo, pool, g_by_gi)
                if m["prob"] > 0:   # descarta combos logicamente impossíveis (ex.
                    evaluated.append(m)  # under1.5 + btts:sim no mesmo jogo)
    return evaluated, sampled


def run(round_n: int | None = None, top: int = 5, verbose: bool = True) -> dict:
    built = _build_pool(round_n)
    chosen_name, round_n = built["chosen_name"], built["round_n"]
    pool, g_by_gi, sem_odds, seed = (built["pool"], built["g_by_gi"],
                                      built["sem_odds"], built["seed"])

    evaluated, sampled = _generate_combos(pool, g_by_gi, seed)

    # Ranking POR FAIXA DE RETORNO; dentro da faixa, prob conjunta desc.
    tiers: dict[str, list] = {label: [] for _, _, label in _RETURN_TIERS}
    for m in evaluated:
        for lo, hi, label in _RETURN_TIERS:
            if lo <= m["ret"] < hi:
                tiers[label].append(m)
                break
    for label in tiers:
        tiers[label].sort(key=lambda m: m["prob"], reverse=True)

    if verbose:
        _print_report(chosen_name, round_n, len(g_by_gi), len(pool), sem_odds,
                      sampled, tiers, top)

    return {"model": chosen_name, "round": round_n, "n_games": len(g_by_gi),
            "n_legs": len(pool), "tiers": tiers, "sem_odds": sem_odds}


_R = 4  # casas decimais no JSON exportado (10k+ combos -- precisão maior só infla o arquivo)


def _round_leg(l: dict) -> dict:
    return {"family": l["family"], "sel": l["sel"], "prob": round(l["prob"], _R),
            "odd": round(l["odd"], _R), "ev": round(l["ev"], _R), "derived": l["derived"],
            "gi": l["gi"], "game": l["game"]}


def export(round_n: int | None = None) -> dict:
    """Serializa TODO o conjunto `evaluated` (não só o top-N por faixa) pra JSON estático,
    consumido pelo frontend (aba Apostas: Sugeridas + Monta-livre filtram no client, sem
    round-trip a Python). Mesmo motor/pool/regras do --multiples-menu. READ-ONLY, sob demanda
    -- fora do pipeline --auto-round."""
    built = _build_pool(round_n)
    chosen_name, round_n = built["chosen_name"], built["round_n"]
    pool, g_by_gi, sem_odds, seed = (built["pool"], built["g_by_gi"],
                                      built["sem_odds"], built["seed"])
    evaluated, sampled = _generate_combos(pool, g_by_gi, seed)

    out = {
        "model": chosen_name, "round": round_n, "n_games": len(g_by_gi),
        "n_legs": len(pool), "sem_odds": sem_odds, "sampled": sampled,
        "evaluated": [
            {"prob": round(m["prob"], _R), "ret": round(m["ret"], _R), "ev": round(m["ev"], _R),
             "tipo": m["tipo"], "n_legs": m["n_legs"], "flags": m["flags"],
             "legs": [_round_leg(l) for l in m["legs"]]}
            for m in evaluated
        ],
    }
    # Sem indent: arquivo é consumido por código (fetch+JSON.parse), não por humanos -- com
    # 10k+ combos, indentação sozinha ~dobra o tamanho do arquivo.
    dest = path("5_outputs", "multiplas_2026.json")
    dest.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    fe_dest = ROOT / "frontend" / "public" / "data" / "multiplas_2026.json"
    if fe_dest.parent.exists():
        shutil.copy(dest, fe_dest)
        print(f"  {len(evaluated)} combinações -> {dest} (+ cópia para {fe_dest})")
    else:
        print(f"  {len(evaluated)} combinações -> {dest}")
    return out


def _format_combo_lines(m: dict, idx: int | None = None) -> list[str]:
    """Bloco de impressão de UMA combinação (header + pernas + flags). Reusado pelo
    ranking por faixa (`--multiples-menu`) e pelo monta-livre (`--multiples-target`)."""
    tag = "+EV" if m["ev"] >= 0 else "−EV"
    header = f"prob conjunta {m['prob']:>5.1%}  retorno {m['ret']:>5.2f}x  " \
              f"EV {m['ev']:>+6.1%} [{tag}]  tipo {m['tipo']} ({m['n_legs']} pernas)"
    lines = [f"  #{idx}  {header}" if idx is not None else f"  {header}"]
    for l in m["legs"]:
        mark = " *" if l["derived"] else ""
        lines.append(f"        {l['prob']:>5.1%} @ {l['odd']:>5.2f}{mark}  "
                      f"{l['game']:<26s} {l['sel']}")
    for f in m["flags"]:
        lines.append(f"        ⚠ {f}")
    return lines


def _print_report(model, round_n, n_games, n_legs, sem_odds, sampled, tiers, top):
    line = "=" * 78
    print(line)
    print("CARDÁPIO DE MÚLTIPLAS — 1X2 + totals + BTTS com EV real (odds reais)")
    print(line)
    rotulo = f"rodada {round_n}" if round_n is not None else "todos os jogos pendentes"
    print(f"modelo: {model} | {rotulo} | {n_games} jogos com odds | {n_legs} pernas no cardápio")
    print("motor: same-game = prob conjunta pela grade; cross-game = produto entre jogos.")
    print("ranking POR FAIXA DE RETORNO (nunca por prob global). EV = etiqueta, não veto.")
    if sem_odds:
        print(f"\n  (sem odds ainda: {', '.join(sem_odds)})")
    if sampled:
        print(f"  (combos de {', '.join(str(k) for k in sampled)} pernas AMOSTRADOS — "
              f"teto de {_COMBO_CAP}/tamanho atingido)")
    if n_legs == 0:
        print("\n  ⚠ Nenhuma perna com odd real nesta rodada — sem cardápio. "
              "Rode --fetch-odds / --fetch-market-odds quando as odds saírem.")
        return

    for _, _, label in _RETURN_TIERS:
        ms = tiers[label]
        print(f"\n── FAIXA DE RETORNO {label} ─────────────────────────────────────────────")
        if not ms:
            print("   (nenhuma múltipla nesta faixa)")
            continue
        for i, m in enumerate(ms[:top], 1):
            print()
            for combo_line in _format_combo_lines(m, idx=i):
                print(combo_line)

    print(f"\n{line}")
    print("Leitura: dentro de cada faixa, mais prob = mais provável bater; EV ao lado diz se,")
    print("a longo prazo, o preço compensa. * = odd derivada (DC de h2h). same-game/misto: o")
    print("retorno por produto é otimista (casa paga SGP abaixo). É passatempo, não conselho.")


def _target_odd_search(evaluated: list[dict], target: float, band: float = 0.15) -> list[dict]:
    """Combos com retorno dentro de ±`band` de `target`, maior prob conjunta primeiro
    (empate → menos pernas). Mesmas regras/flags do --multiples-menu, sem exceção."""
    lo, hi = target * (1 - band), target * (1 + band)
    cands = [m for m in evaluated if lo <= m["ret"] <= hi]
    cands.sort(key=lambda m: (-m["prob"], m["n_legs"]))
    return cands


def _target_prob_search(evaluated: list[dict], min_prob: float) -> list[dict]:
    """Combos com prob conjunta >= `min_prob`, maior retorno primeiro (empate → menos
    pernas). Mesmas regras/flags do --multiples-menu, sem exceção."""
    cands = [m for m in evaluated if m["prob"] >= min_prob]
    cands.sort(key=lambda m: (-m["ret"], m["n_legs"]))
    return cands


def run_target(round_n: int | None = None, target_odd: float | None = None,
                target_prob: float | None = None, band: float = 0.15, top: int = 3,
                verbose: bool = True) -> dict:
    if (target_odd is None) == (target_prob is None):
        raise ValueError("run_target: passe exatamente um de target_odd/target_prob")

    built = _build_pool(round_n)
    chosen_name, round_n = built["chosen_name"], built["round_n"]
    pool, g_by_gi, sem_odds, seed = (built["pool"], built["g_by_gi"],
                                      built["sem_odds"], built["seed"])
    n_games, n_legs = len(g_by_gi), len(pool)

    if n_legs == 0:
        if verbose:
            _print_target_report(chosen_name, round_n, n_games, n_legs, sem_odds,
                                  target_odd, target_prob, band, [], [], top)
        return {"model": chosen_name, "round": round_n, "n_games": n_games,
                "n_legs": n_legs, "mode": "odd" if target_odd is not None else "prob",
                "target": target_odd if target_odd is not None else target_prob,
                "best": None, "alternates": [], "sem_odds": sem_odds}

    evaluated, sampled = _generate_combos(pool, g_by_gi, seed)
    if target_odd is not None:
        cands = _target_odd_search(evaluated, target_odd, band)
    else:
        cands = _target_prob_search(evaluated, target_prob)

    best = cands[0] if cands else None
    alternates = cands[1:top] if cands else []

    if verbose:
        _print_target_report(chosen_name, round_n, n_games, n_legs, sem_odds,
                              target_odd, target_prob, band, evaluated, cands, top, sampled)

    return {"model": chosen_name, "round": round_n, "n_games": n_games, "n_legs": n_legs,
            "mode": "odd" if target_odd is not None else "prob",
            "target": target_odd if target_odd is not None else target_prob,
            "best": best, "alternates": alternates, "sem_odds": sem_odds}


def _print_target_report(model, round_n, n_games, n_legs, sem_odds, target_odd, target_prob,
                          band, evaluated, cands, top=3, sampled=None):
    line = "=" * 78
    print(line)
    print("MONTA-LIVRE DE MÚLTIPLAS — fixa um eixo, resolve o melhor conjunto pro outro")
    print(line)
    rotulo = f"rodada {round_n}" if round_n is not None else "todos os jogos pendentes"
    print(f"modelo: {model} | {rotulo} | {n_games} jogos com odds | {n_legs} pernas no cardápio")
    if target_odd is not None:
        lo, hi = target_odd * (1 - band), target_odd * (1 + band)
        print(f"alvo: odd ≈ {target_odd:.2f}x (banda ±{band:.0%}: {lo:.2f}x–{hi:.2f}x) "
              f"→ maximiza prob conjunta")
    else:
        print(f"alvo: prob conjunta ≥ {target_prob:.1%} → maximiza retorno")
    if sem_odds:
        print(f"\n  (sem odds ainda: {', '.join(sem_odds)})")
    if sampled:
        print(f"  (combos de {', '.join(str(k) for k in sampled)} pernas AMOSTRADOS — "
              f"teto de {_COMBO_CAP}/tamanho atingido)")
    if n_legs == 0:
        print("\n  ⚠ Nenhuma perna com odd real nesta rodada — sem cardápio. "
              "Rode --fetch-odds / --fetch-market-odds quando as odds saírem.")
        return

    if not cands:
        print("\n  ⚠ Nenhuma combinação encontrada nesse alvo (dentro das regras de "
              "compatibilidade e do pool de pernas com odd real).")
        if evaluated:
            if target_odd is not None:
                rets = [m["ret"] for m in evaluated]
                print(f"     retorno disponível no pool (duplas/triplas): "
                      f"{min(rets):.2f}x – {max(rets):.2f}x")
            else:
                probs = [m["prob"] for m in evaluated]
                print(f"     maior prob conjunta disponível no pool (duplas/triplas): "
                      f"{max(probs):.1%}")
            print("     tente ampliar a banda/afrouxar o alvo, ou mudar --round-multi.")
        return

    print("\n── MELHOR COMBINAÇÃO ───────────────────────────────────────────────────")
    for line_txt in _format_combo_lines(cands[0]):
        print(line_txt)

    alternates = cands[1:top]
    if alternates:
        print("\n── ALTERNATIVAS (mesmo alvo) ───────────────────────────────────────────")
        for i, m in enumerate(alternates, 2):
            print()
            for line_txt in _format_combo_lines(m, idx=i):
                print(line_txt)

    print(f"\n{line}")
    print("Leitura: mesma honestidade do --multiples-menu -- EV é etiqueta, não veto;")
    print("* = odd derivada (DC de h2h); same-game/misto: retorno por produto é otimista")
    print("(casa paga SGP abaixo); flags de EV não-confiável em totals/BTTS. Passatempo.")


if __name__ == "__main__":
    run()
