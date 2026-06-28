"""[Montador de múltiplas] Sugere apostas combinadas de MAIOR PROBABILIDADE DE ACERTO
para os próximos jogos da Copa 2026 (fixtures ainda não jogados).

Princípios (todos consequência das validações já feitas):
- Só usa mercados em que o modelo é BEM CALIBRADO (probabilidade confiável):
  1X2 / dupla chance (ECE baixo) e gols por time (casa>1.5/2.5, fora>1.5 -- têm skill).
  Over/under TOTAL e BTTS ficam de fora (mal calibrados, ver markets_check).
- Probabilidade conjunta de uma múltipla = PRODUTO das pernas, e SÓ combina pernas de
  jogos DIFERENTES (pernas do mesmo jogo são correlacionadas -- multiplicar daria número
  errado). Isso deixa explícito o custo de cada perna a mais: a chance cai rápido.
- Não calcula "valor"/EV: o objetivo aqui é maximizar a chance de a múltipla inteira
  bater, não o retorno (já estabelecido que, contra a casa, o EV médio é negativo).

`perfil`:
  "seguro"   -> perna = dupla chance do favorito (maior prob por jogo)
  "vitoria"  -> perna = vitória seca do favorito
  "gols"     -> perna = favorito marca 1+ gol (over 0.5 do time mais forte)
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))
from config_loader import load_config, path  # noqa: E402
import dataset  # noqa: E402
import models_registry as reg  # noqa: E402
import markets as MK  # noqa: E402


def _load_chosen(cfg):
    sel_path = path("5_outputs", "selected_model.json")
    if sel_path.exists():
        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        return sel.get("chosen_model", "dixon_coles"), sel.get("ensemble_weights")
    return "dixon_coles", None


def _leg_for_match(home, away, hda, grid, disp, profile) -> dict | None:
    """Constrói a melhor perna do jogo segundo o perfil. Retorna
    {jogo, mercado, selecao, prob} ou None se o perfil exige grade e não há."""
    pH, pD, pA = float(hda[0]), float(hda[1]), float(hda[2])
    H, A = disp.get(home, home), disp.get(away, away)
    fav_home = pH >= pA

    if profile == "vitoria":
        prob, sel = (pH, f"Vitória {H}") if fav_home else (pA, f"Vitória {A}")
        return {"jogo": f"{H} x {A}", "mercado": "1X2", "selecao": sel, "prob": prob}

    if profile == "gols":
        if grid is None:
            return None
        side = "home" if fav_home else "away"
        prob = MK.team_over_under(grid, side, [0.5])[0.5]["over"]
        team = H if fav_home else A
        return {"jogo": f"{H} x {A}", "mercado": "Gols time", "selecao": f"{team} marca 1+",
                "prob": float(prob)}

    # "seguro" (default): dupla chance de maior probabilidade (= 1 - menor resultado).
    options = [("1X", pH + pD, f"{H} ou empate"), ("12", pH + pA, f"{H} ou {A}"),
               ("X2", pD + pA, f"empate ou {A}")]
    key, prob, sel = max(options, key=lambda o: o[1])
    return {"jogo": f"{H} x {A}", "mercado": f"Dupla chance ({key})", "selecao": sel,
            "prob": float(prob)}


def _ev_legs_from_odds(upcoming, model, disp) -> tuple[list[dict], list[str]]:
    """Para cada jogo com odds 1X2 reais (odds_2026), escolhe o resultado de MAIOR EV
    (EV = prob_modelo × odd − 1). Retorna (pernas, jogos_sem_odds)."""
    import db_client
    odds_rows = db_client.fetch_all("odds_2026") or []
    odds_map = {r["match_id"]: r for r in odds_rows}

    legs, sem_odds = [], []
    for _, r in upcoming.iterrows():
        home, away, neutral = str(r["home_team"]), str(r["away_team"]), bool(r["neutral"])
        H, A = disp.get(home, home), disp.get(away, away)
        gid = f"{home}-{away}-{r['date'].strftime('%Y%m%d')}"
        od = odds_map.get(gid)
        if od is None or od.get("odd_home") is None:
            sem_odds.append(f"{H} x {A}")
            continue
        p = model.predict_proba(home, away, neutral=neutral)
        outcomes = [
            ("home", float(p[0]), od["odd_home"], f"Vitória {H}", od.get("implied_home")),
            ("draw", float(p[1]), od["odd_draw"], "Empate", od.get("implied_draw")),
            ("away", float(p[2]), od["odd_away"], f"Vitória {A}", od.get("implied_away")),
        ]
        best = max(outcomes, key=lambda o: o[1] * o[2] - 1)
        _, prob, odd, sel, implied = best
        legs.append({"jogo": f"{H} x {A}", "selecao": sel, "prob": prob, "odd": float(odd),
                     "ev": prob * float(odd) - 1, "implied": implied})
    return legs, sem_odds


def build_ev(legs: int = 3, n_suggestions: int = 3, round_n: int | None = None,
             verbose: bool = True) -> dict:
    """Múltiplas de MAIOR EV (precisa de odds 1X2 reais em odds_2026). Combina só pernas
    de EV positivo, de jogos distintos. EV conjunto = produto(1+EV) − 1; mostra também a
    probabilidade conjunta (produto das probs) para deixar o trade-off explícito."""
    legs = max(3, legs)  # usuário pediu pelo menos 3
    cfg = load_config()
    lam = cfg["preprocess"]["temporal_decay_lambda"]
    today = cfg["data"]["today"]
    disp = cfg.get("display_names", {})

    chosen_name, weights = _load_chosen(cfg)
    train = dataset.training_frame(today, lam=lam)
    model = (reg.build_ensemble(weights=weights, cfg=cfg, bayesian_mode="full")
             if chosen_name == "ensemble"
             else reg.build_member(chosen_name, cfg, mode="full")).fit(train)

    fixtures = dataset.get_wc2026_fixtures()
    upcoming = fixtures[~fixtures["played"]]
    if round_n is not None and "round" in upcoming.columns:
        upcoming = upcoming[upcoming["round"] == round_n]

    all_legs, sem_odds = _ev_legs_from_odds(upcoming, model, disp)
    pos = sorted([l for l in all_legs if l["ev"] > 0], key=lambda l: l["ev"], reverse=True)

    suggestions = []
    if len(pos) >= legs:
        for start in range(min(n_suggestions, len(pos) - legs + 1)):
            chosen = pos[start:start + legs]
            odd_comb = float(np.prod([l["odd"] for l in chosen]))
            prob_comb = float(np.prod([l["prob"] for l in chosen]))
            ev_comb = float(np.prod([1 + l["ev"] for l in chosen])) - 1
            suggestions.append({"legs": chosen, "odd_comb": odd_comb,
                                "prob_comb": prob_comb, "ev_comb": ev_comb})

    if verbose:
        print(f"  modelo: {chosen_name} | {len(all_legs)} jogos com odds | "
              f"{len(pos)} pernas de EV positivo\n")
        if sem_odds:
            print(f"  (sem odds ainda: {', '.join(sem_odds)})\n")
        print(f"  pernas de EV positivo (modelo vê mais chance que a casa):")
        for l in pos:
            print(f"    EV {l['ev']:>+6.1%}  prob {l['prob']:>5.1%} @ {l['odd']:>5.2f}  "
                  f"{l['jogo']:<26s} {l['selecao']}")
        if not suggestions:
            print(f"\n  ⚠ Só {len(pos)} perna(s) de EV positivo — não dá pra montar "
                  f"múltipla de {legs} pernas. Sem aposta recomendada nesta fase.")
        else:
            print(f"\n  múltiplas de maior EV ({legs} pernas):")
            for i, s in enumerate(suggestions, 1):
                print(f"\n  #{i}  EV conjunto: {s['ev_comb']:>+.0%}  |  "
                      f"odd combinada: {s['odd_comb']:.2f}  |  "
                      f"chance de bater TODAS: {s['prob_comb']:.1%}")
                for l in s["legs"]:
                    print(f"        EV {l['ev']:>+6.1%}  prob {l['prob']:>5.1%} @ {l['odd']:>5.2f}  "
                          f"{l['jogo']:<26s} {l['selecao']}")
            print(f"\n  ⚠ EV alto = onde o modelo MAIS discorda da casa (azarões). Empilhar "
                  f"multiplica o EV mas também o risco: repare na 'chance de bater TODAS'.")

    return {"model": chosen_name, "legs": legs, "positive_ev_legs": pos,
            "suggestions": suggestions, "sem_odds": sem_odds}


def build(profile: str = "seguro", legs: int = 3, n_suggestions: int = 3,
          round_n: int | None = None, verbose: bool = True) -> dict:
    cfg = load_config()
    lam = cfg["preprocess"]["temporal_decay_lambda"]
    today = cfg["data"]["today"]
    disp = cfg.get("display_names", {})

    chosen_name, weights = _load_chosen(cfg)
    train = dataset.training_frame(today, lam=lam)
    model = (reg.build_ensemble(weights=weights, cfg=cfg, bayesian_mode="full")
             if chosen_name == "ensemble"
             else reg.build_member(chosen_name, cfg, mode="full")).fit(train)
    score_model = model if getattr(model, "supports_scoreline", False) \
        else reg.build_member("dixon_coles", cfg).fit(train)

    fixtures = dataset.get_wc2026_fixtures()
    upcoming = fixtures[~fixtures["played"]]
    if round_n is not None and "round" in upcoming.columns:
        upcoming = upcoming[upcoming["round"] == round_n]

    candidate_legs = []
    for _, r in upcoming.iterrows():
        home, away, neutral = str(r["home_team"]), str(r["away_team"]), bool(r["neutral"])
        hda = model.predict_proba(home, away, neutral=neutral)
        grid = score_model.predict_scoreline(home, away, neutral)
        leg = _leg_for_match(home, away, hda, grid, disp, profile)
        if leg is not None:
            candidate_legs.append(leg)

    candidate_legs.sort(key=lambda l: l["prob"], reverse=True)

    # Múltipla ótima de tamanho `legs` = as `legs` pernas de maior prob (jogos distintos,
    # garantido pois há 1 perna por jogo). Sugestões adicionais = janelas deslizantes
    # no ranking (top-N, depois 2..legs+1, etc.) para dar opções de risco parecido.
    suggestions = []
    for start in range(min(n_suggestions, max(1, len(candidate_legs) - legs + 1))):
        chosen = candidate_legs[start:start + legs]
        if len(chosen) < legs:
            break
        joint = float(np.prod([l["prob"] for l in chosen]))
        suggestions.append({"legs": chosen, "joint_prob": joint})

    if verbose:
        nomes = {"seguro": "dupla chance do favorito", "vitoria": "vitória seca do favorito",
                 "gols": "favorito marca 1+ gol"}
        print(f"  modelo: {chosen_name} | perfil: {profile} ({nomes[profile]}) | "
              f"{len(candidate_legs)} jogos disponíveis\n")
        print(f"  ranking de pernas (maior prob. de acerto primeiro):")
        for l in candidate_legs:
            print(f"    {l['prob']:>6.1%}  {l['jogo']:<28s} {l['selecao']} [{l['mercado']}]")
        print(f"\n  múltiplas sugeridas ({legs} pernas, prob. conjunta = produto):")
        for i, s in enumerate(suggestions, 1):
            print(f"\n  #{i}  — probabilidade de bater TODAS: {s['joint_prob']:.1%}")
            for l in s["legs"]:
                print(f"        {l['prob']:>6.1%}  {l['jogo']:<28s} {l['selecao']}")
        print(f"\n  Lembrete: cada perna a mais derruba a probabilidade conjunta "
              f"(é multiplicação, não soma).")

    return {"model": chosen_name, "profile": profile, "legs": legs,
            "candidates": candidate_legs, "suggestions": suggestions}


if __name__ == "__main__":
    build()
