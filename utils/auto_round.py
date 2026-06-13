"""[Deploy Netlify] Orquestração pós-rodada com push automático.

`run_auto_round(n)` encadeia, em ordem:
  1. update_round(n)  -- re-treina/re-seleciona e regenera predictions_2026.json
     (sequential_backtest; também copia para frontend/public/data).
  2. Monte Carlo       -- nova distribuição de pontos (montecarlo_sim).
  3. push_to_github(n) -- commita os JSONs servidos pelo site e faz push,
     disparando o rebuild automático do Netlify (a menos de --no-push).

NÃO toca em modelos, pipeline de treino nem utils/scoring.py.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "4_validation"))
sys.path.insert(0, str(ROOT / "5_outputs"))

# Arquivos servidos pelo site (e config.yaml, que --update-round avança em `today`).
_PUSH_PATHS = [
    "5_outputs/predictions_2026.json",
    "5_outputs/selected_model.json",
    "5_outputs/odds_after.json",
    "frontend/public/data/predictions_2026.json",
    "frontend/public/data/selected_model.json",
    "config.yaml",
]


def push_to_github(round_n: int, verbose: bool = True) -> bool:
    """Commita e faz push das previsões atualizadas para o GitHub.
    Dispara rebuild automático no Netlify. Retorna True se o push foi concluído."""
    # Só adiciona caminhos que existem (evita erro do git add em arquivo ausente).
    paths = [p for p in _PUSH_PATHS if (ROOT / p).exists()]
    try:
        subprocess.run(["git", "-C", str(ROOT), "add", *paths], check=True)
        subprocess.run(
            ["git", "-C", str(ROOT), "commit", "-m",
             f"chore: atualiza previsões pós-rodada {round_n}"],
            check=True,
        )
        subprocess.run(["git", "-C", str(ROOT), "push"], check=True)
        if verbose:
            print(f"✅ Push concluído — Netlify iniciará rebuild em ~1-2min")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠ Push falhou: {e}. Previsões locais estão corretas.")
        print("  Rode manualmente: git add . && git commit -m 'fix' && git push")
        return False


def run_auto_round(n: int, push: bool = True, force: bool = False,
                   verbose: bool = True) -> None:
    """Fluxo completo pós-rodada: update_round -> Monte Carlo -> push (opcional)."""
    from sequential_backtest import update_round
    from montecarlo_sim import run as mc_run, format_report, N_SIMS_DEFAULT, SEED_DEFAULT

    # 1. Re-treina/re-seleciona e regenera as previsões da rodada N.
    update_round(n, verbose=verbose, force=force)

    # 2. Monte Carlo (nova distribuição de pontos a partir das previsões atualizadas).
    if verbose:
        print(f"\n● Monte Carlo pós-rodada {n} ({N_SIMS_DEFAULT} torneios) ...")
    result = mc_run(n_sims=N_SIMS_DEFAULT, strategy="max_ev", seed=SEED_DEFAULT)
    other = mc_run(n_sims=N_SIMS_DEFAULT, strategy="modal", seed=SEED_DEFAULT)
    print()
    print(format_report(result, other_strategy_result=other))

    # 3. Push (último passo) -> rebuild no Netlify.
    if push:
        print()
        push_to_github(n, verbose=verbose)
    elif verbose:
        print("\n(--no-push: previsões atualizadas localmente, sem push para o GitHub)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Auto-rodada: update + Monte Carlo + push")
    ap.add_argument("n", type=int, help="número da rodada")
    ap.add_argument("--no-push", action="store_true", help="não fazer push para o GitHub")
    ap.add_argument("--force", action="store_true", help="ignora jogos pendentes da rodada")
    args = ap.parse_args()
    run_auto_round(args.n, push=not args.no_push, force=args.force)
