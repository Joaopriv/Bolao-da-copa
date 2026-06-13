"""[Iteração 3 / Prompt F4] Revalidação dos pesos por torneio (`competition_weights`) —
regra do IC.

ATENÇÃO -- por que isto roda em DOIS PROCESSOS:
`config_loader.load_config()` e `dataset.load_matches()` são `@lru_cache(maxsize=1)` e
SEM PARÂMETROS -- a primeira chamada em todo o processo lê config.yaml do disco e
congela `w_comp` (peso de competição) para o resto da execução. Passar um `cfg` dict
diferente para `run_comparison()` NÃO muda `w_comp`/`weight` (squad_offset_check.py
funciona porque `squad_offset_weight` é lido do `cfg` passado a `build_member`, não do
`load_matches()` cacheado). Para comparar de fato "pesos antigos" vs "pesos novos" de
`competition_weights`, é preciso reler config.yaml do zero -- daí o subprocesso para a
rodada "antigo", com config.yaml temporariamente reescrito e restaurado no `finally`.

Regra do IC (por modelo, diff = rps_novo - rps_antigo; negativo = "novo" melhor):
  diff_hi < 0 (não cruza zero, "novo" melhor)  -> APROVADO
  diff cruza zero                              -> DESCARTADO (Occam: mantém pesos antigos)
  diff_lo > 0 (não cruza zero, "novo" pior)    -> DESCARTADO (pior)

O ensemble decide a recomendação agregada (config final é decisão do usuário).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import load_config, CONFIG_PATH  # noqa: E402
import models_registry as reg  # noqa: E402
from tournament_comparison import run_comparison  # noqa: E402
from significance import compare_pair  # noqa: E402

# Valores de `preprocess.competition_weights` / `default_competition_weight` ANTES do
# Prompt F1 (snapshot do config.yaml para a comparação "antigo").
OLD_COMPETITION_WEIGHTS = {
    "FIFA World Cup": 1.00,
    "Confederations Cup": 0.90,
    "UEFA Euro": 0.90,
    "Copa América": 0.90,
    "AFC Asian Cup": 0.85,
    "African Cup of Nations": 0.85,
    "Gold Cup": 0.80,
    "CONCACAF Championship": 0.80,
    "UEFA Nations League": 0.75,
    "CONCACAF Nations League": 0.72,
    "FIFA World Cup qualification": 0.65,
    "UEFA Euro qualification": 0.60,
    "African Cup of Nations qualification": 0.55,
    "AFC Asian Cup qualification": 0.55,
    "Friendly": 0.30,
}
OLD_DEFAULT_COMPETITION_WEIGHT = 0.50

_SUBPROCESS_SCRIPT = """
import json, sys
sys.path.insert(0, {root!r})
sys.path.insert(0, {valdir!r})
from tournament_comparison import run_comparison
import models_registry as reg
model_names = {model_names!r}
comp = run_comparison(model_names=model_names, iters={iters!r}, verbose=False)
print(json.dumps({{m: comp["rps_by_model"][m] for m in model_names}}))
"""


def _rps_with_old_weights(model_names: list[str], iters: int) -> dict[str, list[float]]:
    """Roda run_comparison() num subprocesso à parte, com config.yaml temporariamente
    revertido aos pesos ANTERIORES ao F1 -- necessário p/ contornar o cache de
    load_matches()/load_config() (ver docstring do módulo)."""
    original_text = CONFIG_PATH.read_text(encoding="utf-8")
    try:
        cfg_old = yaml.safe_load(original_text)
        cfg_old["preprocess"]["competition_weights"] = OLD_COMPETITION_WEIGHTS
        cfg_old["preprocess"]["default_competition_weight"] = OLD_DEFAULT_COMPETITION_WEIGHT
        CONFIG_PATH.write_text(yaml.safe_dump(cfg_old, allow_unicode=True, sort_keys=False),
                                encoding="utf-8")
        script = _SUBPROCESS_SCRIPT.format(
            root=str(ROOT), valdir=str(Path(__file__).resolve().parent),
            model_names=model_names, iters=iters,
        )
        proc = subprocess.run([sys.executable, "-c", script], cwd=str(ROOT),
                               capture_output=True, text=True)
    finally:
        CONFIG_PATH.write_text(original_text, encoding="utf-8")

    if proc.returncode != 0:
        raise RuntimeError(f"subprocesso (pesos antigos) falhou:\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_competition_weight_comparison(iters: int | None = None, seed: int | None = None,
                                       verbose: bool = True) -> dict:
    cfg = load_config()
    vcfg = cfg["validation"]
    iters = iters or vcfg["bootstrap_iterations"]
    seed = seed if seed is not None else vcfg["random_seed"]

    model_names = reg.available_members() + ["ensemble"]

    if verbose:
        print("  rodando com pesos NOVOS (F1, config.yaml atual) ...")
    comp_novo = run_comparison(model_names=model_names, iters=iters, verbose=verbose)

    if verbose:
        print("  rodando com pesos ANTIGOS (pré-F1, subprocesso) ...")
    rps_antigo = _rps_with_old_weights(model_names, iters)

    results = {}
    for mname in model_names:
        rps_novo = comp_novo["rps_by_model"][mname]
        cmp = compare_pair(rps_novo, rps_antigo[mname], "novo_pesos", "antigo_pesos",
                            iters=iters, seed=seed)
        if cmp["diff_hi"] < 0:
            decision = "APROVADO"
        elif cmp["diff_lo"] > 0:
            decision = "DESCARTADO (pior)"
        else:
            decision = "DESCARTADO (Occam, IC cruza zero)"
        results[mname] = {
            **cmp, "decision": decision,
            "rps_antigo_mean": float(np.nanmean(rps_antigo[mname])),
            "rps_novo_mean": float(np.nanmean(rps_novo)),
        }

    ensemble_decision = results["ensemble"]["decision"]

    if verbose:
        print(f"\n  {'modelo':18s} {'RPS antigo':>10s} {'RPS novo':>10s} {'IC95% diff (novo-antigo)':>26s}  decisão")
        for mname, r in results.items():
            print(f"  {mname:18s} {r['rps_antigo_mean']:>10.4f} {r['rps_novo_mean']:>10.4f} "
                  f"[{r['diff_lo']:+.4f}, {r['diff_hi']:+.4f}]        {r['decision']}")
        print(f"\n  Recomendação (ensemble decide o config final): {ensemble_decision}")

    return {"iters": iters, "seed": seed, "results": results, "ensemble_decision": ensemble_decision}


def run(iters: int | None = None, seed: int | None = None, verbose: bool = True) -> dict:
    return run_competition_weight_comparison(iters=iters, seed=seed, verbose=verbose)


if __name__ == "__main__":
    run()
