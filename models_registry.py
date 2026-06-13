"""Registro central de modelos.

A pasta `2_models/` começa com dígito (não importável como pacote), então a adicionamos
ao sys.path e importamos os módulos por nome de arquivo. Este é o ÚNICO lugar que lida
com esse detalhe — todo o resto importa daqui.
"""
from __future__ import annotations

import sys

from config_loader import load_config, path

# Torna os módulos de 2_models importáveis por nome (m1_poisson, base_model, ...).
_MODELS_DIR = str(path("2_models"))
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

import m1_poisson            # noqa: E402
import m2_dixon_coles        # noqa: E402
import m3_dixon_coles_xg     # noqa: E402
import m4_bivariate_poisson  # noqa: E402
import m5_elo                # noqa: E402
import m6_bayesian           # noqa: E402
import m8_dynamic_dc          # noqa: E402
from m7_ensemble import EnsembleModel  # noqa: E402

# Mapa nome -> builder(cfg).
_BUILDERS = {
    "poisson": m1_poisson.build,
    "dixon_coles": m2_dixon_coles.build,
    "dixon_coles_xg": m3_dixon_coles_xg.build,
    "bivariate_poisson": m4_bivariate_poisson.build,
    "elo": m5_elo.build,
    "bayesian": m6_bayesian.build,
    "dynamic_dc": m8_dynamic_dc.build,
}


def available_members() -> list[str]:
    """Modelos-membro (sem o ensemble) definidos em config.models.members."""
    return list(load_config()["models"]["members"])


def build_member(name: str, cfg=None, mode: str | None = None):
    """Instancia (não treina) um modelo-membro pelo nome.

    `mode` só se aplica ao "bayesian" (fast/full/sequential — ver config.models.bayesian);
    ignorado pelos demais.

    Se `cfg.squad_strength.squad_offset_weight` > 0 (D6.2), envolve o modelo em
    `SquadAdjustedModel` (tilting do grid pela força de elenco, D5/D6.1). Peso 0
    (default sem squad_strength) -> sem wrapper, comportamento idêntico à Iteração 1.

    Se `cfg.h2h.weight` > 0 (Iteração 3 / F7), envolve (também) em `H2HAdjustedModel`
    (tilting do grid pelo histórico de confronto direto). Peso 0 -> sem wrapper.
    """
    cfg = cfg or load_config()
    if name == "bayesian":
        m = _BUILDERS[name](cfg, mode=mode)
    else:
        m = _BUILDERS[name](cfg)

    weight = cfg.get("squad_strength", {}).get("squad_offset_weight", 0.0)
    if weight:
        from squad_adjustment import SquadAdjustedModel
        m = SquadAdjustedModel(m, weight)

    h2h_cfg = cfg.get("h2h", {})
    h2h_weight = h2h_cfg.get("weight", 0.0)
    if h2h_weight:
        from h2h_adjustment import H2HAdjustedModel
        m = H2HAdjustedModel(m, h2h_weight, h2h_cfg.get("min_matches", 3))
    return m


def build_all_members(cfg=None, bayesian_mode: str | None = None) -> dict:
    """Instancia todos os membros do config como dict {nome: modelo}."""
    cfg = cfg or load_config()
    return {name: build_member(name, cfg, mode=bayesian_mode if name == "bayesian" else None)
            for name in available_members()}


def build_ensemble(weights: dict | list | None = None, cfg=None,
                    bayesian_mode: str | None = None) -> EnsembleModel:
    """Constrói o ensemble sobre os membros do config. `weights` pode ser dict {nome: w}."""
    cfg = cfg or load_config()
    members = available_members()
    objs = [build_member(n, cfg, mode=bayesian_mode if n == "bayesian" else None)
            for n in members]
    if isinstance(weights, dict):
        w = [weights.get(n, 0.0) for n in members]
    else:
        w = weights
    return EnsembleModel(objs, weights=w)
