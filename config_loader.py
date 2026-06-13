"""Carregador único de configuração (config.yaml) e utilidades de caminho.

Centraliza o acesso ao config.yaml para que todos os módulos leiam os mesmos
hiperparâmetros e tabelas de normalização — nada hardcoded espalhado pelo código.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

# Raiz do projeto = pasta deste arquivo.
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Lê e cacheia o config.yaml."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def path(*parts: str) -> Path:
    """Caminho absoluto a partir da raiz do projeto."""
    return ROOT.joinpath(*parts)
