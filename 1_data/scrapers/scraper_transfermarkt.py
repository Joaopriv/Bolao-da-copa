"""[Iteração 2 / D2] Download do dataset Transfermarkt (Kaggle, dcaribou/transfermarkt-datasets).

Usa `kagglehub` (não o pacote `kaggle` legado) — autentica via env var
`KAGGLE_API_TOKEN` (já em .env), sem precisar de `~/.kaggle/kaggle.json`.
`kagglehub.dataset_download` faz cache local em `~/.cache/kagglehub/...` e só baixa
de novo se uma versão nova do dataset for publicada.

Usado por `1_data/match_players.py` (D2) para mapear jogador de clube (Understat) ->
seleção nacional (`players.csv`, coluna `country_of_citizenship`).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config_loader import load_config  # noqa: E402


def download_player_scores(verbose: bool = True) -> Path:
    import kagglehub

    cfg = load_config()
    dataset = cfg["transfermarkt"]["kaggle_dataset"]
    if verbose:
        print(f"  Baixando dataset Kaggle '{dataset}' (cache local se já presente) ...")
    path = Path(kagglehub.dataset_download(dataset))
    if verbose:
        print(f"  Dataset disponível em: {path}")
    return path


if __name__ == "__main__":
    download_player_scores()
