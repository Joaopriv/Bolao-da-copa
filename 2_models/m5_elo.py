"""m5 — Elo dinâmico. penaltyblog.ratings.Elo (via EloModel em base_model)."""
from base_model import EloModel


def build(cfg) -> EloModel:
    e = cfg["models"]["elo"]
    return EloModel(
        name="elo", k=e["k"], home_field_advantage=e["home_field_advantage"],
        draw_base=e["draw_base"], draw_width=e["draw_width"],
    )
