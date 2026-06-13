"""m1 — Poisson simples (baseline de MODELO). penaltyblog.models.PoissonGoalsModel."""
import penaltyblog as pb
from base_model import PenaltyblogGoalModel


def build(cfg) -> PenaltyblogGoalModel:
    return PenaltyblogGoalModel(
        pb.models.PoissonGoalsModel, name="poisson", max_goals=cfg["models"]["max_goals"],
        max_train_matches=cfg["models"]["poisson"].get("max_train_matches"),
    )
