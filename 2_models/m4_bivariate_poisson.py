"""m4 — Poisson bivariado. penaltyblog.models.BivariatePoissonGoalModel.

Modela correlação entre os gols dos dois times (mesmo jogo), capturando dependência
que os modelos independentes ignoram.
"""
import penaltyblog as pb
from base_model import PenaltyblogGoalModel


def build(cfg) -> PenaltyblogGoalModel:
    return PenaltyblogGoalModel(
        pb.models.BivariatePoissonGoalModel, name="bivariate_poisson",
        max_goals=cfg["models"]["max_goals"],
        max_train_matches=cfg["models"]["bivariate_poisson"].get("max_train_matches"),
    )
