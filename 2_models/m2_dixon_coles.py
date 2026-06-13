"""m2 — Dixon-Coles sobre gols. penaltyblog.models.DixonColesGoalModel.

Corrige a dependência de placares baixos (1-0, 0-0, 1-1) que o Poisson puro erra.
O decay temporal é aplicado via o peso `weight` (w_t × w_comp) passado no fit.
"""
import penaltyblog as pb
from base_model import PenaltyblogGoalModel


def build(cfg) -> PenaltyblogGoalModel:
    return PenaltyblogGoalModel(
        pb.models.DixonColesGoalModel, name="dixon_coles", max_goals=cfg["models"]["max_goals"]
    )
