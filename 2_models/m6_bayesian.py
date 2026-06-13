"""m6 — Bayesiano hierárquico. penaltyblog.models.HierarchicalBayesianGoalModel.

Base do aprendizado sequencial da Iteração 2 (posterior de uma rodada vira prior da
próxima). MCMC é lento; o custo é controlado por `mode` (config.models.bayesian):
  - "fast": usado por padrão em --compare/--select (trace pequeno, viável em rotina).
  - "full": usado em --predict-2026 (fit único no dataset completo, mais preciso).
  - "sequential": [Iteração 2] --update-round.
"""
import penaltyblog as pb
from base_model import PenaltyblogGoalModel


def build(cfg, mode: str | None = None) -> PenaltyblogGoalModel:
    b = cfg["models"]["bayesian"]
    mode = mode or b["default_mode"]
    m = b[mode]
    return PenaltyblogGoalModel(
        pb.models.HierarchicalBayesianGoalModel, name="bayesian",
        max_goals=cfg["models"]["max_goals"],
        fit_kwargs={"n_samples": m["n_samples"], "burn": m["burn"], "n_chains": m["n_chains"]},
        max_train_matches=m.get("max_train_matches"),
    )
