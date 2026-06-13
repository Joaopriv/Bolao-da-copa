"""Gera comparison_report.html (tabela + barras com error bars + funil) e
selected_model.json.

Sem dependência de skill externa: usa matplotlib para os gráficos (embutidos como PNG
base64) e jinja2 para o HTML. Mostra IC em toda métrica, a matriz modelo×torneio, o
funil de eliminação e o destaque out-of-sample.
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from jinja2 import Template  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_loader import path  # noqa: E402


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _bar_with_ci(aggregate: dict, metric: str, title: str) -> str:
    models = list(aggregate)
    pts = [aggregate[m][metric].point for m in models]
    los = [aggregate[m][metric].point - aggregate[m][metric].lo for m in models]
    his = [aggregate[m][metric].hi - aggregate[m][metric].point for m in models]
    order = sorted(range(len(models)), key=lambda i: pts[i])
    models = [models[i] for i in order]; pts = [pts[i] for i in order]
    los = [los[i] for i in order]; his = [his[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(models) + 1.2))
    ax.barh(models, pts, xerr=[los, his], color="#3a7bd5", ecolor="#222", capsize=4)
    ax.set_xlabel(metric); ax.set_title(title); ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    return _png(fig)


_HTML = Template("""<!doctype html><html lang="pt-br"><head><meta charset="utf-8">
<title>Bolão Copa 2026 — Relatório de Comparação</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:32px;color:#1a1a1a;max-width:1100px}
 h1{border-bottom:3px solid #3a7bd5;padding-bottom:6px} h2{margin-top:34px;color:#234}
 table{border-collapse:collapse;margin:12px 0;font-size:13px} td,th{border:1px solid #ccc;padding:5px 8px;text-align:center}
 th{background:#f0f4fa} .chosen{background:#e6f7e6;font-weight:bold} .elim{color:#b00;}
 .small{color:#666;font-size:12px} img{max-width:100%;margin:8px 0}
 code{background:#f3f3f3;padding:1px 5px;border-radius:3px}
 .card{background:#fafcff;border:1px solid #dde;border-radius:8px;padding:14px 18px;margin:10px 0}
</style></head><body>
<h1>🏆 Bolão Copa 2026 — Comparação de Modelos</h1>
<p class="small">λ (decay temporal) = {{ lam }} · bootstrap = {{ iters }} reamostras · IC95% ·
torneios de teste: {{ tournaments|join(', ') }}</p>
<p class="small">⚠️ Iteração 1: modelagem sobre GOLS (martj42). Modelos de xG, impacto de elenco
(RAPM) e bayesiano sequencial ficam para a Iteração 2. Euro ≠ Copa América ≠ Copa do Mundo
— métricas continentais são proxy imperfeito da Copa.</p>

<h2>1. Métricas agregadas (todos os jogos de teste) — menor RPS é melhor</h2>
<img src="{{ rps_chart }}"><img src="{{ brier_chart }}">
<table><tr><th>Modelo</th>{% for mt in metric_names %}<th>{{ mt }}</th>{% endfor %}</tr>
{% for m in models %}<tr class="{{ 'elim' if m in eliminated }}">
<td style="text-align:left">{{ m }}</td>
{% for mt in metric_names %}<td>{{ aggregate[m][mt] }}</td>{% endfor %}</tr>{% endfor %}
</table>
<p class="small">Linhas em vermelho = descartadas pelo funil (claramente piores que o melhor).</p>

<h2>2. Matriz modelo × torneio (RPS com IC95%)</h2>
<table><tr><th>Modelo</th>{% for t in tournaments %}<th>{{ t }}</th>{% endfor %}</tr>
{% for m in models %}<tr><td style="text-align:left">{{ m }}</td>
{% for t in tournaments %}<td>{{ cell_rps[m][t] }}</td>{% endfor %}</tr>{% endfor %}</table>

<h2>3. Funil de eliminação (a ferramenta de decisão)</h2>
<div class="card">
<p><b>Melhor por RPS:</b> <code>{{ funnel.best_by_rps }}</code></p>
<p><b>Descartados</b> (IC da diferença vs melhor NÃO cruza zero → claramente piores):
{% if funnel.eliminated %}{% for m in funnel.eliminated %}<code>{{ m }}</code> {% endfor %}{% else %}—{% endif %}</p>
<p><b>Equivalentes</b> (IC cruza zero → empate estatístico, ordenados por simplicidade):
{% for m in funnel.equivalent %}<code>{{ m }}</code> {% endfor %}</p>
<p><b>✅ Escolhido</b> (mais simples entre os equivalentes):
<code style="background:#cdeccd">{{ funnel.chosen }}</code></p>
</div>
<table><tr><th>Modelo</th><th>ΔRPS vs melhor</th><th>IC95% da diferença</th><th>cruza zero?</th><th>p-valor (teste t pareado)</th></tr>
{% for m, d in funnel.details.items() %}<tr><td style="text-align:left">{{ m }}</td>
<td>{{ '%+.4f'|format(d.diff_mean) }}</td>
<td>[{{ '%+.4f'|format(d.diff_lo) }}, {{ '%+.4f'|format(d.diff_hi) }}]</td>
<td>{{ 'sim (empate)' if d.crosses_zero else 'não' }}</td>
<td>{{ '%.3f'|format(d.t_p) }}</td></tr>{% endfor %}</table>

{% if selection %}
<h2>4. Seleção honesta — in-sample vs out-of-sample</h2>
<div class="card">
<p>Seleção feita nos torneios ≤ {{ selection.cutoff_year }}
({{ selection.validation_tournaments|join(', ') }}); teste ÚNICO nos posteriores
({{ selection.test_tournaments|join(', ') }}).</p>
<p><b>Modelo escolhido:</b> <code>{{ selection.chosen }}</code></p>
<p><b>Pesos do ensemble (otimizados só na validação):</b>
{% for k,v in selection.ensemble_weights.items() %}{{ k }}={{ '%.2f'|format(v) }} {% endfor %}</p>
<table><tr><th>Métrica</th><th>In-sample (≤{{ selection.cutoff_year }})</th><th>Out-of-sample (&gt;{{ selection.cutoff_year }})</th></tr>
{% for mt in metric_names if mt in selection.in_sample %}<tr><td style="text-align:left">{{ mt }}</td>
<td>{{ selection.in_sample[mt].point|round(4) }} [{{ selection.in_sample[mt].lo|round(4) }}, {{ selection.in_sample[mt].hi|round(4) }}]</td>
<td><b>{{ selection.out_of_sample[mt].point|round(4) }}</b> [{{ selection.out_of_sample[mt].lo|round(4) }}, {{ selection.out_of_sample[mt].hi|round(4) }}]</td></tr>{% endfor %}</table>
<p class="small">O número honesto é o out-of-sample.</p>
</div>
{% endif %}
<p class="small">Gerado pelo pipeline Bolão Copa 2026 · Iteração 1.</p>
</body></html>""")


def write_comparison_report(comparison: dict, selection: dict | None = None) -> Path:
    agg = comparison["aggregate"]
    models = comparison["models"]
    tournaments = comparison["tournaments"]
    metric_names = list(next(iter(agg.values())).keys())
    cell_rps = {m: {t: comparison["cell"][(m, t)]["RPS"] for t in tournaments} for m in models}

    html = _HTML.render(
        lam=comparison["lambda"], iters=comparison["iters"],
        tournaments=tournaments, models=models, metric_names=metric_names,
        aggregate=agg, cell_rps=cell_rps,
        eliminated=comparison["funnel"]["eliminated"],
        funnel=comparison["funnel"], selection=selection,
        rps_chart=_bar_with_ci(agg, "RPS", "RPS agregado (± IC95%)"),
        brier_chart=_bar_with_ci(agg, "Brier", "Brier agregado (± IC95%)"),
    )
    dest = path("5_outputs", "comparison_report.html")
    dest.write_text(html, encoding="utf-8")
    print(f"  relatório -> {dest}")
    return dest


def write_selected_model(selection: dict, justification: str) -> Path:
    out = {
        "chosen_model": selection["chosen"],
        "ensemble_weights": selection["ensemble_weights"],
        "cutoff_year": selection["cutoff_year"],
        "validation_tournaments": selection["validation_tournaments"],
        "test_tournaments": selection["test_tournaments"],
        "in_sample_RPS": selection["in_sample"]["RPS"],
        "out_of_sample_RPS": selection["out_of_sample"]["RPS"],
        "funnel": {
            "best_by_rps": selection["funnel"]["best_by_rps"],
            "eliminated": selection["funnel"]["eliminated"],
            "equivalent": selection["funnel"]["equivalent"],
            "chosen": selection["funnel"]["chosen"],
        },
        "justification": justification,
    }
    dest = path("5_outputs", "selected_model.json")
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  selected_model.json -> {dest}")
    return dest
