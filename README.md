# 🏆 Bolão Copa 2026 — Sistema de Previsão de Placares

Sistema que **valida cientificamente** modelos de previsão de placares para a Copa do
Mundo 2026. Princípio central: **honestidade estatística acima de complexidade** — toda
métrica vem com intervalo de confiança (IC) e a decisão entre modelos usa testes de
significância (empate estatístico → escolhe-se o mais simples).

## Estado atual: Iteração 1 (modelagem sobre GOLS)

Construída a espinha dorsal end-to-end com dados **gratuitos** de gols
([martj42/international_results](https://github.com/martj42/international_results)):

- **Dados:** resultados 2006–2026 normalizados e ponderados (peso temporal × competição).
  Os 72 jogos da fase de grupos da Copa 2026 já vêm no dataset (placar a prever).
- **Modelos** (via [penaltyblog](https://pypi.org/project/penaltyblog/)): Poisson,
  Dixon-Coles, Poisson bivariado, Elo + **ensemble** por pooling log-odds.
- **Validação:** backtest cronológico (sem vazamento) em 7 torneios passados; Brier, RPS,
  Log-Loss, Acurácia 1X2 e Top-1 placar exato, **cada um com IC95% via bootstrap**;
  baselines obrigatórios; Diebold-Mariano + **funil de eliminação**; seleção
  out-of-sample anti-overfitting.
- **Saídas:** `comparison_report.html`, `selected_model.json`, `predictions_2026.json`
  (schema do frontend `bolao-copa-2026.jsx`).

### Deferido para a Iteração 2 (stubs documentados, **sem dados sintéticos**)
| Módulo | Destrava com |
|---|---|
| xG (m3) + scrapers de xG | FootyStats (seleção) / Understat (clube) |
| RAPM + impacto/força de elenco | Kaggle + dcaribou/transfermarkt-datasets + Understat |
| Bayesiano sequencial (`--update-round`) | já implementado (m6); ativa ao vivo na Copa |
| Cruzamento de odds | The Odds API (plano Business) |

O modelo bayesiano (m6) está implementado, mas **fora da comparação de rotina**: o MCMC do
penaltyblog é lento demais para o backtest com bootstrap. Ele é o motor do aprendizado
sequencial da Iteração 2.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Persistência (Supabase)

Supabase é a fonte de verdade dos dados; CSVs locais (`1_data/processed/`) seguem só
como fallback/exportação — se `SUPABASE_URL`/`SUPABASE_KEY` não estiverem configuradas,
o pipeline inteiro funciona normalmente em cima do CSV (nada quebra).

1. Copie `.env.example` para `.env` e preencha `SUPABASE_URL`/`SUPABASE_KEY` (NUNCA
   commitar `.env`).
2. Rode `supabase/schema.sql` no SQL Editor do projeto Supabase (cria as 6 tabelas:
   `matches`, `players`, `player_impact`, `squad_2026`, `predictions`,
   `copa_2026_results`). `supabase-py` não executa DDL, então essa etapa é manual.
3. `python main.py --check-db` confirma a conexão e lista as tabelas encontradas.

Com Supabase configurado: `--scrape` faz UPSERT de `results.csv` em `matches` (sem
duplicar) e `--predict-2026` grava cada previsão em `predictions`, além de emitir
`predictions_2026.json` normalmente.

## Uso

```bash
python main.py --scrape          # baixa dados (martj42) + sync Supabase (matches)
python main.py --prep            # matches_weighted.csv (pesos temporal × competição)
python main.py --compare         # comparação modelo × torneio (com IC) + relatório HTML
python main.py --select          # seleção anti-overfitting + selected_model.json + relatório
python main.py --predict-2026    # predictions_2026.json p/ frontend + tabela predictions
python main.py --check-db        # status da conexão/tabelas Supabase

# Atalhos úteis
python main.py --compare --tournaments WC2018 WC2022 --iters 400   # rodada rápida
python main.py --compare --models poisson dixon_coles elo          # subconjunto de modelos
```

Fluxo completo: `--scrape → --prep → --compare → --select → --predict-2026`.

## Estrutura

```
config.yaml            # TODOS os hiperparâmetros + aliases + pesos (nada hardcoded)
config_loader.py       # leitor único do config
db_client.py           # cliente Supabase (lazy; None se .env não configurado)
dataset.py             # acesso aos dados (Supabase ou CSV) + peso temporal sob demanda
models_registry.py     # registro central de modelos
main.py                # CLI orquestrador
supabase/schema.sql    # DDL das 6 tabelas (rodar manualmente no SQL Editor)
.env.example           # SUPABASE_URL / SUPABASE_KEY (copiar para .env)
1_data/                # scraper_results (+ sync Supabase) + preprocess (+ stubs de xG/jogadores/escalações)
2_models/              # base_model + m1,m2,m4 (penaltyblog), m5 (Elo), m6 (bayes), m7 (ensemble)
3_player_impact/       # [Iteração 2] RAPM, transferência clube→seleção, força de elenco
4_validation/          # metrics, temporal_split, baselines, backtest, comparison, significance, selection
5_outputs/             # report (HTML), predict_2026 (JSON + tabela predictions) + arquivos gerados
```

## Avisos de honestidade
1. Toda métrica tem IC. Empate estatístico (IC da diferença cruza zero) → modelo mais simples.
2. O número honesto é o **out-of-sample** (torneios > cutoff de seleção).
3. Euro ≠ Copa América ≠ Copa do Mundo — torneios continentais são proxy imperfeito.
4. Baselines são pisos: aparecem na tabela, mas nunca são "o modelo escolhido".
