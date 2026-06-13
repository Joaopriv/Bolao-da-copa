-- Bolão Copa 2026 — schema Supabase (fonte de verdade).
--
-- Como aplicar: cole este arquivo no SQL Editor do projeto Supabase e execute.
-- supabase-py (REST/PostgREST) não executa DDL, então a criação das tabelas é
-- manual nesta etapa; depois disso todo o pipeline lê/escreve via supabase-py.
--
-- IMPORTANTE sobre w_temporal / w_final em `matches`:
-- o peso temporal w_t = exp(-lambda * dias) depende da DATA DE REFERÊNCIA (início do
-- torneio no backtest, ou hoje na previsão da Copa) — congelar um valor único por linha
-- reintroduziria vazamento. Estas colunas guardam um SNAPSHOT informativo (relativo a
-- "hoje", calculado no --scrape/--prep) para uso externo (ex.: dashboards); o pipeline de
-- validação SEMPRE recalcula w_t sob demanda em dataset.with_weights().

create table if not exists matches (
    id              bigserial primary key,
    date            date not null,
    home_team       text not null,
    away_team       text not null,
    home_score      integer,
    away_score      integer,
    home_xg         double precision,
    away_xg         double precision,
    tournament      text not null,
    neutral         boolean not null default false,
    w_temporal      double precision,
    w_competition   double precision,
    w_overlap       double precision,
    w_final         double precision,
    source          text not null default 'martj42',
    unique (date, home_team, away_team, tournament)
);
create index if not exists idx_matches_date on matches (date);
create index if not exists idx_matches_tournament on matches (tournament);

-- Iteração 2 — RAPM / força de elenco (FootyStats / Kaggle+transfermarkt-datasets).
create table if not exists players (
    player_id     text primary key,
    name          text not null,
    nationality   text,
    position      text,
    market_value  double precision
);

create table if not exists player_impact (
    player_id      text not null references players (player_id),
    attack_delta   double precision,
    defense_delta  double precision,
    std_error      double precision,
    source         text not null,
    updated_at     timestamptz not null default now(),
    unique (player_id, source)
);

create table if not exists squad_2026 (
    team       text not null,
    player_id  text not null references players (player_id),
    available  boolean not null default true,
    position   text,
    unique (team, player_id)
);

-- Previsões emitidas (espelha predictions_2026.json, uma linha por modelo×jogo).
create table if not exists predictions (
    game_id            text not null,
    home_team          text not null,
    away_team          text not null,
    model_name         text not null,
    prob_home          double precision,
    prob_draw          double precision,
    prob_away          double precision,
    confidence         double precision,
    top_scores         jsonb,
    odds_home          double precision,
    odds_draw          double precision,
    odds_away          double precision,
    market_prob_home   double precision,
    market_prob_draw   double precision,
    market_prob_away   double precision,
    generated_at       timestamptz not null default now(),
    unique (game_id, model_name)
);
create index if not exists idx_predictions_game on predictions (game_id);

-- Iteração 2 — resultados reais da Copa 2026 conforme acontecem (aprendizado sequencial).
create table if not exists copa_2026_results (
    game_id     text primary key,
    home_team   text not null,
    away_team   text not null,
    date        date not null,
    round       integer,
    group_name  text,
    home_score  integer,
    away_score  integer,
    home_xg     double precision,
    away_xg     double precision,
    played_at   timestamptz
);

-- Iteração 2 — xG de jogador por temporada: clube (Understat, D1) e seleção
-- (StatsBomb, Fase 0/"Prompt C3").
create table if not exists player_seasons (
    player_id     text not null references players (player_id),
    team          text not null,        -- clube (D1) ou seleção (source='statsbomb_national')
    season        text not null,        -- "2022"/"2023"/"2024" (Understat) ou "2018-2024" (StatsBomb)
    competition   text,
    minutes       double precision,
    games         double precision,
    xg90          double precision,
    xa90          double precision,
    npxg90        double precision,
    xgchain90     double precision,
    xgbuildup90   double precision,
    source        text not null,        -- 'understat' | 'statsbomb_national'
    updated_at    timestamptz not null default now(),
    unique (player_id, team, season, source)
);

-- Iteração 2 — mapeamento jogador de clube -> seleção (D2, fuzzy match Transfermarkt).
create table if not exists player_national_team (
    player_id        text not null references players (player_id),
    national_team    text not null,
    position         text,
    transfermarkt_id text,
    unique (player_id, national_team)
);

-- Iteração 2 — agregados de TIME por temporada (Understat, D1) — alvo (y) do RAPM-lite.
create table if not exists team_seasons (
    team    text not null,
    league  text not null,
    season  text not null,
    xg90    double precision,
    xga90   double precision,
    games   double precision,
    source  text not null default 'understat',
    unique (team, league, season)
);

-- Iteração 2 — força de elenco por seleção da Copa 2026 (D5/D6).
create table if not exists squad_strength (
    team             text primary key,   -- nome canônico martj42 (uma das 48 da Copa 2026)
    attack_adjusted  double precision,
    defense_adjusted double precision,
    attack_z_pct     double precision,   -- percentil 0-100 entre as 48 (D8 squad_note)
    xgf90            double precision,   -- D8 xg_context
    xga90            double precision,
    w_overlap        double precision,
    data_coverage    double precision,
    updated_at       timestamptz not null default now()
);

-- Iteração 2 / Prompt E — odds de mercado ao vivo (The Odds API), 1 snapshot por jogo
-- (upsert on_conflict=match_id a cada --fetch-odds). Odds NUNCA entram nos modelos --
-- aqui só para exibição no frontend (régua) e alerta de divergência.
create table if not exists odds_2026 (
    match_id          text primary key,   -- mesmo formato de game_id (home-away-YYYYMMDD)
    home_team         text not null,
    away_team         text not null,
    match_date        date not null,
    odd_home          double precision,
    odd_draw          double precision,
    odd_away          double precision,
    implied_home      double precision,   -- prob. de mercado sem vig (de-vig)
    implied_draw      double precision,
    implied_away      double precision,
    diff_pp           double precision,   -- max |prob_modelo - prob_mercado| (pp)
    divergence_alert  boolean not null default false,  -- diff_pp > odds_api.divergence_alert_pp
    fetched_at        timestamptz not null default now()
);
