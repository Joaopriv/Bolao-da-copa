"""Cliente Supabase — fonte de verdade do projeto.

Lê SUPABASE_URL / SUPABASE_KEY do ambiente (via .env, python-dotenv). Se ausentes,
get_client() retorna None e o pipeline cai no fallback de CSV em
1_data/processed/matches_weighted.csv — nada quebra sem credenciais.

Schema das tabelas: ver supabase/schema.sql.
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

_warned = False


@lru_cache(maxsize=1)
def get_client():
    """Retorna o cliente Supabase, ou None se SUPABASE_URL/SUPABASE_KEY não configuradas."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        global _warned
        if not _warned:
            print("  (Supabase não configurado — SUPABASE_URL/SUPABASE_KEY ausentes; "
                  "usando CSV local como fonte de dados)")
            _warned = True
        return None

    from supabase import create_client
    return create_client(url, key)


def fetch_all(table: str, columns: str = "*", page_size: int = 1000) -> list[dict] | None:
    """Lê todas as linhas de `table` (paginado). Retorna None se Supabase não configurado."""
    client = get_client()
    if client is None:
        return None

    rows: list[dict] = []
    start = 0
    while True:
        resp = (client.table(table).select(columns)
                .range(start, start + page_size - 1).execute())
        batch = resp.data
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
    return rows


def fetch_one(table: str, column: str, value, columns: str = "*") -> dict | None:
    """Lê a primeira linha de `table` onde `column == value`. Retorna None se
    Supabase não configurado ou se nenhuma linha for encontrada."""
    client = get_client()
    if client is None:
        return None
    resp = client.table(table).select(columns).eq(column, value).limit(1).execute()
    return resp.data[0] if resp.data else None


def upsert(table: str, rows: list[dict], on_conflict: str, batch_size: int = 500) -> bool:
    """UPSERT em lotes. Retorna False (sem erro) se Supabase não configurado."""
    client = get_client()
    if client is None or not rows:
        return False
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        client.table(table).upsert(chunk, on_conflict=on_conflict).execute()
    return True


def delete_all(table: str, pk_col: str = "player_id") -> bool:
    """Remove TODAS as linhas de `table` (coluna texto `pk_col` usada só para o filtro
    `gte("")`, que casa qualquer string). Usado por scripts que reconstroem a tabela
    inteira a cada execução (ex. D2 player_national_team/squad_2026), para não deixar
    linhas órfãs de uma rodada anterior com matching diferente."""
    client = get_client()
    if client is None:
        return False
    client.table(table).delete().gte(pk_col, "").execute()
    return True
