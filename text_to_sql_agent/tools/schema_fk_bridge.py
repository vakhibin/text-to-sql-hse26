"""FK graph bridge: add intermediate tables so LLM-selected tables are connected by foreign keys."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import networkx as nx


def _canonical_table_names(schema: dict[str, Any], names: list[str]) -> list[str]:
    lower_map = {str(t.get("name", "")).lower(): str(t["name"]) for t in schema.get("tables", []) if t.get("name")}
    out: list[str] = []
    seen: set[str] = set()
    for raw in names:
        key = str(raw).strip().lower()
        canon = lower_map.get(key)
        if canon and canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def expand_selected_tables_via_fk_graph(schema: dict[str, Any], selected_tables: list[str]) -> list[str]:
    """Union of shortest paths between all pairs of selected tables (undirected FK graph).

    Tables not present in the schema are ignored. Single or empty selection is returned as-is
    (after canonicalization).
    """
    terminals = _canonical_table_names(schema, selected_tables)
    if len(terminals) < 2:
        return terminals

    G = nx.Graph()
    for t in schema.get("tables", []):
        name = str(t.get("name", "")).strip()
        if name:
            G.add_node(name)
    for t in schema.get("tables", []):
        tname = str(t.get("name", "")).strip()
        if not tname:
            continue
        for fk in t.get("foreign_keys", []) or []:
            ref = str(fk.get("ref_table", "")).strip()
            if ref and G.has_node(tname) and G.has_node(ref):
                G.add_edge(tname, ref)

    expanded: set[str] = set(terminals)
    for u, v in combinations(terminals, 2):
        if not G.has_node(u) or not G.has_node(v):
            continue
        if nx.has_path(G, u, v):
            expanded.update(nx.shortest_path(G, u, v))

    schema_order = [str(t.get("name", "")) for t in schema.get("tables", []) if t.get("name")]
    return [name for name in schema_order if name in expanded]
