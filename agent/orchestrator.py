#!/usr/bin/env python3
"""PacketShield-Agentic control plane.

FastAPI service orchestrating a LangGraph multi-agent swarm that reacts to
flags raised by the C++/CUDA data plane:

    IngestionAgent  -> parse raw packet headers + hex dump, triage
    ThreatAnalyst   -> query the pgvector CVE store, score severity
    RemediationAgent-> synthesize a block rule, append it to
                       rules/blocklist.rules, and publish it on Redis
                       (channel: packetshield:rules) so the data plane can
                       reload signatures with zero downtime.

Run:
    pip install -r agent/requirements.txt
    uvicorn agent.orchestrator:app --host 0.0.0.0 --port 8000

Environment:
    RULES_PATH            path to blocklist.rules (default: rules/blocklist.rules)
    REDIS_URL             e.g. redis://localhost:6379/0 (optional)
    DATABASE_URL          postgres URL for pgvector CVE search (optional)
    CVE_SEED_PATH         fallback CVE corpus (default: agent/cve_seed.json)
    DATAPLANE_RELOAD_URL  optional HTTP hook, e.g. http://dataplane:9000/reload,
                          POSTed after a rule push so the C++ engine calls
                          reload_signatures() immediately.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

import operator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langgraph.graph import END, StateGraph

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embeddings import DIM, embed_text, to_pgvector_literal  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RULES_PATH = Path(os.environ.get("RULES_PATH", BASE_DIR / "rules" / "blocklist.rules"))
REDIS_URL = os.environ.get("REDIS_URL")
DATABASE_URL = os.environ.get("DATABASE_URL")
CVE_SEED_PATH = Path(os.environ.get("CVE_SEED_PATH", BASE_DIR / "agent" / "cve_seed.json"))
DATAPLANE_RELOAD_URL = os.environ.get("DATAPLANE_RELOAD_URL")
RULES_CHANNEL = "packetshield:rules"

_SUSPICIOUS_HINTS = (
    "jndi", "ldap", "eval(", "base64", "/bin/", "cmd.exe", "powershell",
    "select", "union", "drop table", "../", "%2e", "<script", "wget", "curl ",
    "exploit", "shell", "passwd",
)
_PRINTABLE_RE = re.compile(rb"[\x20-\x7e]{4,}")
_SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """LangGraph state: keys written by nodes, `notes` uses a list reducer."""

    packet_index: int
    payload_hex: str
    matched_signature: Optional[str]
    headers: dict
    printable_tokens: list[str]
    suspicious_tokens: list[str]
    entropy: float
    triage: str
    cve_hits: list[dict]
    cve_source: str
    severity: str
    severity_score: float
    remediation_rule: Optional[str]
    rule_action: str
    rule_delivery: str
    notes: Annotated[list[str], operator.add]


def _note(state: AgentState, msg: str) -> dict:
    return {"notes": [f"[{datetime.now(timezone.utc).isoformat()}] {msg}"]}


# ---------------------------------------------------------------------------
# IngestionAgent — parse headers + hex dump, triage the alert
# ---------------------------------------------------------------------------


def _parse_headers(raw: bytes) -> dict:
    """Best-effort Ethernet / IPv4 / TCP header parse from raw bytes."""
    headers: dict[str, Any] = {"raw_len": len(raw)}
    if len(raw) >= 14:
        headers["eth_dst"] = raw[0:6].hex(":")
        headers["eth_src"] = raw[6:12].hex(":")
        headers["eth_type"] = f"0x{int.from_bytes(raw[12:14], 'big'):04x}"
    if len(raw) >= 34 and raw[12:14] == b"\x08\x00":  # IPv4
        ihl = (raw[14] & 0x0F) * 4
        headers["ip_src"] = ".".join(str(b) for b in raw[26:30])
        headers["ip_dst"] = ".".join(str(b) for b in raw[30:34])
        headers["ip_proto"] = raw[23]
        if len(raw) >= 14 + ihl + 4:  # TCP ports
            off = 14 + ihl
            headers["tcp_sport"] = int.from_bytes(raw[off : off + 2], "big")
            headers["tcp_dport"] = int.from_bytes(raw[off + 2 : off + 4], "big")
    return headers


def _shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    n = len(data)
    return -sum(
        (c / n) * math.log2(c / n) for c in counts if c
    )


def ingestion_agent(state: AgentState) -> dict:
    try:
        raw = bytes.fromhex(state["payload_hex"])
    except ValueError:
        raw = state["payload_hex"].encode("utf-8", "ignore")
        note = "payload_hex was not valid hex; treated as UTF-8 text"
    else:
        note = "decoded payload hex dump"

    headers = _parse_headers(raw)
    tokens = sorted(
        {m.group().decode("ascii", "ignore") for m in _PRINTABLE_RE.finditer(raw)},
        key=len,
        reverse=True,
    )[:25]
    entropy = _shannon_entropy(raw)

    matched = state.get("matched_signature")
    suspicious = [t for t in tokens if any(h in t.lower() for h in _SUSPICIOUS_HINTS)]
    triage = (
        "known_signature"
        if matched
        else ("anomalous" if (entropy > 7.0 or suspicious) else "benign")
    )

    out: dict[str, Any] = {
        "headers": headers,
        "printable_tokens": tokens,
        "entropy": round(entropy, 3),
        "triage": triage,
        "suspicious_tokens": suspicious[:10],
    }
    out.update(_note(state, f"IngestionAgent: {note}; triage={triage}, "
                            f"entropy={entropy:.2f}, tokens={len(tokens)}"))
    return out


# ---------------------------------------------------------------------------
# ThreatAnalystAgent — pgvector CVE lookup, severity scoring
# ---------------------------------------------------------------------------


def _local_cve_search(query_text: str, limit: int = 3) -> list[dict]:
    """Keyword-overlap fallback when pgvector is unreachable."""
    try:
        seed = json.loads(CVE_SEED_PATH.read_text(encoding="utf-8"))
    except OSError:
        return []
    q = set(re.findall(r"[a-z0-9]{3,}", query_text.lower()))
    scored: list[tuple[float, dict]] = []
    for entry in seed:
        hay = (
            entry["cve_id"] + " " + entry["description"] + " "
            + " ".join(entry.get("keywords", []))
        ).lower()
        hits = sum(1 for tok in q if tok in hay)
        if hits:
            scored.append((hits / max(1, len(q)), entry))
    scored.sort(key=lambda x: -x[0])
    return [
        {
            "cve_id": e["cve_id"],
            "description": e["description"],
            "severity": e["severity"],
            "cvss": e["cvss"],
            "similarity": round(s, 3),
            "source": "local-seed",
        }
        for s, e in scored[:limit]
    ]


async def _pgvector_search(query_text: str, limit: int = 3) -> Optional[list[dict]]:
    if not DATABASE_URL:
        return None
    try:
        import asyncpg

        conn = await asyncpg.connect(DATABASE_URL, timeout=5)
        try:
            vec = to_pgvector_literal(embed_text(query_text, DIM))
            rows = await conn.fetch(
                """
                SELECT cve_id, description, severity, cvss,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM cve_vectors
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                vec,
                limit,
            )
            return [
                {
                    "cve_id": r["cve_id"],
                    "description": r["description"],
                    "severity": r["severity"],
                    "cvss": float(r["cvss"]),
                    "similarity": round(float(r["similarity"]), 3),
                    "source": "pgvector",
                }
                for r in rows
            ]
        finally:
            await conn.close()
    except Exception:
        return None


def threat_analyst_agent(state: AgentState) -> dict:
    query_bits = [
        state.get("matched_signature") or "",
        " ".join(state.get("printable_tokens", [])[:8]),
        " ".join(state.get("suspicious_tokens", [])),
    ]
    query_text = " ".join(b for b in query_bits if b).strip() or "network intrusion"

    try:
        cve_hits = asyncio.run(_pgvector_search(query_text)) or []
        source = "pgvector" if cve_hits else "local-seed"
    except RuntimeError:
        # Already inside an event loop (tests): run the fallback path.
        cve_hits, source = [], "local-seed"
    if not cve_hits:
        cve_hits = _local_cve_search(query_text)
        source = "local-seed"

    # Severity: max of CVE severity and triage-derived floor.
    sev = "LOW"
    score = 2.0
    for hit in cve_hits:
        if _SEVERITY_ORDER.get(hit["severity"], 0) > _SEVERITY_ORDER[sev]:
            sev = hit["severity"]
        score = max(score, float(hit.get("cvss", 0.0)))
    if state.get("triage") == "known_signature":
        sev = sev if _SEVERITY_ORDER[sev] >= 2 else "HIGH"
        score = max(score, 7.5)
    elif state.get("triage") == "anomalous":
        score = max(score, 5.0)
        if _SEVERITY_ORDER[sev] < 1:
            sev = "MEDIUM"

    out = {
        "cve_hits": cve_hits,
        "severity": sev,
        "severity_score": round(score, 1),
        "cve_source": source,
    }
    out.update(
        _note(
            state,
            f"ThreatAnalystAgent: {len(cve_hits)} CVE candidates via {source}; "
            f"severity={sev} ({score:.1f})",
        )
    )
    return out


# ---------------------------------------------------------------------------
# RemediationAgent — synthesize + push a block rule to the data plane
# ---------------------------------------------------------------------------


def _synthesize_rule(state: AgentState) -> tuple[Optional[str], str]:
    matched = state.get("matched_signature")
    if matched:
        return matched, "reassert-known-signature"

    # Zero-day path: pick the longest distinctive suspicious token as the
    # literal block rule; fall back to the longest printable token.
    candidates = state.get("suspicious_tokens") or state.get("printable_tokens") or []
    for tok in candidates:
        clean = tok.strip()
        if 4 <= len(clean) <= 128 and clean.isprintable():
            return clean, "synthesized-zero-day-rule"
    return None, "no-viable-rule"


def _append_rule_file(rule: str) -> bool:
    RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if RULES_PATH.exists():
        existing = {
            ln.strip()
            for ln in RULES_PATH.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        }
    if rule in existing:
        return False
    with RULES_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"{rule}\n")
    return True


async def _publish_rule(rule: str, payload: dict) -> str:
    """Publish to Redis; optionally nudge the data plane's reload hook."""
    outcomes: list[str] = []
    if REDIS_URL:
        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(REDIS_URL)
            try:
                await client.publish(RULES_CHANNEL, json.dumps(payload))
                outcomes.append("redis-published")
            finally:
                await client.aclose()
        except Exception as exc:  # noqa: BLE001 — degraded mode is fine
            outcomes.append(f"redis-unreachable:{type(exc).__name__}")
    else:
        outcomes.append("redis-not-configured")

    if DATAPLANE_RELOAD_URL:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5) as http:
                r = await http.post(DATAPLANE_RELOAD_URL, json={"rule": rule})
                outcomes.append(f"dataplane-reload:{r.status_code}")
        except Exception as exc:  # noqa: BLE001
            outcomes.append(f"dataplane-unreachable:{type(exc).__name__}")
    else:
        outcomes.append("dataplane-hook-not-configured")
    return ",".join(outcomes)


def remediation_agent(state: AgentState) -> dict:
    rule, action = _synthesize_rule(state)
    if rule is None:
        out = {"remediation_rule": None, "rule_action": action}
        out.update(_note(state, "RemediationAgent: no viable rule synthesized; "
                                "alert escalated for human review"))
        return out

    added = _append_rule_file(rule)
    payload = {
        "rule": rule,
        "action": "append",
        "packet_index": state.get("packet_index"),
        "severity": state.get("severity"),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    try:
        delivery = asyncio.run(_publish_rule(rule, payload))
    except RuntimeError:
        delivery = "event-loop-busy:rule-persisted-locally"

    out = {
        "remediation_rule": rule,
        "rule_action": ("appended" if added else "already-present") + f" [{action}]",
        "rule_delivery": delivery,
    }
    out.update(
        _note(
            state,
            f"RemediationAgent: rule '{rule}' {out['rule_action']}; "
            f"delivery={delivery}",
        )
    )
    return out


# ---------------------------------------------------------------------------
# LangGraph wiring
# ---------------------------------------------------------------------------


def build_graph() -> Any:
    graph = StateGraph(AgentState)
    graph.add_node("ingestion", ingestion_agent)
    graph.add_node("threat_analyst", threat_analyst_agent)
    graph.add_node("remediation", remediation_agent)
    graph.set_entry_point("ingestion")
    graph.add_edge("ingestion", "threat_analyst")
    graph.add_edge("threat_analyst", "remediation")
    graph.add_edge("remediation", END)
    return graph.compile()


_GRAPH = None


def get_graph() -> Any:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_pipeline(packet_index: int, payload_hex: str,
                 matched_signature: Optional[str] = None) -> dict:
    """Run the full agent swarm synchronously; returns the final state."""
    initial: AgentState = {
        "packet_index": packet_index,
        "payload_hex": payload_hex,
        "matched_signature": matched_signature,
        "notes": [],
    }
    result = get_graph().invoke(initial)
    return dict(result)


# ---------------------------------------------------------------------------
# FastAPI service
# ---------------------------------------------------------------------------


class AlertIn(BaseModel):
    packet_index: int = Field(ge=0)
    payload_hex: str = Field(min_length=1)
    matched_signature: Optional[str] = None


class AlertReport(BaseModel):
    packet_index: int
    triage: Optional[str] = None
    severity: Optional[str] = None
    severity_score: Optional[float] = None
    cve_hits: list[dict] = []
    remediation_rule: Optional[str] = None
    rule_action: Optional[str] = None
    headers: dict = {}
    notes: list[str] = []


class RuleIn(BaseModel):
    rule: str = Field(min_length=4, max_length=128)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = get_graph()
    yield


app = FastAPI(
    title="PacketShield-Agentic Control Plane",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    pg, rd = "not-configured", "not-configured"
    if DATABASE_URL:
        try:
            import asyncpg

            conn = await asyncpg.connect(DATABASE_URL, timeout=3)
            await conn.close()
            pg = "reachable"
        except Exception:
            pg = "unreachable"
    if REDIS_URL:
        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(REDIS_URL)
            await client.ping()
            await client.aclose()
            rd = "reachable"
        except Exception:
            rd = "unreachable"
    return {
        "status": "ok",
        "agents": ["ingestion", "threat_analyst", "remediation"],
        "pgvector": pg,
        "redis": rd,
        "rules_path": str(RULES_PATH),
    }


@app.post("/alert", response_model=AlertReport)
async def submit_alert(alert: AlertIn) -> AlertReport:
    try:
        final = await asyncio.to_thread(
            run_pipeline, alert.packet_index, alert.payload_hex,
            alert.matched_signature,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AlertReport(
        packet_index=final.get("packet_index", alert.packet_index),
        triage=final.get("triage"),
        severity=final.get("severity"),
        severity_score=final.get("severity_score"),
        cve_hits=final.get("cve_hits", []),
        remediation_rule=final.get("remediation_rule"),
        rule_action=final.get("rule_action"),
        headers=final.get("headers", {}),
        notes=final.get("notes", []),
    )


@app.get("/rules")
async def list_rules() -> dict:
    if not RULES_PATH.exists():
        return {"rules": []}
    rules = [
        ln.strip()
        for ln in RULES_PATH.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    return {"count": len(rules), "rules": rules}


@app.post("/rules/append")
async def append_rule(body: RuleIn) -> dict:
    added = await asyncio.to_thread(_append_rule_file, body.rule)
    return {"rule": body.rule, "appended": added}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("orchestrator:app", host="0.0.0.0", port=8000, reload=False)
