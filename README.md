# PacketShield-Agentic — GPU-Accelerated DPI with an Autonomous Agent Swarm

A production-grade deep packet inspection (DPI) engine that pairs a
**C++20 + CUDA data plane** (wire-speed parallel signature matching) with a
**Python control plane** — a LangGraph multi-agent swarm (FastAPI) that
triages GPU-flagged packets, scores them against a pgvector CVE store, and
pushes freshly synthesized block rules back to the GPU engine **without
service interruption**.

## Why this exists

Software intrusion detection on multi-threaded CPUs chokes at 10 Gbps+ line
rates: string-matching sweeps over payloads cause cache thrashing and packet
drops, and static rule sets cannot adapt to polymorphic or zero-day payloads
without human analysts in the loop. PacketShield-Agentic splits the problem:

- **Data plane (C++/CUDA)** — streams 100K-packet batches through pinned host
  memory into a custom CUDA kernel (`match_signatures_kernel`) that inspects
  packets concurrently across warp-organized thread blocks.
- **Control plane (Python/LangGraph)** — when the data plane flags an
  anomalous payload, three agents collaborate: intake/triage, threat analysis
  against historical CVE vectors, and remediation that synthesizes a new block
  rule and hot-reloads it into the running engine.

## System architecture

```
  ┌──────────────────────── DATA PLANE (C++20 / CUDA) ────────────────────────┐
  │  PacketBatch (pinned, cudaHostAlloc)                                      │
  │      │ cudaMemcpyAsync H2D (non-blocking stream)                           │
  │      ▼                                                                    │
  │  match_signatures_kernel<<<blocks, 256 threads>>>   1 thread = 1 packet    │
  │      │ cudaMemcpyAsync D2H                                                 │
  │      ▼                                                                    │
  │  MatchResult[] ──► flag ──► POST /alert ──────────────────────────┐        │
  └──────────────────────────────────────────────────────────────────┼────────┘
                                                                      │
  ┌─────────────────────── CONTROL PLANE (Python) ─────────────────────┼────────┐
  │  FastAPI + LangGraph                                              │        │
  │                                                                    ▼        │
  │  IngestionAgent: parse Ethernet/IPv4/TCP headers + hex dump,       │        │
  │      extract printable tokens, Shannon entropy → triage           │        │
  │                    │                                              │        │
  │                    ▼                                              │        │
  │  ThreatAnalystAgent: pgvector cosine search over CVE embeddings → │        │
  │      severity score (falls back to local CVE seed if DB is down)  │        │
  │                    │                                              │        │
  │                    ▼                                              │        │
  │  RemediationAgent: synthesize block rule → append to               │        │
  │      rules/blocklist.rules → publish on Redis channel              │        │
  │      `packetshield:rules` → optional data-plane reload hook        │        │
  └────────────────────────────────────────────────────────────────────────────┘
                                                                      │
                      data-plane operator calls                         │
                      reload_signatures("rules/blocklist.rules")        │
                      on Redis notification ──► next batch uses        │
                      the new rule. No restart.                       ◄┘
```

### Pinned-memory transfer pipeline (data plane)

1. `create_batch()` packs the full 100K-packet batch into one contiguous
   buffer allocated with `cudaHostAlloc` (page-locked). Pinned memory lets the
   GPU DMA engine pull data at full PCIe bandwidth — no hidden driver-side
   staging copy.
2. All transfers use `cudaMemcpyAsync` on a dedicated non-blocking CUDA
   stream, so copies and kernel execution overlap across batches.
3. Device buffers are owned by an RAII `DeviceBuffer` wrapper
   (`cudaMalloc`/`cudaFree`); exceptions anywhere in the pipeline free
   everything — no leaks on the error path.
4. Timing uses CUDA events around transfers + kernel, excluding host setup
   noise.

### Warp execution design

- **One thread per packet** (`pkt = blockIdx.x * blockDim.x + threadIdx.x`),
  256 threads/block = 8 warps. No inter-thread communication, so no
  `__syncthreads()` stalls.
- Consecutive threads read consecutive packets → **coalesced** global-memory
  accesses.
- The signature blob is read via `__ldg()` through the **read-only data
  cache** (every thread scans the same signatures → near-100% hit rate).
- First-byte filter + short-circuit on first match per thread.

### Agentic feedback loop (control plane)

The signature table inspected by both `cpu_inspect()` and
`run_gpu_inspection()` is **live and mutable** (`std::shared_mutex`: many
concurrent readers, exclusive writer). The Remediation agent's rule push
lands via `reload_signatures(path)` / `append_signature(rule)` and takes
effect on the very next batch — the CUDA kernel rebuilds its signature blob
from the current table on every launch.

## File layout

| Path | Contents |
|------|----------|
| `CMakeLists.txt` | C++20 + optional CUDA build, `-O3 --use_fast_math -Wall`, CTest |
| `include/gpu_matcher.cuh` | `PacketBatch`, `MatchResult`, live signature-table API |
| `src/gpu_matcher.cu` | `match_signatures_kernel`, async pipeline, RAII device memory |
| `src/cpu_matcher.cpp` | Thread-pool CPU baseline + `shared_mutex` signature table |
| `src/main.cpp` | Benchmark: 100K packets, intrusion injected at #4200 |
| `tests/test_matcher.cpp` | CTest suite incl. the agentic hot-reload contract test |
| `rules/blocklist.rules` | Live blocklist (one signature per line, `#` comments) |
| `agent/orchestrator.py` | FastAPI + LangGraph: 3-agent swarm |
| `agent/embeddings.py` | Deterministic text embeddings (dependency-free) |
| `agent/load_cves.py` | Seeds pgvector from `agent/cve_seed.json` |
| `agent/schema.sql` | pgvector schema for the CVE store |
| `agent/cve_seed.json` | Fallback CVE corpus |
| `agent/requirements.txt` | Python 3.11+ dependencies |
| `docker-compose.yml` | pgvector Postgres + Redis services |

## Build the data plane

Requires CMake ≥ 3.24 and a C++20 compiler. CUDA Toolkit (≥ 11.8, `nvcc`)
is optional — without it the project builds CPU-only and GPU tests skip.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/packetshield_bench
```

## Benchmark

The harness generates **100,000 synthetic packets averaging 512 bytes** and
injects a test intrusion signature into **packet #4200**. Target performance
on reference discrete-GPU hardware (PCIe 4.0):

| Implementation | 100K-packet batch | Speedup |
|----------------|-------------------|---------|
| CPU baseline (`std::thread`, N = cores) | ~48.2 ms | 1.0x |
| GPU (`match_signatures_kernel`, pinned + async) | ~6.1 ms | **~7.9x** |

`./build/packetshield_bench` prints the measured numbers on your hardware,
verifies the injected intrusion is detected, and checks CPU/GPU verdict
agreement before reporting speedup. Update this table with your measured
figures.

## Run the control plane

```bash
pip install -r agent/requirements.txt

# Optional backing services (pgvector CVE store + Redis pub/sub):
docker compose up -d
DATABASE_URL=postgresql://packetshield:packetshield@localhost:5432/packetshield \
    python agent/load_cves.py

# Start the agent swarm API:
REDIS_URL=redis://localhost:6379/0 \
DATABASE_URL=postgresql://packetshield:packetshield@localhost:5432/packetshield \
uvicorn agent.orchestrator:app --host 0.0.0.0 --port 8000
```

Without `DATABASE_URL`/`REDIS_URL` the service still runs: the Threat
Analyst falls back to the local CVE seed corpus and the Remediation agent
persists rules to `rules/blocklist.rules` (delivery field reports
`redis-not-configured`). Health status is visible at `GET /health`.

### Try the full loop

```bash
# 1. Data plane flags packet 4200 (intrusion signature detected):
PAYLOAD=$(python3 -c "print('00'*64 + 'EXPLOIT_CVE_2024-8811'.encode().hex() + '00'*64)")

# 2. Control plane triages, scores, and remediates:
curl -s -X POST localhost:8000/alert \
  -H 'Content-Type: application/json' \
  -d "{\"packet_index\": 4200, \"payload_hex\": \"$PAYLOAD\",
       \"matched_signature\": \"EXPLOIT_CVE_2024-8811\"}" | python3 -m json.tool

# 3. Confirm the rule landed in the live blocklist:
curl -s localhost:8000/rules | python3 -m json.tool
```

A data-plane operator subscribes to the Redis channel `packetshield:rules`
and calls `reload_signatures("rules/blocklist.rules")` (or sets
`DATAPLANE_RELOAD_URL` to an HTTP hook that does it) — the next GPU batch
then enforces the new rule with no restart. `tests/test_matcher.cpp`
(`test_dynamic_signature_reload`) covers this contract in CTest.

## Free cloud GPU: Google Colab

Colab's free tier provides a T4 GPU with `nvcc` preinstalled — enough to
compile and run the full data plane:

1. Go to `colab.new` → **Runtime → Change runtime type → T4 GPU** → save.
2. In a code cell, verify the GPU: `!nvidia-smi`
3. Clone, build, test, and benchmark (replace `<you>`):

```python
!git clone https://github.com/<you>/packetshield-agentic.git
%cd packetshield-agentic
!cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
!cmake --build build -j
!ctest --test-dir build --output-on-failure
!./build/packetshield_bench
```

4. For the control plane on Colab:
```python
!pip install -r agent/requirements.txt
!python - <<'EOF'
from agent.orchestrator import run_pipeline
report = run_pipeline(4200, "00" * 64 + "EXPLOIT_CVE_2024-8811".encode().hex(),
                      matched_signature="EXPLOIT_CVE_2024-8811")
print(report["severity"], report["rule_action"])
print(report["notes"][-1])
EOF
```

Notes: Colab VMs are ephemeral — results and the rules file reset when the
runtime recycles. Commit any rule changes you care about from the cloned
repo before disconnecting.

## Free cloud GPU: Kaggle

1. Create a notebook at kaggle.com → **Settings → Accelerator → GPU T4 x2**
   → **Internet: ON** (needed for `git clone`).
2. Run the same shell sequence as Colab above (`!git clone …`, `!cmake …`,
   `!ctest …`, `!./build/packetshield_bench`). Kaggle sessions last up to
   ~12h; attach a dataset or commit outputs to keep artifacts.

## API reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service status; pgvector/redis reachability; agents list |
| POST | `/alert` | Submit a data-plane flag → runs the 3-agent swarm |
| GET | `/rules` | Current live blocklist |
| POST | `/rules/append` | Manually append a rule (`{"rule": "..."}`) |

`POST /alert` body:

```json
{
  "packet_index": 4200,
  "payload_hex": "00ff…",
  "matched_signature": "EXPLOIT_CVE_2024-8811"
}
```

Response includes `triage`, `severity` / `severity_score`, `cve_hits`
(`cve_id`, `description`, `cvss`, `similarity`, `source`), the synthesized
`remediation_rule`, `rule_action`, parsed `headers`, and a timestamped
`notes` trace of each agent's reasoning.

## GitHub deployment

```bash
git init -b main
git add .
git commit -m "Initial commit: PacketShield-Agentic high-throughput GPU DPI engine"
gh repo create packetshield-agentic --public --source=. --push
```

Or with a manual remote:

```bash
git remote add origin git@github.com:<you>/packetshield-agentic.git
git push -u origin main
```

## Tests

```bash
ctest --test-dir build --output-on-failure
```

- `test_detection_correctness` — every signature detected at the right packet
  with the right id; clean packets produce no false positives.
- `test_injected_intrusion_4200` — the benchmark's headline case.
- `test_memory_safety` — empty payloads/batches, null frees, pinned
  allocation round-trip, `destroy_batch` nulling.
- `test_standard_intrusion_payloads` — realistic IDS attack strings.
- `test_dynamic_signature_reload` — the agentic contract: a zero-day rule
  pushed at runtime is enforced on the next batch; bad reloads never clobber
  the live table.
- `test_gpu_cpu_parity` (CUDA builds) — GPU and CPU verdicts identical.

## License

MIT — see `LICENSE`.
