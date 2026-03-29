
# Aether — Adaptive Empathic Thinking, Hearing & Experiential Reasoning
**v1.2 · Developer Reference**

Aether is a production-ready, **empathy-first** conversational AI agent built on **Phi-3.5-mini-instruct**. It combines a hierarchical three-tier memory system, an upstream emotion-injection pipeline, a ReAct tool loop, per-user tone optimisation via Thompson Sampling, and a full observability stack — all in a single FastAPI service.

**Python 3.10+** **FastAPI 0.110+** **Redis 7+** **Phi-3.5-mini-instruct** **MIT License**

## 1. Overview

Aether is designed around a single guiding principle: **the emotional context of a conversation is not metadata — it is the primary signal**. Every architectural decision flows from that. Emotion detection runs first, safety checks run second, and the LLM receives an emotionally-informed prompt rather than a neutral one. The result is a system that responds differently to someone venting about a hard day versus someone asking a factual question, without requiring any explicit instruction.

The v1.2 release consolidates 24 months of iterative improvement into a stable, deployable codebase. The major advances over v1.0 are the **hierarchical memory system**, the shift from post-generation empathy rewriting to **upstream emotion injection**, and the introduction of a **ReAct tool loop** for sequential reasoning over external data sources.

### 1.1 Design Principles

- **Empathy-first pipeline**: Emotion detection is the first operation on every request — not an afterthought.
- **No-surprise degradation**: Every optional component (Redis, vLLM, Whisper, BLIP-2, LangChain) has a graceful fallback. The system runs on CPU with no external services if needed.
- **Implicit learning**: The tone bandit updates from follow-up message sentiment and response length — no explicit rating UI required.
- **Production awareness**: Rate limiting, auth, Prometheus metrics, and session management are first-class features.
- **Privacy by default**: PII is rejected before storage. Memory retention is configurable. Users can inspect and delete their own stored facts.

## 2. Architecture

### 2.1 Request Pipeline

Every request passes through the following stages in strict order:

| Stage | Component                        | What happens                                                                 | Metrics                  |
|-------|----------------------------------|------------------------------------------------------------------------------|--------------------------|
| 1     | `fuse_modalities()`              | Whisper transcript + BLIP-2 caption embedded and attention-weighted against the text query. Most relevant modality appears closest to the query. | — |
| 2     | `_handle_memory_command()`       | Intercepts "remember X", "forget X", "what do you know about me?" before the main pipeline. Short-circuits immediately. | — |
| 3     | `detect_emotions()`              | RoBERTa emotion classifier (6 classes). Results inform upstream prompt, tone bandit, and empathy rewrite. | LATENCY_EMPATHY |
| 4     | `is_harmful()` / `has_pii()`     | Toxic-BERT + DistilBERT PII. Hard gate — harmful or PII content is rejected immediately. | — |
| 5     | `cache_lookup()`                 | 64-dim compressed cosine similarity against Redis semantic cache (threshold 0.92). Cache hit skips LLM and tools. | CACHE_HITS |
| 6     | `get_context()`                  | Three-tier retrieval: key-facts + HNSW episodic search + recency fallback. Injected into system prompt. | — |
| 7     | `react_tool_loop()`              | MLP ToolPolicy gate first (threshold 0.35). Then ReAct loop (OpenAI) or parallel execution. | LATENCY_TOOLS |
| 8     | `_generate_base()`               | Emotions injected upstream into system prompt. Uses LangChain/GPT-4o-mini or Phi-3.5 direct/vLLM. Includes last 6 turns. | LATENCY_LLM |
| 9     | `ToneBandit.choose()`            | Thompson Sampling selects tone (empathetic / witty / direct). Witty suppressed on negative emotions. Empathetic path calls rewriter. | LATENCY_REWRITE |
| 10    | Post-processing                  | Stream response, then store memory doc, cache response, summarise session, extract key-facts (fire-and-forget). | LATENCY_TOTAL |

### 2.2 Component Map

| Component                    | Role |
|-----------------------------|------|
| **EmpatheticContextEngine** | Singleton loaded at startup. Owns embedder, emotion/harm/PII pipelines, autoencoder, Phi-3.5, Whisper, BLIP-2. |
| **LatentAutoencoder**       | 384→64-dim compression for Redis. Used for memory docs and semantic cache (~6× storage reduction). |
| **ToolPolicy (MLP)**        | Learned gate: [query ‖ context ‖ emotion] → P(use_tool). Weights saved to `aether_tool_policy.pt`. |
| **UserMemory**              | Per-user Redis interface. Manages all three memory tiers, semantic cache, fact storage/deletion, and privacy pruning. |
| **ConversationBuffer**      | In-process deque keyed by `conv_id`. Holds last 20 turns. Expires after `session_ttl`. |
| **ToneBandit**              | Per-user Thompson Sampling bandit (empathetic / witty / direct). Updated via explicit feedback or implicit signals. |
| **AetherEngine**            | Main orchestrator. Owns the `process()` generator, memory commands, coherence tracker, and bandit registry. |
| **RateLimiter**             | Redis INCR+EXPIRE (multi-worker safe) with in-process deque fallback. |

### 2.3 Memory Hierarchy

| Tier | Description |
|------|-------------|
| **Tier 1 — Session buffer** | `ConversationBuffer` (in-process). Last 20 turns of the active session. Sub-millisecond access. Expires after 30 min. |
| **Tier 2 — Session summary** | Generated async by Phi-3.5. Stored in Redis as narrative prose. TTL: ~7 days. |
| **Tier 3 — Key facts** | Typed structured facts (`preference` / `topic` / `name` / `event`) extracted by the LLM each turn. Indefinite TTL (or governed by `MEMORY_RETENTION_DAYS`). |

**Note**: `get_context()` assembles all tiers intelligently. Tier 3 facts appear first so they influence generation most strongly.

## 3. Quick Start

### 3.1 Prerequisites
- Python 3.10+
- Redis 7+ (strongly recommended)
- CUDA GPU with ≥8 GB VRAM (CPU fallback works but slower)
- OpenAI API key (optional — enables ReAct loop)
- Tavily API key (optional — web search)

### 3.2 Installation
```bash
git clone https://github.com/your-org/aether.git && cd aether
python -m venv .venv && source .venv/bin/activate

pip install fastapi uvicorn[standard] pydantic pydantic-settings httpx \
            torch transformers sentence-transformers peft trl datasets \
            sse-starlette numpy redis[asyncio] prometheus-client

# Optional components
pip install langchain langchain-openai          # ReAct loop
pip install openai-whisper                      # Audio
pip install salesforce-lavis                    # Image captioning
pip install unsloth                             # Fast fine-tuning
cp .env.example .env
```

### 3.3 Key Environment Variables
- `REDIS_URL`
- `OPENAI_API_KEY`
- `TAVILY_API_KEY`
- `MEMORY_RETENTION_DAYS` (default: 30, 0 = unlimited)
- `VLLM_BASE_URL`
- `API_KEY` (for authentication)
- `RATE_LIMIT_RPM` (default: 30)
- `USE_UNSLOTH`

### 3.4 Running the Server
```bash
uvicorn aether:app --host 0.0.0.0 --port 8000

# With multimodal
python aether.py --multimodal

# Fine-tune
python aether.py --finetune --dataset tatsu-lab/alpaca --epochs 1
```

Prometheus metrics are available at `http://localhost:8001/metrics`.

## 4. API Reference

### 4.1 POST /chat (Main Endpoint)
Returns Server-Sent Events (word-by-word streaming).

### 4.2 POST /feedback
Explicit tone bandit reward.

### 4.3 POST /memory
Direct memory commands: `remember <fact>`, `forget <topic>`, `list`.

### 4.4 In-Chat Memory Commands
- `remember <fact>`
- `forget <topic>`
- `what do you know about me?`

## 5. Observability

Prometheus metrics exposed on port 8001:
- `aether_requests_total`
- `aether_latency_*` (per stage)
- `aether_semantic_cache_hits_total`
- `aether_turn_coherence_score`
- `aether_bandit_implicit_reward`
- `aether_memory_facts_stored_total`

## 6. Security & Privacy
- Optional `X-API-Key` authentication
- Redis-backed rate limiting
- PII and harm detection before any processing/storage
- Configurable memory retention + user-controlled deletion

## 7. Deployment
- Single worker (dev)
- Multi-worker with Gunicorn + Redis (production)
- **Recommended**: Use `vLLM_BASE_URL` for high throughput and shared model serving

## 8. Fine-Tuning
Supports QLoRA with Unsloth (fast) or standard TRL. LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`.

```bash
python aether.py --finetune --dataset tatsu-lab/alpaca --epochs 1
python aether.py --load-finetuned
```

## 9. Suggested Extensions
(See original document sections 9.1–9.4 for detailed prioritized list including Streaming Fact Extraction, Emotion Trend Tracking, Persona Profiles, Knowledge Graph Memory, Multi-Agent Orchestration, etc.)

## 10. Changelog

**v1.2 (current)** — Major release with hierarchical memory, upstream emotion injection, ReAct tool loop, compressed semantic cache, privacy slider, user memory commands, and enhanced observability.

**v1.1** — LatentAutoencoder, semantic cache, Phi-3.5 default, implicit feedback, vLLM/Unsloth support, etc.

**v1.0** — Initial release.

---

**Aether v1.2** • MIT License • `https://github.com/Satoshi88818/Aether-Adaptive-Empathic-Thinking-Hearing-Experiential-Reasoning`
```



