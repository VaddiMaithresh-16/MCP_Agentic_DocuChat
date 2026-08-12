# Agentic RAG DocuChat

Agentic RAG DocuChat lets you ask questions about PDF documents using retrieval-augmented generation. The Gradio app supports llama.cpp, Ollama, and Gemini, remembers useful personal facts automatically, caches repeated answers, and can create a protected public share link.

Everything lives in `rag_backend.py` (logic) and `gradio_app.py` (UI). The original standalone CLI (`main.py`) has been removed — it duplicated logic that now lives in `rag_backend.py` and wasn't used by the Gradio app.

## Features

- Upload, add, or remove one or more PDFs — multiple documents are searched together, not one at a time
- Process or rebuild the document index
- Choose between llama.cpp (local, default), Gemini, and Ollama for each question
- llama.cpp default model: `Qwen3-4B-GGUF:Q4_0`, served locally by `llama-server`
- Ollama default model: `gemma3:4b`
- Both llama.cpp and Ollama use direct grounded generation by default (faster, streams token by token); they automatically switch to the same MCP tool-calling agent Gemini uses when Composio tools are configured, and stream that agent's answer tokens too when the model supports it
- MCP tools via Composio — configurable toolkits (search, calendar, email, docs, etc.) with no code change, reloadable from the UI without restarting
- Switchable embeddings, three ways: local (HuggingFace, private, default), online (Google API), or llamacpp (your own llama-server's `/v1/embeddings`, fully local with no HuggingFace download) — toggle in the UI or via `EMBEDDING_PROVIDER`, rebuilds the index automatically
- Inline citations: every provider is prompted to cite claims as `(filename.pdf, p. N)` right after the sentence, not just in a separate source list
- Automatic Ollama/llama.cpp readiness status
- Stable Chroma IDs, MMR retrieval, and index rebuilds are skipped when nothing changed
- Source filename and page display
- Persistent conversation history in SQLite
- Automatic capture of clear personal facts after responses
- Persistent, size- and age-bounded response cache
- Per-user rate limiting to prevent spam-clicking a local model mid-generation
- Retry with backoff on local-model network hiccups
- Copy last answer and export full chat transcript
- Light/dark theme toggle
- Clear conversation and memory controls
- Helpful / Not helpful feedback buttons
- Optional Gradio public sharing protected by username and password
- Simple, plain workspace UI
- Context engineering, provider-specific prompt engineering, and answer-repair loops

## Project layout

```text
MCP_Agentic_DocuChat-main/
├── rag_backend.py           # Provider-aware RAG backend
├── persistent_memory.py     # SQLite memory, cache, and feedback
├── ollama_utils.py          # Ollama and GPU health checks
├── llamacpp_utils.py        # llama.cpp (llama-server) health checks
├── prompt_engineering.py    # Context, prompt, and repair-loop templates
├── gradio_app.py            # Gradio interface and launch options
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   └── .gitkeep
└── tests/
    └── test_persistent_memory.py
```

Generated or private files are ignored:

- `.env` for API keys and local settings
- `data/uploads/` for uploaded PDFs
- `chroma_langchain_db/` for vectors
- `agentic_rag_memory.sqlite3` for conversations, memories, cache, and feedback

## Installation

```bash
python3 -m venv avenv
source avenv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv avenv
avenv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Configuration

Create a private environment file:

```bash
cp .env.example .env
```

At minimum, configure:

```env
GEMINI_API_KEY=your_gemini_api_key
DOCUMENT_PATH=./data/research_paper.pdf
```

The full `.env.example` also contains settings for embeddings, memory, Chroma, Ollama, Gradio sharing, and Composio (MCP) tools.

### Embeddings — local, online, or llama.cpp

```env
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_MODEL_ONLINE=models/text-embedding-004
LLAMACPP_EMBED_BASE_URL=http://localhost:8081/v1
LLAMACPP_EMBED_MODEL=nomic-embed-text
```

- `local` (default): a HuggingFace sentence-transformer runs on your machine — private, no API calls, no per-token cost. `BAAI/bge-base-en-v1.5` is the default (stronger retrieval than the previous `bge-small`, still small enough to run on CPU).
- `online`: uses Google's embedding API instead — no local model download, needs `GEMINI_API_KEY` (the same key used for the chat model).
- `llamacpp`: uses your own `llama-server`'s `/v1/embeddings` endpoint — fully local, no HuggingFace download either. An embedding model is usually a different GGUF than your chat model, so this typically means a **second** `llama-server` process on its own port, e.g.:
  ```bash
  llama-server -hf nomic-ai/nomic-embed-text-v1.5-GGUF --port 8081 --embedding
  ```
  `LLAMACPP_EMBED_BASE_URL` defaults to that second port; if you leave it unset it falls back to `LLAMACPP_BASE_URL` (only workable if that same server was also started with `--embedding` support).

Switch providers from the sidebar radio in the UI, or set `EMBEDDING_PROVIDER` in `.env` — either way the app rebuilds the index automatically, since vectors from different embedding models aren't comparable.

### Composio (MCP) tools

```env
COMPOSIO_API_KEY=your_composio_api_key
COMPOSIO_USER_ID=your_composio_user_id
COMPOSIO_TOOLKITS=TAVILY
```

The Gemini agent always has retrieval + any configured Composio tools. **llama.cpp and Ollama get MCP tools too**: as soon as `COMPOSIO_TOOLKITS` resolves to at least one tool, local providers automatically route through the same tool-calling agent instead of the direct-grounded fast path — no separate flag to flip. Every agent-based answer (Gemini always, local providers when MCP is on) still streams its own text tokens when the model supports it; it only falls back to one non-streaming update on turns where nothing streamable came through (e.g. a tool-call-only step). `COMPOSIO_TOOLKITS` is comma-separated, so you can add more than web search — e.g. `COMPOSIO_TOOLKITS=TAVILY,GOOGLECALENDAR,GMAIL` — without touching code, as long as those toolkits are connected on your Composio account. If Composio isn't configured, every provider just answers from the document with no tools.

The sidebar status panel shows an MCP tools line (on/off + tool count). Added a toolkit on your Composio account without restarting the app? Click **Reload MCP tools** in the sidebar to pick it up immediately.

### Gemini

Set `GEMINI_API_KEY` and choose a model with `GEMINI_MODEL`.

### llama.cpp (local, default)

Grab a binary release from the [llama.cpp releases page](https://github.com/ggml-org/llama.cpp/releases) (Windows: `llama-*-bin-win-avx2-x64.zip` for CPU, `win-cuda` for Nvidia GPU), unzip it, then run `llama-server` from that folder — or add the folder to PATH so `llama-server` works from anywhere:

```bash
llama-server -hf Qwen/Qwen3-4B-GGUF:Q4_0 --port 8080 -ngl 99 -c 4096
```

- `-ngl 99` offloads all layers to GPU if you have one. Drop it (or lower the number for partial offload) on CPU-only machines.
- `-c 4096` sets the server's context window.
- `llama-server` prints its own web UI at `http://localhost:8080` too — useful to sanity-check the model loaded before pointing this app at it.

This exposes an OpenAI-compatible API at `http://localhost:8080/v1`, which the app talks to directly (no extra client needed). The app uses:

```env
LLAMACPP_MODEL=Qwen3-4B-GGUF:Q4_0
LLAMACPP_BASE_URL=http://localhost:8080/v1
LLAMACPP_API_KEY=not-needed
LLAMACPP_NUM_CTX=4096
```

If the model name `llama-server` reports doesn't exactly match `LLAMACPP_MODEL` (e.g. you loaded a local `.gguf` path instead of an HF repo id), the app auto-adopts whatever the server reports — no need to keep the two in sync manually.

If you already have the GGUF file locally instead of pulling by repo, point `llama-server` at it with `-m /path/to/model.gguf --port 8080` instead of `-hf`.

### Ollama

Install Ollama, start it, and download the default model:

```bash
ollama serve
ollama pull gemma3:4b
```

The application uses:

```env
OLLAMA_MODEL=gemma3:4b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_KEEP_ALIVE=10m
OLLAMA_NUM_CTX=4096
```

Ollama normally chooses the best available hardware itself. The app checks for NVIDIA GPUs with `nvidia-smi`, Apple GPUs on macOS, and the Ollama local API. It reports whether Ollama is running and whether GPU hardware was detected. If no GPU is available, Ollama runs on the CPU.

## Run Gradio

```bash
python gradio_app.py
```

Open the local URL shown in the terminal. Choose `Llama.cpp`, `Gemini`, or `Ollama` before asking a question. The workspace is a simple two-column layout: setup, document, and status on the left; chat on the right.

### Upload documents — one or several

Use **Upload PDF(s) → Process document(s)**. Files are copied to `data/uploads/` and indexed. The **On process** toggle controls what happens:
- **Add to existing** (default): the new file(s) join whatever's already loaded — questions are answered against all of them together.
- **Replace all**: the first uploaded file replaces the entire loaded set (matches the old single-document behavior).

The **Loaded documents** dropdown lists every currently-indexed file; **Remove selected document** drops one from the set and rebuilds the index (you can't remove the last remaining document — load a replacement instead). Processing documents does not delete your SQLite memories.

Every chunk still carries its own filename in its metadata, so the "Sources" panel and inline citations correctly say which specific PDF a claim came from, even with several loaded at once.

### Automatic memory

You do not need to name or manually save memories. After each response, the app checks the user’s question for clear personal statements, for example:

```text
I prefer concise explanations.
I am working on a finance project.
I use Ollama for private questions.
```

Useful facts are saved automatically for the same `User ID`. Conversation turns are also persisted. The app intentionally saves short explicit facts rather than blindly storing every answer, which keeps long-term memory useful and private.

### Public sharing with login

Keep sharing disabled for private documents:

```env
GRADIO_SHARE=false
```

To create a temporary public Gradio URL, set both credentials:

```env
GRADIO_SHARE=true
GRADIO_USERNAME=myuser
GRADIO_PASSWORD=use-a-strong-password
```

Then run:

```bash
python gradio_app.py
```

The app refuses to create a public link when sharing is enabled without both a username and password. You can also override the setting for one run:

```bash
python gradio_app.py --share
python gradio_app.py --no-share
```

The share URL is temporary. Anyone who has the URL can attempt to access the app, so use a strong password and avoid sharing sensitive documents.

## How the upgraded backend works

```text
Question + selected provider
          ↓
Persistent response cache (keyed on durable memory, not on the
constantly-changing recent conversation — so repeated questions
in an ongoing session can actually hit cache)
          ↓
Context engineering: durable memory + recent conversation, kept in
separate labelled sections, for EVERY provider
          ↓
Single Chroma MMR retrieval pass over ALL loaded documents — reused
for generation context AND for the sources shown in the UI
          ↓
Prompt engineering: tool-calling agent (Gemini always; llama.cpp/
Ollama too when MCP tools are configured) or direct grounded
response — both ask for inline (filename.pdf, p. N) citations
          ↓
Streaming: real tokens on the direct-grounded path; agent answers
stream their own text tokens when the model supports it, otherwise
one non-streaming update
          ↓
Loop engineering: grounding check → repair pass, on every provider
          ↓
Sources, feedback, and memory update
```

### Engineering guardrails

- **Context engineering:** every provider receives labelled task, memory, conversation, and retrieved-document sections with size limits, built from the same `build_context_package`. Gemini previously got no pre-fetched document context at all (it relied entirely on the agent choosing to call the retrieval tool) — it now gets the same retrieved passages every provider does, while keeping the tool available for adaptive re-querying.
- **Prompt engineering:** agent-based providers (Gemini always; llama.cpp/Ollama when MCP tools are configured) use a tool-aware system prompt; the direct-grounded path uses a grounded local prompt that treats PDF text as evidence rather than instructions. Every prompt asks for the same inline citation format — `(filename.pdf, p. N)` immediately after the claim — so citations are consistent across providers and multiple documents.
- **Loop engineering:** every provider checks whether the answer looks grounded (empty/too short, or missing a page citation when evidence existed) and runs a repair pass if not, via the shared `needs_grounding_repair` check, which recognizes both the word "page" and the `p. N` inline format. The limit is controlled by `ANSWER_LOOP_MAX_ATTEMPTS`.
- **One retrieval → generate → cache → memory pipeline**, not two: `ask()` (non-streaming) and `astream_answer()` (streaming) both delegate to the same internal generator (`_astream_answer_impl`) — `ask()` just consumes it to the end. There's no separate non-streaming code path to keep in sync with the streaming one.
- **MCP tool access is provider-agnostic:** `_use_agent(provider)` decides per-question whether a provider needs the tool-calling agent — Gemini always does, local providers do only when `_get_tools()` returns at least one Composio tool. Composio's client + `tools.get()` call is fetched once and cached (`_get_tools`) and shared across every provider's agent, instead of being rebuilt per agent. `refresh_tools()` clears that cache (and any agents built on it) so a newly-connected toolkit doesn't need a restart.
- **Agent streaming:** `_agent_stream_text` watches the LangGraph agent's `astream_events` for `on_chat_model_stream` events and yields only the model's own text tokens — tool-call and tool-result events (retrieval, MCP calls) pass by unshown, so the UI never displays raw tool chatter. If a turn produces no streamable text (some tool-only turns, or a backend that doesn't support event streaming), it transparently falls back to one `ainvoke` call.
- **Multi-document indexing:** each chunk keeps the source path PyPDFLoader stamped on its parent document, so `source_file` is correct per-chunk even with several PDFs loaded — this was previously hardcoded to a single `self.document_path` and would have mislabeled multi-doc chunks. `add_document`/`remove_document`/`load_document` all key the Chroma collection off a fingerprint of the *whole* document set (paths + sizes + mtimes + embedding config), so a different document set never shares a collection with another one.

### Performance

- **Retrieval runs once per question**, not twice, across every loaded document. The MMR search results are reused for both the generation context and the "Sources used" panel.
- **Embedding, chat, agent, and MCP-tool objects are built once per config and cached** (`_get_embeddings`, `_get_model`, `_get_agent`, `_get_tools`) — uploading or rebuilding a document no longer reloads the embedding model into memory, switching providers doesn't reconstruct a client that's already warm, and Composio isn't re-queried per question or per agent.
- **Ollama's health/GPU probe is cached** (`OLLAMA_STATUS_CACHE_SECONDS`, default 15s) instead of spawning `nvidia-smi` and hitting the Ollama HTTP API on every single message. The Gradio "Refresh status" button always forces a fresh probe.
- **The Gradio UI uses native async handlers and `demo.queue()`** instead of calling `asyncio.run()` per click, and offloads blocking calls (PDF upload/add/remove, index rebuild, embedding-provider switch, MCP reload) to a thread via `asyncio.to_thread` so they don't freeze the UI. Every provider streams through the same `astream_answer` call now — direct-grounded providers stream real tokens, agent-based providers stream their own text tokens when possible and fall back to one update otherwise.
- **Local-provider calls retry with exponential backoff** (`_invoke_with_retry`, 2 retries by default) — a server that's still warming up or a dropped connection doesn't immediately surface as an error.
- **The response cache is bounded**, not unbounded SQLite growth: entries expire after `CACHE_TTL_SECONDS` (default 7 days) and the table is pruned to `CACHE_MAX_ENTRIES` (default 2000, oldest first) on every write.
- **llama.cpp auto-detects the loaded chat model** from `llama-server`'s `/v1/models` response, so `LLAMACPP_MODEL` doesn't have to exactly match what the server reports (e.g. a local `.gguf` path vs an HF repo id).
- **`_refresh_index` skips rebuilding** when the document set, embedding provider, and embedding model are all unchanged from the last build — a plain restart (or an accidental double-call) doesn't re-read and re-embed the PDFs.
- **Per-user rate limiting** (`RATE_LIMIT_SECONDS`, default 1.5s) rejects a second question from the same `user_id` before the cooldown passes, so spam-clicking "Ask" or double-submitting doesn't queue up redundant local-model generations.
- **Config is validated at startup** (`_validate_config`), not discovered mid-conversation — an invalid `EMBEDDING_PROVIDER`, a missing `GEMINI_API_KEY` when online embeddings are selected, or `RETRIEVAL_FETCH_K < RETRIEVAL_K` all raise a clear error immediately.
- **Structured logging** replaces stray `print()` calls — `LOG_LEVEL` controls verbosity (`INFO` by default), and index builds, embedding loads, retries, and Composio tool loading all log through the standard `logging` module.

The Chroma collection name includes a fingerprint of the active document set, embedding provider, and embedding model. This prevents vectors from different PDFs, providers, or embedding dimensions being mixed together, and previously-embedded chunks are skipped on restart (stable content-hash IDs), so reopening the app doesn't re-embed documents that haven't changed.

## Testing

Run the local tests without API keys or model servers:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

These cover `persistent_memory.py` (memory, cache TTL/eviction), `prompt_engineering.py` (context building, grounding checks, the inline-citation prompt format), and `ollama_utils.py`/`llamacpp_utils.py` (health-check shape). They deliberately don't import `rag_backend.py` directly, because doing so pulls in the full langchain/langgraph/chroma/sentence-transformers dependency chain at import time — the same reason a syntax check below is run separately rather than as a test. Backend-level behavior that's new in this pass (multi-document rebuild, rate limiting, the MCP-triggered agent switch, embedding-provider swapping) is compiled and manually reviewed rather than covered by an automated test; if you want that coverage, installing the full `requirements.txt` in your test environment and writing `RagBackend`-level tests against a small sample PDF is the natural next step.

Run a syntax check:

```bash
python -m py_compile rag_backend.py persistent_memory.py ollama_utils.py llamacpp_utils.py prompt_engineering.py gradio_app.py
```

A full end-to-end test requires installed dependencies, a valid `.env`, an accessible PDF, and at least one working provider (Gemini key, a running Ollama server, or a running `llama-server`).

## Troubleshooting

### Gemini does not work

Check `GEMINI_API_KEY` and `GEMINI_MODEL`. You can switch the UI to Ollama while troubleshooting.

### llama.cpp does not work

Confirm `llama-server` is running and reachable at `LLAMACPP_BASE_URL` (default `http://localhost:8080/v1`):

```bash
curl http://localhost:8080/v1/models
```

If that fails, start the server with the model you downloaded, e.g. `llama-server -hf Qwen/Qwen3-4B-GGUF:Q4_0 --port 8080`.

### Ollama does not work

Check that Ollama is running:

```bash
ollama list
```

Then download the default model if needed:

```bash
ollama pull gemma3:4b
```

Confirm that `OLLAMA_BASE_URL` points to the correct server. If the app reports that Ollama is offline, start it with `ollama serve`. If it reports that the model is missing, pull the model shown in the message. Small Ollama models use direct PDF context rather than tool calling for more reliable answers.

### The uploaded PDF fails

Confirm it is a readable PDF and that the file upload completed before pressing **Process document**.

### Clear all local memory

Stop the app and remove the SQLite file:

```bash
rm agentic_rag_memory.sqlite3
```

This removes conversations, automatic memories, cache entries, and feedback. It does not remove Chroma vectors.

## Security

- Never commit `.env` or real API keys.
- Keep sharing disabled for private documents.
- Use a strong Gradio password when sharing is enabled.
- Uploaded PDFs, memories, conversations, and feedback are local application data.
- Rotate any credential that is accidentally exposed.

## License

This project is intended for educational, research, and learning purposes.
