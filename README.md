# Agentic RAG DocuChat

Agentic RAG DocuChat lets you ask questions about PDF documents using retrieval-augmented generation. The upgraded Gradio app supports Gemini and local Ollama, remembers useful personal facts automatically, caches repeated answers, and can create a protected public share link.

`main.py` remains the original command-line application. The new experience is provided by `rag_backend.py` and `gradio_app.py`.

## Features

- Upload a PDF directly in Gradio
- Process or rebuild the document index
- Choose between llama.cpp (local, default), Gemini, and Ollama for each question
- llama.cpp default model: `Qwen3-4B-GGUF:Q4_0`, served locally by `llama-server`
- Ollama default model: `gemma3:4b`
- Both llama.cpp and Ollama use direct grounded generation by default (faster, streams token by token); they automatically switch to the same MCP tool-calling agent Gemini uses when Composio tools are configured
- Switchable embeddings: local (HuggingFace, private, default) or online (Google API) — toggle in the UI or via `EMBEDDING_PROVIDER`, rebuilds the index automatically
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
- MCP tools via Composio — web search by default, configurable to add more toolkits (calendar, email, docs, etc.) with no code change
- Simple, plain workspace UI
- Context engineering, provider-specific prompt engineering, and answer-repair loops

## Project layout

```text
MCP_Agentic_DocuChat/
├── main.py                  # Original CLI application
├── rag_backend.py           # Provider-aware RAG backend
├── persistent_memory.py     # SQLite memory, cache, and feedback
├── ollama_utils.py          # Ollama and GPU health checks
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
python -m pip install --upgrade pip
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

### Embeddings — local or online

```env
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_MODEL_ONLINE=models/text-embedding-004
```

- `local` (default): a HuggingFace sentence-transformer runs on your machine — private, no API calls, no per-token cost. `BAAI/bge-base-en-v1.5` is the default (stronger retrieval than the previous `bge-small`, still small enough to run on CPU).
- `online`: uses Google's embedding API instead — no local model download, needs `GEMINI_API_KEY` (the same key used for the chat model).

Switch providers from the sidebar dropdown in the UI, or set `EMBEDDING_PROVIDER` in `.env` — either way the app rebuilds the index automatically, since vectors from different embedding models aren't comparable.

### Composio (MCP) tools

```env
COMPOSIO_API_KEY=your_composio_api_key
COMPOSIO_USER_ID=your_composio_user_id
COMPOSIO_TOOLKITS=TAVILY
```

The Gemini agent always has retrieval + any configured Composio tools. **llama.cpp and Ollama get MCP tools too**: as soon as `COMPOSIO_TOOLKITS` resolves to at least one tool, local providers automatically route through the same tool-calling agent instead of the direct-grounded fast path — no separate flag to flip. Streaming is only available on the direct-grounded path, so a local answer becomes a single non-streaming response once MCP tools are active (the same trade-off Gemini already has). `COMPOSIO_TOOLKITS` is comma-separated, so you can add more than web search — e.g. `COMPOSIO_TOOLKITS=TAVILY,GOOGLECALENDAR,GMAIL` — without touching code, as long as those toolkits are connected on your Composio account. If Composio isn't configured, every provider just answers from the document with no tools, and local providers keep streaming.

The sidebar status panel shows an MCP tools line (on/off + tool count) so it's obvious which mode you're in.

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

### Upload a document

Use **Upload PDF → Process document**. The file is copied to `data/uploads/`, indexed, and becomes the active document for the running app. Processing a new document does not delete your SQLite memories.

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

## Run the original CLI

```bash
python main.py
python main.py --query "Summarize the main contributions of the paper."
```

The original `main.py` is intentionally preserved. Its in-memory LangGraph checkpoint and original tool flow remain available separately from the upgraded Gradio backend.

## How the upgraded backend works

```text
Question + selected provider
          ↓
Persistent response cache (keyed on durable memory, not on the
constantly-changing recent conversation — so repeated questions
in an ongoing session can actually hit cache)
          ↓
Context engineering: durable memory + recent conversation, kept in
separate labelled sections, for BOTH providers
          ↓
Single Chroma MMR retrieval pass — reused for generation context
AND for the sources shown in the UI (previously run twice)
          ↓
Prompt engineering: Gemini agent (pre-fetched context + tools for
adaptive re-querying) or direct grounded Ollama response
          ↓
Loop engineering: grounding check → repair pass, on BOTH providers
          ↓
Sources, feedback, and memory update
```

### Engineering guardrails

- **Context engineering:** every provider receives labelled task, memory, conversation, and retrieved-document sections with size limits, built from the same `build_context_package`. Gemini previously got no pre-fetched document context at all (it relied entirely on the agent choosing to call the retrieval tool) — it now gets the same retrieved passages Ollama does, while keeping the tool available for adaptive re-querying.
- **Prompt engineering:** agent-based providers (Gemini always; llama.cpp/Ollama when MCP tools are configured) use a tool-aware system prompt; the direct-grounded path uses a grounded local prompt that treats PDF text as evidence rather than instructions.
- **Loop engineering:** every provider checks whether the answer looks grounded (empty/too short, or missing a page citation when evidence existed) and runs a repair pass if not, via the shared `needs_grounding_repair` check. The limit is controlled by `ANSWER_LOOP_MAX_ATTEMPTS`.
- **MCP tool access is provider-agnostic:** `_use_agent(provider)` decides per-question whether a provider needs the tool-calling agent — Gemini always does, local providers do only when `_get_tools()` returns at least one Composio tool. Composio's client + `tools.get()` call is fetched once and cached (`_get_tools`) and shared across every provider's agent, instead of being rebuilt per agent.

### Performance

- **Retrieval runs once per question**, not twice. The MMR search results are reused for both the generation context and the "Sources used" panel.
- **Embedding, chat, agent, and MCP-tool objects are built once per config and cached** (`_get_embeddings`, `_get_model`, `_get_agent`, `_get_tools`) — uploading or rebuilding a document no longer reloads the embedding model into memory, switching providers doesn't reconstruct a client that's already warm, and Composio isn't re-queried per question or per agent.
- **Ollama's health/GPU probe is cached** (`OLLAMA_STATUS_CACHE_SECONDS`, default 15s) instead of spawning `nvidia-smi` and hitting the Ollama HTTP API on every single message. The Gradio "Refresh system status" button always forces a fresh probe.
- **The Gradio UI uses native async handlers and `demo.queue()`** instead of calling `asyncio.run()` per click, and offloads blocking calls (PDF upload, index rebuild, embedding-provider switch) to a thread via `asyncio.to_thread` so they don't freeze the UI. Local providers stream tokens live via `astream_answer` when the direct-grounded path is used; any provider running through the tool-calling agent (Gemini, or a local provider with MCP tools on) shows a progressive "Thinking…" state instead, since agents don't stream cleanly through this path.
- **Local-provider calls retry with exponential backoff** (`_invoke_with_retry`, 2 retries by default) — a server that's still warming up or a dropped connection doesn't immediately surface as an error.
- **The response cache is bounded**, not unbounded SQLite growth: entries expire after `CACHE_TTL_SECONDS` (default 7 days) and the table is pruned to `CACHE_MAX_ENTRIES` (default 2000, oldest first) on every write.
- **llama.cpp auto-detects the loaded model** from `llama-server`'s `/v1/models` response, so `LLAMACPP_MODEL` doesn't have to exactly match what the server reports (e.g. a local `.gguf` path vs an HF repo id).
- **`_refresh_index` skips rebuilding** when the document, embedding provider, and embedding model are all unchanged from the last build — a plain restart (or an accidental double-call) doesn't re-read and re-embed the PDF.
- **Per-user rate limiting** (`RATE_LIMIT_SECONDS`, default 1.5s) rejects a second question from the same `user_id` before the cooldown passes, so spam-clicking "Ask" or double-submitting doesn't queue up redundant local-model generations.
- **Config is validated at startup** (`_validate_config`), not discovered mid-conversation — an invalid `EMBEDDING_PROVIDER`, a missing `GEMINI_API_KEY` when online embeddings are selected, or `RETRIEVAL_FETCH_K < RETRIEVAL_K` all raise a clear error immediately.
- **Structured logging** replaces stray `print()` calls — `LOG_LEVEL` controls verbosity (`INFO` by default), and index builds, embedding loads, retries, and Composio tool loading all log through the standard `logging` module.

The Chroma collection name includes a fingerprint of the active document, embedding provider, and embedding model. This prevents vectors from different PDFs, providers, or embedding dimensions being mixed together, and previously-embedded chunks are skipped on restart (stable content-hash IDs), so reopening the app doesn't re-embed the whole document.

## Testing

Run the local test without API keys:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Run a syntax check:

```bash
python -m py_compile main.py rag_backend.py persistent_memory.py ollama_utils.py gradio_app.py
```

A full end-to-end test requires installed dependencies, a valid `.env`, an accessible PDF, and either a working Gemini key or a running Ollama server with the selected model.

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
