"""Reusable agentic RAG backend for the CLI and Gradio interface.

Performance notes (see README "Engineering guardrails" for the full picture):
  - Retrieval runs at most once per question (context text and displayed
    sources are derived from the same MMR search, not two).
  - The embedding model and chat model/agent objects are built once per
    config and reused, instead of being reconstructed on every call.
  - Ollama's health/GPU probe is cached with a short TTL instead of running
    on every single message.
  - The response cache key is based on durable user memory, not the
    constantly-changing recent-conversation text, so repeated questions in
    an ongoing session can actually hit cache.
  - Rebuilding the index for the same document+embedding-model combo is a
    no-op (`_refresh_index` short-circuits on an unchanged document_version).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver

from llamacpp_utils import llamacpp_status
from ollama_utils import ollama_status
from persistent_memory import PersistentMemory, ResponseCache
from prompt_engineering import (
    agent_system_prompt,
    build_context_package,
    needs_grounding_repair,
    ollama_prompt,
    repair_prompt,
)

LOCAL_PROVIDERS = {"ollama", "llamacpp"}
EMBEDDING_PROVIDERS = {"local", "online"}

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("docuchat")


async def _invoke_with_retry(model, payload, retries: int = 2, base_delay: float = 0.6):
    """Call model.ainvoke with exponential-backoff retries.

    Network hiccups (server still loading the model, a dropped connection)
    are common with local model servers on first use — retry a couple of
    times before surfacing the error to the user.
    """
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await model.ainvoke(payload)
        except Exception as error:  # noqa: BLE001 - re-raised after retries
            last_error = error
            if attempt == retries:
                break
            logger.warning("Model call failed (attempt %s/%s): %s", attempt + 1, retries + 1, error)
            await asyncio.sleep(base_delay * (2 ** attempt))
    raise last_error


class RagBackend:
    """Manage document indexing, provider selection, memory, and response caching."""

    def __init__(self, document_path: str | None = None) -> None:
        self.document_path = document_path or os.getenv("DOCUMENT_PATH", "./data/research_paper.pdf")

        # Embeddings: "local" runs a HuggingFace sentence-transformer on
        # this machine (private, no API calls); "online" calls Google's
        # embedding API (needs GEMINI_API_KEY, no local model download).
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5").strip()
        self.embedding_model_online = os.getenv("EMBEDDING_MODEL_ONLINE", "models/text-embedding-004").strip()

        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
        self.ollama_model = os.getenv("OLLAMA_MODEL", "gemma3:4b").strip()
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
        self.ollama_keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "10m").strip()
        self.ollama_num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
        self.ollama_status_ttl = float(os.getenv("OLLAMA_STATUS_CACHE_SECONDS", "15"))

        # llama.cpp: a local `llama-server` process exposing an OpenAI-
        # compatible API (see llamacpp_utils.py). The model is loaded by
        # llama-server itself, so LLAMACPP_MODEL only needs to match what
        # the server reports (or can be left as a display label).
        self.llamacpp_model = os.getenv("LLAMACPP_MODEL", "Qwen3-4B-GGUF:Q4_0").strip()
        self.llamacpp_base_url = os.getenv("LLAMACPP_BASE_URL", "http://localhost:8080/v1").strip()
        self.llamacpp_api_key = os.getenv("LLAMACPP_API_KEY", "not-needed").strip() or "not-needed"
        self.llamacpp_num_ctx = int(os.getenv("LLAMACPP_NUM_CTX", "4096"))
        self.llamacpp_status_ttl = float(os.getenv("LLAMACPP_STATUS_CACHE_SECONDS", "15"))

        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.hf_token = os.getenv("HF_TOKEN", "").strip() or None

        # Composio (MCP): which toolkits get pulled in for the Gemini agent.
        # Comma-separated so extra tools (search, docs, calendar, whatever
        # your Composio account has) can be added without a code change.
        self.composio_api_key = os.getenv("COMPOSIO_API_KEY")
        self.composio_user_id = os.getenv("COMPOSIO_USER_ID")
        self.composio_toolkits = [
            toolkit.strip().upper()
            for toolkit in os.getenv("COMPOSIO_TOOLKITS", "TAVILY").split(",")
            if toolkit.strip()
        ]

        self.database_path = os.getenv("MEMORY_DATABASE", "./agentic_rag_memory.sqlite3")
        self.chroma_dir = os.getenv("CHROMA_DIR", "./chroma_langchain_db")
        self.retrieval_k = int(os.getenv("RETRIEVAL_K", "4"))
        self.retrieval_fetch_k = int(os.getenv("RETRIEVAL_FETCH_K", "12"))
        self.rate_limit_seconds = float(os.getenv("RATE_LIMIT_SECONDS", "1.5"))

        self._validate_config()

        self.memory = PersistentMemory(self.database_path)
        self.cache = ResponseCache(
            self.memory,
            ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", str(60 * 60 * 24 * 7))),
            max_entries=int(os.getenv("CACHE_MAX_ENTRIES", "2000")),
        )

        # Reusable object caches — built once per config, not once per call.
        self._embedding_models: dict[str, Any] = {}
        self._models: dict[str, Any] = {}
        self._agents: dict[str, Any] = {}
        self._tools_cache: list[Any] | None = None
        self._ollama_status_cache: dict[str, Any] | None = None
        self._ollama_status_time: float = 0.0
        self._llamacpp_status_cache: dict[str, Any] | None = None
        self._llamacpp_status_time: float = 0.0
        self._last_request_time: dict[str, float] = {}
        self.document_version = ""

        self._refresh_index()

    # -------------------------------------------------------------- boot

    def _validate_config(self) -> None:
        """Fail fast and clearly at startup instead of mid-conversation."""
        if self.embedding_provider not in EMBEDDING_PROVIDERS:
            raise RuntimeError(
                f"EMBEDDING_PROVIDER must be 'local' or 'online', got '{self.embedding_provider}'."
            )
        if self.embedding_provider == "online" and not self.gemini_api_key:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=online requires GEMINI_API_KEY (used for the embedding API too)."
            )
        if self.retrieval_k <= 0 or self.retrieval_fetch_k <= 0:
            raise RuntimeError("RETRIEVAL_K and RETRIEVAL_FETCH_K must be positive.")
        if self.retrieval_fetch_k < self.retrieval_k:
            raise RuntimeError("RETRIEVAL_FETCH_K must be >= RETRIEVAL_K.")
        logger.info(
            "Config OK — embeddings=%s (%s), document=%s",
            self.embedding_provider,
            self.embedding_model_online if self.embedding_provider == "online" else self.embedding_model_name,
            self.document_path,
        )

    def _check_rate_limit(self, user_id: str) -> None:
        now = time.monotonic()
        last = self._last_request_time.get(user_id)
        if last is not None and (now - last) < self.rate_limit_seconds:
            wait = round(self.rate_limit_seconds - (now - last), 1)
            raise RuntimeError(f"You're asking a bit fast — wait {wait}s and try again.")
        self._last_request_time[user_id] = now

    # ---------------------------------------------------------------- index

    def _document_version(self, force_nonce: str = "") -> str:
        embedding_key = (
            f"online:{self.embedding_model_online}"
            if self.embedding_provider == "online"
            else f"local:{self.embedding_model_name}"
        )
        path = Path(self.document_path)
        if path.exists():
            return f"{path.resolve()}:{path.stat().st_mtime_ns}:{path.stat().st_size}:{embedding_key}:{force_nonce}"
        return f"{self.document_path}:{embedding_key}:{force_nonce}"

    def _refresh_index(self, force_nonce: str = "") -> None:
        new_version = self._document_version(force_nonce)
        if new_version == self.document_version and getattr(self, "vector_store", None) is not None:
            # Same document, same embedding config, no forced rebuild —
            # nothing changed since the last build, skip re-reading and
            # re-embedding the PDF entirely.
            logger.debug("Index unchanged, skipping rebuild for %s", self.document_path)
            return
        self.document_version = new_version
        base_collection = os.getenv("CHROMA_COLLECTION", "agentic_rag_research_collection")
        collection_suffix = hashlib.sha1(self.document_version.encode()).hexdigest()[:10]
        self.collection_name = f"{base_collection}_{collection_suffix}"
        logger.info("Building index for %s (collection %s)", self.document_path, self.collection_name)
        self.documents, self.vector_store = self._build_index()

    def _load_documents(self):
        try:
            documents = PyPDFLoader(self.document_path).load()
        except FileNotFoundError as error:
            raise RuntimeError(f"DOCUMENT_PATH not found: {self.document_path}") from error
        except requests.exceptions.RequestException as error:
            raise RuntimeError(f"Could not fetch DOCUMENT_PATH: {error}") from error
        except Exception as error:
            raise RuntimeError(f"Could not load PDF: {error}") from error
        if not documents:
            raise RuntimeError("The PDF contains no pages.")
        return documents

    def _get_embeddings(self):
        """Embedding model objects are expensive to build (model load into
        memory, or an API client) and don't depend on which document is
        loaded — cache by provider+model so switching/reloading a document
        doesn't reload it."""
        if self.embedding_provider == "online":
            key = f"online:{self.embedding_model_online}"
            if key not in self._embedding_models:
                try:
                    from langchain_google_genai import GoogleGenerativeAIEmbeddings
                except ImportError as error:
                    raise RuntimeError(
                        "Install langchain-google-genai to use EMBEDDING_PROVIDER=online."
                    ) from error
                logger.info("Loading online embedding model %s", self.embedding_model_online)
                self._embedding_models[key] = GoogleGenerativeAIEmbeddings(
                    model=self.embedding_model_online,
                    google_api_key=self.gemini_api_key,
                )
            return self._embedding_models[key]

        key = f"local:{self.embedding_model_name}"
        if key not in self._embedding_models:
            embedding_kwargs: dict[str, Any] = {"model_name": self.embedding_model_name}
            if self.hf_token:
                embedding_kwargs["model_kwargs"] = {"token": self.hf_token}
            logger.info("Loading local embedding model %s", self.embedding_model_name)
            self._embedding_models[key] = HuggingFaceEmbeddings(**embedding_kwargs)
        return self._embedding_models[key]

    def _build_index(self):
        documents = self._load_documents()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", "700")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
            add_start_index=True,
        )
        chunks = splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["source_file"] = Path(self.document_path).name
            chunk.metadata["page_number"] = int(chunk.metadata.get("page", 0)) + 1
        embeddings = self._get_embeddings()
        store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=self.chroma_dir,
        )
        existing_ids = set(store.get().get("ids", []))
        new_chunks = []
        new_ids = []
        for chunk in chunks:
            stable_id = hashlib.sha256(
                f"{chunk.metadata['source_file']}|{chunk.metadata['page_number']}|{chunk.metadata.get('start_index', 0)}|{chunk.page_content}".encode()
            ).hexdigest()
            if stable_id not in existing_ids:
                new_chunks.append(chunk)
                new_ids.append(stable_id)
        if new_chunks:
            store.add_documents(new_chunks, ids=new_ids)
        return documents, store

    # ----------------------------------------------------------- retrieval

    def _retrieve(self, question: str, k: int | None = None, fetch_k: int | None = None):
        """Single retrieval pass — the returned context text, source labels,
        and raw docs are reused everywhere (generation, the agent's tool
        call, and the sources shown in the UI) instead of re-querying."""
        docs = self.vector_store.max_marginal_relevance_search(
            question, k=k or self.retrieval_k, fetch_k=fetch_k or self.retrieval_fetch_k
        )
        if not docs:
            return "No relevant passages were found in the PDF.", [], []
        context_text = "\n\n".join(
            f"Source: {doc.metadata.get('source_file', 'PDF')}, page {doc.metadata.get('page_number', '?')}\n{doc.page_content}"
            for doc in docs
        )
        sources: list[str] = []
        for doc in docs:
            label = f"{doc.metadata.get('source_file', 'PDF')} — page {doc.metadata.get('page_number', '?')}"
            if label not in sources:
                sources.append(label)
        return context_text, sources, docs

    def _retrieval_tool(self):
        @tool
        def retrieve_from_pdf(query: str) -> str:
            """Retrieve relevant passages from the currently loaded PDF."""
            context_text, _sources, _docs = self._retrieve(query)
            return context_text

        return retrieve_from_pdf

    def _web_tools(self) -> list[Any]:
        """Fetch Composio (MCP) tools. Toolkits are configurable via
        COMPOSIO_TOOLKITS (comma-separated) so more than web search can be
        wired in without touching code — e.g.
        `COMPOSIO_TOOLKITS=TAVILY,GOOGLECALENDAR,GMAIL`."""
        if not (self.composio_api_key and self.composio_user_id and self.composio_toolkits):
            return []
        try:
            from composio import Composio
            from composio_langchain import LangchainProvider

            composio = Composio(api_key=self.composio_api_key, provider=LangchainProvider())
            tools = composio.tools.get(user_id=self.composio_user_id, toolkits=self.composio_toolkits)
            logger.info("Loaded %s Composio tool(s) from toolkits: %s", len(tools), ", ".join(self.composio_toolkits))
            return tools
        except Exception as error:
            logger.warning("Composio tools disabled (%s): %s", ", ".join(self.composio_toolkits), error)
            return []

    def _get_tools(self) -> list[Any]:
        """Composio client construction + a tools.get() network call is
        wasteful to repeat per agent — fetch once and share across every
        provider's agent (Gemini, and local providers when MCP is on)."""
        if self._tools_cache is None:
            self._tools_cache = self._web_tools()
        return self._tools_cache

    def _use_agent(self, provider: str) -> bool:
        """Gemini always gets the tool-calling agent (retrieval tool for
        adaptive re-querying, plus MCP tools if configured). Local
        providers (llama.cpp, Ollama) only go through the agent when MCP
        tools are actually configured — otherwise the faster, streaming,
        direct-grounded path is used, since small local models don't need
        agent overhead just to answer from pre-fetched context."""
        if provider == "gemini":
            return True
        return provider in LOCAL_PROVIDERS and bool(self._get_tools())

    # ------------------------------------------------------------ providers

    def _cached_ollama_status(self, force: bool = False) -> dict[str, Any]:
        now = time.time()
        stale = (now - self._ollama_status_time) > self.ollama_status_ttl
        if force or self._ollama_status_cache is None or stale:
            self._ollama_status_cache = ollama_status(self.ollama_base_url)
            self._ollama_status_time = now
        return self._ollama_status_cache

    def _cached_llamacpp_status(self, force: bool = False) -> dict[str, Any]:
        now = time.time()
        stale = (now - self._llamacpp_status_time) > self.llamacpp_status_ttl
        if force or self._llamacpp_status_cache is None or stale:
            self._llamacpp_status_cache = llamacpp_status(self.llamacpp_base_url)
            self._llamacpp_status_time = now
        return self._llamacpp_status_cache

    def _resolve_llamacpp_model(self) -> str:
        """Auto-adopt whatever model llama-server reports, if it differs
        from LLAMACPP_MODEL (e.g. a local .gguf path vs an HF repo id)."""
        health = self._cached_llamacpp_status()
        reported_models = health.get("models", [])
        if health["running"] and reported_models and self.llamacpp_model not in reported_models:
            self.llamacpp_model = reported_models[0]
        return self.llamacpp_model

    def _build_llamacpp_client(self):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as error:
            raise RuntimeError("Install langchain-openai to use llama.cpp.") from error
        health = self._cached_llamacpp_status()
        if not health["running"]:
            raise RuntimeError(
                f"llama.cpp server is not reachable at {self.llamacpp_base_url}. "
                f"Start it with `llama-server -hf {self.llamacpp_model} --port 8080`."
            )
        return ChatOpenAI(
            model=self.llamacpp_model,
            base_url=self.llamacpp_base_url,
            api_key=self.llamacpp_api_key,
            temperature=0.2,
            max_tokens=self.llamacpp_num_ctx,
        )

    def _build_ollama_client(self):
        try:
            from langchain_ollama import ChatOllama
        except ImportError as error:
            raise RuntimeError("Install langchain-ollama to use Ollama.") from error
        health = self._cached_ollama_status()
        if not health["running"]:
            raise RuntimeError(
                f"Ollama is not reachable at {self.ollama_base_url}. Start it with `ollama serve`."
            )
        installed_models = health.get("models", [])
        model_base = self.ollama_model.split(":", 1)[0]
        if installed_models and not any(
            name == self.ollama_model or name.split(":", 1)[0] == model_base
            for name in installed_models
        ):
            available = ", ".join(installed_models[:6]) or "none"
            raise RuntimeError(
                f"Ollama model `{self.ollama_model}` is not installed. "
                f"Run `ollama pull {self.ollama_model}`. Available: {available}."
            )
        return ChatOllama(
            model=self.ollama_model,
            base_url=self.ollama_base_url,
            temperature=0.2,
            keep_alive=self.ollama_keep_alive,
            num_ctx=self.ollama_num_ctx,
        )

    def _get_model(self, provider: str):
        """Chat model objects are cached per provider+model — built once,
        reused for every question, instead of reconstructed each call."""
        provider = provider.lower()
        if provider == "gemini":
            if not self.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is missing. Add it to .env or choose Ollama.")
            key = f"gemini:{self.gemini_model}"
            if key not in self._models:
                self._models[key] = init_chat_model(
                    f"google_genai:{self.gemini_model}",
                    api_key=self.gemini_api_key,
                    temperature=0.2,
                )
            return self._models[key], self.gemini_model
        elif provider == "ollama":
            key = f"ollama:{self.ollama_model}"
            if key not in self._models:
                self._models[key] = self._build_ollama_client()
            return self._models[key], self.ollama_model
        elif provider == "llamacpp":
            resolved_model = self._resolve_llamacpp_model()
            key = f"llamacpp:{resolved_model}"
            if key not in self._models:
                self._models[key] = self._build_llamacpp_client()
            return self._models[key], self.llamacpp_model
        raise RuntimeError("Provider must be Gemini, Ollama, or llama.cpp.")

    def _get_agent(self, provider: str = "gemini"):
        """Tool-calling agent (retrieval + MCP tools) for any provider.
        Gemini always uses this; local providers use it only when MCP
        tools are configured (see `_use_agent`)."""
        provider = provider.lower()
        key = f"agent:{provider}"
        if key not in self._agents:
            model, _ = self._get_model(provider)
            tools = self._get_tools() + [self._retrieval_tool()]
            self._agents[key] = create_agent(
                model=model,
                tools=tools,
                system_prompt=agent_system_prompt(),
                checkpointer=InMemorySaver(),
            )
        return self._agents[key]

    @staticmethod
    def _answer_text(response: dict[str, Any]) -> str:
        content = response["messages"][-1].content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content)

    # ----------------------------------------------------------------- ask

    async def ask(self, question: str, user_id: str = "default", provider: str = "gemini") -> dict[str, Any]:
        question = question.strip()
        user_id = user_id.strip() or "default"
        if not question:
            return {"answer": "Please enter a question.", "cached": False, "sources": []}
        self._check_rate_limit(user_id)
        return await self._ask_impl(question, user_id, provider)

    async def _ask_impl(self, question: str, user_id: str, provider: str) -> dict[str, Any]:
        """Shared answer logic, without the rate-limit check — `ask()`
        checks it once; `astream_answer()`'s non-streaming fallback (Gemini)
        already checked it too, so this avoids double-charging one request
        against the per-user cooldown."""
        provider = provider.lower()
        if provider not in {"gemini", "ollama", "llamacpp"}:
            raise RuntimeError("Choose Gemini, Ollama, or llama.cpp.")

        # For local providers, resolve the model early (this is where
        # llama.cpp auto-detects the server's actual loaded model) so the
        # cache key and the model name shown in the UI both use the real
        # model, not a possibly-stale config value.
        if provider == "gemini":
            model_name = self.gemini_model
        else:
            _, model_name = self._get_model(provider)

        # Cache key uses durable memory only. Recent conversation changes
        # every turn, so keying on it (as before) meant a verbatim repeated
        # question almost never hit cache after the first turn.
        memories_text = self.memory.memories_text(user_id)
        key = self.memory.cache_key(
            user_id, question, self.document_version, f"{provider}:{model_name}", memories_text
        )
        cached = self.cache.get(key)
        if cached:
            self.memory.auto_capture(user_id, question)
            cached["cached"] = True
            return cached

        # One retrieval pass, reused for generation context and displayed
        # sources on both providers (previously Ollama ran it twice, and
        # Gemini's agent got no pre-fetched context at all).
        context_text, sources, _docs = self._retrieve(question)
        recent_text = self.memory.recent_turns_text(user_id)
        context_package = build_context_package(question, memories_text, recent_text, context_text)

        max_attempts = max(1, int(os.getenv("ANSWER_LOOP_MAX_ATTEMPTS", "2")))

        if self._use_agent(provider):
            agent = self._get_agent(provider)
            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": context_package}]},
                config={"configurable": {"thread_id": user_id}},
            )
            answer = self._answer_text(response)
            # Loop engineering for agent-based providers too: a single
            # grounding check + repair pass, using the raw model directly
            # rather than re-running the whole tool-calling agent.
            if max_attempts > 1 and needs_grounding_repair(answer, context_text):
                model, _ = self._get_model(provider)
                repair_response = await _invoke_with_retry(model, repair_prompt(context_package, answer))
                answer = self._answer_text({"messages": [repair_response]})
        else:
            model, _ = self._get_model(provider)
            answer = ""
            for attempt in range(1, max_attempts + 1):
                response = await _invoke_with_retry(model, ollama_prompt(context_package, attempt))
                answer = self._answer_text({"messages": [response]})
                if not needs_grounding_repair(answer, context_text) or attempt == max_attempts:
                    break
                response = await _invoke_with_retry(model, repair_prompt(context_package, answer))
                answer = self._answer_text({"messages": [response]})

        result = {
            "answer": answer,
            "cached": False,
            "sources": sources,
            "provider": provider,
            "model": model_name,
        }
        self.memory.auto_capture(user_id, question)
        self.memory.add_turn(user_id, question, answer)
        self.cache.put(key, result)
        return result

    # ------------------------------------------------------------ streaming

    async def astream_answer(self, question: str, user_id: str = "default", provider: str = "llamacpp"):
        """Yield answer text incrementally for local providers (llama.cpp,
        Ollama) when no MCP tools are needed, so the UI can show tokens as
        they're generated instead of waiting for the full response. Falls
        back to a single chunk when the tool-calling agent is in play
        (Gemini always; local providers when MCP tools are configured),
        since agents don't stream cleanly through this path.

        Cache and memory bookkeeping mirror `ask()` — this is a streaming
        twin of it, not a separate code path for retrieval or storage.
        """
        question = question.strip()
        user_id = user_id.strip() or "default"
        provider = provider.lower()
        if not question:
            yield "Please enter a question.", []
            return
        self._check_rate_limit(user_id)
        if provider not in {"gemini", "ollama", "llamacpp"}:
            raise RuntimeError("Choose Gemini, Ollama, or llama.cpp.")

        if self._use_agent(provider):
            # Non-streaming fallback — reuse the normal ask() path. Checked
            # before retrieval runs here, so it isn't run twice.
            result = await self._ask_impl(question, user_id, provider)
            yield result["answer"], result["sources"]
            return

        if provider == "gemini":
            model_name = self.gemini_model
        else:
            _, model_name = self._get_model(provider)

        memories_text = self.memory.memories_text(user_id)
        key = self.memory.cache_key(
            user_id, question, self.document_version, f"{provider}:{model_name}", memories_text
        )
        cached = self.cache.get(key)
        if cached:
            self.memory.auto_capture(user_id, question)
            yield cached["answer"], cached.get("sources", [])
            return

        context_text, sources, _docs = self._retrieve(question)
        recent_text = self.memory.recent_turns_text(user_id)
        context_package = build_context_package(question, memories_text, recent_text, context_text)

        model, _ = self._get_model(provider)
        answer = ""
        async for chunk in model.astream(ollama_prompt(context_package, 1)):
            piece = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            answer += piece
            yield answer, sources

        if needs_grounding_repair(answer, context_text):
            response = await _invoke_with_retry(model, repair_prompt(context_package, answer))
            answer = self._answer_text({"messages": [response]})
            yield answer, sources

        result = {
            "answer": answer,
            "cached": False,
            "sources": sources,
            "provider": provider,
            "model": model_name,
        }
        self.memory.auto_capture(user_id, question)
        self.memory.add_turn(user_id, question, answer)
        self.cache.put(key, result)

    # ------------------------------------------------------------- controls

    def load_document(self, document_path: str) -> str:
        self.document_path = document_path
        self._agents.clear()  # tools/checkpointer tied to the prior document
        self._refresh_index()
        self.cache.clear()
        return f"Loaded {Path(document_path).name} ({len(self.documents)} pages)."

    def rebuild_index(self) -> str:
        self._agents.clear()
        self._refresh_index(force_nonce=str(time.time_ns()))
        self.cache.clear()
        return f"Rebuilt the index for {Path(self.document_path).name}."

    def set_embedding_provider(self, provider: str) -> str:
        """Switch between local (HuggingFace) and online (Google) embeddings
        at runtime and rebuild the index under the new vector space —
        vectors from different embedding models aren't comparable, so a
        plain provider swap without a rebuild would silently return
        garbage retrieval results."""
        provider = provider.strip().lower()
        if provider not in EMBEDDING_PROVIDERS:
            raise RuntimeError("Embedding provider must be 'local' or 'online'.")
        if provider == "online" and not self.gemini_api_key:
            raise RuntimeError("Online embeddings need GEMINI_API_KEY set.")
        if provider == self.embedding_provider:
            return f"Already using {provider} embeddings."
        self.embedding_provider = provider
        self._agents.clear()
        self._refresh_index(force_nonce=str(time.time_ns()))
        self.cache.clear()
        label = self.embedding_model_online if provider == "online" else self.embedding_model_name
        return f"Switched to {provider} embeddings ({label}) and rebuilt the index."

    def remember(self, memory: str, user_id: str = "default") -> str:
        memory = memory.strip()
        if not memory:
            return "Enter a memory first."
        self.memory.add_memory(user_id.strip() or "default", memory)
        self.cache.clear()
        return "Memory saved for future responses."

    def clear_user(self, user_id: str = "default") -> str:
        self.memory.clear_user(user_id.strip() or "default")
        self.cache.clear()
        return "Conversation and permanent memories cleared for this user."

    def feedback(self, user_id: str, rating: str) -> str:
        self.memory.add_feedback(user_id.strip() or "default", rating)
        return "Thanks for the feedback."

    def status(self, refresh: bool = True) -> dict[str, Any]:
        embedding_label = (
            f"{self.embedding_model_online} (online)"
            if self.embedding_provider == "online"
            else f"{self.embedding_model_name} (local)"
        )
        tools = self._get_tools()
        mcp_label = (
            f"{len(tools)} tool(s) from {', '.join(self.composio_toolkits)}" if tools else "off"
        )
        return {
            "document": Path(self.document_path).name,
            "pages": len(self.documents),
            "embedding_model": embedding_label,
            "mcp_tools": mcp_label,
            "ollama": self._cached_ollama_status(force=refresh),
            "llamacpp": self._cached_llamacpp_status(force=refresh),
        }
