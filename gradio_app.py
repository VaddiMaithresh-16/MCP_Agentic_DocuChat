"""Simple Gradio workspace for the agentic RAG backend.

Kept intentionally plain: one column, a status line, and a chat box.
No custom CSS beyond Gradio's built-in dark theme.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import tempfile
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from rag_backend import RagBackend

load_dotenv()

backend: RagBackend | None = None
startup_error = ""

PROVIDERS = ["Llama.cpp", "Gemini", "Ollama"]
EMBEDDING_CHOICES = ["Local", "Online"]
STATUS_DOT = {True: "\U0001F7E2", False: "\U0001F534"}

COPY_JS = """
() => {
  const box = document.querySelector('#last_answer_box textarea');
  if (box && box.value) { navigator.clipboard.writeText(box.value); }
}
"""

THEME_TOGGLE_JS = """
() => { document.body.classList.toggle('dark'); }
"""


# --------------------------------------------------------------- backend ---

def _try_start(path: str | None = None) -> str:
    global backend, startup_error
    try:
        backend = RagBackend(document_path=path)
        startup_error = ""
        return "Backend is ready."
    except Exception as error:
        backend = None
        startup_error = str(error)
        return f"Setup needed: {startup_error}"


_try_start()


def _status_text(refresh: bool = False) -> str:
    if backend is None:
        return f"{STATUS_DOT[False]} Backend not ready \u2014 {startup_error}"
    try:
        status = backend.status(refresh=refresh)
        ollama = status["ollama"]
        llamacpp = status["llamacpp"]
        mcp_running = status["mcp_tools"] != "off"
        lines = [
            f"Document: {status['document']} ({status['pages']} pages)",
            f"Embeddings: {status['embedding_model']}",
            f"{STATUS_DOT[llamacpp['running']]} llama.cpp \u2014 {'running' if llamacpp['running'] else 'offline'}",
            f"{STATUS_DOT[ollama['running']]} Ollama \u2014 {'running' if ollama['running'] else 'offline'}",
            f"{STATUS_DOT[mcp_running]} MCP tools \u2014 {status['mcp_tools']}",
        ]
        return "\n".join(lines)
    except Exception as error:
        return f"{STATUS_DOT[False]} Status unavailable \u2014 {error}"


def _refresh_status() -> str:
    return _status_text(refresh=True)


# ----------------------------------------------------------------- chat ---

async def answer_question(question: str, user_id: str, provider: str, history: list[dict] | None):
    history = history or []

    if backend is None:
        message = "The backend is not ready. Upload a PDF or fix the setup message below."
        yield (
            history + [{"role": "user", "content": question}, {"role": "assistant", "content": message}],
            "",
            startup_error,
            "",
            "",
        )
        return

    if not question or not question.strip():
        yield history, "", "Write a question first.", question, ""
        return

    yield (
        history + [{"role": "user", "content": question}, {"role": "assistant", "content": "Thinking\u2026"}],
        gr.update(),
        f"Asking {provider}\u2026",
        question,
        "",
    )

    provider_key = provider.lower().replace(".", "").replace(" ", "")

    if provider_key in {"ollama", "llamacpp"} and backend is not None:
        try:
            sources_text = ""
            updated = history
            async for partial_answer, sources in backend.astream_answer(question, user_id, provider_key):
                sources_list = "\n".join(f"- {source}" for source in sources)
                sources_text = f"**Sources**\n{sources_list}" if sources_list else "No document sources returned."
                updated = history + [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": partial_answer or "\u2026"},
                ]
                yield updated, sources_text, f"Streaming from {provider}\u2026", "", partial_answer
            yield updated, sources_text, f"{provider} \u00b7 done", "", updated[-1]["content"]
        except Exception as error:
            message = f"I couldn't answer with {provider}.\n\n**Reason:** {error}"
            updated = history + [{"role": "user", "content": question}, {"role": "assistant", "content": message}]
            yield updated, "", f"Provider error: {error}", "", ""
        return

    try:
        result = await backend.ask(question, user_id, provider_key)
        sources = "\n".join(f"- {source}" for source in result.get("sources", []))
        source_text = f"**Sources**\n{sources}" if sources else "No document sources returned."
        cache_text = "cached" if result.get("cached") else "fresh"
        status = f"{provider} \u00b7 {result.get('model', '')} \u00b7 {cache_text} answer"
        updated = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": result["answer"]},
        ]
        yield updated, source_text, status, "", result["answer"]
    except Exception as error:
        message = f"I couldn't answer with {provider}.\n\n**Reason:** {error}"
        updated = history + [{"role": "user", "content": question}, {"role": "assistant", "content": message}]
        yield updated, "", f"Provider error: {error}", "", ""


def export_chat(history: list[dict] | None) -> str | None:
    if not history:
        return None
    lines = []
    for turn in history:
        role = "You" if turn.get("role") == "user" else "DocuChat"
        lines.append(f"{role}: {turn.get('content', '')}")
    text = "\n\n".join(lines)
    path = Path(tempfile.gettempdir()) / "docuchat-transcript.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


# ------------------------------------------------------------- document ---

async def upload_document(file_path: str | None):
    if not file_path:
        yield "Choose a PDF first.", _status_text()
        return
    yield "Processing document\u2026", _status_text()
    try:
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        destination = upload_dir / Path(file_path).name
        shutil.copy2(file_path, destination)
        # load_document/_try_start rebuild the index (PDF parse + embed),
        # which blocks — run it off the event loop so the UI doesn't freeze.
        if backend is None:
            message = await asyncio.to_thread(_try_start, str(destination))
        else:
            message = await asyncio.to_thread(backend.load_document, str(destination))
        yield message, _status_text(refresh=True)
    except Exception as error:
        yield f"Upload failed: {error}", _status_text()


async def rebuild_index():
    if backend is None:
        yield startup_error
        return
    yield "Rebuilding index\u2026"
    try:
        yield await asyncio.to_thread(backend.rebuild_index)
    except Exception as error:
        yield f"Rebuild failed: {error}"


async def switch_embeddings(choice: str):
    if backend is None:
        yield startup_error, _status_text()
        return
    yield f"Switching to {choice.lower()} embeddings\u2026", _status_text()
    try:
        message = await asyncio.to_thread(backend.set_embedding_provider, choice.lower())
        yield message, _status_text(refresh=True)
    except Exception as error:
        yield f"Couldn't switch embeddings: {error}", _status_text()


def clear_user(user_id: str):
    return startup_error if backend is None else backend.clear_user(user_id)


def save_feedback(user_id: str, rating: str):
    return startup_error if backend is None else backend.feedback(user_id, rating)


# --------------------------------------------------------------- launch ---

def _launch():
    parser = argparse.ArgumentParser(description="Agentic RAG DocuChat Gradio app")
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio URL")
    parser.add_argument("--no-share", action="store_true", help="Disable public sharing")
    args = parser.parse_args()
    env_share = os.getenv("GRADIO_SHARE", "false").lower() in {"1", "true", "yes", "on"}
    share = False if args.no_share else (True if args.share else env_share)
    username = os.getenv("GRADIO_USERNAME", "").strip()
    password = os.getenv("GRADIO_PASSWORD", "").strip()
    if username == "choose_a_username" or password == "choose_a_strong_password":
        username = password = ""
    auth = (username, password) if username and password else None
    if share and auth is None:
        raise RuntimeError("Public sharing requires GRADIO_USERNAME and GRADIO_PASSWORD in .env.")
    demo.launch(share=share, auth=auth)


# ------------------------------------------------------------------- UI ---

with gr.Blocks(title="DocuChat", theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate")) as demo:
    with gr.Row():
        gr.Markdown("## DocuChat\nAsk questions about a PDF using a local llama.cpp model, Gemini, or Ollama.")
        theme_toggle = gr.Button("\U0001F319 / \u2600\ufe0f", size="sm", scale=0)
        theme_toggle.click(None, None, None, js=THEME_TOGGLE_JS)

    if startup_error:
        gr.Markdown(f"\u26a0\ufe0f **Setup needed:** {startup_error}")

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            user_id = gr.Textbox(label="User ID", value="default")
            provider = gr.Radio(PROVIDERS, value="Llama.cpp", label="Model")
            embedding_choice = gr.Radio(
                EMBEDDING_CHOICES,
                value="Online" if (backend and backend.embedding_provider == "online") else "Local",
                label="Embeddings",
                info="Local = private, runs here. Online = Google API, needs GEMINI_API_KEY.",
            )
            system_status = gr.Markdown(_status_text())
            refresh = gr.Button("Refresh status", size="sm")
            refresh.click(_refresh_status, outputs=system_status)
            embedding_message = gr.Markdown()
            embedding_choice.change(switch_embeddings, inputs=embedding_choice, outputs=[embedding_message, system_status])

            gr.Markdown("---")
            pdf_upload = gr.File(label="Upload a PDF", file_types=[".pdf"], type="filepath")
            process = gr.Button("Process document", variant="primary")
            rebuild = gr.Button("Rebuild index", size="sm")
            document_message = gr.Markdown()
            process.click(upload_document, inputs=pdf_upload, outputs=[document_message, system_status])
            rebuild.click(rebuild_index, outputs=document_message)

            gr.Markdown("---")
            clear_memory = gr.Button("Clear my chat & memory", size="sm")
            with gr.Row():
                helpful = gr.Button("\U0001F44D", size="sm")
                not_helpful = gr.Button("\U0001F44E", size="sm")

        with gr.Column(scale=2, min_width=560):
            chatbot = gr.Chatbot(type="messages", height=460, show_label=False, placeholder="Ask a question about your document.")
            question = gr.Textbox(label="Your question", placeholder="What are the main findings?", lines=2)
            with gr.Row():
                ask = gr.Button("Ask", variant="primary", scale=3)
                clear_chat = gr.ClearButton([question, chatbot], value="Clear chat", scale=1, size="sm")
            with gr.Row():
                copy_answer = gr.Button("\U0001F4CB Copy last answer", size="sm")
                export_btn = gr.DownloadButton("\U0001F4E5 Export chat", size="sm")
            response_status = gr.Markdown()
            sources = gr.Markdown()
            last_answer_box = gr.Textbox(visible=False, elem_id="last_answer_box")

            chat_outputs = [chatbot, sources, response_status, question, last_answer_box]
            ask.click(answer_question, inputs=[question, user_id, provider, chatbot], outputs=chat_outputs)
            question.submit(answer_question, inputs=[question, user_id, provider, chatbot], outputs=chat_outputs)
            clear_memory.click(clear_user, inputs=user_id, outputs=response_status)
            helpful.click(save_feedback, inputs=[user_id, gr.State("helpful")], outputs=response_status)
            not_helpful.click(save_feedback, inputs=[user_id, gr.State("not helpful")], outputs=response_status)
            copy_answer.click(None, None, None, js=COPY_JS)
            export_btn.click(export_chat, inputs=chatbot, outputs=export_btn)


if __name__ == "__main__":
    demo.queue()
    _launch()
