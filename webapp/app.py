from flask import Flask, jsonify, render_template, request
from rag import answer

app = Flask(__name__)


def current_user() -> str:
    # Injected by App Service Authentication (Easy Auth). Absent when running locally.
    return request.headers.get("X-MS-CLIENT-PRINCIPAL-NAME", "local-dev")


@app.get("/")
def index():
    return render_template("index.html", user=current_user())


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify(error="Question is required"), 400
    if len(question) > 1000:
        return jsonify(error="Question is too long"), 400

    # Only accept user/assistant turns from the browser, so a client can't inject a system prompt
    history = [
        {"role": m["role"], "content": str(m["content"])}
        for m in data.get("history", [])
        if isinstance(m, dict) and m.get("role") in ("user", "assistant")
    ][-10:]

    app.logger.info("Question from %s", current_user())
    try:
        reply, sources = answer(question, history)
    except Exception:
        app.logger.exception("Chat failed")
        return jsonify(error="Something went wrong. If you were idle, refresh the page to sign in again."), 500
    return jsonify(reply=reply, sources=sources)


@app.get("/health")
def health():
    return "ok"
