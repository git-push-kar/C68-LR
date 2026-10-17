import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import TextIteratorStreamer

from polyvalent_lr.models.loader import (
    load_internvl_model_and_tokenizer,
    ensure_internvl_tokens_configured,
)
from polyvalent_lr.config import DEFAULT_BASE_MODEL_PATH

try:
    from peft import PeftModel
except Exception:  # pragma: no cover - peft is optional for the fused checkpoint
    PeftModel = None


DEFAULT_SYSTEM = (
    "You are a rigorous logical reasoning engine. "
    "Solve the problem by validating rules and facts systematically."
)

LOGIC_FORMAT = (
    "Format your response as:\n"
    "<think>\n[Step-by-step deduction/proof steps]\n</think>\n"
    "<answer>[Final Answer]</answer>"
)

STATE = {
    "model": None,
    "tokenizer": None,
    "lock": threading.Lock(),
    "info": {},
}


def build_prompt(messages, system_prompt, logic_mode):
    """Builds an InternVL/Qwen ChatML prompt matching the LR training format."""
    system_text = (system_prompt or DEFAULT_SYSTEM).strip()
    if logic_mode:
        system_text = f"{system_text}\n\n{LOGIC_FORMAT}"

    parts = [f"<|im_start|>system\n{system_text}<|im_end|>\n"]
    for msg in messages:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _input_device():
    model = STATE["model"]
    device = getattr(model, "device", None)
    if device is None or getattr(device, "type", None) == "meta":
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device(STATE["info"].get("device", "cpu"))
    if getattr(device, "type", None) == "meta":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


STOP_STRINGS = ["<|im_end|>", "<|im_start|>"]


def iter_generation(messages, system_prompt, logic_mode, max_new_tokens, temperature, top_p,
                    repetition_penalty, no_repeat_ngram_size=3):
    """Yields generated text chunks for the given chat history."""
    model = STATE["model"]
    tokenizer = STATE["tokenizer"]
    prompt = build_prompt(messages, system_prompt, logic_mode)

    inputs = tokenizer(prompt, return_tensors="pt").to(_input_device())
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=600.0
    )

    do_sample = temperature > 0.0
    gen_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask"),
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
        "top_p": float(top_p),
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "repetition_penalty": float(repetition_penalty),
        "no_repeat_ngram_size": int(no_repeat_ngram_size),
        "stop_strings": STOP_STRINGS,
        "tokenizer": tokenizer,
        "streamer": streamer,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)

    errors = {}

    def _run():
        try:
            with torch.no_grad():
                model.generate(**gen_kwargs)
        except Exception as exc:  # surfaced to the client below
            errors["error"] = str(exc)
            streamer.end()

    with STATE["lock"]:
        ensure_internvl_tokens_configured(model, tokenizer)
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        for chunk in streamer:
            if chunk:
                yield chunk
        thread.join(timeout=5.0)

    if "error" in errors:
        raise RuntimeError(errors["error"])


def generate_text(messages, system_prompt, logic_mode, max_new_tokens, temperature, top_p,
                  repetition_penalty, no_repeat_ngram_size=3):
    return "".join(
        iter_generation(
            messages, system_prompt, logic_mode,
            max_new_tokens, temperature, top_p, repetition_penalty, no_repeat_ngram_size,
        )
    )


PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Polyvalent LR Chat</title>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --panel-2: #1c222b; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #2f81f7; --accent-2: #1f6feb;
    --user: #1f6feb; --assistant: #21262d; --warn: #d29922; --ok: #3fb950;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 15px;
  }
  .app { display: flex; flex-direction: column; height: 100vh; }
  header {
    display: flex; align-items: center; gap: 12px; padding: 12px 20px;
    background: var(--panel); border-bottom: 1px solid var(--border);
  }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  .badge {
    font-size: 12px; color: var(--muted); background: var(--panel-2);
    border: 1px solid var(--border); border-radius: 999px; padding: 3px 10px;
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); display: inline-block; margin-right: 6px; }
  .dot.on { background: var(--ok); }
  .layout { display: flex; flex: 1; min-height: 0; }
  .chat-col { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  .messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 14px; }
  .msg { display: flex; gap: 12px; max-width: 900px; width: 100%; margin: 0 auto; }
  .avatar {
    flex: 0 0 32px; height: 32px; border-radius: 8px; display: flex; align-items: center;
    justify-content: center; font-size: 12px; font-weight: 700; color: #fff;
  }
  .msg.user .avatar { background: var(--user); }
  .msg.assistant .avatar { background: #6e7681; }
  .bubble {
    flex: 1; min-width: 0; background: var(--assistant); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 14px; white-space: pre-wrap; word-wrap: break-word;
    line-height: 1.5; overflow-x: auto;
  }
  .msg.user .bubble { background: #12233f; border-color: #1f6feb55; }
  .empty { color: var(--muted); text-align: center; margin: auto; max-width: 480px; line-height: 1.6; }
  .think { border-left: 3px solid var(--warn); background: #ffffff08; border-radius: 6px; padding: 8px 10px; margin: 8px 0; color: #c9d1d9; }
  .think-h { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--warn); margin-bottom: 4px; }
  .answer { border-left: 3px solid var(--ok); background: #3fb95011; border-radius: 6px; padding: 8px 10px; margin: 8px 0; }
  .answer .lbl { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--ok); margin-right: 8px; }
  .composer { border-top: 1px solid var(--border); background: var(--panel); padding: 12px 20px; }
  .composer-inner { max-width: 900px; margin: 0 auto; display: flex; gap: 10px; align-items: flex-end; }
  textarea {
    flex: 1; resize: none; background: var(--panel-2); color: var(--text);
    border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px;
    font: inherit; max-height: 200px; min-height: 46px; outline: none;
  }
  textarea:focus { border-color: var(--accent); }
  button {
    cursor: pointer; border: 1px solid var(--border); background: var(--panel-2);
    color: var(--text); border-radius: 10px; padding: 11px 16px; font: inherit; font-weight: 600;
  }
  button:hover { background: #242b35; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button.primary:hover { background: var(--accent-2); }
  button:disabled { opacity: .5; cursor: not-allowed; }
  .sidebar {
    width: 300px; flex: 0 0 300px; border-left: 1px solid var(--border);
    background: var(--panel); padding: 18px; overflow-y: auto;
  }
  .sidebar h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .07em; color: var(--muted); margin: 0 0 10px; }
  .field { margin-bottom: 16px; }
  .field label { display: block; font-size: 13px; margin-bottom: 6px; color: var(--text); }
  .field .hint { font-size: 11px; color: var(--muted); margin-top: 4px; }
  .field input[type=number], .field textarea {
    width: 100%; background: var(--panel-2); color: var(--text);
    border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font: inherit;
  }
  .field textarea { min-height: 90px; resize: vertical; }
  .check { display: flex; align-items: center; gap: 8px; font-size: 13px; }
  .check input { width: 16px; height: 16px; }
  .row { display: flex; gap: 8px; }
  .status { font-size: 12px; color: var(--muted); margin-top: 8px; }
  @media (max-width: 820px) { .sidebar { display: none; } }
</style>
</head>
<body>
<div class="app">
  <header>
    <h1>Polyvalent &middot; Logical Reasoning Chat</h1>
    <span class="badge" id="model-badge"><span class="dot" id="dot"></span><span id="model-name">loading...</span></span>
  </header>
  <div class="layout">
    <div class="chat-col">
      <div class="messages" id="messages">
        <div class="empty" id="empty">
          Chat with the LR fine-tuned model. Use <b>Logical reasoning mode</b> for
          <code>&lt;think&gt;</code> / <code>&lt;answer&gt;</code> structured proofs.
        </div>
      </div>
      <div class="composer">
        <div class="composer-inner">
          <textarea id="input" rows="1" placeholder="Ask a logical reasoning question...  (Enter to send, Shift+Enter for newline)"></textarea>
          <button class="primary" id="send">Send</button>
          <button id="stop" disabled>Stop</button>
        </div>
      </div>
    </div>
    <aside class="sidebar">
      <div class="field">
        <h2>Generation</h2>
        <label for="max-new">Max new tokens</label>
        <input type="number" id="max-new" value="512" min="16" max="4096" step="16" />
      </div>
      <div class="field">
        <label for="temp">Temperature</label>
        <input type="number" id="temp" value="0" min="0" max="2" step="0.1" />
        <div class="hint">0 = greedy / deterministic</div>
      </div>
      <div class="field">
        <label for="topp">Top-p</label>
        <input type="number" id="topp" value="1" min="0" max="1" step="0.05" />
      </div>
      <div class="field">
        <label for="rep">Repetition penalty</label>
        <input type="number" id="rep" value="1.1" min="1" max="2" step="0.01" />
      </div>
      <div class="field">
        <label for="no-repeat">No-repeat n-gram size</label>
        <input type="number" id="no-repeat" value="3" min="0" max="10" step="1" />
      </div>
      <div class="field">
        <h2>Prompting</h2>
        <div class="check">
          <input type="checkbox" id="logic-mode" checked />
          <label for="logic-mode" style="margin:0">Logical reasoning mode</label>
        </div>
      </div>
      <div class="field">
        <label for="system">System prompt</label>
        <textarea id="system">You are a rigorous logical reasoning engine. Solve the problem by validating rules and facts systematically.</textarea>
      </div>
      <div class="row">
        <button id="clear" style="flex:1">Clear chat</button>
      </div>
      <div class="status" id="status"></div>
    </aside>
  </div>
</div>
<script>
const messagesEl = document.getElementById('messages');
const emptyEl = document.getElementById('empty');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const stopBtn = document.getElementById('stop');
const statusEl = document.getElementById('status');

let history = [];
let controller = null;
let streaming = false;

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function formatContent(s) {
  let e = escapeHtml(s);
  e = e.replace(/&lt;think&gt;([\s\S]*?)&lt;\/think&gt;/g,
    (m, p) => '<div class="think"><div class="think-h">reasoning</div>' + p.trim() + '</div>');
  e = e.replace(/&lt;answer&gt;([\s\S]*?)&lt;\/answer&gt;/g,
    (m, p) => '<div class="answer"><span class="lbl">answer</span>' + p.trim() + '</div>');
  e = e.replace(/&lt;\/?think&gt;/g, '').replace(/&lt;\/?answer&gt;/g, '');
  return e;
}

function addBubble(role) {
  if (emptyEl) emptyEl.style.display = 'none';
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? 'You' : 'LR';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

function setStreaming(on) {
  streaming = on;
  sendBtn.disabled = on;
  stopBtn.disabled = !on;
  statusEl.textContent = on ? 'generating...' : '';
}

async function refreshInfo() {
  try {
    const r = await fetch('/api/info');
    const info = await r.json();
    document.getElementById('model-name').textContent = info.label + ' · ' + info.device;
    document.getElementById('dot').classList.add('on');
  } catch (e) {
    document.getElementById('model-name').textContent = 'offline';
  }
}

async function send() {
  const text = inputEl.value.trim();
  if (!text || streaming) return;
  inputEl.value = '';
  inputEl.style.height = 'auto';

  history.push({ role: 'user', content: text });
  const userBubble = addBubble('user');
  userBubble.textContent = text;
  const botBubble = addBubble('assistant');
  botBubble.innerHTML = '<span style="color:#8b949e">...</span>';

  const payload = {
    messages: history,
    system: document.getElementById('system').value,
    logic_mode: document.getElementById('logic-mode').checked,
    max_new_tokens: parseInt(document.getElementById('max-new').value, 10) || 512,
    temperature: parseFloat(document.getElementById('temp').value) || 0,
    top_p: parseFloat(document.getElementById('topp').value) || 1,
    repetition_penalty: parseFloat(document.getElementById('rep').value) || 1.1,
    no_repeat_ngram_size: parseInt(document.getElementById('no-repeat').value, 10) || 0,
  };

  controller = new AbortController();
  setStreaming(true);
  let acc = '';
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!res.ok || !res.body) throw new Error('HTTP ' + res.status);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split('\n\n');
      buffer = events.pop();
      for (const ev of events) {
        const line = ev.trim();
        if (!line.startsWith('data:')) continue;
        let data;
        try { data = JSON.parse(line.slice(5).trim()); } catch (e) { continue; }
        if (data.type === 'token') {
          acc += data.text;
          botBubble.innerHTML = formatContent(acc);
          messagesEl.scrollTop = messagesEl.scrollHeight;
        } else if (data.type === 'error') {
          acc += '\n\n[error] ' + data.text;
          botBubble.innerHTML = formatContent(acc);
        }
      }
    }
    if (!acc) botBubble.innerHTML = '<span style="color:#d29922">(no output)</span>';
    history.push({ role: 'assistant', content: acc });
  } catch (e) {
    if (e.name === 'AbortError') {
      botBubble.innerHTML = formatContent(acc + '\n\n[stopped]');
      history.push({ role: 'assistant', content: acc });
    } else {
      botBubble.innerHTML = '<span style="color:#d29922">Request failed: ' + escapeHtml(String(e)) + '</span>';
    }
  } finally {
    controller = null;
    setStreaming(false);
    inputEl.focus();
  }
}

sendBtn.addEventListener('click', send);
stopBtn.addEventListener('click', () => { if (controller) controller.abort(); });
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 200) + 'px';
});
document.getElementById('clear').addEventListener('click', () => {
  history = [];
  messagesEl.querySelectorAll('.msg').forEach(n => n.remove());
  if (emptyEl) emptyEl.style.display = '';
});

refreshInfo();
inputEl.focus();
</script>
</body>
</html>
"""


class ChatHandler(BaseHTTPRequestHandler):
    server_version = "PolyvalentLR/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("[LR-UI] " + (fmt % args) + "\n")

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/info":
            self._send_json(STATE["info"])
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        if self.path != "/api/chat":
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            payload = self._read_json()
        except Exception as exc:
            self._send_json({"error": f"bad request: {exc}"}, status=400)
            return

        messages = payload.get("messages") or []
        system_prompt = payload.get("system") or DEFAULT_SYSTEM
        logic_mode = bool(payload.get("logic_mode", True))
        max_new_tokens = int(payload.get("max_new_tokens", 512))
        temperature = float(payload.get("temperature", 0.0))
        top_p = float(payload.get("top_p", 1.0))
        repetition_penalty = float(payload.get("repetition_penalty", 1.1))
        no_repeat_ngram_size = int(payload.get("no_repeat_ngram_size", 3))

        if payload.get("stream", True):
            self._stream_response(
                messages, system_prompt, logic_mode,
                max_new_tokens, temperature, top_p, repetition_penalty, no_repeat_ngram_size,
            )
        else:
            try:
                text = generate_text(
                    messages, system_prompt, logic_mode,
                    max_new_tokens, temperature, top_p, repetition_penalty, no_repeat_ngram_size,
                )
                self._send_json({"response": text})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)

    def _stream_response(self, messages, system_prompt, logic_mode,
                         max_new_tokens, temperature, top_p, repetition_penalty,
                         no_repeat_ngram_size=3):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for chunk in iter_generation(
                messages, system_prompt, logic_mode,
                max_new_tokens, temperature, top_p, repetition_penalty, no_repeat_ngram_size,
            ):
                data = json.dumps({"type": "token", "text": chunk})
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
            self.wfile.write(b'data: {"type": "done"}\n\n')
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as exc:
            try:
                data = json.dumps({"type": "error", "text": str(exc)})
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass


def load_model(args):
    model_path = Path(args.model_path)
    if not model_path.exists() and "/" not in args.model_path and "\\" not in args.model_path:
        pass  # allow a bare HF repo id

    print(f"[LR-UI] Loading model: {args.model_path}")
    model, tokenizer = load_internvl_model_and_tokenizer(
        model_name_or_path=args.model_path,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=True,
        is_eval=True,
    )

    label = model_path.name or args.model_path
    if args.adapter_dir:
        if PeftModel is None:
            raise RuntimeError("--adapter_dir was given but peft is not installed.")
        print(f"[LR-UI] Attaching LoRA adapter: {args.adapter_dir}")
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        model.eval()
        label = f"{label} + {Path(args.adapter_dir).name}"

    STATE["model"] = model
    STATE["tokenizer"] = tokenizer

    device = _input_device()
    STATE["info"] = {
        "loaded": True,
        "label": label,
        "model_path": args.model_path,
        "adapter_dir": args.adapter_dir,
        "device": str(device),
    }
    print(f"[LR-UI] Model ready on {device}")

    if args.self_test:
        print("[LR-UI] Running self-test generation...")
        out = generate_text(
            [{"role": "user", "content": "All cats are mammals. Tom is a cat. Is Tom a mammal?"}],
            DEFAULT_SYSTEM, True, 64, 0.0, 1.0, 1.1,
        )
        print("[LR-UI] Self-test output:\n" + out)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Zero-dependency chat UI for the fine-tuned LR model.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_BASE_MODEL_PATH,
                        help="Fused checkpoint dir, base model dir, or HF repo id (default: local InternVL3-2B base).")
    parser.add_argument("--adapter_dir", type=str, default=None,
                        help="Optional LoRA adapter to attach to the base model.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device_map", type=str, default="auto" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "fp16", "float32"])
    parser.add_argument("--self_test", action="store_true", help="Load the model, run one generation, then exit.")
    args = parser.parse_args()

    if args.dtype == "bfloat16":
        args.torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        args.torch_dtype = torch.float16
    elif args.dtype == "float32":
        args.torch_dtype = torch.float32
    else:
        if torch.cuda.is_available():
            args.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            args.torch_dtype = torch.float32

    if not load_model(args):
        return

    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    url = f"http://{args.host}:{args.port}"
    print("=" * 60)
    print(f"[LR-UI] Chat interface running at {url}")
    print("[LR-UI] Press Ctrl+C to stop.")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[LR-UI] Shutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
