import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# expandable_segments is unstable on Windows (crashes the CUDA caching allocator); force off
# before torch reads PYTORCH_CUDA_ALLOC_CONF at import time.
_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments:true" in _conf.replace(" ", "").lower():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

import torch
from transformers import TextIteratorStreamer

from polyvalent_lr.models.loader import (
    load_internvl_model_and_tokenizer,
    ensure_internvl_tokens_configured,
)

try:
    from peft import PeftModel
except Exception:  # pragma: no cover - peft optional for fused checkpoints
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

STOP_STRINGS = ["<|im_end|>", "<|im_start|>"]

try:
    from polyvalent_lr.reasoning_v3 import decoding as v3_decoding
    from polyvalent_lr.reasoning_v3 import prompts as v3_prompts
    V3_SYSTEM = v3_prompts.inference_system_prompt()
except Exception:  # pragma: no cover - v3 package is optional
    v3_decoding = None
    V3_SYSTEM = DEFAULT_SYSTEM


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


class Runtime:
    """Holds one loaded model + tokenizer and serializes generation on the shared GPU."""

    def __init__(self, key, label, model, tokenizer):
        self.key = key
        self.label = label
        self.model = model
        self.tokenizer = tokenizer
        self.lock = threading.Lock()

    def input_device(self):
        device = getattr(self.model, "device", None)
        if device is None or getattr(device, "type", None) == "meta":
            try:
                device = next(self.model.parameters()).device
            except Exception:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if getattr(device, "type", None) == "meta":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    def iter_generation(self, messages, system_prompt, logic_mode, max_new_tokens,
                        temperature, top_p, repetition_penalty, no_repeat_ngram_size=3):
        tokenizer = self.tokenizer
        prompt = build_prompt(messages, system_prompt, logic_mode)
        inputs = tokenizer(prompt, return_tensors="pt").to(self.input_device())
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
                    self.model.generate(**gen_kwargs)
            except Exception as exc:
                errors["error"] = str(exc)
                streamer.end()

        with self.lock:
            ensure_internvl_tokens_configured(self.model, tokenizer)
            thread = threading.Thread(target=_run, daemon=True)
            thread.start()
            for chunk in streamer:
                if chunk:
                    yield chunk
            thread.join(timeout=5.0)

        if "error" in errors:
            raise RuntimeError(errors["error"])

    def generate_text(self, messages, system_prompt, logic_mode, max_new_tokens,
                      temperature, top_p, repetition_penalty, no_repeat_ngram_size=3,
                      self_consistency=1):
        if self_consistency and self_consistency > 1:
            samples = [
                "".join(self.iter_generation(
                    messages, system_prompt, logic_mode,
                    max_new_tokens, temperature, top_p, repetition_penalty, no_repeat_ngram_size,
                ))
                for _ in range(self_consistency)
            ]
            if v3_decoding is not None:
                return v3_decoding.majority_vote(samples)
            return samples[0]
        return "".join(
            self.iter_generation(
                messages, system_prompt, logic_mode,
                max_new_tokens, temperature, top_p, repetition_penalty, no_repeat_ngram_size,
            )
        )


STATE = {"runtimes": {}, "order": [], "device": ""}


def _resolve_dtype(name):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _detect_adapter(model_path):
    """Finds a LoRA adapter co-located with a model copy (models/<name>/adapter/ or top level)."""
    for candidate in (Path(model_path) / "adapter", Path(model_path)):
        if (candidate / "adapter_config.json").exists():
            return str(candidate)
    return None


def load_runtime(key, label, model_path, adapter_dir, args):
    if adapter_dir is None:
        adapter_dir = _detect_adapter(model_path)

    print(f"[LR-Compare] Loading '{label}' from {model_path}"
          + (f" (+adapter {adapter_dir})" if adapter_dir else ""))
    model, tokenizer = load_internvl_model_and_tokenizer(
        model_name_or_path=model_path,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=True,
        is_eval=True,
    )
    if adapter_dir:
        already_attached = getattr(model, "peft_config", None) is not None
        if PeftModel is not None and isinstance(model, PeftModel):
            already_attached = True
        if already_attached:
            print("[LR-Compare]      base already has an adapter attached; using it as-is")
            model.eval()
        elif PeftModel is None:
            raise RuntimeError("--adapter was given but peft is not installed.")
        else:
            model = PeftModel.from_pretrained(model, adapter_dir)
            model.eval()
    runtime = Runtime(key, label, model, tokenizer)
    STATE["runtimes"][key] = runtime
    STATE["order"].append(key)
    print(f"[LR-Compare] '{label}' ready on {runtime.input_device()}")
    return runtime


PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Polyvalent LR - Compare Models</title>
<style>
  :root {
    --bg:#0d1117; --panel:#161b22; --panel-2:#1c222b; --border:#30363d;
    --text:#e6edf3; --muted:#8b949e; --accent:#2f81f7; --accent-2:#1f6feb;
    --user:#1f6feb; --assistant:#21262d; --warn:#d29922; --ok:#3fb950; --b:#a371f7;
  }
  * { box-sizing:border-box; }
  html,body { height:100%; }
  body { margin:0; background:var(--bg); color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; font-size:14px; }
  .app { display:flex; flex-direction:column; height:100vh; }
  header { display:flex; align-items:center; gap:12px; padding:10px 18px;
    background:var(--panel); border-bottom:1px solid var(--border); }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  .badge { font-size:12px; color:var(--muted); background:var(--panel-2);
    border:1px solid var(--border); border-radius:999px; padding:3px 10px; }
  .dot { width:8px; height:8px; border-radius:50%; background:var(--muted); display:inline-block; margin-right:6px; }
  .dot.on { background:var(--ok); }
  .composer { border-bottom:1px solid var(--border); background:var(--panel); padding:12px 18px; }
  .composer-inner { max-width:1500px; margin:0 auto; display:flex; gap:10px; align-items:flex-end; }
  textarea { flex:1; resize:none; background:var(--panel-2); color:var(--text);
    border:1px solid var(--border); border-radius:10px; padding:10px 12px; font:inherit;
    max-height:160px; min-height:44px; outline:none; }
  textarea:focus { border-color:var(--accent); }
  button { cursor:pointer; border:1px solid var(--border); background:var(--panel-2);
    color:var(--text); border-radius:10px; padding:10px 15px; font:inherit; font-weight:600; }
  button:hover { background:#242b35; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  button.primary:hover { background:var(--accent-2); }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .cols { flex:1; min-height:0; display:flex; gap:1px; background:var(--border); }
  .col { flex:1; min-width:0; display:flex; flex-direction:column; background:var(--bg); }
  .col-head { padding:9px 14px; background:var(--panel); border-bottom:1px solid var(--border);
    font-weight:600; font-size:13px; display:flex; align-items:center; gap:8px; }
  .tag { font-size:11px; padding:2px 7px; border-radius:999px; background:#1f6feb22; color:#79c0ff; border:1px solid #1f6feb55; }
  .col:nth-child(2) .tag { background:#a371f722; color:#d2a8ff; border-color:#a371f755; }
  .messages { flex:1; overflow-y:auto; padding:16px; display:flex; flex-direction:column; gap:12px; }
  .msg { display:flex; gap:10px; }
  .avatar { flex:0 0 28px; height:28px; border-radius:8px; display:flex; align-items:center;
    justify-content:center; font-size:11px; font-weight:700; color:#fff; }
  .msg.user .avatar { background:var(--user); }
  .msg.assistant .avatar { background:#6e7681; }
  .bubble { flex:1; min-width:0; background:var(--assistant); border:1px solid var(--border);
    border-radius:10px; padding:10px 12px; white-space:pre-wrap; word-wrap:break-word; line-height:1.5; overflow-x:auto; }
  .msg.user .bubble { background:#12233f; border-color:#1f6feb55; }
  .empty { color:var(--muted); text-align:center; margin:auto; max-width:320px; line-height:1.6; }
  .think { border-left:3px solid var(--warn); background:#ffffff08; border-radius:6px; padding:7px 9px; margin:7px 0; color:#c9d1d9; }
  .think-h { font-size:10px; text-transform:uppercase; letter-spacing:.06em; color:var(--warn); margin-bottom:4px; }
  .answer { border-left:3px solid var(--ok); background:#3fb95011; border-radius:6px; padding:7px 9px; margin:7px 0; }
  .answer .lbl { font-size:10px; text-transform:uppercase; letter-spacing:.06em; color:var(--ok); margin-right:8px; }
  .meta { font-size:11px; color:var(--muted); margin-top:6px; }
  @media (max-width:900px){ .cols{ flex-direction:column; } }
</style>
</head>
<body>
<div class="app">
  <header>
    <h1>Polyvalent &middot; Logical Reasoning &mdash; Model Comparison</h1>
    <span class="badge"><span class="dot" id="dot"></span><span id="status">loading...</span></span>
  </header>
  <div class="composer">
    <div class="composer-inner">
      <textarea id="input" rows="1" placeholder="Ask a logical reasoning question...  (Enter to send to both, Shift+Enter for newline)"></textarea>
      <button class="primary" id="send">Send to both</button>
      <button id="clear">Clear</button>
    </div>
  </div>
  <div class="cols" id="cols"></div>
</div>
<script>
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const colsEl = document.getElementById('cols');
const statusEl = document.getElementById('status');

let models = [];
let streaming = false;

function escapeHtml(s){return s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function formatContent(s){
  let e=escapeHtml(s);
  e=e.replace(/&lt;think&gt;([\s\S]*?)&lt;\/think&gt;/g,(m,p)=>'<div class="think"><div class="think-h">reasoning</div>'+p.trim()+'</div>');
  e=e.replace(/&lt;answer&gt;([\s\S]*?)&lt;\/answer&gt;/g,(m,p)=>'<div class="answer"><span class="lbl">answer</span>'+p.trim()+'</div>');
  e=e.replace(/&lt;\/?think&gt;/g,'').replace(/&lt;\/?answer&gt;/g,'');
  return e;
}

async function init(){
  const colState = {};
  const r = await fetch('/api/info');
  const info = await r.json();
  models = info.models || [];
  document.getElementById('dot').classList.add('on');
  statusEl.textContent = info.device + ' · ' + models.length + ' models';
  colsEl.innerHTML = '';
  for (const m of models){
    const col = document.createElement('div');
    col.className = 'col';
    col.innerHTML =
      '<div class="col-head"><span class="tag">'+escapeHtml(m.label)+'</span></div>'+
      '<div class="messages" id="msg-'+m.key+'"><div class="empty">Waiting for a question...</div></div>';
    colsEl.appendChild(col);
    colState[m.key] = { history: [], el: col.querySelector('.messages') };
  }
  window.colState = colState;
}

function addBubble(container, role){
  const empty = container.querySelector('.empty');
  if (empty) empty.style.display = 'none';
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? 'You' : 'AI';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  wrap.appendChild(avatar); wrap.appendChild(bubble);
  container.appendChild(wrap);
  container.scrollTop = container.scrollHeight;
  return bubble;
}

async function streamTo(key, payload, bubble, container){
  const t0 = performance.now();
  const res = await fetch('/api/chat', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ ...payload, model: key })
  });
  if (!res.ok || !res.body) throw new Error('HTTP ' + res.status);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '', acc = '';
  while (true){
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream:true });
    const events = buffer.split('\n\n');
    buffer = events.pop();
    for (const ev of events){
      const line = ev.trim();
      if (!line.startsWith('data:')) continue;
      let data; try { data = JSON.parse(line.slice(5).trim()); } catch(e){ continue; }
      if (data.type === 'token'){ acc += data.text; bubble.innerHTML = formatContent(acc); container.scrollTop = container.scrollHeight; }
      else if (data.type === 'error'){ acc += '\n\n[error] ' + data.text; bubble.innerHTML = formatContent(acc); }
    }
  }
  const dt = ((performance.now()-t0)/1000).toFixed(1);
  if (!acc) bubble.innerHTML = '<span style="color:#d29922">(no output)</span>';
  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = dt + 's';
  bubble.appendChild(meta);
  return acc;
}

async function send(){
  const text = inputEl.value.trim();
  if (!text || streaming) return;
  inputEl.value = ''; inputEl.style.height = 'auto';
  streaming = true; sendBtn.disabled = true;

  const payload = {
    system: document.getElementById('system') ? document.getElementById('system').value : undefined,
    logic_mode: true,
    max_new_tokens: 512, temperature: 0, top_p: 1, repetition_penalty: 1.1, no_repeat_ngram_size: 3,
  };

  const jobs = models.map(async (m) => {
    const st = window.colState[m.key];
    st.history.push({ role:'user', content:text });
    addBubble(st.el, 'user').textContent = text;
    const bubble = addBubble(st.el, 'assistant');
    bubble.innerHTML = '<span style="color:#8b949e">...</span>';
    try {
      const ans = await streamTo(m.key, { ...payload, messages: st.history }, bubble, st.el);
      st.history.push({ role:'assistant', content: ans });
    } catch(e){
      bubble.innerHTML = '<span style="color:#d29922">Request failed: ' + escapeHtml(String(e)) + '</span>';
    }
  });
  await Promise.all(jobs);
  streaming = false; sendBtn.disabled = false; inputEl.focus();
}

sendBtn.addEventListener('click', send);
inputEl.addEventListener('keydown', (e)=>{ if (e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); } });
inputEl.addEventListener('input', ()=>{ inputEl.style.height='auto'; inputEl.style.height=Math.min(inputEl.scrollHeight,160)+'px'; });
document.getElementById('clear').addEventListener('click', ()=>{ init(); });
init();
inputEl.focus();
</script>
</body>
</html>
"""


class CompareHandler(BaseHTTPRequestHandler):
    server_version = "PolyvalentLRCompare/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("[LR-Compare] " + (fmt % args) + "\n")

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
            self._send_json({
                "device": STATE["device"],
                "models": [
                    {"key": k, "label": STATE["runtimes"][k].label}
                    for k in STATE["order"]
                ],
            })
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

        key = payload.get("model")
        runtime = STATE["runtimes"].get(key)
        if runtime is None:
            self._send_json({"error": f"unknown model '{key}'"}, status=400)
            return

        messages = payload.get("messages") or []
        system_prompt = payload.get("system") or DEFAULT_SYSTEM
        logic_mode = bool(payload.get("logic_mode", True))
        max_new_tokens = int(payload.get("max_new_tokens", 512))
        temperature = float(payload.get("temperature", 0.0))
        top_p = float(payload.get("top_p", 1.0))
        repetition_penalty = float(payload.get("repetition_penalty", 1.1))
        no_repeat_ngram_size = int(payload.get("no_repeat_ngram_size", 3))
        self_consistency = int(payload.get("self_consistency", STATE.get("self_consistency", 1)))
        # Self-consistency needs sampling; if the UI left temp at 0, use the preset default.
        if self_consistency > 1 and temperature <= 0.0:
            temperature = float(STATE.get("default_temperature", 0.6))

        if self_consistency and self_consistency > 1:
            self._stream_voted(
                runtime, messages, system_prompt, logic_mode,
                max_new_tokens, temperature, top_p, repetition_penalty,
                no_repeat_ngram_size, self_consistency,
            )
        else:
            self._stream_response(
                runtime, messages, system_prompt, logic_mode,
                max_new_tokens, temperature, top_p, repetition_penalty, no_repeat_ngram_size,
            )

    def _stream_response(self, runtime, messages, system_prompt, logic_mode,
                         max_new_tokens, temperature, top_p, repetition_penalty,
                         no_repeat_ngram_size=3):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for chunk in runtime.iter_generation(
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

    def _stream_voted(self, runtime, messages, system_prompt, logic_mode,
                      max_new_tokens, temperature, top_p, repetition_penalty,
                      no_repeat_ngram_size, self_consistency):
        """Self-consistency path: sample N, majority-vote, stream the winner."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            text = runtime.generate_text(
                messages, system_prompt, logic_mode,
                max_new_tokens, temperature, top_p, repetition_penalty,
                no_repeat_ngram_size, self_consistency=self_consistency,
            )
            note = f"[self-consistency: {self_consistency} samples, majority-voted]\n\n"
            for piece in (note + text,):
                data = json.dumps({"type": "token", "text": piece})
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


def main():
    parser = argparse.ArgumentParser(
        description="Zero-dependency side-by-side chat UI to compare two LR model variants."
    )
    parser.add_argument("--model_a", type=str, default="models/lr_adapter",
                        help="Model A (default: base + adapter copy)")
    parser.add_argument("--adapter_a", type=str, default=None,
                        help="Optional extra adapter for model A (already baked into models/lr_adapter)")
    parser.add_argument("--label_a", type=str, default="Base + LoRA adapter")
    parser.add_argument("--model_b", type=str, default="models/lr_merged",
                        help="Model B (default: merged/fused copy)")
    parser.add_argument("--adapter_b", type=str, default=None, help="Optional extra adapter for model B")
    parser.add_argument("--label_b", type=str, default="Merged (fused)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device_map", type=str, default="auto" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "fp16", "float32"])
    parser.add_argument("--preset", type=str, default="v3", choices=["v2", "v3"],
                        help="Decoding preset. v3 = T=0.6, rep 1.1 (+ optional self-consistency).")
    parser.add_argument("--self_consistency", type=int, default=1,
                        help="Sample this many completions and majority-vote the final table (v3).")
    parser.add_argument("--self_test", action="store_true",
                        help="Load both models, run one generation on each, print, then exit.")
    args = parser.parse_args()
    args.torch_dtype = _resolve_dtype(args.dtype)
    STATE["self_consistency"] = max(1, args.self_consistency)
    STATE["default_temperature"] = 0.6 if args.preset == "v3" else 0.0

    if not Path(args.model_a).exists():
        parser.error(f"model A not found: {args.model_a}")
    if not Path(args.model_b).exists():
        parser.error(f"model B not found: {args.model_b}")

    load_runtime("a", args.label_a, args.model_a, args.adapter_a, args)
    load_runtime("b", args.label_b, args.model_b, args.adapter_b, args)
    STATE["device"] = args.device

    if args.self_test:
        print("[LR-Compare] Self-test generation...")
        q = [{"role": "user", "content": "All cats are mammals. Tom is a cat. Is Tom a mammal?"}]
        for key in STATE["order"]:
            out = STATE["runtimes"][key].generate_text(q, DEFAULT_SYSTEM, True, 64, 0.0, 1.0, 1.1)
            print(f"\n--- {STATE['runtimes'][key].label} ---\n{out}")
        return

    server = ThreadingHTTPServer((args.host, args.port), CompareHandler)
    print("=" * 60)
    print(f"[LR-Compare] Compare UI running at http://{args.host}:{args.port}")
    print("[LR-Compare] Press Ctrl+C to stop.")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[LR-Compare] Shutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
