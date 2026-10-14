#!/usr/bin/env python3
"""
Clicky Web Companion Server
Runs a local web interface on localhost:7799 with:
- SMRITI Long-Term Memory integration (via localhost:7798)
- Local Ollama AI chat (Mistral, Qwen 3.5 Vision, etc.)
- Native macOS silent screen capture via screencapture CLI
- Browser-based Voice input (Speech-to-Text) and Text-to-Speech with full stop/mute control
"""

import os
import sys
import json
import base64
import subprocess
import urllib.request
import urllib.error
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

PORT = 7800
SMRITI_API_URL = "http://localhost:7798"
OLLAMA_API_URL = "http://localhost:11434"
SCREENSHOT_PATH = "/tmp/clicky_web_screen.jpg"
WEB_DIR = Path(__file__).parent.resolve()

import math
import time
import threading
import ctypes
import ctypes.util

# CoreGraphics binding for instantaneous, permission-free mouse location
_core_graphics = None
class _CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

def get_mouse_location():
    """Returns the current global (x, y) coordinates of the mouse cursor on macOS."""
    global _core_graphics
    try:
        if _core_graphics is None:
            lib_path = ctypes.util.find_library("CoreGraphics")
            if lib_path:
                _core_graphics = ctypes.cdll.LoadLibrary(lib_path)
                _core_graphics.CGEventCreate.restype = ctypes.c_void_p
                _core_graphics.CGEventGetLocation.restype = _CGPoint
                _core_graphics.CGEventGetLocation.argtypes = [ctypes.c_void_p]

        if _core_graphics:
            event = _core_graphics.CGEventCreate(None)
            if event:
                pt = _core_graphics.CGEventGetLocation(event)
                return int(pt.x), int(pt.y)
    except Exception as e:
        print(f"⚠️ Error reading mouse location: {e}")
    return None, None

# Gesture detection state
gesture_enabled = True
last_circle_capture_time = 0.0
last_cursor_info = None
_gesture_thread = None

def play_snap_sound():
    """Plays standard macOS Pop chime for instant tactile audio feedback."""
    try:
        sound_path = "/System/Library/Sounds/Pop.aiff"
        if os.path.exists(sound_path):
            subprocess.Popen(["afplay", sound_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"⚠️ Error playing snap sound: {e}")

def _check_circle_slice(pts):
    """
    Checks if a slice of points forms a valid closed circle/loop.
    Requires intentional gesture:
    - Min 60px bounding box
    - Aspect ratio between 0.45 and 2.2
    - Closure distance < 45% of diagonal
    - Must traverse all 4 quadrants around centroid
    - Path length matches expected perimeter
    - Cumulative signed angle >= 1.78 * pi (~320 degrees)
    """
    if len(pts) < 14:
        return False, None, None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max_x - min_x
    h = max_y - min_y

    # Intentional circle gesture size: at least 60px, up to 750px
    if w < 60 or h < 60 or w > 750 or h > 750:
        return False, None, None

    # Elliptical aspect ratio check (not a slit)
    aspect = w / max(h, 1)
    if not (0.45 <= aspect <= 2.2):
        return False, None, None

    diag = math.hypot(w, h)
    closure_dist = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
    if closure_dist > diag * 0.45:
        return False, None, None

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0

    # Must visit all 4 quadrants around centroid
    q1 = any(p[0] >= cx and p[1] >= cy for p in pts)
    q2 = any(p[0] < cx and p[1] >= cy for p in pts)
    q3 = any(p[0] < cx and p[1] < cy for p in pts)
    q4 = any(p[0] >= cx and p[1] < cy for p in pts)
    if not (q1 and q2 and q3 and q4):
        return False, None, None

    # Check path length vs perimeter
    path_len = sum(math.hypot(pts[k+1][0] - pts[k][0], pts[k+1][1] - pts[k][1]) for k in range(len(pts)-1))
    approx_perimeter = math.pi * ((w + h) / 2.0)
    if path_len < approx_perimeter * 0.70 or path_len > approx_perimeter * 2.5:
        return False, None, None

    angles = [math.atan2(p[1] - cy, p[0] - cx) for p in pts]
    signed = 0.0
    tot_abs = 0.0
    for j in range(len(angles) - 1):
        da = angles[j + 1] - angles[j]
        while da > math.pi:
            da -= 2 * math.pi
        while da < -math.pi:
            da += 2 * math.pi
        signed += da
        tot_abs += abs(da)

    if abs(signed) >= 1.78 * math.pi and tot_abs <= 3.2 * math.pi:
        return True, cx, cy
    return False, None, None

def detect_circle_in_buffer(buffer):
    """
    buffer is list of (x, y, t).
    Searches backwards from the most recent point for a complete circular lasso loop.
    """
    if len(buffer) < 10:
        return False, None, None
    now = buffer[-1][2]
    for i in range(len(buffer) - 10, -1, -1):
        dt = now - buffer[i][2]
        if dt < 0.25:
            continue
        if dt > 1.2:
            break
        ok, cx, cy = _check_circle_slice(buffer[i:])
        if ok:
            return True, cx, cy
    return False, None, None

def _circle_gesture_loop():
    """Background 50Hz daemon loop tracking cursor loops."""
    global last_circle_capture_time, last_cursor_info
    buffer = []
    print("🌀 [Clicky Gesture] Circle trigger listener active (50Hz polling)...")
    while True:
        try:
            if not gesture_enabled:
                time.sleep(0.2)
                buffer.clear()
                continue

            now = time.time()
            mx, my = get_mouse_location()
            if mx is not None and my is not None:
                buffer.append((mx, my, now))
                # Keep points within the last 1.3s
                cutoff = now - 1.3
                while buffer and buffer[0][2] < cutoff:
                    buffer.pop(0)

                # Debounce by 1.8s
                if now - last_circle_capture_time > 1.8:
                    detected, cx, cy = detect_circle_in_buffer(buffer)
                    if detected:
                        last_circle_capture_time = now
                        buffer.clear()
                        play_snap_sound()
                        b64, cursor_info = capture_screen(mode="cursor", center=(int(cx), int(cy)))
                        last_cursor_info = cursor_info
                        print(f"🎯 [Gesture Snap] Circle detected at ({int(cx)}, {int(cy)})! Captured 1280x800 crop.")
            time.sleep(0.02)
        except Exception as e:
            time.sleep(0.1)

def start_gesture_listener():
    """Starts the gesture listener in a background daemon thread."""
    global _gesture_thread
    if _gesture_thread is None or not _gesture_thread.is_alive():
        _gesture_thread = threading.Thread(target=_circle_gesture_loop, daemon=True)
        _gesture_thread.start()

def capture_screen(mode="cursor", crop_w=1280, crop_h=800, center=None):
    """
    Captures screen content.
    - mode='cursor': Captures a crisp 1280x800 crop centered directly around the mouse cursor (or center=(cx, cy)).
    - mode='full': Captures full desktop and downscales to max 1280px via sips.
    Returns (base64_jpeg_string, cursor_info_dict).
    """
    try:
        cursor_info = None
        if mode == "cursor":
            if center and center[0] is not None and center[1] is not None:
                mx, my = int(center[0]), int(center[1])
            else:
                mx, my = get_mouse_location()
            if mx is not None and my is not None:
                rx = max(0, mx - crop_w // 2)
                ry = max(0, my - crop_h // 2)
                cmd = ["screencapture", "-x", "-t", "jpg", f"-R{rx},{ry},{crop_w},{crop_h}", SCREENSHOT_PATH]
                subprocess.run(cmd, check=True, capture_output=True)
                cursor_info = {
                    "global_x": mx,
                    "global_y": my,
                    "crop_x": rx,
                    "crop_y": ry,
                    "rel_x": mx - rx,
                    "rel_y": my - ry,
                    "width": crop_w,
                    "height": crop_h,
                    "mode": "cursor"
                }
            else:
                mode = "full"

        if mode == "full":
            subprocess.run(["screencapture", "-x", "-t", "jpg", SCREENSHOT_PATH], check=True, capture_output=True)
            subprocess.run(["sips", "-Z", "1280", SCREENSHOT_PATH], capture_output=True)
            cursor_info = {"mode": "full"}

        if os.path.exists(SCREENSHOT_PATH) and os.path.getsize(SCREENSHOT_PATH) > 0:
            with open(SCREENSHOT_PATH, "rb") as f:
                img_data = f.read()
            return base64.b64encode(img_data).decode("utf-8"), cursor_info
    except Exception as e:
        print(f"⚠️ Error capturing screen: {e}")
    return None, None

def query_smriti_recall(query, limit=4):
    """Recalls relevant memories from SMRITI API."""
    try:
        req = urllib.request.Request(
            f"{SMRITI_API_URL}/recall",
            data=json.dumps({"query": query, "top_k": limit}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("memories", [])
    except Exception as e:
        print(f"⚠️ Smriti recall warning: {e}")
        return []

def encode_smriti_memory(content, modality="text"):
    """Encodes a conversation turn into SMRITI."""
    try:
        req = urllib.request.Request(
            f"{SMRITI_API_URL}/encode",
            data=json.dumps({
                "content": content,
                "source": "direct",
                "modality": modality
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"⚠️ Smriti encode warning: {e}")
        return None

KEYS_FILE = Path.home() / ".clicky_keys.json"

def get_api_keys():
    """Returns stored API keys from environment variables or ~/.clicky_keys.json."""
    keys = {
        "anthropic": os.environ.get("ANTHROPIC_API_KEY", "").strip(),
        "gemini": os.environ.get("GEMINI_API_KEY", "").strip(),
        "openai": os.environ.get("OPENAI_API_KEY", "").strip()
    }
    if KEYS_FILE.exists():
        try:
            with open(KEYS_FILE, "r") as f:
                data = json.load(f)
                for provider in ["anthropic", "gemini", "openai"]:
                    if data.get(provider) and not keys[provider]:
                        keys[provider] = data[provider].strip()
        except Exception as e:
            print(f"⚠️ Error reading keys file: {e}")
    return keys

def save_api_keys(new_keys):
    """Saves API keys to ~/.clicky_keys.json with restricted 0600 permissions."""
    current = {}
    if KEYS_FILE.exists():
        try:
            with open(KEYS_FILE, "r") as f:
                current = json.load(f)
        except Exception:
            current = {}
    for provider, key in new_keys.items():
        if key is not None:
            clean = str(key).strip()
            if clean:
                current[provider] = clean
            elif provider in current and clean == "":
                del current[provider]
    try:
        with open(KEYS_FILE, "w") as f:
            json.dump(current, f, indent=2)
        os.chmod(KEYS_FILE, 0o600)
    except Exception as e:
        print(f"⚠️ Error saving keys file: {e}")
    return get_api_keys()

def mask_key(key):
    if not key or len(key) < 8:
        return ""
    return key[:6] + "..." + key[-4:]

def query_anthropic(model, system_prompt, user_prompt, image_b64=None, api_key=None):
    """Sends chat completion request to Anthropic Messages API."""
    if not api_key:
        api_key = get_api_keys().get("anthropic")
    if not api_key:
        raise ValueError("Anthropic API key is not configured. Click ⚙️ API Keys in the top bar to add your key.")

    content_blocks = []
    if image_b64:
        content_blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_b64
            }
        })
    content_blocks.append({"type": "text", "text": user_prompt})

    payload = {
        "model": model,
        "max_tokens": 1500,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": content_blocks}
        ]
    }

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        content_list = data.get("content", [])
        return "".join([c.get("text", "") for c in content_list if c.get("type") == "text"])

def query_gemini(model, system_prompt, user_prompt, image_b64=None, api_key=None):
    """Sends chat completion request to Google Gemini API."""
    if not api_key:
        api_key = get_api_keys().get("gemini")
    if not api_key:
        raise ValueError("Google Gemini API key is not configured. Click ⚙️ API Keys in the top bar to add your key.")

    parts = []
    if image_b64:
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": image_b64
            }
        })
    parts.append({"text": user_prompt})

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1500
        }
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join([p.get("text", "") for p in parts])
        return ""

def query_openai(model, system_prompt, user_prompt, image_b64=None, api_key=None):
    """Sends chat completion request to OpenAI API."""
    if not api_key:
        api_key = get_api_keys().get("openai")
    if not api_key:
        raise ValueError("OpenAI API key is not configured. Click ⚙️ API Keys in the top bar to add your key.")

    content_blocks = []
    if image_b64:
        content_blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
        })
    content_blocks.append({"type": "text", "text": user_prompt})

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_blocks}
        ],
        "temperature": 0.7,
        "max_tokens": 1500
    }

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""

def is_model_vision(model_name):
    """Returns True if the model has multimodal vision capabilities."""
    name = (model_name or "").lower()
    if any(k in name for k in ["claude-3-5", "claude-3-opus", "gemini", "gpt-4o", "gpt-4-turbo"]):
        return True
    if any(k in name for k in ["vision", "vl", "pixtral", "llava", "minicpm", "moondream", "bakllava", "qwen3.5"]):
        return True
    return False

def get_ollama_models():
    """Gets available models from local Ollama."""
    try:
        req = urllib.request.Request(f"{OLLAMA_API_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []

def query_ollama(model, system_prompt, user_prompt, image_b64=None):
    """Sends chat completion request to Ollama using native /api/chat with multimodal image support."""
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    is_vision = is_model_vision(model)
    user_msg = {"role": "user", "content": user_prompt}
    if is_vision and image_b64:
        user_msg["images"] = [image_b64]
    messages.append(user_msg)

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.7
        }
    }

    req = urllib.request.Request(
        f"{OLLAMA_API_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        return res.get("message", {}).get("content", "")

class ClickyWebHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Keep terminal clean, log only API queries
        if args and isinstance(args[0], str) and "/api/" in args[0]:
            print(f"🌐 [ClickyWeb] {args[0]}")

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        clean_path = self.path.split("?")[0]
        if clean_path == "/" or clean_path == "/index.html":
            index_file = WEB_DIR / "index.html"
            if index_file.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_cors_headers()
                self.end_headers()
                with open(index_file, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"index.html not found")

        elif clean_path == "/api/status":
            models = get_ollama_models()
            smriti_stats = {}
            try:
                req = urllib.request.Request(f"{SMRITI_API_URL}/health")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    smriti_stats = json.loads(resp.read().decode("utf-8"))
            except Exception:
                pass

            keys = get_api_keys()
            data = {
                "ollama_running": len(models) > 0,
                "models": models,
                "smriti_connected": smriti_stats.get("status") == "ok",
                "smriti_memories": smriti_stats.get("memories_count", 0),
                "has_screenshot": os.path.exists(SCREENSHOT_PATH),
                "gesture_enabled": gesture_enabled,
                "last_circle_capture_time": last_circle_capture_time,
                "last_cursor_info": last_cursor_info,
                "has_anthropic": bool(keys.get("anthropic")),
                "has_gemini": bool(keys.get("gemini")),
                "has_openai": bool(keys.get("openai"))
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))

        elif clean_path == "/api/keys":
            keys = get_api_keys()
            data = {
                "has_anthropic": bool(keys.get("anthropic")),
                "has_gemini": bool(keys.get("gemini")),
                "has_openai": bool(keys.get("openai")),
                "masked": {
                    "anthropic": mask_key(keys.get("anthropic", "")),
                    "gemini": mask_key(keys.get("gemini", "")),
                    "openai": mask_key(keys.get("openai", ""))
                }
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))

        elif clean_path == "/api/screenshot":
            if os.path.exists(SCREENSHOT_PATH):
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_cors_headers()
                self.end_headers()
                with open(SCREENSHOT_PATH, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/gesture_toggle":
            global gesture_enabled
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 0:
                try:
                    body = json.loads(self.rfile.read(content_length).decode("utf-8"))
                    if "enabled" in body:
                        gesture_enabled = bool(body["enabled"])
                    else:
                        gesture_enabled = not gesture_enabled
                except Exception:
                    gesture_enabled = not gesture_enabled
            else:
                gesture_enabled = not gesture_enabled

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"gesture_enabled": gesture_enabled}).encode("utf-8"))

        elif self.path == "/api/keys":
            content_length = int(self.headers.get("Content-Length", 0))
            body = {}
            if content_length > 0:
                try:
                    body = json.loads(self.rfile.read(content_length).decode("utf-8"))
                except Exception:
                    pass
            save_api_keys(body.get("keys", {}))
            keys = get_api_keys()
            data = {
                "status": "ok",
                "has_anthropic": bool(keys.get("anthropic")),
                "has_gemini": bool(keys.get("gemini")),
                "has_openai": bool(keys.get("openai")),
                "masked": {
                    "anthropic": mask_key(keys.get("anthropic", "")),
                    "gemini": mask_key(keys.get("gemini", "")),
                    "openai": mask_key(keys.get("openai", ""))
                }
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))

        elif self.path == "/api/capture":
            content_length = int(self.headers.get("Content-Length", 0))
            body = {}
            if content_length > 0:
                try:
                    body = json.loads(self.rfile.read(content_length).decode("utf-8"))
                except Exception:
                    pass
            mode = body.get("mode", "cursor")
            b64, cursor_info = capture_screen(mode=mode)
            self.send_response(200 if b64 else 500)
            self.send_header("Content-Type", "application/json")
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok" if b64 else "error",
                "has_image": b64 is not None,
                "image_data": f"data:image/jpeg;base64,{b64}" if b64 else None,
                "cursor_info": cursor_info
            }).encode("utf-8"))

        elif self.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            try:
                body = json.loads(body_bytes.decode("utf-8"))
            except Exception:
                body = {}

            prompt = body.get("prompt", "").strip()
            model = body.get("model", "mistral:latest")
            include_screen = body.get("include_screen", True)
            capture_mode = body.get("capture_mode", "cursor")
            image_b64 = body.get("image_b64")
            cursor_info = body.get("cursor_info")

            if not prompt:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Empty prompt"}).encode("utf-8"))
                return

            # If screen requested, use current intentional snapshot or capture fresh
            if include_screen and not image_b64:
                # Prioritize existing intentional snapshot (from circle gesture or snap button)
                # to prevent overwriting user's target with a capture of the Clicky web chat window
                if os.path.exists(SCREENSHOT_PATH) and os.path.getsize(SCREENSHOT_PATH) > 0 and (last_circle_capture_time > 0 or last_cursor_info):
                    with open(SCREENSHOT_PATH, "rb") as f:
                        image_b64 = base64.b64encode(f.read()).decode("utf-8")
                    cursor_info = last_cursor_info or {"mode": "snapshot"}
                else:
                    image_b64, cursor_info = capture_screen(mode=capture_mode)

            # Recall from SMRITI
            memories = query_smriti_recall(prompt, limit=4)
            recalled_context = ""
            if memories:
                recalled_context = "\n\n[RECALLED SMRITI MEMORIES]:\n" + "\n".join(
                    [f"- {m.get('content', '')}" for m in memories]
                )

            # Spatial cursor context
            cursor_context = ""
            if cursor_info and cursor_info.get("mode") == "cursor":
                cursor_context = (
                    f"\n[Cursor Spatial Awareness: The attached image is a 1280x800 crop centered directly around the user's mouse cursor. "
                    f"The cursor was positioned at relative pixel ({cursor_info.get('rel_x')}, {cursor_info.get('rel_y')}). "
                    f"Pay particular attention to the text, code, or UI elements immediately adjacent to the cursor.]\n"
                )

            now_str = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")
            system_prompt = (
                f"You are Clicky, a friendly, intelligent AI companion with persistent long-term memory via SMRITI.\n"
                f"Today is: {now_str}.\n"
                f"You can see what the user is working on if a screen capture is attached.\n"
                f"Be direct, insightful, and clear. If referring to code or visual elements on screen, describe them precisely.\n"
                f"Keep responses conversational and readable."
            ) + cursor_context + recalled_context

            is_vision = is_model_vision(model)
            text_model_warning = ""
            if include_screen and not is_vision:
                text_model_warning = f"⚠️ [Notice: '{model}' is a text-only model and cannot view images. To analyze what's on your screen, select a Vision model like Gemini 2.0 Flash, Claude 3.5 Sonnet, or GPT-4o in the top dropdown.]\n\n"

            try:
                # Dispatch to appropriate provider
                if model.startswith("claude-"):
                    reply = query_anthropic(model=model, system_prompt=system_prompt, user_prompt=prompt, image_b64=image_b64)
                elif model.startswith("gemini-"):
                    reply = query_gemini(model=model, system_prompt=system_prompt, user_prompt=prompt, image_b64=image_b64)
                elif model.startswith("gpt-"):
                    reply = query_openai(model=model, system_prompt=system_prompt, user_prompt=prompt, image_b64=image_b64)
                else:
                    reply = query_ollama(model=model, system_prompt=system_prompt, user_prompt=prompt, image_b64=image_b64)
                    if not reply:
                        reply = "I received your message, but didn't get a response from the model. Make sure Ollama is running."

                if text_model_warning:
                    reply = text_model_warning + reply

                # Encode exchange to SMRITI
                spatial_tag = f" [Cursor at ({cursor_info['global_x']}, {cursor_info['global_y']})]" if (cursor_info and "global_x" in cursor_info) else ""
                encode_text = f"User asked: {prompt}{spatial_tag}\nClicky answered: {reply[:300]}"
                encode_smriti_memory(encode_text, modality="text")

                response_data = {
                    "reply": reply,
                    "model": model,
                    "recalled_memories": memories,
                    "has_screenshot": image_b64 is not None,
                    "cursor_info": cursor_info
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

def run_server():
    server_address = ("127.0.0.1", PORT)
    httpd = HTTPServer(server_address, ClickyWebHandler)
    print(f"✨ Clicky Web Companion running at http://localhost:{PORT}")
    print(f"🧠 SMRITI Memory linked at {SMRITI_API_URL}")
    print(f"🦙 Ollama AI linked at {OLLAMA_API_URL}")
    start_gesture_listener()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Clicky Web Server...")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
