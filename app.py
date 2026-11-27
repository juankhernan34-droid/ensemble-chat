"""
Atlas Chat API - Serverless Inference API variant
Flask API with Stripe payment integration

NO-GPU VARIANT: instead of downloading Mistral-7B and Falcon-7B locally
(which needs ~28GB+ of GPU VRAM), this version calls the Hugging Face
Serverless Inference API. It runs on any CPU-only host (e.g. free
Hugging Face Spaces).

Required env: HUGGINGFACE_TOKEN (or HF_TOKEN) - a Hugging Face Read token.
You must also accept the gated-model license for
mistralai/Mistral-7B-Instruct-v0.1 on huggingface.co while logged in.
"""

import os
import json
import re
import hashlib
import html
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from functools import wraps
import sqlite3
import time
from threading import Lock
import math
import urllib.request
import urllib.parse

from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
import stripe
from dotenv import load_dotenv

from huggingface_hub import InferenceClient
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key-change-this")
CORS(app)

# Stripe configuration
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY")

# Pricing tiers
PRICING_TIERS = {
    "free": {"name": "Free", "price": 0, "requests_per_month": 100, "stripe_price_id": None},
    "starter": {"name": "Starter", "price": 2.99, "requests_per_month": 5000, "stripe_price_id": os.getenv("STRIPE_STARTER_PRICE_ID")},
    "pro": {"name": "Pro", "price": 9.99, "requests_per_month": 50000, "stripe_price_id": os.getenv("STRIPE_PRO_PRICE_ID")},
}

# Database initialization
def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect("ensemble_chat.db")
    c = conn.cursor()

    # Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            email TEXT UNIQUE,
            tier TEXT DEFAULT 'free',
            api_key TEXT UNIQUE,
            stripe_customer_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Usage tracking table
    c.execute("""
        CREATE TABLE IF NOT EXISTS usage (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            requests INTEGER DEFAULT 0,
            month TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # Transactions table
    c.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            stripe_charge_id TEXT,
            amount REAL,
            tier TEXT,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # Visits table (simple visitor counter)
    c.execute("""
        CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY,
            path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Vern's Pizza orders (pay at the store for pickup, by card at the door
    # for delivery — no payment data anywhere). order_type is 'pickup' or
    # 'delivery'; delivery orders carry delivery_address and delivery_fee.
    # Numbers run per ready day (V-001, V-002... each day), so the number is
    # only unique together with its date. Older DBs declared order_number
    # UNIQUE on its own — a pre-order for tomorrow could then never take
    # V-001 if today already had one, so rebuild the table once if so.
    row = c.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='orders'"
    ).fetchone()
    if row and "order_number TEXT UNIQUE" in (row[0] or ""):
        c.execute("ALTER TABLE orders RENAME TO orders_old")
        c.execute("""
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                order_number TEXT,
                order_date TEXT,
                customer_name TEXT,
                customer_phone TEXT,
                pickup_type TEXT,
                pickup_time TEXT,
                items_json TEXT,
                subtotal REAL,
                status TEXT,
                order_type TEXT DEFAULT 'pickup',
                delivery_address TEXT,
                delivery_fee REAL DEFAULT 0,
                delivery_distance_km REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(order_number, order_date)
            )
        """)
        c.execute(
            "INSERT INTO orders (id, order_number, order_date, customer_name,"
            " customer_phone, pickup_type, pickup_time, items_json, subtotal,"
            " status, order_type, delivery_address, delivery_fee, created_at)"
            " SELECT id, order_number, order_date, customer_name, customer_phone,"
            " pickup_type, pickup_time, items_json, subtotal, status, order_type,"
            " delivery_address, delivery_fee, created_at FROM orders_old"
        )
        c.execute("DROP TABLE orders_old")
    c.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            order_number TEXT,
            order_date TEXT,
            customer_name TEXT,
            customer_phone TEXT,
            pickup_type TEXT,
            pickup_time TEXT,
            items_json TEXT,
            subtotal REAL,
            status TEXT,
            order_type TEXT DEFAULT 'pickup',
            delivery_address TEXT,
            delivery_fee REAL DEFAULT 0,
            delivery_distance_km REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(order_number, order_date)
        )
    """)

    # Delivery distance (km from the store) was added after launch; older
    # databases lack the column, so add it when it is missing.
    cols = [r[1] for r in c.execute("PRAGMA table_info(orders)").fetchall()]
    if "delivery_distance_km" not in cols:
        c.execute("ALTER TABLE orders ADD COLUMN delivery_distance_km REAL")

    # One row per ordered line, for the kitchen view
    c.execute("""
        CREATE TABLE IF NOT EXISTS order_items (
            id INTEGER PRIMARY KEY,
            order_id INTEGER,
            category TEXT,
            name TEXT,
            detail TEXT,
            unit_price REAL,
            quantity INTEGER,
            line_total REAL,
            FOREIGN KEY(order_id) REFERENCES orders(id)
        )
    """)

    conn.commit()
    conn.close()

def log_visit(path):
    """Log a page visit (simple counter, no personal data)"""
    try:
        conn = sqlite3.connect("ensemble_chat.db")
        c = conn.cursor()
        c.execute("INSERT INTO visits (path) VALUES (?)", (path,))
        conn.commit()
        conn.close()
    except Exception:
        pass

def get_visit_count(path=None):
    """Get total visit count, optionally filtered by path"""
    try:
        conn = sqlite3.connect("ensemble_chat.db")
        c = conn.cursor()
        if path:
            c.execute("SELECT COUNT(*) FROM visits WHERE path=?", (path,))
        else:
            c.execute("SELECT COUNT(*) FROM visits")
        result = c.fetchone()
        conn.close()
        return result[0] if result else 0
    except Exception:
        return 0

# Initialize models via Hugging Face Serverless Inference API (no local GPU needed)
def load_models():
    """Set up the Inference API client and return it with the two ensemble model IDs"""
    print("Setting up Hugging Face Inference API client (no local GPU needed)...")

    # Accept either env var name (this repo uses HUGGINGFACE_TOKEN, ai-api-stripe uses HF_TOKEN)
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token:
        print("WARNING: HUGGINGFACE_TOKEN (or HF_TOKEN) is not set - gated models (Mistral) will fail")

    try:
        # huggingface_hub >= 0.23 uses api_key=; force the new router endpoint
        # (api-inference.huggingface.co was deprecated and no longer resolves)
        client = InferenceClient(api_key=hf_token, base_url="https://router.huggingface.co")
    except TypeError:
        # older huggingface_hub (e.g. 0.19.x) uses token=
        client = InferenceClient(token=hf_token)

    if os.getenv("GROQ_API_KEY"):
        # Groq free-tier models (OpenAI-compatible, no card required).
        # Verified 2026-11-09 against console.groq.com/docs/models: these
        # are the two strongest self-serve production chat models on Groq.
        # Llama 3.3 70B is Enterprise-only and Qwen 3.8-27B is preview-only
        # (not for production), so they are deliberately not used here.
        MODEL_A = "openai/gpt-oss-20b"
        MODEL_B = "openai/gpt-oss-120b"
    else:
        MODEL_A = "Qwen/Qwen2.5-7B-Instruct"
        MODEL_B = "HuggingFaceH4/zephyr-7b-beta"
    return client, [MODEL_A, MODEL_B]

# Set up inference client on startup
try:
    HF_CLIENT, MODELS = load_models()
    print("Inference API client ready")
except Exception as e:
    print(f"Error setting up Inference API client: {e}")
    HF_CLIENT, MODELS = None, None

# Helper functions
def generate_api_key(email):
    """Generate unique API key for user"""
    return hashlib.sha256(f"{email}{datetime.now().isoformat()}".encode()).hexdigest()

def get_current_month():
    """Get current month in YYYY-MM format"""
    return datetime.now().strftime("%Y-%m")

def get_user_usage(user_id, month=None):
    """Get user's API request usage for a month"""
    if month is None:
        month = get_current_month()

    conn = sqlite3.connect("ensemble_chat.db")
    c = conn.cursor()
    c.execute("SELECT requests FROM usage WHERE user_id=? AND month=?", (user_id, month))
    result = c.fetchone()
    conn.close()

    return result[0] if result else 0

def increment_usage(user_id, requests=1):
    """Increment user's API request count"""
    month = get_current_month()

    conn = sqlite3.connect("ensemble_chat.db")
    c = conn.cursor()

    c.execute("SELECT id FROM usage WHERE user_id=? AND month=?", (user_id, month))
    if c.fetchone():
        c.execute("UPDATE usage SET requests = requests + ? WHERE user_id=? AND month=?",
                 (requests, user_id, month))
    else:
        c.execute("INSERT INTO usage (user_id, requests, month) VALUES (?, ?, ?)",
                 (user_id, requests, month))

    conn.commit()
    conn.close()

def call_gen(client, model_id, prompt, max_new_tokens=2000):
    """Call a single model through the configured inference provider via direct HTTP.

    Provider selection (env):
      GROQ_API_KEY set  -> Groq (free tier, OpenAI-compatible)
      otherwise          -> Hugging Face router (needs credits/providers)
    """
    import requests
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        # Groq: OpenAI-compatible, free tier, no card required
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
    else:
        hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        # Use the OpenAI-compatible chat completions endpoint on the new router
        url = "https://router.huggingface.co/v1/chat/completions"
        headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    out = resp.json()
    # OpenAI-compatible format: choices[0].message.content
    if isinstance(out, dict) and "choices" in out and out["choices"]:
        text = out["choices"][0].get("message", {}).get("content", "")
    elif isinstance(out, list) and out:
        text = out[0].get("generated_text", "")
    elif isinstance(out, dict):
        text = out.get("generated_text", "")
    else:
        text = str(out)
    return text.strip()

def safe_call_gen(client, model_id, prompt, max_new_tokens=2000):
    """Call one model, returning error string instead of raising.

    Retries once when the model returns an empty reply (free-tier
    flakiness: Groq sometimes returns empty content). If the retry is
    also empty, the empty string is returned and the caller reports the
    usual "All models failed" error.
    """
    try:
        text = call_gen(client, model_id, prompt, max_new_tokens)
        if not text:
            # Empty reply: retry once before giving up.
            text = call_gen(client, model_id, prompt, max_new_tokens)
        return text
    except Exception as e:
        # Don't leak tokens; HF errors are safe to surface
        # Include response body for HTTP errors to aid debugging
        detail = str(e)[:200]
        if hasattr(e, 'response') and e.response is not None:
            try:
                body = e.response.text[:300]
                detail += f" | Body: {body}"
            except:
                pass
        return f"__ERROR__{type(e).__name__}: {detail}"

def ask_ensemble(prompt):
    """Query both models and return best answer"""
    if not HF_CLIENT or not MODELS:
        return {"error": "Models not loaded: HF client failed to initialize (check HUGGINGFACE_TOKEN)"}

    with ThreadPoolExecutor(max_workers=len(MODELS)) as ex:
        futures = [ex.submit(safe_call_gen, HF_CLIENT, m, prompt) for m in MODELS]
        results = [f.result() for f in futures]

    errors = [r for r in results if r.startswith("__ERROR__")]
    cleaned = [r for r in results if r and r.strip() and not r.startswith("__ERROR__")]

    if not cleaned:
        detail = errors[0].replace("__ERROR__", "") if errors else "unknown"
        return {"error": f"All models failed: {detail}"}

    cnt = Counter(cleaned)
    top, freq = cnt.most_common(1)[0]

    if freq >= 2:
        return {"answer": top, "strategy": "majority"}

    longest = max(cleaned, key=len)
    return {"answer": longest, "strategy": "longest"}

# Decorators
def require_api_key(f):
    """Decorator to validate API key"""
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return jsonify({"error": "Missing API key"}), 401

        conn = sqlite3.connect("ensemble_chat.db")
        c = conn.cursor()
        c.execute("SELECT id, tier FROM users WHERE api_key=?", (api_key,))
        result = c.fetchone()
        conn.close()

        if not result:
            return jsonify({"error": "Invalid API key"}), 401

        user_id, tier = result

        # Check rate limit
        usage = get_user_usage(user_id)
        limit = PRICING_TIERS[tier]["requests_per_month"]

        if usage >= limit:
            return jsonify({"error": f"Rate limit exceeded. Max {limit} requests/month"}), 429

        request.user_id = user_id
        request.tier = tier
        return f(*args, **kwargs)

    return decorated

# Routes - Public
@app.route("/", methods=["GET"])
def index():
    """Home page - SEO-friendly landing page"""
    log_visit("/")
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Atlas Chat - AI Chatbot API</title>
<meta name="description" content="Atlas Chat is an AI-powered chatbot API. Two AI models working together to give you the best answer. Free tier with 100 requests/month. Try the live demo.">
<meta name="keywords" content="AI chatbot, chatbot API, AI API, ensemble AI, GPT, artificial intelligence">
<meta property="og:title" content="Atlas Chat - AI Chatbot API">
<meta property="og:description" content="Two AI models working together. Free tier, live demo, simple API.">
<meta property="og:type" content="website">
<meta name="twitter:card" content="summary">
<meta name="twitter:title" content="Atlas Chat - AI Chatbot API">
<meta name="twitter:description" content="Two AI models working together. Free tier, live demo, simple API.">
<style>
  body { font-family: system-ui, sans-serif; max-width: 700px; margin: 3rem auto; padding: 0 1.5rem; line-height: 1.6; color: #1a1a1a; }
  h1 { font-size: 2.2rem; margin-bottom: 0.25rem; }
  p.tagline { font-size: 1.2rem; color: #444; margin-top: 0; }
  .cta { display: inline-block; margin: 1rem 0.5rem 0 0; padding: 0.7rem 1.6rem; font-size: 1rem; text-decoration: none; border-radius: 8px; }
  .cta-primary { background: #1a73e8; color: #fff; }
  .cta-secondary { border: 1px solid #ccc; color: #1a1a1a; }
  .features { margin-top: 2rem; }
  .features li { margin-bottom: 0.5rem; }
  footer { margin-top: 3rem; font-size: 0.85rem; color: #888; }
</style>
</head>
<body>
  <h1>Atlas Chat</h1>
  <p class="tagline">Two AI minds. One best answer.</p>
  <p>Atlas Chat is an AI chatbot API powered by an ensemble of two large language models. Ask once, get the smartest answer — with a free tier to get you started.</p>
  <a class="cta cta-primary" href="/demo">Try the Live Demo</a>
  <a class="cta cta-secondary" href="/docs">API Docs</a>
  <a class="cta cta-secondary" href="/pricing">Pricing</a>
  <a class="cta cta-secondary" href="/menu">Vern's Pizza Menu</a>
  <a class="cta cta-secondary" href="/order">Order Pickup — Vern's Pizza</a>
  <div class="features">
    <h2>Why Atlas Chat?</h2>
    <ul>
      <li><strong>Ensemble intelligence</strong> — two models vote on the best answer</li>
      <li><strong>Simple API</strong> — one POST request, JSON in and out</li>
      <li><strong>Free tier</strong> — 100 requests/month, no credit card</li>
      <li><strong>Paid plans</strong> — Starter $2.99/mo, Pro $9.99/mo</li>
    </ul>
  </div>
  <footer>Atlas Chat API &middot; <a href="/docs">Docs</a> &middot; <a href="/pricing">Pricing</a> &middot; <a href="/menu">Vern's Pizza Menu</a> &middot; <a href="/order">Order Pickup</a></footer>
</body>
</html>
    """

# Vern's Pizza menu (display only — no ordering, cart, or checkout)
_MENU_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verns_menu.json")
_MENU_CACHE = None


def load_menu():
    """Load the Vern's Pizza menu JSON (cached after first read)"""
    global _MENU_CACHE
    if _MENU_CACHE is None:
        try:
            with open(_MENU_PATH, "r", encoding="utf-8") as f:
                _MENU_CACHE = json.load(f)
        except Exception:
            _MENU_CACHE = {}
    return _MENU_CACHE


def _money(value):
    """Format a price as $X.XX, or a dash when that size is not offered"""
    if value is None:
        return "&mdash;"
    return f"${value:.2f}"


def _size_label(key):
    """Human-readable label for a price-map key (reg/lg/each/12in/3/5/8...)"""
    if key == "reg":
        return "Reg"
    if key == "lg":
        return "Lg"
    if key == "each":
        return "Each"
    if key.endswith("in"):
        return f'{key[:-2]}"'
    return f"{key} pc"


def _price_text(price):
    """Render a price that is a number or a {size: price} map"""
    if isinstance(price, dict):
        def sort_key(kv):
            k = kv[0]
            if k in ("reg", "each"):
                return (0, k)
            if k == "lg":
                return (1, k)
            if k.endswith("in"):
                return (2, int(k[:-2]))
            return (3, int(k))
        parts = [f"{_size_label(k)} {_money(v)}" for k, v in sorted(price.items(), key=sort_key)]
        return " / ".join(parts)
    return _money(price)


def _pizza_table(pizzas):
    """Render one pizza category as a table with Original Thick / Vern's Lite prices"""
    rows = []
    for p in pizzas:
        thick = " / ".join(_money(p["original_thick"].get(s)) for s in ("10", "12", "15", "18"))
        lite = " / ".join(_money(p["verns_lite"].get(s)) for s in ("10", "12", "15"))
        rows.append(
            f"<tr><td><strong>{html.escape(p['name'])}</strong><br>"
            f"<span class=\"tops\">{html.escape(p['toppings'])}</span></td>"
            f"<td class=\"price\">{thick}</td><td class=\"price\">{lite}</td></tr>"
        )
    return (
        "<div class=\"tbl-wrap\"><table><thead><tr><th>Pizza</th>"
        "<th>Original Thick<br><span class=\"sz\">10&Prime; / 12&Prime; / 15&Prime; / 18&Prime;</span></th>"
        "<th>Vern&rsquo;s Lite<br><span class=\"sz\">10&Prime; / 12&Prime; / 15&Prime;</span></th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _simple_list(items):
    """Render a list of {name, price, note?, items?} as name — price rows"""
    rows = []
    for it in items:
        line = f"<strong>{html.escape(it['name'])}</strong>"
        if it.get("items"):
            line += f"<br><span class=\"tops\">{html.escape(it['items'])}</span>"
        if it.get("note"):
            line += f"<br><span class=\"tops\">{html.escape(it['note'])}</span>"
        rows.append(f"<li><span>{line}</span><span class=\"price\">{_price_text(it['price'])}</span></li>")
    return "<ul class=\"menu-list\">" + "".join(rows) + "</ul>"


@app.route("/menu", methods=["GET"])
def menu():
    """Vern's Pizza menu page (display only — no ordering or checkout)"""
    log_visit("/menu")
    m = load_menu()
    if not m:
        return "Menu is unavailable right now. Please try again later.", 503
    store = m["store"]
    hours = store["hours"]
    sizes = m["sizes"]
    ot_legend = " &middot; ".join(f'{s}&Prime; ({html.escape(v)})' for s, v in sizes["original_thick"].items())
    vl_legend = " &middot; ".join(f'{s}&Prime; ({html.escape(v)})' for s, v in sizes["verns_lite"].items())
    phone_digits = store["phone"].replace("-", "").replace(" ", "")
    body = (
        "<h1>Vern&rsquo;s Pizza &mdash; Menu</h1>"
        f"<p class=\"store\">{html.escape(store['name'])} &middot; {html.escape(store['address'])}<br>"
        f"Phone: <a href=\"tel:{phone_digits}\">{html.escape(store['phone'])}</a> &middot; "
        f"Order online: <a href=\"https://{html.escape(store['order_online'])}\">{html.escape(store['order_online'])}</a></p>"
        "<p><a class=\"order-btn\" href=\"/order\">Order for Pickup &mdash; pay at the store</a></p>"
        "<p class=\"hours\">"
        f"Mon&ndash;Tue: {html.escape(hours['mon_tue'])}<br>"
        f"Wed&ndash;Sat: {html.escape(hours['wed_sat'])}<br>"
        f"Sun &amp; Holidays: {html.escape(hours['sun_holidays'])}</p>"
        f"<p>{html.escape(store['notes'])}</p>"
        "<h2>Pizza sizes</h2>"
        f"<p><strong>Original Thick:</strong> {ot_legend}<br>"
        f"<strong>Vern&rsquo;s Lite:</strong> {vl_legend}</p>"
        f"<p>{html.escape(m['customize_note'])}</p>"
        "<h2>Traditional Pizzas</h2>" + _pizza_table(m["pizzas"]["traditional"]) +
        "<h2>Gourmet Pizzas</h2>" + _pizza_table(m["pizzas"]["gourmet"]) +
        "<h2>Classic Pizzas</h2>" + _pizza_table(m["pizzas"]["classic"]) +
        "<h2>Snacks &amp; Appetizers</h2>" + _simple_list(m["snacks_appetizers"]) +
        "<h2>Combos</h2>" + _simple_list(m["combos"]) +
        "<h2>Salads</h2>" + _simple_list(m["salads"]) +
        "<h2>Lasagna</h2>" + _simple_list(m["lasagna"]) +
        "<h2>Desserts</h2>" + _simple_list(m["desserts"]) +
        f"<h2>Beverages</h2><p>{html.escape(m['beverages'])}</p>"
        "<p><a class=\"order-btn\" href=\"/order\">Order for Pickup &mdash; pay at the store</a></p>"
    )
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>Vern&rsquo;s Pizza Menu &mdash; Atlas Chat</title>\n"
        "<meta name=\"description\" content=\"Vern&rsquo;s Pizza menu in Calgary: traditional, gourmet and classic pizzas, snacks, combos, salads, lasagna and desserts. 575 28th Street S.E., 403-508-0220.\">\n"
        "<style>\n"
        "  body { font-family: system-ui, sans-serif; max-width: 700px; margin: 3rem auto; padding: 0 1.5rem; line-height: 1.6; color: #1a1a1a; }\n"
        "  h1 { font-size: 2rem; margin-bottom: 0.25rem; }\n"
        "  h2 { margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.25rem; }\n"
        "  a { color: #1a73e8; }\n"
        "  .store { font-size: 1.05rem; }\n"
        "  .hours { color: #444; }\n"
        "  .tbl-wrap { overflow-x: auto; }\n"
        "  table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }\n"
        "  th, td { text-align: left; padding: 0.45rem 0.4rem; border-bottom: 1px solid #eee; vertical-align: top; }\n"
        "  th { white-space: nowrap; }\n"
        "  .sz { font-weight: normal; font-size: 0.8rem; color: #666; }\n"
        "  .tops { color: #555; font-size: 0.85rem; }\n"
        "  .price { white-space: nowrap; }\n"
        "  .menu-list { list-style: none; padding: 0; }\n"
        "  .menu-list li { display: flex; justify-content: space-between; gap: 1rem; padding: 0.45rem 0; border-bottom: 1px solid #eee; }\n"
        "  .order-btn { display: inline-block; padding: 0.7rem 1.4rem; font-size: 1.05rem; font-weight: 700; background: #1a73e8; color: #fff !important; text-decoration: none; border-radius: 8px; }\n"
        "  footer { margin-top: 3rem; font-size: 0.85rem; color: #888; }\n"
        "</style>\n</head>\n<body>\n"
        + body +
        "\n  <footer>Atlas Chat &middot; <a href=\"/\">Home</a> &middot; <a href=\"/demo\">Demo</a> &middot; <a href=\"/pricing\">Pricing</a><br>"
        "Menu reproduced with permission from Vern&rsquo;s Pizza, Calgary.</footer>\n"
        "</body>\n</html>\n"
    )

# ---------------------------------------------------------------------------
# Vern's Pizza pickup ordering — pay at the store, no payment code anywhere.
# Prices are always recomputed server-side from verns_menu.json; anything the
# customer's browser sends as a total is ignored.
# Honest limit: orders live in the local SQLite file, which Render's free
# plan wipes on redeploy/restart. An order's useful life is hours (placed
# today, picked up today) — the kitchen page is the source of truth during
# the day, not the database. Do not promise orders are kept forever.
# ---------------------------------------------------------------------------

_CALGARY_TZ = ZoneInfo("America/Edmonton")

# Store hours, from verns_menu.json (store.hours):
# Mon-Tue 10AM-10PM, Wed-Sat 10AM-11PM, Sun 11AM-10PM (holidays: Sunday hours)
_STORE_HOURS = {
    0: (10, 22),  # Monday
    1: (10, 22),  # Tuesday
    2: (10, 23),  # Wednesday
    3: (10, 23),  # Thursday
    4: (10, 23),  # Friday
    5: (10, 23),  # Saturday
    6: (11, 22),  # Sunday
}
# The kitchen needs at least 45 minutes; the app quotes "45 minutes to an hour"
_PICKUP_LEAD_MINUTES = 45

# Delivery: the fee depends on how far the customer is from the store —
# $1.00 per kilometre on top of the $3.50 base, plus the $1.00 card-machine
# fee. The driver takes the card machine to the door, so delivery is always
# paid by card on arrival — no payment is taken in the app.
#
# STARTING FIGURES — Klen and Lili can change any number here.
# Over 25 km (e.g. Airdrie / Okotoks) the delivery part caps at $10.00 and
# the store calls to confirm the fee — the app never invents a higher
# out-of-town fee.
_DELIVERY_FEE = 3.50
_CARD_FEE = 1.00
_DELIVERY_PER_KM = 1.00
_DELIVERY_FEE_CAP = 10.00
_DELIVERY_CAP_KM = 25.0

# Vern's Pizza, #127, 575 28th Street SE, Calgary — the point delivery
# distance is measured from.
_STORE_LAT = 51.0475
_STORE_LON = -113.9905

_GEOCODE_CACHE = {}
_GEOCODE_LOCK = Lock()


def _haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line distance in km between two lat/lon points."""
    radius = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _delivery_fee_for_distance(distance_km):
    """Delivery fee (before the card-machine fee) for a distance in km:
    $3.50 base + $1.00 per km. Past 25 km (Airdrie / Okotoks) the fee caps
    at $10.00 — the store confirms the real fee by phone instead."""
    if distance_km > _DELIVERY_CAP_KM:
        return _DELIVERY_FEE_CAP
    return round(_DELIVERY_FEE + _DELIVERY_PER_KM * distance_km, 2)


def _geocode_address(address):
    """(lat, lon) for an address via OpenStreetMap Nominatim, or None.
    Free, no key. A short timeout and a cache keep it from slowing orders;
    anything that goes wrong returns None — the caller falls back to the
    flat fee and never fails the order because of geocoding."""
    key = (address or "").strip().lower()
    if not key:
        return None
    with _GEOCODE_LOCK:
        if key in _GEOCODE_CACHE:
            return _GEOCODE_CACHE[key]
    result = _geocode_query(address.strip())
    if result is None:
        # Nominatim often misses spelled-out street types ("28th Street SE");
        # retry once with the usual abbreviations ("28th St SE").
        short = _abbreviate_address(address.strip())
        if short != address.strip():
            result = _geocode_query(short)
    with _GEOCODE_LOCK:
        _GEOCODE_CACHE[key] = result
    return result


_ABBREVIATIONS = [
    ("street", "St"), ("avenue", "Ave"), ("boulevard", "Blvd"),
    ("drive", "Dr"), ("northwest", "NW"), ("northeast", "NE"),
    ("southwest", "SW"), ("southeast", "SE"),
]


def _abbreviate_address(address):
    """'575 28th Street SE' -> '575 28 St SE' (word-boundary matches).
    Nominatim also misses ordinal suffixes ('28th'), so those go too."""
    out = address
    for long, short in _ABBREVIATIONS:
        out = re.sub(r"\b" + long + r"\b", short, out, flags=re.IGNORECASE)
    out = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", out, flags=re.IGNORECASE)
    return out


def _geocode_query(query_text):
    """One Nominatim lookup; (lat, lon) or None."""
    try:
        query = urllib.parse.urlencode({
            "q": query_text + ", Alberta, Canada",
            "format": "json",
            "limit": 1,
        })
        req = urllib.request.Request(
            "https://nominatim.openstreetmap.org/search?" + query,
            headers={"User-Agent": "ensemble-chat-verms/1.0 (Vern's Pizza ordering)"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data:
            return (float(data[0]["lat"]), float(data[0]["lon"]))
    except Exception:
        return None
    return None


def _delivery_fee_for_address(address):
    """(delivery_fee, distance_km or None, note) for a delivery address.
    delivery_fee already includes the $1.00 card-machine fee. When the
    address cannot be geocoded, the flat $3.50 + $1.00 fee applies and the
    note says the fee is confirmed by phone."""
    latlon = _geocode_address(address)
    if not latlon:
        return round(_DELIVERY_FEE + _CARD_FEE, 2), None, "fee confirmed by phone"
    distance = round(_haversine_km(_STORE_LAT, _STORE_LON, latlon[0], latlon[1]), 1)
    fee = round(_delivery_fee_for_distance(distance) + _CARD_FEE, 2)
    if distance > 25:
        return fee, distance, "Airdrie / Okotoks — the store will call to confirm the fee"
    return fee, distance, ""

# Shared kitchen access code, set as an env var in Render (KITCHEN_CODE).
# No default code — if it is not set, the kitchen page says so instead of
# letting anyone in.
KITCHEN_CODE = os.getenv("KITCHEN_CODE")


def _calgary_now():
    """Current time in Calgary (America/Edmonton)"""
    return datetime.now(_CALGARY_TZ)


def _store_hours_today(now=None):
    """(open, close) datetimes for the store today, Calgary time"""
    now = now or _calgary_now()
    open_h, close_h = _STORE_HOURS[now.weekday()]
    open_dt = now.replace(hour=open_h, minute=0, second=0, microsecond=0)
    close_dt = now.replace(hour=close_h, minute=0, second=0, microsecond=0)
    return open_dt, close_dt


def _clock_label(hhmm):
    """'18:00' -> '6:00 PM'"""
    try:
        h, m = (int(x) for x in hhmm.split(":"))
    except Exception:
        return hhmm
    ampm = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {ampm}"


def _day_word(date_str, today_str):
    """'Today', 'Tomorrow', or the weekday name for a YYYY-MM-DD date,
    relative to the YYYY-MM-DD day it is being shown on."""
    if date_str == today_str:
        return "Today"
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        t = datetime.strptime(today_str, "%Y-%m-%d").date()
    except ValueError:
        return ""
    if (d - t).days == 1:
        return "Tomorrow"
    return d.strftime("%A")


def _slot_value(t):
    """A slot datetime as 'YYYY-MM-DD HH:MM' — the date rides along so the
    server knows which day the slot is on."""
    return t.strftime("%Y-%m-%d %H:%M")


def _slots_for_day(day, earliest):
    """15-minute slots on `day` (a datetime that day), from `earliest`
    rounded up to the next 15 minutes, up to closing time."""
    open_dt, close_dt = _store_hours_today(day)
    minutes = earliest.minute
    rounded = earliest.replace(minute=0, second=0, microsecond=0) + timedelta(
        minutes=((minutes + 14) // 15) * 15
    )
    slots = []
    # The kitchen needs 45 minutes from opening, so before opening the
    # first bookable slot is opening+45min, not opening itself.
    t = max(rounded, open_dt + timedelta(minutes=_PICKUP_LEAD_MINUTES))
    while t <= close_dt:
        slots.append(t)
        t += timedelta(minutes=15)
    return slots


def _pickup_slots(now=None):
    """Pickup/delivery slots as (value, label) pairs. Today's remaining
    slots when there are any, otherwise the next open day's full day
    (opening+45min to closing). Values carry the date ('2026-11-26 10:00');
    labels name the day ('Today 6:00 PM', 'Tomorrow 10:00 AM')."""
    now = now or _calgary_now()
    today = now.strftime("%Y-%m-%d")
    slots = _slots_for_day(now, now + timedelta(minutes=_PICKUP_LEAD_MINUTES))
    if slots:
        return [(_slot_value(t), "Today " + _clock_label(t.strftime("%H:%M"))) for t in slots]
    next_day = now + timedelta(days=1)
    open_dt, _ = _store_hours_today(next_day)
    slots = _slots_for_day(next_day, open_dt + timedelta(minutes=_PICKUP_LEAD_MINUTES))
    return [
        (_slot_value(t), _day_word(t.strftime("%Y-%m-%d"), today) + " " + _clock_label(t.strftime("%H:%M")))
        for t in slots
    ]


def _asap_ready(now=None):
    """When the kitchen can have an ASAP order ready. Open now and
    now+45min before closing: today at now+45min. Before opening: today at
    opening+45min. After closing (or now+45min past closing): the next open
    day at opening+45min. There is always a next opening, so ASAP never
    fails for being too early or too late."""
    now = now or _calgary_now()
    open_dt, close_dt = _store_hours_today(now)
    ready = now + timedelta(minutes=_PICKUP_LEAD_MINUTES)
    if now < open_dt:
        return open_dt + timedelta(minutes=_PICKUP_LEAD_MINUTES)
    if ready <= close_dt:
        return ready
    next_day = now + timedelta(days=1)
    next_open, _ = _store_hours_today(next_day)
    return next_open + timedelta(minutes=_PICKUP_LEAD_MINUTES)


def _next_open_dt(now=None):
    """Next time the kitchen opens: today's opening if it has not passed,
    otherwise tomorrow's opening."""
    now = now or _calgary_now()
    open_dt, _ = _store_hours_today(now)
    if now < open_dt:
        return open_dt
    next_day = now + timedelta(days=1)
    next_open, _ = _store_hours_today(next_day)
    return next_open


_PIZZA_GROUPS = ("traditional", "gourmet", "classic")
_MENU_SECTIONS = {
    "snack": "snacks_appetizers",
    "combo": "combos",
    "salad": "salads",
    "lasagna": "lasagna",
    "dessert": "desserts",
}


def _lookup_price(menu, category, name, variant):
    """Server-side price authority. Returns (unit_price, detail) or (None, None).

    Pizza variant is 'original_thick:12' or 'verns_lite:15'.
    Other items: variant is the price-map key ('reg', 'lg', '3', '5', '8',
    '12in', '15in', 'each') or '' when the price is a single number.
    """
    if category == "pizza":
        try:
            crust, size = variant.split(":", 1)
        except (ValueError, AttributeError):
            return None, None
        if crust not in ("original_thick", "verns_lite"):
            return None, None
        for group in _PIZZA_GROUPS:
            for p in menu.get("pizzas", {}).get(group, []):
                if p["name"] == name:
                    price = p.get(crust, {}).get(size)
                    if price is None:
                        return None, None
                    crust_label = "Original Thick" if crust == "original_thick" else "Vern's Lite"
                    return price, f'{crust_label} {size}"'
        return None, None
    section = _MENU_SECTIONS.get(category)
    if section is None:
        return None, None
    for it in menu.get(section, []):
        if it["name"] == name:
            price = it["price"]
            if isinstance(price, dict):
                unit = price.get(variant)
                if unit is None:
                    return None, None
                return unit, _size_label(variant)
            if variant:
                return None, None
            return price, ""
    return None, None


def create_order(customer_name, customer_phone, pickup_type, pickup_time, items,
                 order_type="pickup", delivery_address="", order_date=None):
    """Price the items from verns_menu.json and insert the order + its lines.
    Returns (order_dict, None) on success, (None, error_message) on failure.
    The client-sent total (if any) is never used. Delivery orders are charged
    by distance from the store (_delivery_fee_for_address) — recomputed here,
    never taken from the browser. order_date is the READY day (today unless the
    caller worked out a later ready day); order numbers run per ready day."""
    menu = load_menu()
    if not menu:
        return None, "The menu is unavailable right now."
    lines = []
    subtotal = 0.0
    for item in items:
        if not isinstance(item, dict):
            return None, "Invalid item."
        try:
            qty = int(item.get("quantity", 0))
        except (TypeError, ValueError):
            return None, "Invalid quantity."
        if qty < 1 or qty > 20:
            return None, "Quantity must be between 1 and 20."
        unit, detail = _lookup_price(
            menu, item.get("category", ""), item.get("name", ""), item.get("variant", "")
        )
        if unit is None:
            return None, f"Sorry, that item is not on the menu: {item.get('name', '')}"
        lines.append({
            "category": item["category"],
            "name": item["name"],
            "detail": detail,
            "unit_price": unit,
            "quantity": qty,
            "line_total": round(unit * qty, 2),
        })
        subtotal += unit * qty
    if not lines:
        return None, "Your order is empty."
    subtotal = round(subtotal, 2)
    if order_type == "delivery":
        delivery_fee, delivery_distance, _fee_note = _delivery_fee_for_address(delivery_address)
    else:
        order_type = "pickup"
        delivery_fee = 0.0
        delivery_address = ""
        delivery_distance = None
    order_date = order_date or _calgary_now().strftime("%Y-%m-%d")
    conn = sqlite3.connect("ensemble_chat.db")
    try:
        # Daily order numbers V-001, V-002...; order_number is UNIQUE, so a
        # same-second collision rolls back and retries with the next number.
        for _ in range(10):
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM orders WHERE order_date=?", (order_date,))
            n = (c.fetchone()[0] or 0) + 1
            number = f"V-{n:03d}"
            try:
                c.execute(
                    "INSERT INTO orders (order_number, order_date, customer_name,"
                    " customer_phone, pickup_type, pickup_time, items_json, subtotal,"
                    " status, order_type, delivery_address, delivery_fee,"
                    " delivery_distance_km)"
                    " VALUES (?,?,?,?,?,?,?,?, 'new', ?, ?, ?, ?)",
                    (number, order_date, customer_name, customer_phone,
                     pickup_type, pickup_time, json.dumps(lines), subtotal,
                     order_type, delivery_address, delivery_fee, delivery_distance),
                )
                order_id = c.lastrowid
                for ln in lines:
                    c.execute(
                        "INSERT INTO order_items (order_id, category, name, detail,"
                        " unit_price, quantity, line_total) VALUES (?,?,?,?,?,?,?)",
                        (order_id, ln["category"], ln["name"], ln["detail"],
                         ln["unit_price"], ln["quantity"], ln["line_total"]),
                    )
                conn.commit()
            except sqlite3.IntegrityError:
                conn.rollback()
                continue
            return {
                "order_number": number, "order_date": order_date,
                "customer_name": customer_name, "customer_phone": customer_phone,
                "pickup_type": pickup_type, "pickup_time": pickup_time,
                "items": lines, "subtotal": subtotal, "status": "new",
                "order_type": order_type, "delivery_address": delivery_address,
                "delivery_fee": delivery_fee,
                "delivery_distance_km": delivery_distance,
                "total": round(subtotal + delivery_fee, 2),
            }, None
        return None, "Could not place your order — please try again."
    finally:
        conn.close()


def get_order(order_number):
    """One order (with its items) by order number, or None. Order numbers
    repeat each day, so the newest match wins — that is the order just
    placed, which is what the confirmation page is showing."""
    conn = sqlite3.connect("ensemble_chat.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM orders WHERE order_number=? ORDER BY id DESC LIMIT 1", (order_number,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    order = dict(row)
    try:
        order["items"] = json.loads(order["items_json"])
    except Exception:
        order["items"] = []
    return order


def get_todays_orders():
    """Today's orders (Calgary day), newest first"""
    order_date = _calgary_now().strftime("%Y-%m-%d")
    conn = sqlite3.connect("ensemble_chat.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM orders WHERE order_date=? ORDER BY id DESC", (order_date,))
    orders = [dict(r) for r in c.fetchall()]
    conn.close()
    for order in orders:
        try:
            order["items"] = json.loads(order["items_json"])
        except Exception:
            order["items"] = []
    return orders


def update_order_status(order_number, new_status):
    """new -> ready -> picked_up (new -> picked_up is also allowed).
    Returns False for unknown orders, bad statuses, or already-picked-up orders.
    The kitchen only ever acts on today's orders, so the update is scoped to
    today's ready day — order numbers repeat each day, and a pre-order for
    tomorrow must never be touched by today's kitchen screen."""
    if new_status not in ("ready", "picked_up"):
        return False
    today = _calgary_now().strftime("%Y-%m-%d")
    conn = sqlite3.connect("ensemble_chat.db")
    c = conn.cursor()
    c.execute("SELECT status FROM orders WHERE order_number=? AND order_date=?", (order_number, today))
    row = c.fetchone()
    if not row or row[0] == "picked_up":
        conn.close()
        return False
    c.execute("UPDATE orders SET status=? WHERE order_number=? AND order_date=?", (new_status, order_number, today))
    conn.commit()
    conn.close()
    return True


def _order_page_head(title, extra_css=""):
    """Shared page shell for the ordering pages (same style as /menu)"""
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{title} &mdash; Vern&rsquo;s Pizza</title>\n<style>\n"
        "  body { font-family: system-ui, sans-serif; max-width: 700px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; color: #1a1a1a; }\n"
        "  h1 { font-size: 1.8rem; margin-bottom: 0.25rem; }\n"
        "  h2 { margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.25rem; }\n"
        "  a { color: #1a73e8; }\n"
        "  .tops { color: #555; font-size: 0.85rem; }\n"
        "  .price { white-space: nowrap; }\n"
        "  .item { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; padding: 0.6rem 0; border-bottom: 1px solid #eee; }\n"
        "  .iname { flex: 1 1 200px; }\n"
        "  select, input[type=text], input[type=tel] { font-size: 1rem; padding: 0.45rem; box-sizing: border-box; }\n"
        "  .stepper { display: inline-flex; align-items: center; gap: 0.4rem; }\n"
        "  .stepper button { width: 2.2rem; height: 2.2rem; font-size: 1.2rem; cursor: pointer; border: 1px solid #ccc; background: #fff; border-radius: 6px; }\n"
        "  .stepper .qty { min-width: 1.4rem; text-align: center; font-weight: 700; font-size: 1.1rem; }\n"
        "  .line-total { min-width: 4.2rem; text-align: right; font-weight: 700; }\n"
        "  .subtotal-bar { position: sticky; bottom: 0; background: #fff; border-top: 2px solid #1a1a1a; padding: 0.8rem 0; display: flex; justify-content: space-between; align-items: center; font-size: 1.2rem; font-weight: 700; }\n"
        "  .big-btn { display: block; width: 100%; padding: 0.9rem; font-size: 1.15rem; font-weight: 700; background: #1a73e8; color: #fff; border: none; border-radius: 8px; cursor: pointer; margin-top: 1rem; }\n"
        "  .big-btn:disabled { background: #999; cursor: wait; }\n"
        "  .order-btn { display: inline-block; padding: 0.7rem 1.4rem; font-size: 1.05rem; font-weight: 700; background: #1a73e8; color: #fff !important; text-decoration: none; border-radius: 8px; }\n"
        "  label { font-weight: 600; display: block; margin-top: 1rem; }\n"
        "  label input, label select { display: block; width: 100%; margin-top: 0.25rem; }\n"
        "  .radio-row { margin-top: 0.5rem; font-weight: 400; }\n"
        "  .radio-row input { margin-right: 0.4rem; }\n"
        "  .error { display: none; margin-top: 1rem; padding: 0.8rem; background: #fdecea; border: 1px solid #e57373; border-radius: 8px; color: #7f1d1d; }\n"
        "  .note { color: #555; font-size: 0.95rem; }\n"
        "  .big-number { font-size: 2.6rem; font-weight: 800; margin: 0.5rem 0; }\n"
        "  .pay-note { background: #e8f5e9; border: 1px solid #81c784; border-radius: 8px; padding: 0.9rem; font-size: 1.1rem; font-weight: 700; margin-top: 1rem; }\n"
        "  .order-card { border: 2px solid #333; border-radius: 10px; padding: 1rem; margin-bottom: 1.2rem; font-size: 1.15rem; }\n"
        "  .order-card .num { font-size: 1.8rem; font-weight: 800; }\n"
        "  .badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 20px; font-size: 0.9rem; font-weight: 700; }\n"
        "  .badge-new { background: #fff3cd; color: #664d03; }\n"
        "  .badge-ready { background: #d1e7dd; color: #0f5132; }\n"
        "  .badge-picked_up { background: #e2e3e5; color: #41464b; }\n"
        "  .kbtn { padding: 0.6rem 1.2rem; font-size: 1.05rem; font-weight: 700; border-radius: 8px; border: 1px solid #333; background: #fff; cursor: pointer; margin: 0.3rem 0.3rem 0 0; }\n"
        "  .kbtn-ready { background: #198754; color: #fff; border-color: #198754; }\n"
        "  .kbtn-picked { background: #333; color: #fff; }\n"
        "  footer { margin-top: 3rem; font-size: 0.85rem; color: #888; }\n"
        + extra_css +
        "</style>\n</head>\n<body>\n"
    )


def _order_page_foot():
    return (
        "\n  <footer>Atlas Chat &middot; <a href=\"/\">Home</a> &middot; <a href=\"/menu\">Menu</a> &middot; <a href=\"/order\">Order for Pickup</a><br>"
        "Menu reproduced with permission from Vern&rsquo;s Pizza, Calgary.</footer>\n"
        "</body>\n</html>\n"
    )


def _pizza_variant_options(pizza):
    """<option> list for one pizza: every crust+size that has a price"""
    opts = []
    for crust, label, sizes in (
        ("original_thick", "Original Thick", ("10", "12", "15", "18")),
        ("verns_lite", "Vern's Lite", ("10", "12", "15")),
    ):
        for size in sizes:
            price = pizza[crust].get(size)
            if price is None:
                continue
            opts.append(
                f"<option value=\"{crust}:{size}\" data-price=\"{price:.2f}\">"
                f"{label} {size}&Prime; &mdash; {_money(price)}</option>"
            )
    return "".join(opts)


def _variant_options(price):
    """<option> list for a {size: price} map, in menu order (Reg, Lg, sizes...)"""
    def sort_key(kv):
        k = kv[0]
        if k in ("reg", "each"):
            return (0, k)
        if k == "lg":
            return (1, k)
        if k.endswith("in"):
            return (2, int(k[:-2]))
        return (3, int(k))
    opts = []
    for k, v in sorted(price.items(), key=sort_key):
        opts.append(
            f"<option value=\"{k}\" data-price=\"{v:.2f}\">"
            f"{_size_label(k)} &mdash; {_money(v)}</option>"
        )
    return "".join(opts)


def _order_item_row(category, name, price, note=None, toppings=None, is_pizza=False):
    """One orderable row: name + variant/size picker + quantity stepper"""
    line = f"<strong>{html.escape(name)}</strong>"
    if toppings:
        line += f"<br><span class=\"tops\">{html.escape(toppings)}</span>"
    if note:
        line += f"<br><span class=\"tops\">{html.escape(note)}</span>"
    if is_pizza:
        picker = f"<select class=\"variant\" aria-label=\"Crust and size\">{_pizza_variant_options(price)}</select>"
        data_price = ""
    elif isinstance(price, dict):
        picker = f"<select class=\"variant\" aria-label=\"Size\">{_variant_options(price)}</select>"
        data_price = ""
    else:
        picker = f"<span class=\"price\">{_money(price)}</span>"
        data_price = f" data-price=\"{price:.2f}\""
    return (
        f"<div class=\"item\" data-category=\"{category}\" data-name=\"{html.escape(name, quote=True)}\"{data_price}>"
        f"<div class=\"iname\">{line}</div>{picker}"
        "<div class=\"stepper\"><button type=\"button\" class=\"minus\" aria-label=\"Remove one\">&minus;</button>"
        "<span class=\"qty\">0</span>"
        "<button type=\"button\" class=\"plus\" aria-label=\"Add one\">+</button></div>"
        "<div class=\"line-total\">$0.00</div></div>"
    )


@app.route("/order", methods=["GET"])
def order_page():
    """Pickup or delivery order form (phone-first, no account). Orders are
    taken any time — before opening or after closing they are made when the
    kitchen next opens, so the form never closes."""
    log_visit("/order")
    m = load_menu()
    if not m:
        return "Menu is unavailable right now. Please try again later.", 503
    store = m["store"]
    now = _calgary_now()
    open_dt, close_dt = _store_hours_today(now)
    slots = _pickup_slots(now)
    ready_dt = _asap_ready(now)
    phone_digits = store["phone"].replace("-", "").replace(" ", "")
    closed_note = ""
    if now < open_dt or now > close_dt:
        next_open = _next_open_dt(now)
        day_word = "today" if next_open.date() == now.date() else "tomorrow"
        closed_note = (
            f"<p class=\"note\">We&rsquo;re closed right now &mdash; orders placed now are made "
            f"when we open {day_word} at {_clock_label(next_open.strftime('%H:%M'))}.</p>"
            f"<p class=\"note\">Store hours: Mon&ndash;Tue {html.escape(store['hours']['mon_tue'])}, "
            f"Wed&ndash;Sat {html.escape(store['hours']['wed_sat'])}, "
            f"Sun &amp; Holidays {html.escape(store['hours']['sun_holidays'])}.</p>"
        )
    sections = []
    for group, title in (
        ("traditional", "Traditional Pizzas"),
        ("gourmet", "Gourmet Pizzas"),
        ("classic", "Classic Pizzas"),
    ):
        rows = "".join(
            _order_item_row("pizza", p["name"], p, toppings=p["toppings"], is_pizza=True)
            for p in m["pizzas"][group]
        )
        sections.append(f"<h2>{title}</h2>" + rows)
    for key, title in (
        ("snacks_appetizers", "Snacks &amp; Appetizers"),
        ("combos", "Combos"),
        ("salads", "Salads"),
        ("lasagna", "Lasagna"),
        ("desserts", "Desserts"),
    ):
        rows = "".join(
            _order_item_row(
                {"snacks_appetizers": "snack", "combos": "combo", "salads": "salad",
                 "lasagna": "lasagna", "desserts": "dessert"}[key],
                it["name"], it["price"], note=it.get("note"), toppings=it.get("items"),
            )
            for it in m[key]
        )
        sections.append(f"<h2>{title}</h2>" + rows)
    slot_options = "".join(
        f"<option value=\"{v}\">{html.escape(label)}</option>" for v, label in slots
    )
    if open_dt <= now <= close_dt:
        asap_label = "ASAP &mdash; ready in about 45 minutes to an hour"
    else:
        asap_day = _day_word(ready_dt.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"))
        asap_label = (
            f"ASAP &mdash; ready {asap_day.lower() if asap_day in ('Today', 'Tomorrow') else asap_day} "
            f"at {_clock_label(ready_dt.strftime('%H:%M'))}"
        )
    asap_radio = (
        "<label class=\"radio-row\"><input type=\"radio\" name=\"pickup_type\" value=\"asap\" checked> "
        + asap_label + "</label>"
    )
    slots_today = slots[0][0][:10] == now.strftime("%Y-%m-%d")
    sched_radio = (
        "<label class=\"radio-row\"><input type=\"radio\" name=\"pickup_type\" value=\"scheduled\"> "
        f"Pick a time {'today' if slots_today else 'tomorrow'}</label>"
        f"<select id=\"pickup_time\" style=\"width:100%;margin-top:0.25rem;\">{slot_options}</select>"
    )
    body = (
        "<h1>Order — Pickup or Delivery</h1>"
        f"<p class=\"store\">{html.escape(store['name'])} &middot; {html.escape(store['address'])}<br>"
        f"Phone: <a href=\"tel:{phone_digits}\">{html.escape(store['phone'])}</a></p>"
        + closed_note +
        "<p class=\"note\">Pick your items, then tell us your name and phone. "
        "Pickup: pay at the store. Delivery: the driver brings the card machine "
        "&mdash; pay by card at the door. No payment is taken here. "
        "Prices are the menu prices; GST is added at the store till.</p>"
        + "".join(sections) +
        "<h2>Drinks</h2>"
        f"<p class=\"note\">Drinks ({html.escape(m['beverages'])}) are available at the store "
        "&mdash; just ask when you pick up. They are not priced in the app.</p>"
        "<h2>Your details</h2>"
        "<form id=\"order-form\">"
        "<label>How do you want your order?</label>"
        "<label class=\"radio-row\"><input type=\"radio\" name=\"fulfilment\" value=\"pickup\" checked> "
        "Pickup &mdash; pay at the store</label>"
        "<label class=\"radio-row\"><input type=\"radio\" name=\"fulfilment\" value=\"delivery\"> "
        "Delivery &mdash; driver brings the card machine</label>"
        "<div id=\"delivery-info\" class=\"note\" style=\"display:none;\">"
        f"Delivery fee depends on your area &mdash; {_money(_DELIVERY_FEE)} plus "
        f"{_money(_DELIVERY_PER_KM)} per kilometre from the store, "
        f"plus a {_money(_CARD_FEE)} card-machine fee. "
        "About 45 minutes to an hour. Airdrie and Okotoks sometimes &mdash; "
        "the store will call to confirm the fee.</div>"
        "<label>Your name<input type=\"text\" id=\"cust_name\" autocomplete=\"name\" required></label>"
        "<label>Phone number<input type=\"tel\" id=\"cust_phone\" autocomplete=\"tel\" required "
        "placeholder=\"e.g. 403-555-0123\"></label>"
        "<div id=\"delivery-address-block\" style=\"display:none;\">"
        "<label>Delivery address<input type=\"text\" id=\"delivery_address\" "
        "autocomplete=\"street-address\" placeholder=\"e.g. 123 45 Ave SE, Calgary\"></label></div>"
        "<label><span id=\"time-label\">Pickup time</span></label>"
        + asap_radio + sched_radio +
        "<div class=\"error\" id=\"order-error\"></div>"
        "<div id=\"fee-lines\" class=\"note\" style=\"display:none;\">"
        "Delivery fee <span id=\"fee-amount\">$4.50</span> (includes $1.00 card-machine fee)"
        "<span id=\"fee-note\"></span></div>"
        "<div class=\"subtotal-bar\"><span id=\"subtotal-label\">Subtotal</span><span id=\"subtotal\">$0.00</span></div>"
        "<button type=\"submit\" class=\"big-btn\" id=\"place-order\">Place Pickup Order</button>"
        "<p class=\"note\">Pickup: pay at the store. Delivery: pay by card at the door. GST is added at the store.</p>"
        "</form>"
        "<script>\n"
        "function money(n){ return '$' + n.toFixed(2); }\n"
        "function rowPrice(row){\n"
        "  var sel = row.querySelector('select.variant');\n"
        "  if (sel) { return parseFloat(sel.options[sel.selectedIndex].getAttribute('data-price') || '0'); }\n"
        "  return parseFloat(row.getAttribute('data-price') || '0');\n"
        "}\n"
        "var deliveryFee = " + f"{_DELIVERY_FEE + _CARD_FEE:.2f}" + ";\n"
        "var feeTimer = null;\n"
        "function fetchDeliveryFee(){\n"
        "  var addr = document.getElementById('delivery_address').value.trim();\n"
        "  if (fulfilment() !== 'delivery' || addr.length < 5) {\n"
        "    deliveryFee = " + f"{_DELIVERY_FEE + _CARD_FEE:.2f}" + ";\n"
        "    document.getElementById('fee-note').textContent = '';\n"
        "    updateSubtotal();\n"
        "    return;\n"
        "  }\n"
        "  fetch('/api/delivery-fee', {method: 'POST', headers: {'Content-Type': 'application/json'},\n"
        "    body: JSON.stringify({address: addr})})\n"
        "  .then(function(r){ return r.json(); })\n"
        "  .then(function(j){\n"
        "    if (typeof j.fee === 'number') { deliveryFee = j.fee; }\n"
        "    var note = '';\n"
        "    if (j.distance_km !== null && j.distance_km !== undefined) { note = ' (~' + j.distance_km + ' km from the store)'; }\n"
        "    if (j.note) { note += (note ? ' - ' : ' ') + j.note; }\n"
        "    document.getElementById('fee-note').textContent = note;\n"
        "    updateSubtotal();\n"
        "  })\n"
        "  .catch(function(){ updateSubtotal(); });\n"
        "}\n"
        "function fulfilment(){\n"
        "  var r = document.querySelector('input[name=fulfilment]:checked');\n"
        "  return r ? r.value : 'pickup';\n"
        "}\n"
        "function updateSubtotal(){\n"
        "  var total = 0;\n"
        "  document.querySelectorAll('.item').forEach(function(row){\n"
        "    var qty = parseInt(row.querySelector('.qty').textContent, 10) || 0;\n"
        "    var line = qty * rowPrice(row);\n"
        "    row.querySelector('.line-total').textContent = money(line);\n"
        "    total += line;\n"
        "  });\n"
        "  var isDelivery = fulfilment() === 'delivery';\n"
        "  document.getElementById('subtotal').textContent = money(total + (isDelivery ? deliveryFee : 0));\n"
        "  document.getElementById('fee-amount').textContent = money(deliveryFee);\n"
        "  document.getElementById('subtotal-label').textContent = isDelivery ? 'Total' : 'Subtotal';\n"
        "  document.getElementById('fee-lines').style.display = isDelivery ? 'block' : 'none';\n"
        "  document.getElementById('delivery-info').style.display = isDelivery ? 'block' : 'none';\n"
        "  document.getElementById('delivery-address-block').style.display = isDelivery ? 'block' : 'none';\n"
        "  document.getElementById('time-label').textContent = isDelivery ? 'Delivery time' : 'Pickup time';\n"
        "  document.getElementById('place-order').textContent = isDelivery ? 'Place Delivery Order' : 'Place Pickup Order';\n"
        "}\n"
        "document.addEventListener('click', function(e){\n"
        "  var btn = e.target.closest('.plus, .minus');\n"
        "  if (!btn) return;\n"
        "  var qtyEl = btn.parentElement.querySelector('.qty');\n"
        "  var qty = parseInt(qtyEl.textContent, 10) || 0;\n"
        "  qty += btn.classList.contains('plus') ? 1 : -1;\n"
        "  if (qty < 0) qty = 0;\n"
        "  if (qty > 20) qty = 20;\n"
        "  qtyEl.textContent = qty;\n"
        "  updateSubtotal();\n"
        "});\n"
        "document.addEventListener('change', function(e){\n"
        "  if (e.target.classList.contains('variant')) updateSubtotal();\n"
        "  if (e.target.name === 'fulfilment') fetchDeliveryFee();\n"
        "});\n"
        "document.getElementById('delivery_address').addEventListener('input', function(){\n"
        "  clearTimeout(feeTimer);\n"
        "  feeTimer = setTimeout(fetchDeliveryFee, 600);\n"
        "});\n"
        "document.getElementById('order-form').addEventListener('submit', function(e){\n"
        "  e.preventDefault();\n"
        "  var err = document.getElementById('order-error');\n"
        "  err.style.display = 'none';\n"
        "  var items = [];\n"
        "  document.querySelectorAll('.item').forEach(function(row){\n"
        "    var qty = parseInt(row.querySelector('.qty').textContent, 10) || 0;\n"
        "    if (qty <= 0) return;\n"
        "    var sel = row.querySelector('select.variant');\n"
        "    items.push({category: row.getAttribute('data-category'),\n"
        "      name: row.getAttribute('data-name'),\n"
        "      variant: sel ? sel.value : '', quantity: qty});\n"
        "  });\n"
        "  var pickupType = document.querySelector('input[name=pickup_type]:checked');\n"
        "  var fulfil = fulfilment();\n"
        "  var address = document.getElementById('delivery_address').value.trim();\n"
        "  if (fulfil === 'delivery' && address.length < 5) {\n"
        "    err.textContent = 'Please enter your delivery address.';\n"
        "    err.style.display = 'block';\n"
        "    return;\n"
        "  }\n"
        "  var pickupTime = '';\n"
        "  if (pickupType && pickupType.value === 'scheduled') {\n"
        "    var timeSel = document.getElementById('pickup_time');\n"
        "    pickupTime = timeSel ? timeSel.value : '';\n"
        "  }\n"
        "  var btn = document.getElementById('place-order');\n"
        "  btn.disabled = true;\n"
        "  btn.textContent = 'Placing your order...';\n"
        "  fetch('/api/order', {method: 'POST', headers: {'Content-Type': 'application/json'},\n"
        "    body: JSON.stringify({name: document.getElementById('cust_name').value,\n"
        "      phone: document.getElementById('cust_phone').value,\n"
        "      pickup_type: pickupType ? pickupType.value : 'asap',\n"
        "      pickup_time: pickupTime, order_type: fulfil, delivery_address: address, items: items})})\n"
        "  .then(function(r){ return r.json().then(function(j){ return {status: r.status, body: j}; }); })\n"
        "  .then(function(res){\n"
        "    if (res.status === 201) { window.location = '/order/' + res.body.order_number; return; }\n"
        "    err.textContent = res.body.error || 'Something went wrong. Please try again or call the store.';\n"
        "    err.style.display = 'block';\n"
        "    btn.disabled = false;\n"
        "    btn.textContent = fulfil === 'delivery' ? 'Place Delivery Order' : 'Place Pickup Order';\n"
        "  })\n"
        "  .catch(function(){\n"
        "    err.textContent = 'Could not reach the store. Please try again or call the store.';\n"
        "    err.style.display = 'block';\n"
        "    btn.disabled = false;\n"
        "    btn.textContent = fulfil === 'delivery' ? 'Place Delivery Order' : 'Place Pickup Order';\n"
        "  });\n"
        "});\n"
        "updateSubtotal();\n"
        "</script>\n"
    )
    return _order_page_head("Order — Vern's Pizza") + body + _order_page_foot()


@app.route("/api/delivery-fee", methods=["POST"])
def api_delivery_fee():
    """Estimated delivery fee for an address: {"fee", "distance_km", "note"}.
    The fee shown here is an estimate — the order itself recomputes it
    server-side, so any fee the browser sends is ignored."""
    data = request.get_json(silent=True) or {}
    address = (data.get("address") or "").strip()
    if len(address) < 5:
        return jsonify({
            "fee": round(_DELIVERY_FEE + _CARD_FEE, 2),
            "distance_km": None,
            "note": "Enter your full address for an exact fee.",
        })
    fee, distance, note = _delivery_fee_for_address(address)
    return jsonify({"fee": fee, "distance_km": distance, "note": note})


@app.route("/api/order", methods=["POST"])
def api_order():
    """Create a pickup or delivery order. All prices are recomputed
    server-side from verns_menu.json — any total the browser sends is ignored.
    The delivery fee depends on the distance from the store
    (_delivery_fee_for_address) and is recomputed here too.
    Orders are taken any time — ASAP and scheduled times are worked out
    against the next time the kitchen can make them, and the order is stored
    under its ready day so the kitchen sees it on the morning it is made."""
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    phone = (data.get("phone") or "").strip()
    pickup_type = data.get("pickup_type") or "asap"
    pickup_time = (data.get("pickup_time") or "").strip()
    order_type = (data.get("order_type") or "pickup").strip().lower()
    delivery_address = (data.get("delivery_address") or "").strip()
    items = data.get("items") or []
    if order_type not in ("pickup", "delivery"):
        return jsonify({"error": "Please choose pickup or delivery."}), 400
    if order_type == "delivery" and len(delivery_address) < 5:
        return jsonify({"error": "Please enter your delivery address."}), 400
    if not name:
        return jsonify({"error": "Please enter your name."}), 400
    if len(re.sub(r"\D", "", phone)) < 7:
        return jsonify({"error": "Please enter a phone number we can reach you at."}), 400
    if not isinstance(items, list) or not items:
        return jsonify({"error": "Please add at least one item."}), 400
    now = _calgary_now()
    if pickup_type == "asap":
        # Ready when the kitchen next can make it; never rejected for being
        # too early or too late. Stored under its ready day.
        ready_dt = _asap_ready(now)
        pickup_time = ready_dt.strftime("%H:%M")
        order_date = ready_dt.strftime("%Y-%m-%d")
    elif pickup_type == "scheduled":
        slot_values = [v for v, _ in _pickup_slots(now)]
        if pickup_time not in slot_values:
            return jsonify({"error": "Please pick a valid pickup time."}), 400
        order_date, pickup_time = pickup_time.split(" ")
    else:
        return jsonify({"error": "Please choose ASAP or a pickup time."}), 400
    order, err = create_order(name, phone, pickup_type, pickup_time, items,
                              order_type, delivery_address, order_date)
    if err:
        return jsonify({"error": err}), 400
    log_visit("/order")
    return jsonify({
        "order_number": order["order_number"],
        "pickup_type": pickup_type,
        "pickup_time": pickup_time,
        "subtotal": order["subtotal"],
        "order_type": order["order_type"],
        "delivery_address": order["delivery_address"],
        "delivery_fee": order["delivery_fee"],
        "total": order["total"],
    }), 201


_WAIT_CHAT_CSS = (
    "  .wait-chat textarea { width: 100%; min-height: 80px; padding: 0.5rem; font-size: 1rem; box-sizing: border-box; margin-top: 0.25rem; }\n"
    "  .wait-answer { margin-top: 1rem; padding: 1rem; border: 1px solid #ccc; border-radius: 8px; background: #fafafa; white-space: pre-wrap; }\n"
)

_WAIT_CHAT_HTML = """
<h2>Chat while you wait</h2>
<p class="note">Your pizza is being made &mdash; chat with Atlas, our AI, while you wait. No signup, just ask.</p>
<div class="wait-chat">
  <label for="wait-prompt">Ask Atlas anything</label>
  <textarea id="wait-prompt" rows="3" placeholder="e.g. What toppings go well with pepperoni?"></textarea>
  <button type="button" class="big-btn" id="wait-ask">Ask Atlas</button>
  <div id="wait-answer" class="wait-answer" style="display:none;"></div>
  <p class="note">Free while you wait &mdash; a few questions per visit. <a href="/demo">Try the full Atlas Chat demo</a>.</p>
</div>
<script>
document.getElementById('wait-ask').addEventListener('click', function(){
  var box = document.getElementById('wait-prompt');
  var ans = document.getElementById('wait-answer');
  var btn = document.getElementById('wait-ask');
  var q = box.value.trim();
  if (!q) { ans.style.display = 'block'; ans.textContent = 'Type a question first.'; return; }
  btn.disabled = true;
  btn.textContent = 'Atlas is thinking...';
  ans.style.display = 'block';
  ans.textContent = 'Thinking...';
  fetch('/api/wait-chat', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({prompt: q})})
  .then(function(r){ return r.json().then(function(j){ return {status: r.status, body: j}; }); })
  .then(function(res){
    ans.textContent = res.body.answer || res.body.error || 'Something went wrong. Please try again.';
    btn.disabled = false;
    btn.textContent = 'Ask Atlas';
  })
  .catch(function(){
    ans.textContent = 'Could not reach the chat. Please try again.';
    btn.disabled = false;
    btn.textContent = 'Ask Atlas';
  });
});
</script>
"""

@app.route("/order/<order_number>", methods=["GET"])
def order_confirmation(order_number):
    """Confirmation page for one order: number, pickup/delivery time, items,
    subtotal, delivery fee when applicable, and the pay note."""
    order = get_order(order_number)
    if not order:
        return (
            _order_page_head("Order not found — Vern's Pizza")
            + "<h1>Order not found</h1>"
            "<p>We could not find that order number. It may have been placed on a "
            "different day, or the site restarted since it was placed.</p>"
            "<p>Questions? Call the store: <a href=\"tel:4035080220\">403-508-0220</a></p>"
            "<p><a href=\"/order\">Place a new order</a></p>"
            + _order_page_foot()
        ), 404
    store = load_menu().get("store", {})
    store_name = store.get("name", "Vern's Pizza")
    store_addr = store.get("address", "#127, 575 28th Street S.E., Calgary")
    store_phone = store.get("phone", "403-508-0220")
    phone_digits = store_phone.replace("-", "").replace(" ", "")
    is_delivery = (order.get("order_type") or "pickup") == "delivery"
    fulfil_label = "Delivery" if is_delivery else "Pickup"
    today = _calgary_now().strftime("%Y-%m-%d")
    ready_date = order.get("order_date") or today
    day = _day_word(ready_date, today)
    if order["pickup_type"] == "asap":
        if ready_date == today:
            when = (f"ASAP &mdash; ready in about 45 minutes to an hour "
                   f"(around {_clock_label(order['pickup_time'])})")
        else:
            day_phrase = day.lower() if day in ("Today", "Tomorrow") else day
            when = f"ASAP &mdash; ready {day_phrase} at {_clock_label(order['pickup_time'])}"
    else:
        when = f"{day} at {_clock_label(order['pickup_time'])}"
    rows = "".join(
        f"<li><span>{ln['quantity']} &times; {html.escape(ln['name'])}"
        + (f" &mdash; {html.escape(ln['detail'])}" if ln["detail"] else "")
        + f"</span><span class=\"price\">{_money(ln['line_total'])}</span></li>"
        for ln in order["items"]
    )
    if is_delivery:
        fee = order.get("delivery_fee") or 0.0
        distance = order.get("delivery_distance_km")
        distance_html = (f" (~{distance:g} km from the store)" if distance is not None else "")
        totals_html = (
            f"<p><strong>Subtotal: {_money(order['subtotal'])}</strong><br>"
            f"<strong>Delivery fee: {_money(fee)}</strong>{distance_html} "
            f"({_money(round(fee - _CARD_FEE, 2))} delivery + {_money(_CARD_FEE)} card-machine fee)<br>"
            f"<strong>Total: {_money(round(order['subtotal'] + fee, 2))}</strong><br>"
            "<span class=\"note\">GST is added at the store till.</span></p>"
        )
        pay_html = ("<p class=\"pay-note\">Pay by card when the driver arrives "
                    "&mdash; the driver brings the card machine. No payment was taken here.</p>")
        address_html = (f"<p><strong>Deliver to:</strong> "
                        f"{html.escape(order.get('delivery_address') or '')}</p>")
    else:
        totals_html = (
            f"<p><strong>Subtotal: {_money(order['subtotal'])}</strong><br>"
            "<span class=\"note\">GST is added at the store till.</span></p>"
        )
        pay_html = ("<p class=\"pay-note\">Pay at the store when you pick up "
                    "&mdash; no payment was taken here.</p>")
        address_html = ""
    body = (
        "<h1>Order placed!</h1>"
        f"<p>Your order number is:</p><p class=\"big-number\">{html.escape(order['order_number'])}</p>"
        f"<p><strong>{fulfil_label}:</strong> {when}</p>"
        f"<p>We&rsquo;ll call you at {html.escape(order['customer_phone'])} when your order is ready.</p>"
        f"<p><strong>Name:</strong> {html.escape(order['customer_name'])}</p>"
        + address_html +
        "<ul class=\"menu-list\">" + rows + "</ul>"
        + totals_html
        + pay_html +
        f"<p>{html.escape(store_name)} &middot; "
        f"{html.escape(store_addr)}<br>"
        f"Phone: <a href=\"tel:{phone_digits}\">{html.escape(store_phone)}</a> "
        "&mdash; call if anything goes wrong with your order.</p>"
        + _WAIT_CHAT_HTML
        + "<p><a href=\"/menu\">Back to the menu</a></p>"
    )
    return _order_page_head(f"Order {html.escape(order['order_number'])} — Vern's Pizza", extra_css=_WAIT_CHAT_CSS) + body + _order_page_foot()


def _kitchen_login_page(error=None):
    """Access-code gate for the kitchen page"""
    msg = f"<p class=\"error\" style=\"display:block;\">{html.escape(error)}</p>" if error else ""
    return (
        _order_page_head("Kitchen — Vern's Pizza")
        + "<h1>Kitchen</h1>"
        + msg +
        "<p>Enter the kitchen code to see today&rsquo;s orders.</p>"
        "<form method=\"POST\" action=\"/kitchen\">"
        "<label>Kitchen code<input type=\"text\" name=\"code\" inputmode=\"numeric\" "
        "autocomplete=\"off\" required autofocus></label>"
        "<button type=\"submit\" class=\"big-btn\">Open Kitchen</button></form>"
        + _order_page_foot()
    )


def _kitchen_token():
    """Session token proving the code was entered (hash of the code, so a
    guessed 'true' flag in a forged session cookie is not enough)."""
    if not KITCHEN_CODE:
        return None
    return hashlib.sha256(f"kitchen:{KITCHEN_CODE}".encode()).hexdigest()


@app.route("/kitchen", methods=["GET", "POST"])
def kitchen():
    """Today's orders for the store, newest first. Big text, auto-refreshes
    every 30 seconds. Protected by the shared KITCHEN_CODE env var; the code
    is remembered in a session cookie on that device."""
    log_visit("/kitchen")
    if not KITCHEN_CODE:
        return (
            _order_page_head("Kitchen — Vern's Pizza")
            + "<h1>Kitchen</h1>"
            "<p>The kitchen screen is not set up yet &mdash; the owner needs to set "
            "the kitchen access code in the app settings first.</p>"
            + _order_page_foot()
        ), 503
    if request.method == "POST":
        code = (request.form.get("code") or "").strip()
        if code and code == KITCHEN_CODE:
            session["kitchen_token"] = _kitchen_token()
            return redirect(url_for("kitchen"))
        return _kitchen_login_page("That code did not match. Please try again.")
    if session.get("kitchen_token") != _kitchen_token():
        return _kitchen_login_page()
    orders = get_todays_orders()
    if not orders:
        cards = "<p>No orders yet today.</p>"
    else:
        cards = ""
        for o in orders:
            status = o["status"]
            is_delivery = (o.get("order_type") or "pickup") == "delivery"
            if status == "picked_up" and is_delivery:
                badge = "DELIVERED"
            else:
                badge = {"new": "NEW", "ready": "READY", "picked_up": "PICKED UP"}.get(status, status.upper())
            if o["pickup_type"] == "asap":
                when = f"ASAP (ready ~{_clock_label(o['pickup_time'])})"
            else:
                day = _day_word(o.get("order_date") or "", _calgary_now().strftime("%Y-%m-%d"))
                when = f"{day} at {_clock_label(o['pickup_time'])}"
            fulfil_line = f"Delivery: {when}" if is_delivery else f"Pickup: {when}"
            items = "".join(
                f"<li>{ln['quantity']} &times; {html.escape(ln['name'])}"
                + (f" &mdash; {html.escape(ln['detail'])}" if ln["detail"] else "")
                + f" &mdash; {_money(ln['line_total'])}</li>"
                for ln in o["items"]
            )
            buttons = ""
            if status == "new":
                buttons = (
                    f"<button class=\"kbtn kbtn-ready\" data-order=\"{html.escape(o['order_number'])}\" "
                    "data-status=\"ready\">Ready</button>"
                )
            if status in ("new", "ready"):
                buttons += (
                    f"<button class=\"kbtn kbtn-picked\" data-order=\"{html.escape(o['order_number'])}\" "
                    f"data-status=\"picked_up\">{'Delivered' if is_delivery else 'Picked Up'}</button>"
                )
            print_url = "/kitchen/print/" + urllib.parse.quote(o["order_number"], safe="")
            buttons += (
                f"<a class=\"kbtn kbtn-print\" href=\"{print_url}\" "
                "target=\"_blank\" rel=\"noopener\">Print</a>"
            )
            phone_digits = re.sub(r"\D", "", o["customer_phone"])
            delivery_html = ""
            if is_delivery:
                fee = o.get("delivery_fee") or 0.0
                distance = o.get("delivery_distance_km")
                distance_html = (f" &middot; ~{distance:g} km from the store" if distance is not None else "")
                delivery_html = (
                    f"<div><span class=\"badge badge-delivery\">DELIVERY</span> "
                    f"<strong>{html.escape(o.get('delivery_address') or '')}</strong>{distance_html}</div>"
                )
                totals_html = (
                    f"<div><strong>Subtotal: {_money(o['subtotal'])}</strong> &middot; "
                    f"Delivery fee: {_money(fee)} &middot; "
                    f"<strong>Total: {_money(round(o['subtotal'] + fee, 2))}</strong> "
                    "<span class=\"note\">(GST added at the till; pay by card at the door)</span></div>"
                )
            else:
                totals_html = (
                    f"<div><strong>Subtotal: {_money(o['subtotal'])}</strong> "
                    "<span class=\"note\">(GST added at the till)</span></div>"
                )
            cards += (
                f"<div class=\"order-card{' delivery' if is_delivery else ''}\">"
                f"<div><span class=\"num\">{html.escape(o['order_number'])}</span> "
                f"<span class=\"badge badge-{status}\">{badge}</span></div>"
                f"<div><strong>{html.escape(o['customer_name'])}</strong> &middot; "
                f"<a href=\"tel:{html.escape(phone_digits)}\">"
                f"{html.escape(o['customer_phone'])}</a></div>"
                + delivery_html +
                f"<div>{fulfil_line}</div>"
                f"<ul>{items}</ul>"
                + totals_html +
                f"<div>{buttons}</div></div>"
            )
    body = (
        "<h1>Kitchen &mdash; Today&rsquo;s Orders</h1>"
        f"<p class=\"note\">This page refreshes itself every 30 seconds. "
        f"{len(orders)} order(s) today. Orders are kept for today only &mdash; "
        "the site does not store them forever.</p>"
        + cards +
        "<script>\n"
        "document.addEventListener('click', function(e){\n"
        "  var btn = e.target.closest('.kbtn');\n"
        "  if (!btn) return;\n"
        "  btn.disabled = true;\n"
        "  fetch('/api/kitchen/status', {method: 'POST', headers: {'Content-Type': 'application/json'},\n"
        "    body: JSON.stringify({order_number: btn.getAttribute('data-order'),\n"
        "      status: btn.getAttribute('data-status')})})\n"
        "  .then(function(){ window.location.reload(); });\n"
        "});\n"
        "</script>\n"
    )
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<meta http-equiv=\"refresh\" content=\"30\">\n"
        "<title>Kitchen &mdash; Vern&rsquo;s Pizza</title>\n"
        "<style>\n"
        "  body { font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; color: #1a1a1a; }\n"
        "  h1 { font-size: 1.8rem; }\n"
        "  a { color: #1a73e8; }\n"
        "  .note { color: #555; font-size: 0.95rem; }\n"
        "  .order-card { border: 2px solid #333; border-radius: 10px; padding: 1rem; margin-bottom: 1.2rem; font-size: 1.15rem; }\n"
        "  .order-card .num { font-size: 1.8rem; font-weight: 800; }\n"
        "  .badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 20px; font-size: 0.9rem; font-weight: 700; }\n"
        "  .badge-new { background: #fff3cd; color: #664d03; }\n"
        "  .badge-ready { background: #d1e7dd; color: #0f5132; }\n"
        "  .badge-picked_up { background: #e2e3e5; color: #41464b; }\n"
        "  .kbtn { padding: 0.6rem 1.2rem; font-size: 1.05rem; font-weight: 700; border-radius: 8px; border: 1px solid #333; background: #fff; cursor: pointer; margin: 0.3rem 0.3rem 0 0; }\n"
        "  .kbtn-ready { background: #198754; color: #fff; border-color: #198754; }\n"
        "  .kbtn-picked { background: #333; color: #fff; }\n"
        "  .kbtn-print { background: #fff; color: #1a1a1a; text-decoration: none; display: inline-block; }\n"
        "  .badge-delivery { background: #cfe2ff; color: #084298; }\n"
        "  .order-card.delivery { border-color: #1a73e8; }\n"
"</style>\n</head>\n<body>\n" + body + "\n</body>\n</html>\n"
    )


@app.route("/kitchen/print/<order_number>", methods=["GET"])
def kitchen_print(order_number):
    """Printable receipt for one of today's orders. Opens in its own page
    (no 30-second refresh) and brings up the tablet's print dialog, so the
    kitchen can print the ticket for the order on screen. Same code gate
    as the kitchen page."""
    if not KITCHEN_CODE or session.get("kitchen_token") != _kitchen_token():
        return _kitchen_login_page()
    orders = get_todays_orders()
    order = None
    for o in orders:
        if o["order_number"] == order_number:
            order = o
            break
    if order is None:
        return (
            _order_page_head("Order not found — Vern's Pizza")
            + "<h1>Order not found</h1>"
            "<p>That order is not on today's kitchen list any more.</p>"
            + _order_page_foot()
        ), 404
    is_delivery = (order.get("order_type") or "pickup") == "delivery"
    if order["pickup_type"] == "asap":
        when = "ASAP (ready ~" + _clock_label(order["pickup_time"]) + ")"
    else:
        day = _day_word(order.get("order_date") or "", _calgary_now().strftime("%Y-%m-%d"))
        when = day + " at " + _clock_label(order["pickup_time"])
    fulfil_line = "Delivery: " + when if is_delivery else "Pickup: " + when
    items = "".join(
        "<li>" + str(ln["quantity"]) + " x " + html.escape(ln["name"])
        + (" &mdash; " + html.escape(ln["detail"]) if ln["detail"] else "")
        + " &mdash; " + _money(ln["line_total"]) + "</li>"
        for ln in order["items"]
    )
    phone_digits = re.sub(r"\D", "", order["customer_phone"])
    address_html = ""
    if is_delivery:
        fee = order.get("delivery_fee") or 0.0
        distance = order.get("delivery_distance_km")
        distance_html = (" &middot; ~" + format(distance, "g") + " km from the store") if distance is not None else ""
        address_html = (
            "<p><strong>DELIVERY</strong><br>" + html.escape(order.get("delivery_address") or "")
            + distance_html + "</p>"
        )
        totals_html = (
            "<p>Subtotal: " + _money(order["subtotal"]) + "<br>"
            "Delivery fee: " + _money(fee) + "<br>"
            "<strong>Total: " + _money(round(order["subtotal"] + fee, 2)) + "</strong><br>"
            "<span class=\"note\">GST added at the till; pay by card at the door.</span></p>"
        )
    else:
        totals_html = (
            "<p><strong>Subtotal: " + _money(order["subtotal"]) + "</strong><br>"
            "<span class=\"note\">GST added at the till.</span></p>"
        )
    receipt = (
        "<h1>Vern&rsquo;s Pizza</h1>"
        "<p>575 28th Street SE, Calgary &middot; 403-508-0220</p>"
        "<p class=\"big-number\">" + html.escape(order["order_number"]) + "</p>"
        "<p><strong>" + html.escape(order["customer_name"]) + "</strong><br>"
        "<a href=\"tel:" + html.escape(phone_digits) + "\">"
        + html.escape(order["customer_phone"]) + "</a></p>"
        + address_html +
        "<p>" + fulfil_line + "</p>"
        "<ul>" + items + "</ul>"
        + totals_html +
        "<p class=\"note\">Printed " + _calgary_now().strftime("%Y-%m-%d %I:%M %p") + "</p>"
        "<p class=\"no-print\"><button class=\"big-btn\" onclick=\"window.print()\">Print this order</button></p>"
    )
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>Print " + html.escape(order["order_number"]) + " &mdash; Vern&rsquo;s Pizza</title>\n"
        "<style>\n"
        "  body { font-family: system-ui, sans-serif; max-width: 72mm; margin: 1rem auto; padding: 0 0.5rem; line-height: 1.5; color: #1a1a1a; }\n"
        "  h1 { font-size: 1.4rem; margin-bottom: 0; }\n"
        "  .big-number { font-size: 2.4rem; font-weight: 800; margin: 0.3rem 0; }\n"
        "  ul { padding-left: 1.2rem; }\n"
        "  .note { color: #555; font-size: 0.9rem; }\n"
        "  .big-btn { display: block; width: 100%; padding: 0.9rem; font-size: 1.15rem; font-weight: 700; background: #1a73e8; color: #fff; border: none; border-radius: 8px; cursor: pointer; margin-top: 1rem; }\n"
        "  @media print { .no-print { display: none; } body { margin: 0; max-width: none; } }\n"
        "</style>\n</head>\n<body>\n" + receipt +
        "<script>\n"
        "window.addEventListener('load', function(){ setTimeout(function(){ window.print(); }, 400); });\n"
        "</script>\n</body>\n</html>\n"
    )


@app.route("/api/kitchen/status", methods=["POST"])
def api_kitchen_status():
    """Mark an order Ready / Picked Up (kitchen session only)"""
    if not KITCHEN_CODE or session.get("kitchen_token") != _kitchen_token():
        return jsonify({"error": "Not authorized."}), 401
    data = request.get_json(silent=True) or {}
    if not update_order_status(data.get("order_number", ""), data.get("status", "")):
        return jsonify({"error": "Could not update that order."}), 400
    return jsonify({"ok": True})


@app.route("/robots.txt", methods=["GET"])
def robots():
    """Robots.txt for search engine crawlers"""
    return "User-agent: *\nAllow: /\nSitemap: https://ensemble-chat.onrender.com/sitemap.xml\n", 200, {"Content-Type": "text/plain"}

@app.route("/sitemap.xml", methods=["GET"])
def sitemap():
    """Sitemap for search engines"""
    return """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://ensemble-chat.onrender.com/</loc><priority>1.0</priority></url>
  <url><loc>https://ensemble-chat.onrender.com/demo</loc><priority>0.9</priority></url>
  <url><loc>https://ensemble-chat.onrender.com/docs</loc><priority>0.7</priority></url>
  <url><loc>https://ensemble-chat.onrender.com/pricing</loc><priority>0.7</priority></url>
  <url><loc>https://ensemble-chat.onrender.com/menu</loc><priority>0.7</priority></url>
  <url><loc>https://ensemble-chat.onrender.com/order</loc><priority>0.7</priority></url>
</urlset>
""", 200, {"Content-Type": "application/xml"}

# Routes - Web pages
@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Simple visitor stats (public)"""
    return jsonify({
        "total_visits": get_visit_count(),
        "landing_page_visits": get_visit_count("/"),
        "demo_visits": get_visit_count("/demo"),
    })

@app.route("/demo", methods=["GET"])
def demo():
    """Simple demo/chat page: enter API key + prompt, see the ensemble response."""
    log_visit("/demo")
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Atlas Chat - Live Demo | Try the AI Chatbot Free</title>
<meta name="description" content="Try Atlas Chat free — ask our AI ensemble anything. Two AI models, one best answer. No signup needed to see it work.">
<meta property="og:title" content="Atlas Chat - Live Demo">
<meta property="og:description" content="Try the AI chatbot free. Two models, one best answer.">
<style>
  body { font-family: system-ui, sans-serif; max-width: 700px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
  h1 { margin-bottom: 0.25rem; }
  p.sub { color: #555; margin-top: 0; }
  label { font-weight: 600; display: block; margin-top: 1rem; }
  input, textarea { width: 100%; padding: 0.5rem; margin-top: 0.25rem; box-sizing: border-box; font-size: 1rem; }
  textarea { min-height: 90px; }
  button { margin-top: 1rem; padding: 0.6rem 1.4rem; font-size: 1rem; cursor: pointer; }
  #result { margin-top: 1.5rem; padding: 1rem; border: 1px solid #ccc; border-radius: 8px; background: #fafafa; white-space: pre-wrap; display: none; }
  #result .meta { color: #666; font-size: 0.85rem; margin-top: 0.75rem; }
  .row { display: flex; gap: 0.5rem; align-items: end; }
  .row .grow { flex: 1; }
  .hint { font-size: 0.85rem; color: #666; }
</style>
</head>
<body>
  <h1>Atlas Chat - Demo</h1>
  <p class="sub">Ask GPT-OSS 120B &amp; GPT-OSS 20B together. Free tier: 100 requests/month.</p>

  <div class="row">
    <div class="grow">
      <label for="email">Email (to get a free API key)</label>
      <input id="email" type="email" placeholder="you@example.com">
    </div>
    <button id="signupBtn" type="button">Get API key</button>
  </div>

  <label for="apikey">API key</label>
  <input id="apikey" type="password" placeholder="paste your API key here">

  <label for="prompt">Prompt</label>
  <textarea id="prompt" placeholder="What is machine learning?"></textarea>

  <button id="askBtn" type="button">Ask the ensemble</button>

  <div id="result"></div>

<script>
async function signup() {
  const email = document.getElementById('email').value.trim();
  if (!email) { alert('Enter your email first.'); return; }
  const res = await fetch('/auth/signup', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({email})
  });
  const data = await res.json();
  if (data.api_key) {
    document.getElementById('apikey').value = data.api_key;
    alert('API key created and filled in below. You are on the free tier.');
  } else {
    alert('Signup: ' + (data.error || 'unknown error'));
  }
}

async function ask() {
  const apiKey = document.getElementById('apikey').value.trim();
  const prompt = document.getElementById('prompt').value.trim();
  const result = document.getElementById('result');
  if (!apiKey) { alert('Enter your API key first.'); return; }
  if (!prompt) { alert('Type a prompt first.'); return; }
  result.style.display = 'block';
  result.textContent = 'Thinking... (models can take a while on the free plan)';
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-API-Key': apiKey},
      body: JSON.stringify({prompt})
    });
    const data = await res.json();
    if (!res.ok) {
      result.textContent = 'Error: ' + (data.error || res.statusText);
      return;
    }
    result.innerHTML = '';
    const answer = document.createElement('div');
    answer.textContent = data.answer;
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = 'Strategy: ' + data.strategy +
      ' | Usage: ' + data.usage.requests_this_month + '/' + data.usage.limit + ' this month';
    result.appendChild(answer);
    result.appendChild(meta);
  } catch (e) {
    result.textContent = 'Network error: ' + e.message;
  }
}

document.getElementById('signupBtn').addEventListener('click', signup);
document.getElementById('askBtn').addEventListener('click', ask);
</script>
</body>
</html>
    """

@app.route("/success", methods=["GET"])
def success():
    """Checkout success page"""
    session_id = request.args.get("session_id", "")
    return f"""
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Subscription successful</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 600px; margin: 3rem auto; padding: 0 1rem;">
  <h1>You're subscribed!</h1>
  <p>Thanks — your payment went through and your plan is being activated. It can take a minute for the upgrade to show on your account.</p>
  <p>Your new limits apply to the API key tied to the email you checked out with.</p>
  <p><a href="/demo">Try the demo</a> &middot; <a href="/docs">API docs</a></p>
  <p style="color:#888; font-size:0.8rem;">Checkout session: {session_id}</p>
</body>
</html>
    """

@app.route("/cancel", methods=["GET"])
def cancel():
    """Checkout cancelled page"""
    return """
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Checkout cancelled</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 600px; margin: 3rem auto; padding: 0 1rem;">
  <h1>Checkout cancelled</h1>
  <p>No worries — nothing was charged. You're still on the free tier (100 requests/month).</p>
  <p><a href="/pricing">See plans</a> &middot; <a href="/demo">Try the demo</a></p>
</body>
</html>
    """

@app.route("/pricing", methods=["GET"])
def pricing():
    """Show pricing tiers"""
    return jsonify(PRICING_TIERS)

@app.route("/docs", methods=["GET"])
def docs():
    """API documentation"""
    return jsonify({
        "title": "Atlas Chat API Documentation",
        "endpoints": {
            "POST /auth/signup": "Create new account",
            "POST /auth/login": "Get API key",
            "POST /api/chat": "Chat with ensemble (requires API key)",
            "GET /api/usage": "Check API usage (requires API key)"
        },
        "authentication": "Include 'X-API-Key' header with all API requests",
        "example": {
            "method": "POST",
            "url": "https://api.ensemblechat.com/api/chat",
            "headers": {"X-API-Key": "your-api-key-here"},
            "body": {"prompt": "What is machine learning?"}
        }
    })

# Routes - Authentication
@app.route("/auth/signup", methods=["POST"])
def signup():
    """Create new user account"""
    data = request.json
    email = data.get("email")

    if not email:
        return jsonify({"error": "Email required"}), 400

    api_key = generate_api_key(email)

    try:
        conn = sqlite3.connect("ensemble_chat.db")
        c = conn.cursor()
        c.execute("INSERT INTO users (email, tier, api_key) VALUES (?, ?, ?)",
                 (email, "free", api_key))
        conn.commit()
        user_id = c.lastrowid
        conn.close()

        return jsonify({
            "success": True,
            "email": email,
            "api_key": api_key,
            "tier": "free",
            "message": "Account created! Use your API key to make requests."
        }), 201

    except sqlite3.IntegrityError:
        return jsonify({"error": "Email already exists"}), 400

@app.route("/auth/login", methods=["POST"])
def login():
    """Login and get API key"""
    data = request.json
    email = data.get("email")

    conn = sqlite3.connect("ensemble_chat.db")
    c = conn.cursor()
    c.execute("SELECT api_key, tier FROM users WHERE email=?", (email,))
    result = c.fetchone()
    conn.close()

    if not result:
        return jsonify({"error": "User not found"}), 404

    api_key, tier = result
    return jsonify({
        "email": email,
        "api_key": api_key,
        "tier": tier
    })

# Routes - API
@app.route("/api/chat", methods=["POST"])
@require_api_key
def chat():
    """Chat with ensemble chatbot"""
    data = request.json
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "Prompt required"}), 400

    # Get ensemble response
    response = ask_ensemble(prompt)

    if "error" in response:
        return jsonify(response), 500

    # Increment usage
    increment_usage(request.user_id)
    usage = get_user_usage(request.user_id)
    limit = PRICING_TIERS[request.tier]["requests_per_month"]

    return jsonify({
        "prompt": prompt,
        "answer": response["answer"],
        "strategy": response["strategy"],
        "usage": {
            "requests_this_month": usage,
            "limit": limit,
            "remaining": limit - usage
        }
    })

# Waiting-room chat: no API key, small per-visitor cap so one person cannot
# drain the shared Groq free tier. No DB writes, no new files.
# (Render sits behind a proxy, so the real visitor IP comes from
# X-Forwarded-For; remote_addr alone would be the proxy for everyone.)
_WAIT_CHAT_LIMIT = 5          # questions per visitor per hour
_WAIT_CHAT_WINDOW = 3600      # seconds
_WAIT_CHAT_MAX_PROMPT = 500   # characters
_wait_chat_hits = {}
_wait_chat_lock = Lock()


def _wait_chat_allowed(ip):
    now = time.time()
    with _wait_chat_lock:
        hits = [t for t in _wait_chat_hits.get(ip, []) if now - t < _WAIT_CHAT_WINDOW]
        if len(hits) >= _WAIT_CHAT_LIMIT:
            return False
        hits.append(now)
        _wait_chat_hits[ip] = hits
        return True


@app.route("/api/wait-chat", methods=["POST"])
def wait_chat():
    """Guest chat for waiting Vern's customers: the same ensemble as /demo,
    no signup. Capped per visitor so the shared free-tier models are safe."""
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Type a question first."}), 400
    if len(prompt) > _WAIT_CHAT_MAX_PROMPT:
        return jsonify({"error": "Please keep your question shorter."}), 400
    forwarded = request.headers.get("X-Forwarded-For", "")
    ip = forwarded.split(",")[0].strip() if forwarded else (request.remote_addr or "unknown")
    if not _wait_chat_allowed(ip):
        return jsonify({"error": "That is enough questions for now — please try again in a bit."}), 429
    response = ask_ensemble(prompt)
    if "error" in response:
        return jsonify({"error": "The chat is having trouble right now. Please try again in a minute."}), 502
    return jsonify({"answer": response["answer"]})


@app.route("/api/usage", methods=["GET"])
@require_api_key
def usage():
    """Get user's API usage"""
    usage_count = get_user_usage(request.user_id)
    tier = request.tier
    limit = PRICING_TIERS[tier]["requests_per_month"]

    return jsonify({
        "tier": tier,
        "requests_this_month": usage_count,
        "limit": limit,
        "remaining": limit - usage_count,
        "percentage_used": round((usage_count / limit * 100) if limit > 0 else 0, 2)
    })

# Routes - Payments (Stripe)
@app.route("/auth/upgrade", methods=["POST"])
def upgrade():
    """Upgrade to paid tier using Stripe"""
    data = request.json
    email = data.get("email")
    tier = data.get("tier")  # "starter" or "pro"

    if tier not in ["starter", "pro"]:
        return jsonify({"error": "Invalid tier"}), 400

    conn = sqlite3.connect("ensemble_chat.db")
    c = conn.cursor()
    c.execute("SELECT id, stripe_customer_id FROM users WHERE email=?", (email,))
    result = c.fetchone()
    conn.close()

    if not result:
        return jsonify({"error": "User not found"}), 404

    user_id, stripe_customer_id = result

    try:
        # Create Stripe checkout session
        session_obj = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price": PRICING_TIERS[tier]["stripe_price_id"],
                "quantity": 1,
            }],
            mode="subscription",
            success_url="https://ensemble-chat.onrender.com/success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url="https://ensemble-chat.onrender.com/cancel",
            customer=stripe_customer_id or None,
            customer_email=email if not stripe_customer_id else None,
            metadata={"tier": tier, "email": email},
        )

        return jsonify({
            "checkout_url": session_obj.url,
            "session_id": session_obj.id
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/webhook/stripe", methods=["POST"])
def stripe_webhook():
    """Handle Stripe webhook events"""
    payload = request.get_data()
    sig_header = request.headers.get("Stripe-Signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, os.getenv("STRIPE_WEBHOOK_SECRET")
        )
    except ValueError:
        return jsonify({"error": "Invalid payload"}), 400
    except stripe.error.SignatureVerificationError:
        return jsonify({"error": "Invalid signature"}), 400

    # Handle subscription events
    if event["type"] == "checkout.session.completed":
        session_obj = event["data"]["object"]
        metadata = session_obj.get("metadata") or {}
        tier = metadata.get("tier")
        customer_email = metadata.get("email") or session_obj.get("customer_email")
        stripe_customer = session_obj.get("customer")
        amount_total = (session_obj.get("amount_total") or 0) / 100.0

        if customer_email and tier in PRICING_TIERS:
            conn = sqlite3.connect("ensemble_chat.db")
            c = conn.cursor()
            c.execute("SELECT id FROM users WHERE email=?", (customer_email,))
            row = c.fetchone()
            if row:
                user_id = row[0]
                c.execute(
                    "UPDATE users SET tier=?, stripe_customer_id=COALESCE(?, stripe_customer_id), "
                    "updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (tier, stripe_customer, user_id),
                )
                c.execute(
                    "INSERT INTO transactions (user_id, stripe_charge_id, amount, tier, status) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (user_id, session_obj.get("id"), amount_total, tier, "completed"),
                )
                conn.commit()
            conn.close()

    elif event["type"] == "customer.subscription.deleted":
        subscription = event["data"]["object"]
        stripe_customer = subscription.get("customer")

        if stripe_customer:
            conn = sqlite3.connect("ensemble_chat.db")
            c = conn.cursor()
            c.execute(
                "UPDATE users SET tier='free', updated_at=CURRENT_TIMESTAMP "
                "WHERE stripe_customer_id=?",
                (stripe_customer,),
            )
            conn.commit()
            conn.close()

    return jsonify({"status": "success"})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500

# Initialize database on startup (works under gunicorn too, not just __main__)
init_db()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
