"""
Ensemble Chat API - Serverless Inference API variant
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
import hashlib
from datetime import datetime, timedelta
from functools import wraps
import sqlite3

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
    "starter": {"name": "Starter", "price": 4.99, "requests_per_month": 5000, "stripe_price_id": os.getenv("STRIPE_STARTER_PRICE_ID")},
    "pro": {"name": "Pro", "price": 14.99, "requests_per_month": 50000, "stripe_price_id": os.getenv("STRIPE_PRO_PRICE_ID")},
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

    conn.commit()
    conn.close()

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

    MODEL_A = "meta-llama/Llama-3.2-3B-Instruct"
    MODEL_B = "mistralai/Mistral-7B-Instruct-v0.3"
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

def call_gen(client, model_id, prompt, max_new_tokens=180):
    """Call a single model through the Hugging Face Inference API via direct HTTP"""
    import requests
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

def safe_call_gen(client, model_id, prompt, max_new_tokens=180):
    """Call one model, returning error string instead of raising"""
    try:
        return call_gen(client, model_id, prompt, max_new_tokens)
    except Exception as e:
        # Don't leak tokens; HF errors are safe to surface
        return f"__ERROR__{type(e).__name__}: {str(e)[:200]}"

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
    """Home page"""
    return jsonify({
        "name": "Ensemble Chat API",
        "version": "1.0.0",
        "description": "AI-powered ensemble chatbot API with Mistral & Falcon (serverless inference, no GPU)",
        "docs": "/docs",
        "pricing": "/pricing",
        "demo": "/demo"
    })

# Routes - Web pages
@app.route("/demo", methods=["GET"])
def demo():
    """Simple demo/chat page: enter API key + prompt, see the ensemble response."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ensemble Chat - Demo</title>
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
  <h1>Ensemble Chat - Demo</h1>
  <p class="sub">Ask Mistral &amp; Falcon together. Free tier: 100 requests/month.</p>

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
        "title": "Ensemble Chat API Documentation",
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
