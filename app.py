"""
Ensemble Chat API - Main Application
Flask API with Stripe payment integration
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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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

# Initialize models
def load_models():
    """Load Mistral and Falcon models"""
    print("Loading models...")
    
    MODEL_A = "mistralai/Mistral-7B-Instruct-v0.1"
    MODEL_B = "tiiuae/falcon-7b-instruct"
    
    def safe_load(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            framework="pt",
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        return gen
    
    gen_a = safe_load(MODEL_A)
    gen_b = safe_load(MODEL_B)
    return [gen_a, gen_b]

# Load models on startup
try:
    MODELS = load_models()
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    MODELS = None

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

def call_gen(gen, prompt, max_new_tokens=180):
    """Call a single generator"""
    out = gen(prompt, max_new_tokens=max_new_tokens, return_full_text=False)
    text = out[0].get("generated_text") or out[0].get("text") or str(out[0])
    return text.strip()

def ask_ensemble(prompt):
    """Query both models and return best answer"""
    if not MODELS:
        return {"error": "Models not loaded"}
    
    with ThreadPoolExecutor(max_workers=len(MODELS)) as ex:
        futures = [ex.submit(call_gen, g, prompt) for g in MODELS]
        results = [f.result() for f in futures]
    
    cleaned = [r for r in results if r and r.strip()]
    
    if not cleaned:
        return {"answer": "", "strategy": "empty"}
    
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
        "description": "AI-powered ensemble chatbot API with Mistral & Falcon",
        "docs": "/docs",
        "pricing": "/pricing"
    })

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
            success_url="https://yourdomain.com/success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url="https://yourdomain.com/cancel",
            customer=stripe_customer_id or None,
            customer_email=email if not stripe_customer_id else None,
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
        customer_email = session_obj.get("customer_email")
        
        # Extract tier from metadata or session details
        # Update user tier in database
        # (Implement based on your tier mapping)
    
    return jsonify({"status": "success"})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)
