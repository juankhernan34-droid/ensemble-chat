# Launch Checklist — ensemble-chat

Follow these steps in order. Nothing here costs money until Step 5, and Step 5 has a free option.

## Step 1 — Get your Hugging Face token (free, ~3 minutes)

1. Go to **huggingface.co** and log in (create a free account if you don't have one).
2. Click your profile picture (top right) → **Settings** → **Access Tokens** (or go straight to huggingface.co/settings/tokens).
3. Click **Create new token**.
4. Name it `ensemble-chat`, type **Read**, click **Create**.
5. **Copy the token** (it starts with `hf_...`). Save it somewhere safe — you can't see it again.

## Step 2 — Accept the Mistral license (free, ~1 minute)

The app uses a Mistral model that is "gated" — Hugging Face blocks it until you accept the terms.

1. While logged in, go to **huggingface.co/mistralai/Mistral-7B-Instruct-v0.1**.
2. You'll see a form asking you to agree to the model's conditions. Fill it in and submit.
3. Wait for access to be granted (usually instant, sometimes takes a while).

## Step 3 — Put the fixed files into your GitHub repo (~5 minutes)

Two files were fixed: `app.py` (the token bug) and `Dockerfile` (secrets no longer baked into the image).

1. Go to **github.com/juankhernan34-droid/ensemble-chat**.
2. Click on **app.py** in the file list.
3. Click the **pencil icon** (Edit) at the top right of the file view.
4. Select ALL the old text (Ctrl+A) and delete it.
5. Paste the new fixed `app.py` content (get it from Atlas).
6. Scroll down, click **Commit changes**.
7. Repeat for **Dockerfile**.

## Step 4 — Set your environment variables at deploy time

Whichever host you pick, you'll enter these as environment variables (never paste secrets into chat or code):

| Variable | Where to get it | Required? |
|---|---|---|
| `HUGGINGFACE_TOKEN` | Step 1 above (`hf_...`) | **Yes** |
| `FLASK_SECRET_KEY` | Make up any long random string | Yes |
| `STRIPE_SECRET_KEY` | dashboard.stripe.com/apikeys (`sk_live_...`) | Only for payments |
| `STRIPE_PUBLISHABLE_KEY` | Same page (`pk_live_...`) | Only for payments |
| `STRIPE_WEBHOOK_SECRET` | dashboard.stripe.com/webhooks (`whsec_...`) | Only for payments |
| `STRIPE_STARTER_PRICE_ID` | Create a $4.99/mo product in Stripe dashboard (`price_...`) | Only for payments |
| `STRIPE_PRO_PRICE_ID` | Create a $14.99/mo product in Stripe dashboard (`price_...`) | Only for payments |
| `FLASK_ENV` | Set to `production` | Recommended |

**Stripe webhook:** in the Stripe dashboard, add a webhook endpoint pointing at
`https://YOUR-DOMAIN/webhook/stripe` and tick the events
`checkout.session.completed` and `customer.subscription.deleted`.
Paid users are upgraded automatically; cancelled users drop back to free.

## Step 5 — Pick hosting

### Option A — FREE (recommended): no-GPU version on Hugging Face Spaces

Use the file **`app_inference_api.py`** (in this repo folder). It's the same app,
but instead of downloading the two 7B models (which needs an expensive GPU),
it calls Hugging Face's servers to do the AI work. Same endpoints, same Stripe,
same database. Costs $0 on a free Space.

1. Create a new Space at huggingface.co/spaces → choose the **Docker** SDK.
2. Upload the repo files, but rename `app_inference_api.py` → `app.py`
   (or point your Dockerfile/CMD at it).
3. In the Space settings → **Variables and secrets**, add the env vars from Step 4.
4. It builds and goes live. Done.

### Option B — Full local models on a rented GPU (paid, by the hour)

Only pick this if you specifically want the models running on your own machine.
Two 7B models in fp16 need **~28GB+ VRAM**, so you need a 48GB card minimum
(RTX A6000 / L40S / RTX 6000 Ada). Prices checked October 2026:

| Provider | Cheapest 48GB GPU | $/hr | Gotchas |
|---|---|---|---|
| **RunPod Community Cloud** | RTX A6000 48GB | **~$0.33** | Cheapest; hosts vary in reliability, can be preempted |
| **RunPod Secure Cloud** | RTX A6000 48GB | **$0.53** | Reliable datacenter GPUs, per-second billing |
| Vast.ai (marketplace) | RTX A6000 48GB | ~$0.35–0.50 | Auction-style; price/reliability varies by host, spot can be interrupted |
| Lambda Labs | RTX A6000 48GB | $1.09 | Simple pricing, no egress fees; GPUs often sold out |
| DigitalOcean GPU Droplet | L40S 48GB | $1.57 | Easiest setup, but **billed even when powered off** — you must DESTROY the droplet to stop paying; no spot discount |

**Honest math:** even the cheapest reliable option ($0.53/hr) costs ~$380/month
running 24/7. For hourly testing it's fine; for always-on, **Option A (free) wins
by a mile.** Note the 24GB cards (RTX 4090 ~$0.30–0.74/hr) are cheaper but CANNOT
fit both models at once — don't rent one expecting the app to work as-is.

**One-line recommendation:** launch on **Option A (Hugging Face Spaces, $0)**
with `app_inference_api.py`. Rent a RunPod A6000 by the hour only if you later
need the full local-model version.

## Step 6 — Verify it's live

```bash
curl https://YOUR-DOMAIN/
```

You should get back:

```json
{"name": "Ensemble Chat API", "version": "1.0.0", ...}
```

Then test signup + chat:

```bash
curl -X POST https://YOUR-DOMAIN/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com"}'
# copy the api_key from the response, then:
curl -X POST https://YOUR-DOMAIN/api/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"prompt":"Hello!"}'
```
