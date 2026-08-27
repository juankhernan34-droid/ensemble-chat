# Deployment Guide

## Prerequisites

- Docker & Docker Compose
- Stripe Account (https://stripe.com)
- Hugging Face Account (https://huggingface.co)
- Cloud hosting (AWS, Google Cloud, DigitalOcean, etc.)

## Step 1: Get API Keys

### Stripe
1. Go to https://dashboard.stripe.com/apikeys
2. Copy your **Secret Key** and **Publishable Key**
3. Create products for each tier:
   - "Starter" - $4.99/month - 5,000 requests
   - "Pro" - $14.99/month - 50,000 requests
4. Copy the **Price IDs** for each product
5. Set up webhook: https://dashboard.stripe.com/webhooks
   - Endpoint: `https://yourdomain.com/webhook/stripe`
   - Events: `checkout.session.completed`

### Hugging Face
1. Go to https://huggingface.co/settings/tokens
2. Create a new **Read** token
3. Copy it

## Step 2: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:
```
FLASK_SECRET_KEY=generate-a-random-string-here
STRIPE_SECRET_KEY=sk_live_...
STRIPE_PUBLISHABLE_KEY=pk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_STARTER_PRICE_ID=price_...
STRIPE_PRO_PRICE_ID=price_...
HUGGINGFACE_TOKEN=hf_...
```

## Step 3: Deploy with Docker

### Local Testing
```bash
docker-compose up --build
```

Test at: http://localhost:5000

### AWS EC2

1. Launch GPU instance (g4dn.xlarge or better)
2. Install Docker:
```bash
sudo apt update
sudo apt install docker.io docker-compose-v2 nvidia-docker2 -y
sudo usermod -aG docker $USER
```

3. Clone repo & deploy:
```bash
git clone https://github.com/yourusername/ensemble-chat.git
cd ensemble-chat
cp .env.example .env
# Edit .env with your keys
sudo docker-compose up -d
```

4. Set up reverse proxy (Nginx):
```bash
sudo apt install nginx -y
```

Create `/etc/nginx/sites-available/ensemblechat`:
```nginx
server {
    listen 80;
    server_name api.ensemblechat.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Enable it:
```bash
sudo ln -s /etc/nginx/sites-available/ensemblechat /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

5. Get SSL certificate (Let's Encrypt):
```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d api.ensemblechat.com
```

### DigitalOcean App Platform

1. Push to GitHub
2. Go to https://cloud.digitalocean.com/apps
3. Click "Create App"
4. Connect your GitHub repo
5. Set environment variables in settings
6. Deploy!

### Google Cloud Run

```bash
gcloud auth login
gcloud builds submit --tag gcr.io/your-project/ensemble-chat
gcloud run deploy ensemble-chat \
  --image gcr.io/your-project/ensemble-chat \
  --platform managed \
  --region us-central1 \
  --set-env-vars="STRIPE_SECRET_KEY=sk_live_..."
```

## Step 4: Verify Deployment

```bash
curl https://api.ensemblechat.com/
```

Should return:
```json
{
  "name": "Ensemble Chat API",
  "version": "1.0.0",
  "description": "AI-powered ensemble chatbot API..."
}
```

## Monitoring

- Check logs: `docker logs -f ensemble-chat_web_1`
- Monitor usage: `sqlite3 ensemble_chat.db "SELECT * FROM usage;"`
- Set up alerts in Stripe dashboard

## Scaling Tips

1. **Load Balancing**: Put multiple instances behind a load balancer
2. **Caching**: Add Redis for response caching
3. **Database**: Upgrade to PostgreSQL for production
4. **CDN**: Use Cloudflare for API responses
5. **Monitoring**: Set up DataDog or New Relic

## Troubleshooting

**Out of GPU Memory?**
- Use smaller models
- Enable model quantization (int8)
- Use CPU fallback

**Stripe webhook not working?**
- Check endpoint URL in Stripe dashboard
- Verify webhook secret in .env
- Check logs for errors

**Models not loading?**
- Ensure Hugging Face token is valid
- Check available disk space
- Try re-authenticating
