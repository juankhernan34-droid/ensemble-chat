# Ensemble Chat API - Usage Guide

## Getting Started

### 1. Sign Up

```bash
curl -X POST https://api.ensemblechat.com/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email": "your-email@example.com"}'
```

**Response:**
```json
{
  "success": true,
  "email": "your-email@example.com",
  "api_key": "abc123def456...",
  "tier": "free",
  "message": "Account created! Use your API key to make requests."
}
```

### 2. Make API Requests

Use your API key in the `X-API-Key` header:

```bash
curl -X POST https://api.ensemblechat.com/api/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{"prompt": "What is machine learning?"}'
```

**Response:**
```json
{
  "prompt": "What is machine learning?",
  "answer": "Machine learning is a subset of artificial intelligence...",
  "strategy": "majority",
  "usage": {
    "requests_this_month": 42,
    "limit": 100,
    "remaining": 58
  }
}
```

### 3. Check Your Usage

```bash
curl -X GET https://api.ensemblechat.com/api/usage \
  -H "X-API-Key: your-api-key-here"
```

**Response:**
```json
{
  "tier": "free",
  "requests_this_month": 42,
  "limit": 100,
  "remaining": 58,
  "percentage_used": 42.0
}
```

## Pricing Tiers

| Tier | Price | Requests/Month | Best For |
|------|-------|-----------------|----------|
| **Free** | $0 | 100 | Testing, Learning |
| **Starter** | $4.99 | 5,000 | Small Projects |
| **Pro** | $14.99 | 50,000 | Production Apps |

## Upgrading

```bash
curl -X POST https://api.ensemblechat.com/auth/upgrade \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your-email@example.com",
    "tier": "starter"
  }'
```

You'll get a Stripe checkout link. Complete payment to upgrade!

## Response Strategy

The API uses an **ensemble approach**:

- **Majority**: If both models give similar answers, you get the consensus
- **Longest**: If they disagree, you get the most detailed response

Both raw model outputs are also returned for comparison.

## Error Codes

| Code | Meaning |
|------|----------|
| 401 | Missing/Invalid API Key |
| 429 | Rate limit exceeded |
| 400 | Bad request (missing prompt) |
| 500 | Server error |

## Python Example

```python
import requests

api_key = "your-api-key-here"
headers = {"X-API-Key": api_key}

# Chat
response = requests.post(
    "https://api.ensemblechat.com/api/chat",
    json={"prompt": "What is Python?"},
    headers=headers
)

print(response.json()["answer"])
```

## JavaScript Example

```javascript
const apiKey = "your-api-key-here";

fetch("https://api.ensemblechat.com/api/chat", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": apiKey
  },
  body: JSON.stringify({
    prompt: "What is JavaScript?"
  })
})
  .then(res => res.json())
  .then(data => console.log(data.answer));
```

## Support

Email: support@ensemblechat.com
Docs: https://docs.ensemblechat.com
Status: https://status.ensemblechat.com
