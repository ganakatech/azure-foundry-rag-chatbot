# Product Chatbot with Azure AI Foundry + RAG

A learning project: a product Q&A chatbot that uses Retrieval-Augmented Generation (RAG) with Azure AI Foundry models and Azure AI Search, deployed as a Flask web app on Azure App Service and secured with Microsoft Entra ID.

```
Browser ──(Entra ID sign-in, App Service Easy Auth)──► Flask app on App Service
                                                          │ Managed Identity (no API keys)
                                     ┌────────────────────┴────────────────────┐
                                     ▼                                         ▼
                     Azure AI Foundry                                Azure AI Search
             (gpt-4o-mini + text-embedding-3-small)              (hybrid vector + keyword index)
```

## What's here

| File | Purpose |
|---|---|
| `products.json` | Sample product catalog |
| `ingest.py` | Creates the search index, embeds each product, and uploads it (Entra ID auth) |
| `chat.py` | Command-line chat loop (Part 1, API-key version, kept for reference) |
| `webapp/app.py` | Flask API: `/`, `/api/chat`, `/health`; reads the signed-in user from Easy Auth headers |
| `webapp/rag.py` | Retrieval + grounded answer generation using `DefaultAzureCredential` |
| `webapp/templates/index.html` | Chat UI |
| `Azure-Foundry-RAG-Chatbot-Guide.pdf` | Step-by-step guide covering setup, deployment, Entra ID and cleanup |

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r webapp/requirements.txt
cp .env.example .env          # fill in your endpoints
az login                      # DefaultAzureCredential uses this locally

python ingest.py              # build the index (one time)
cd webapp && flask --app app run --debug
```

Your user needs the **Cognitive Services OpenAI User** role on the Foundry resource, and the **Search Service Contributor** and **Search Index Data Contributor** roles on AI Search.

## Deploy

See the PDF guide (Part 2 and Part 3) for App Service deployment, Managed Identity role assignments, and Easy Auth setup.
