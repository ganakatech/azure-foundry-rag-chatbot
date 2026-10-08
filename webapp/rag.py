import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

load_dotenv()  # Locally: reads ../.env. On Azure: no file, uses App Settings instead.

# Locally: uses your `az login` account. On App Service: uses the Managed Identity.
credential = DefaultAzureCredential()

aoai = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_ad_token_provider=get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    ),
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
search = SearchClient(
    os.environ["AZURE_SEARCH_ENDPOINT"],
    os.environ["AZURE_SEARCH_INDEX"],
    credential,
)

SYSTEM_PROMPT = """You are a helpful product assistant for our store.
Answer ONLY using the product information in the <context> section.
If the answer is not in the context, say you don't know and suggest contacting support.
Cite the product name(s) you used in square brackets, e.g. [AeroBook 14 Laptop]."""


def retrieve(question: str, k: int = 3) -> list[dict]:
    vector = aoai.embeddings.create(
        model=os.environ["EMBED_DEPLOYMENT"], input=question
    ).data[0].embedding
    results = search.search(
        search_text=question,
        vector_queries=[VectorizedQuery(vector=vector, k_nearest_neighbors=k, fields="content_vector")],
        select=["name", "content"],
        top=k,
    )
    return [{"name": r["name"], "content": r["content"]} for r in results]


def answer(question: str, history: list[dict]) -> tuple[str, list[str]]:
    docs = retrieve(question)
    context = "\n\n".join(f"[{d['name']}]\n{d['content']}" for d in docs)
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": f"<context>\n{context}\n</context>\n\nQuestion: {question}"}]
    )
    reply = aoai.chat.completions.create(
        model=os.environ["CHAT_DEPLOYMENT"], messages=messages, temperature=0.2
    ).choices[0].message.content
    return reply, [d["name"] for d in docs]
