import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

load_dotenv()
aoai = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
search = SearchClient(os.environ["AZURE_SEARCH_ENDPOINT"], os.environ["AZURE_SEARCH_INDEX"],
                      AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"]))

SYSTEM_PROMPT = """You are a helpful product assistant for our store.
Answer ONLY using the product information in the <context> section.
If the answer is not in the context, say you don't know and suggest contacting support.
Cite the product name(s) you used in square brackets, e.g. [AeroBook 14 Laptop]."""


def retrieve(question: str, k: int = 3) -> str:
    vector = aoai.embeddings.create(model=os.environ["EMBED_DEPLOYMENT"], input=question).data[0].embedding
    results = search.search(
        search_text=question,                                   # keyword part (hybrid)
        vector_queries=[VectorizedQuery(vector=vector, k_nearest_neighbors=k, fields="content_vector")],
        select=["name", "content"],
        top=k,
    )
    return "\n\n".join(f"[{r['name']}]\n{r['content']}" for r in results)


history = [{"role": "system", "content": SYSTEM_PROMPT}]

while True:
    question = input("\nYou: ").strip()
    if question.lower() in {"exit", "quit"}:
        break
    context = retrieve(question)
    print(context)
    messages = history + [{"role": "user",
                           "content": f"<context>\n{context}\n</context>\n\nQuestion: {question}"}]
    reply = aoai.chat.completions.create(
        model=os.environ["CHAT_DEPLOYMENT"], messages=messages, temperature=0.2
    ).choices[0].message.content
    print(f"\nBot: {reply}")
    # Keep the plain question and answer in history, not the bulky context
    history += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]
