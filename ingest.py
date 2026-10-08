import json, os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
)

load_dotenv()

# Uses your `az login` account (needs the roles from Part 3, Step 3)
cred = DefaultAzureCredential()
aoai = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_ad_token_provider=get_bearer_token_provider(
        cred, "https://cognitiveservices.azure.com/.default"
    ),
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
endpoint, index_name = os.environ["AZURE_SEARCH_ENDPOINT"], os.environ["AZURE_SEARCH_INDEX"]

# 1. Define the index schema
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="name", type=SearchFieldDataType.String),
    SimpleField(name="category", type=SearchFieldDataType.String, filterable=True),
    SearchableField(name="content", type=SearchFieldDataType.String),
    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=1536,          # text-embedding-3-small size
        vector_search_profile_name="vprofile",
    ),
]
vector_search = VectorSearch(
    algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
    profiles=[VectorSearchProfile(name="vprofile", algorithm_configuration_name="hnsw")],
)
SearchIndexClient(endpoint, cred).create_or_update_index(
    SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
)

# 2. Embed and upload
products = json.load(open("products.json"))
docs = []
for p in products:
    text = f"{p['name']} ({p['category']}): {p['description']}"
    vector = aoai.embeddings.create(model=os.environ["EMBED_DEPLOYMENT"], input=text).data[0].embedding
    docs.append({"id": p["id"], "name": p["name"], "category": p["category"],
                 "content": text, "content_vector": vector})

SearchClient(endpoint, index_name, cred).upload_documents(docs)
print(f"Uploaded {len(docs)} documents to '{index_name}'")