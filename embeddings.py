"""Qwen text-embedding-v4 via Alibaba Model Studio (DashScope) OpenAI-compatible API."""
import os
from openai import OpenAI

# International (Singapore) endpoint. Change region here if your account differs.
BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)
EMBED_MODEL = "text-embedding-v4"
EMBED_DIM = 1024            # must match the pgvector column dimension
BATCH = 10                 # DashScope hard limit: <=10 texts per embeddings request

_client = None
def client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=BASE_URL)
    return _client

def embed_texts(texts):
    """Embed a list of strings, batching at 10. Returns list[list[float]]."""
    out = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        resp = client().embeddings.create(
            model=EMBED_MODEL, input=batch, dimensions=EMBED_DIM
        )
        out.extend([d.embedding for d in resp.data])
    return out

def embed_query(text):
    return embed_texts([text])[0]
