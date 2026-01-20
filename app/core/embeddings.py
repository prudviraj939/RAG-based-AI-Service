from openai import OpenAI


client = OpenAI()


EMBED_MODEL = "text-embedding-3-small"


def embed_text(text: str) -> list[float]:
    resp = client.embeddings.create(
    model=EMBED_MODEL,
    input=text
    )
    return resp.data[0].embedding