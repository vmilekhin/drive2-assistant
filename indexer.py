# indexer.py
"""Индексация статей Drive2 в ChromaDB."""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

import config


CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "drive2_articles"
EMBED_MODEL = "intfloat/multilingual-e5-small"

# Размер чанка (символов) и перекрытие
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def chunk_text(text: str) -> list[str]:
    """Разбивает текст на чанки с перекрытием."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


def main() -> None:
    print("Загружаю модель эмбеддингов...")
    model = SentenceTransformer(EMBED_MODEL)
    print(f"Модель загружена: {EMBED_MODEL}")

    # ChromaDB клиент (persistent — сохраняет на диск)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Удаляем коллекцию, если была — чтобы пересоздать
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Удалена старая коллекция: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Читаем все JSON
    articles_dir = Path(config.ARTICLES_DIR)
    files = sorted(articles_dir.glob("*.json"))
    print(f"Найдено статей: {len(files)}\n")

    total_chunks = 0

    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        text = data["text"]
        title = data["title"]
        url = data["url"]
        brand = data.get("brand") or ""
        date = data.get("date", "")[:10]
        tires = ", ".join(data.get("tire_sizes", []))

        chunks = chunk_text(text)
        if not chunks:
            print(f"[!] {f.name}: нет текста, пропускаю")
            continue

        # Префикс для e5-модели (важно для качества!)
        prefixed = [f"passage: {c}" for c in chunks]

        embeddings = model.encode(prefixed, show_progress_bar=False).tolist()

        ids = [f"{f.stem}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": f.name,
                "title": title,
                "url": url,
                "brand": brand,
                "date": date,
                "tire_sizes": tires,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        total_chunks += len(chunks)
        print(f"[{f.name[:50]}]")
        print(f"    title:  {title[:60]}")
        print(f"    brand:  {brand}")
        print(f"    chunks: {len(chunks)}")
        print(f"    text:   {len(text)} симв.\n")

    print(f"\n[✓] Проиндексировано чанков: {total_chunks}")
    print(f"    Коллекция: {COLLECTION_NAME}")
    print(f"    Путь:      {CHROMA_DIR}")


if __name__ == "__main__":
    main()
