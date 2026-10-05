"""Q&A по подогреву сидений — на основе 3 статей Drive2.

Отдельная коллекция ChromaDB, интерактивный режим.
"""

import json
from pathlib import Path

import chromadb
import ollama
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------

ARTICLES = [
    "data/articles/article_041.json",
    "data/articles/article_042.json",
    "data/articles/article_043.json",
]

CHROMA_DIR = "chroma_db"
COLLECTION = "heating_seats"
EMBED_MODEL = "intfloat/multilingual-e5-small"
OLLAMA_MODEL = "qwen2.5:3b"
TOP_K = 5
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

SYSTEM_PROMPT = """Ты — ассистент по автомобильной тематике, специализирующийся на подогреве сидений.

Отвечай на вопрос, опираясь ТОЛЬКО на фрагменты статей и комментариев с Drive2.ru.

Правила:
1. Если в источниках есть информация — дай развёрнутый ответ с конкретными деталями:
   микросхемы, реле, схемы подключения, размеры, артикулы, последовательности действий.
2. Если информации нет — честно скажи: "В найденных статьях нет информации по этому вопросу".
3. Если авторы противоречат — упомяни: "мнения авторов расходятся".
4. НЕ выдумывай детали, которых нет в источниках.
5. Отвечай по делу, без воды. Максимум 300 слов.
"""


# ---------------------------------------------------------------------------
# Индексация
# ---------------------------------------------------------------------------

def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


def build_index():
    print("=" * 60)
    print("ИНДЕКСАЦИЯ 3 СТАТЕЙ ПРО ПОДОГРЕВ СИДЕНИЙ")
    print("=" * 60)

    print("\nЗагружаю модель эмбеддингов...")
    model = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Удаляем старую коллекцию
    try:
        client.delete_collection(COLLECTION)
        print(f"Удалена старая коллекция: {COLLECTION}")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0
    for path_str in ARTICLES:
        path = Path(path_str)
        if not path.exists():
            print(f"[!] {path} не найден, пропускаю")
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        text = data["text"]
        chunks = chunk_text(text)

        if not chunks:
            continue

        prefixed = [f"passage: {c}" for c in chunks]
        embeddings = model.encode(prefixed, show_progress_bar=False).tolist()

        ids = [f"{path.stem}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": path.name,
                "title": data["title"],
                "url": data["url"],
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
        print(f"  [{path.name}] {data['title']}")
        print(f"      чанков: {len(chunks)} | символов: {len(text)}")

    print(f"\n[✓] Проиндексировано чанков: {total_chunks}")
    return collection, model


# ---------------------------------------------------------------------------
# Поиск и ответ
# ---------------------------------------------------------------------------

def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[Источник {i}] {c['title']}\n"
            f"URL: {c['url']}\n\n"
            f"{c['text']}"
        )
    return "\n\n" + ("=" * 60) + "\n\n".join(parts)


def ask(collection, model, query: str) -> tuple[str, list[dict]]:
    q_emb = model.encode([f"query: {query}"])[0].tolist()

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "title": meta["title"],
            "url": meta["url"],
            "distance": dist,
        })

    context = build_context(chunks)
    user_prompt = (
        f"Вопрос: {query}\n\n"
        f"Фрагменты статей:\n{context}\n\n"
        f"Дай ответ, опираясь только на эти источники."
    )

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        think=False,
        options={"num_predict": 700, "temperature": 0.3},
    )

    return response["message"]["content"], chunks


def format_sources(chunks: list[dict]) -> str:
    seen = set()
    lines = ["\n" + "=" * 60, "ИСТОЧНИКИ:", "=" * 60]
    for c in chunks:
        if c["url"] in seen:
            continue
        seen.add(c["url"])
        lines.append(f"• {c['title']} (distance: {c['distance']:.3f})")
        lines.append(f"  {c['url']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Индексируем при каждом запуске (3 статьи — это 5 секунд)
    collection, model = build_index()

    print("\n" + "=" * 60)
    print("ГОТОВО. Задавайте вопросы про подогрев сидений.")
    print("Введите 'выход' для завершения.")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("Вы: ").strip()
            if not query:
                continue
            if query.lower() in ("выход", "exit", "quit"):
                print("До встречи!")
                break

            print("\n[ищу ответ...]\n")
            answer, chunks = ask(collection, model, query)
            print(f"Ответ:\n{answer}\n")
            print(format_sources(chunks))
            print()

        except KeyboardInterrupt:
            print("\nЗавершаю.")
            break
        except Exception as e:
            print(f"Ошибка: {e}\n")


if __name__ == "__main__":
    main()
