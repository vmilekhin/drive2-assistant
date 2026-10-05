# search.py
"""RAG-поиск по статьям Drive2 через ChromaDB + Ollama."""

import sys
from pathlib import Path

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

import config


CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "drive2_articles"
EMBED_MODEL = "intfloat/multilingual-e5-small"
OLLAMA_MODEL = "qwen2.5:3b"

TOP_K = 5


SYSTEM_PROMPT = """Ты — ассистент по автомобильной тематике.
Отвечай на вопрос пользователя, опираясь на фрагменты статей и комментариев с Drive2.ru.

Правила:
1. Если в источниках есть информация — дай развёрнутый ответ, используя конкретные детали: марки, модели, размеры, мнения.
2. Если информации действительно нет — кратко скажи: "В найденных источниках нет прямого ответа, но можно отметить..." и процитируй, что есть близкого.
3. Если источники противоречат друг другу — упомяни это: "мнения авторов расходятся".
4. НЕ выдумывай факты, которых нет в источниках. Но и не бойся делать выводы на основе того, что есть.
5. Отвечай на русском, по делу, без воды.
"""

def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME)


def retrieve(collection, model, query: str, k: int = TOP_K):
    """Находит top-k чанков по запросу."""
    # e5-модель требует префикс "query: " для поисковых запросов
    q_emb = model.encode([f"query: {query}"])[0].tolist()

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
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
            "title": meta.get("title", ""),
            "url": meta.get("url", ""),
            "brand": meta.get("brand", ""),
            "tires": meta.get("tire_sizes", ""),
            "date": meta.get("date", ""),
            "distance": dist,
        })
    return chunks


def build_context(chunks: list[dict]) -> str:
    """Собирает контекст для LLM."""
    parts = []
    for i, c in enumerate(chunks, 1):
        header = f"[Источник {i}] {c['title']}"
        if c["brand"]:
            header += f" (марка: {c['brand']})"
        if c["tires"]:
            header += f" [размеры: {c['tires']}]"
        parts.append(f"{header}\nURL: {c['url']}\n\n{c['text']}")
    return "\n\n" + ("=" * 60) + "\n\n".join(parts)


def ask(collection, model, query: str) -> tuple[str, list[dict]]:
    """Возвращает (ответ LLM, использованные источники)."""
    chunks = retrieve(collection, model, query, TOP_K)

    # --- ОТЛАДКА: показываем найденные чанки ---
    print("\n" + "-" * 60)
    print("НАЙДЕННЫЕ ЧАНКИ (для отладки):")
    print("-" * 60)
    for i, c in enumerate(chunks, 1):
        print(f"\n[{i}] {c['title']} (distance: {c['distance']:.3f})")
        print(f"    {c['text'][:200]}...")
    print("-" * 60 + "\n")
    # --- конец отладки ---

    context = build_context(chunks)
    user_prompt = (
        f"Вопрос: {query}\n\n"
        f"Ниже — фрагменты статей с Drive2.ru:\n{context}\n\n"
        f"Дай аналитический ответ на вопрос, опираясь только на эти источники."
    )

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        think=False,
        options={
            "num_predict": 700,
            "temperature": 0.3,
        },
    )

    answer = response["message"]["content"]
    return answer, chunks


def format_sources(chunks: list[dict]) -> str:
    """Форматирует список источников."""
    lines = ["\n" + "=" * 60, "ИСТОЧНИКИ:", "=" * 60]
    seen_urls = set()
    for c in chunks:
        url = c["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        lines.append(f"• {c['title']}")
        lines.append(f"  {url}")
    return "\n".join(lines)


def main() -> None:
    print("Загружаю модель эмбеддингов...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    print("Загружаю ChromaDB...")
    collection = load_collection()
    print(f"В коллекции: {collection.count()} чанков")

    print("\n" + "=" * 60)
    print("Drive2-assistant готов. Введите 'выход' для завершения.")
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
            answer, chunks = ask(collection, embed_model, query)
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
