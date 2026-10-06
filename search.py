# search.py
"""RAG-поиск по статьям Drive2 через ChromaDB + Ollama (qwen2.5:7b)."""

import os
import sys
from pathlib import Path

import chromadb
import ollama
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "drive2_articles"
EMBED_MODEL = "intfloat/multilingual-e5-small"
OLLAMA_MODEL = "qwen2.5:7b"

TOP_K = 7
SHOW_DEBUG = os.getenv("DEBUG", "0") == "1"  # включается: DEBUG=1 python search.py


SYSTEM_PROMPT = """Ты — экспертный ассистент по автомобильной тематике, работающий с базой статей и комментариев Drive2.ru.

Твоя задача — синтезировать ответ из реального опыта автовладельцев.

ФОРМАТ ОТВЕТА:
📋 КРАТКИЙ ВЫВОД — 1-2 предложения
🛞 КОНКРЕТИКА — модели, размеры, характеристики, артикулы (если упомянуты)
⚠️ ВАЖНО ЗНАТЬ — нюансы, противоречия, предупреждения

ПРАВИЛА:
1. Опирайся ТОЛЬКО на предоставленные фрагменты. Не выдумывай факты.
2. Если мнения противоречат — скажи: "мнения авторов расходятся".
3. Если информации мало — честно: "в найденных статьях нет прямого ответа".
4. НЕ используй LaTeX, формулы, длинные списки.
5. Пиши кратко и по делу. Максимум 300 слов.
6. Указывай конкретные детали: размеры шин (185/65 R15), цены, модели.
"""


# ---------------------------------------------------------------------------
# Инициализация
# ---------------------------------------------------------------------------

def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME)


def retrieve(collection, model, query: str, k: int = TOP_K):
    """Находит top-k чанков по запросу."""
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


def clean_answer(text: str) -> str:
    """Убирает LaTeX и лишние переносы."""
    import re
    text = re.sub(r"\\\[.*?\\\]", "", text, flags=re.DOTALL)
    text = re.sub(r"\\\(.*?\\\)", "", text, flags=re.DOTALL)
    text = re.sub(r"\\boxed\{([^}]*)\}", r"\1", text)
    text = text.replace("\\times", "×").replace("\\cdot", "·")
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    text = re.sub(r"\$([^$]*)\$", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def ask(collection, model, query: str) -> tuple[str, list[dict]]:
    """Возвращает (ответ LLM, использованные источники)."""
    chunks = retrieve(collection, model, query, TOP_K)

    # Отладка — только если DEBUG=1
    if SHOW_DEBUG:
        print("\n" + "-" * 60)
        print("НАЙДЕННЫЕ ЧАНКИ (для отладки):")
        print("-" * 60)
        for i, c in enumerate(chunks, 1):
            print(f"\n[{i}] {c['title']} (distance: {c['distance']:.3f})")
            print(f"    {c['text'][:200]}...")
        print("-" * 60 + "\n")

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

    answer = clean_answer(response["message"]["content"])

    if not answer:
        answer = "Не удалось сгенерировать ответ. Попробуйте переформулировать."

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
    print(f"Модель LLM: {OLLAMA_MODEL}")
    print(f"TOP_K: {TOP_K}")

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
