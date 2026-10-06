"""Streamlit-интерфейс Drive2-assistant."""

import re
import time
from pathlib import Path

import chromadb
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "intfloat/multilingual-e5-small"

MODELS = {
    "⚡ Быстро (Qwen2.5:3b)": {
        "name": "qwen2.5:3b",
        "num_predict": 350,
        "top_k": 4,
        "description": "20–40 сек на ответ, качество базовое",
    },
    "🎯 Качественно (Qwen2.5:7b)": {
        "name": "qwen2.5:7b",
        "num_predict": 500,
        "top_k": 7,
        "description": "1–3 мин на ответ, качество выше",
    },
}

COLLECTIONS = {
    "📚 Все статьи Drive2 (зимние шины)": "drive2_articles",
    "🔥 Подогрев сидений": "heating_seats",
    "🔋 АКБ для Kia Rio 4G": "akb_kia_rio",
}

SYSTEM_PROMPT = """Ты — экспертный ассистент по автомобильной тематике, работающий с базой статей Drive2.ru.

🛑 КРИТИЧЕСКИ ВАЖНО:
- Отвечай ТОЛЬКО на русском языке.
- НИКОГДА не используй китайские иероглифы, английские слова без перевода, любые другие языки.
- Не повторяйся.

ФОРМАТ ОТВЕТА:
📋 КРАТКИЙ ВЫВОД — 1-2 предложения
🛞 КОНКРЕТИКА — модели, размеры, характеристики
⚠️ ВАЖНО ЗНАТЬ — нюансы, противоречия

ПРАВИЛА:
1. Опирайся ТОЛЬКО на предоставленные фрагменты.
2. Если мнения противоречат — скажи: "мнения авторов расходятся".
3. Если информации мало — честно: "в найденных статьях нет прямого ответа".
4. Без LaTeX. Без формул.
5. Кратко. Максимум 250 слов.
6. Конкретные детали: размеры, модели, цены.
7. Пиши правильно: "Michelin" (не "Мишельен"), "Nokian" (не "Нокиан"), "Hankook" (не "Ханкук").
"""

EXAMPLE_QUESTIONS = {
    "drive2_articles": [
        "Какую резину советуют на Kia Rio?",
        "Сравни Michelin и Nokian — что лучше?",
        "Какой размер шин подходит для Kia Rio?",
        "Что советуют про Nokian Hakkapeliitta?",
    ],
    "heating_seats": [
        "Как сделать, чтобы подогрев сидений работал после автозапуска?",
        "Какая микросхема используется в блоке управления?",
        "Как подключить управляющий провод от сигнализации?",
    ],
    "akb_kia_rio": [
        "Какой аккумулятор лучше для Kia Rio 4G?",
        "Какой размер и полярность АКБ для Kia Rio?",
        "Что советуют про Mutlu, Varta, Bosch?",
    ],
}


# ---------------------------------------------------------------------------
# Кэшированная инициализация
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_available_collections():
    """Возвращает только те коллекции, что реально существуют в ChromaDB."""
    client = get_chroma_client()
    existing = {c.name for c in client.list_collections()}
    return {
        label: name
        for label, name in COLLECTIONS.items()
        if name in existing
    }


# ---------------------------------------------------------------------------
# Callback для примеров вопросов
# ---------------------------------------------------------------------------

def set_question(question: str) -> None:
    """Записывает текст вопроса в session_state до создания виджета."""
    st.session_state.query_input = question


# ---------------------------------------------------------------------------
# Поиск и генерация
# ---------------------------------------------------------------------------

def retrieve(collection, model, query: str, k: int):
    """Ищет top-k чанков по запросу."""
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
            "distance": dist,
        })
    return chunks


def build_context(chunks):
    """Собирает контекст для LLM."""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[Источник {i}] {c['title']}\n"
            f"URL: {c['url']}\n\n"
            f"{c['text']}"
        )
    return "\n\n" + ("=" * 60) + "\n\n".join(parts)


def ask(
    collection,
    model,
    query: str,
    ollama_model: str,
    num_predict: int,
    top_k: int,
):
    """RAG-запрос с выбранной моделью, лимитом токенов и TOP_K."""
    chunks = retrieve(collection, model, query, top_k)

    context = build_context(chunks)
    user_prompt = (
        f"Вопрос: {query}\n\n"
        f"Фрагменты статей:\n{context}\n\n"
        f"Дай ответ, опираясь только на эти источники."
    )

    response = ollama.chat(
        model=ollama_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        think=False,
        options={
            "num_predict": num_predict,
            "temperature": 0.2,
            "repeat_penalty": 1.2,
        },
    )

    # Постобработка: убираем китайские иероглифы и мусор
    answer = response["message"]["content"]
    answer = re.sub(
        r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+",
        "",
        answer,
    )
    answer = re.sub(r"[ \t]+", " ", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    answer = "\n".join(line.rstrip() for line in answer.split("\n"))

    return answer.strip(), chunks


# ---------------------------------------------------------------------------
# Интерфейс
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Drive2-assistant",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Drive2-assistant")
st.markdown(
    "AI-помощник по статьям Drive2.ru. Задайте вопрос — получите "
    "аналитический ответ на основе опыта реальных автовладельцев."
)

# Инициализация session_state
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# Sidebar
with st.sidebar:
    st.header("⚙️ Настройки")

    available = get_available_collections()
    if not available:
        st.error("Нет ни одной коллекции в ChromaDB. Запустите indexer.py.")
        st.stop()

    collection_label = st.selectbox(
        "Коллекция статей",
        options=list(available.keys()),
    )
    collection_name = available[collection_label]

    client = get_chroma_client()
    collection = client.get_collection(collection_name)
    st.info(f"**Чанков в базе:** {collection.count()}")

    st.divider()
    st.markdown("**Как это работает:**")
    st.markdown(
        "1. Вопрос превращается в эмбеддинг.\n"
        "2. Поиск ближайших чанков в ChromaDB.\n"
        "3. LLM (Qwen2.5) синтезирует ответ.\n"
        "4. Указываются источники."
    )

    st.divider()
    st.subheader("🤖 Модель LLM")

    model_label = st.radio(
        "Выберите режим:",
        options=list(MODELS.keys()),
        index=1,
    )
    model_config = MODELS[model_label]
    ollama_model = model_config["name"]
    num_predict = model_config["num_predict"]
    top_k = model_config["top_k"]

    st.caption(f"ℹ️ {model_config['description']}")

    st.divider()
    st.caption(f"Модель: `{ollama_model}`")
    st.caption(f"Эмбеддинги: `{EMBED_MODEL}`")
    st.caption(f"TOP_K: {top_k}")

# Основная область
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "Ваш вопрос:",
        placeholder="Например: какую резину советуют на Kia Rio?",
        key="query_input",
    )

with col2:
    st.write("")
    st.write("")
    ask_button = st.button(
        "🔍 Спросить",
        type="primary",
        use_container_width=True,
    )

# Примеры вопросов
st.markdown("**Примеры вопросов:**")
examples = EXAMPLE_QUESTIONS.get(collection_name, [])
if examples:
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        cols[i].button(
            ex,
            key=f"ex_{i}",
            on_click=set_question,
            args=(ex,),
        )

# Обработка вопроса
if (ask_button or query) and query.strip():
    with st.spinner(f"Ищу ответ в статьях... ({ollama_model})"):
        t0 = time.time()
        model = load_embed_model()
        answer, chunks = ask(
            collection,
            model,
            query,
            ollama_model=ollama_model,
            num_predict=num_predict,
            top_k=top_k,
        )
        elapsed = time.time() - t0

    st.divider()
    st.subheader("💬 Ответ")
    st.markdown(answer)

    st.caption(f"⏱ Сгенерировано за {elapsed:.1f} сек")

    # Источники
    st.divider()
    st.subheader(f"📚 Источники ({len(chunks)})")

    seen_urls = set()
    for i, c in enumerate(chunks, 1):
        if c["url"] in seen_urls:
            continue
        seen_urls.add(c["url"])

        with st.expander(f"[{i}] {c['title']} (distance: {c['distance']:.3f})"):
            st.markdown(f"**URL:** {c['url']}")
            st.markdown("**Фрагмент:**")
            st.text(c["text"][:500] + ("..." if len(c["text"]) > 500 else ""))
