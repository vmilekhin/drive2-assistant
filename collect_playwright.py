# collect_playwright.py
"""Автоматический сбор статей Drive2 через Playwright."""

import json
import random
import re
import time
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import sync_playwright

import config


# Сколько статей скачивать на один запрос
MAX_PER_QUERY = 8

# Пауза между запросами
PAUSE_SEARCH = (3.0, 5.0)     # между поиском и статьями
PAUSE_ARTICLE = (4.0, 7.0)    # между статьями
PAUSE_QUERY = (5.0, 8.0)      # между поисковыми запросами


def make_search_url(query: str) -> str:
    return f"https://www.drive2.ru/search/?text={quote(query)}"


def collect_urls_from_search(page, query: str) -> list[str]:
    """Открывает поиск Drive2, возвращает список URL статей."""
    url = make_search_url(query)
    print(f"\n[+] Поиск: {query}")
    print(f"    URL: {url}")

    try:
        page.goto(url, timeout=30000, wait_until="domcontentloaded")
        page.wait_for_timeout(5000)
    except Exception as e:
        print(f"    Ошибка: {e}")
        return []

    links = page.query_selector_all("a[href*='/l/']")
    urls: list[str] = []
    seen = set()

    for link in links:
        href = link.get_attribute("href")
        if not href:
            continue
        if href.startswith("/l/"):
            href = "https://www.drive2.ru" + href
        # Нормализуем: убираем якоря и параметры
        clean = re.sub(r"[#?].*$", "", href)
        if clean not in seen and "/l/" in clean:
            seen.add(clean)
            urls.append(clean)

    print(f"    Найдено URL: {len(urls)}")
    return urls[:MAX_PER_QUERY]


def save_article_html(page, url: str, out_path: Path) -> bool:
    """Скачивает страницу статьи и сохраняет HTML."""
    try:
        page.goto(url, timeout=30000, wait_until="domcontentloaded")
        page.wait_for_timeout(4000)  # даём JS подгрузить комментарии

        html = page.content()
        out_path.write_text(html, encoding="utf-8")

        size_kb = len(html) // 1024
        print(f"      ✓ {out_path.name} ({size_kb} КБ)")
        return True
    except Exception as e:
        print(f"      ✗ Ошибка: {e}")
        return False


def main() -> None:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Файл-карта: filename -> original_url
    mapping_path = Path("data/articles_map.json")
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    else:
        mapping = {}

    counter = max([int(k.split("_")[1].split(".")[0]) for k in mapping.keys()] + [0])

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=config.USER_AGENT,
            viewport={"width": 1366, "height": 768},
            locale="ru-RU",
        )
        page = context.new_page()

        for i, query in enumerate(config.SEARCH_QUERIES, 1):
            urls = collect_urls_from_search(page, query)

            for url in urls:
                counter += 1
                filename = f"drive2_{counter:03d}.html"
                out_path = raw_dir / filename

                print(f"    [{counter}] {url}")
                ok = save_article_html(page, url, out_path)

                if ok:
                    mapping[filename] = url

                # Пауза между статьями
                time.sleep(random.uniform(*PAUSE_ARTICLE))

            # Пауза между поисковыми запросами
            if i < len(config.SEARCH_QUERIES):
                pause = random.uniform(*PAUSE_QUERY)
                print(f"\n    Пауза {pause:.1f} сек между запросами...")
                time.sleep(pause)

        browser.close()

    # Сохраняем карту filename -> url
    mapping_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[✓] Готово. Сохранено статей: {len(mapping)}")
    print(f"    HTML:  {raw_dir}/")
    print(f"    Карта: {mapping_path}")


if __name__ == "__main__":
    main()
