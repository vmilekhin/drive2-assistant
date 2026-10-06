# collect_akb.py
"""Сбор статей про АКБ для Kia Rio 4G (2017-2020) через Playwright."""

import random
import re
import time
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import sync_playwright

import config


SEARCH_QUERIES = [
    "аккумулятор Kia Rio 4",
    "аккумулятор Kia Rio 2017",
    "аккумулятор Kia Rio 2018",
    "аккумулятор Kia Rio 2019",
    "аккумулятор Kia Rio 2020",
    "какой аккумулятор на Kia Rio 4",
    "замена аккумулятора Kia Rio 4",
    "АКБ Kia Rio 4 поколение",
    "аккумулятор Kia Rio 1.6",
    "аккумулятор Kia Rio 1.4",
    "выбор аккумулятора Kia Rio",
    "аккумулятор для Kia Rio 4",
    "какой АКБ купить на Kia Rio",
    "аккумулятор 60Ah Kia Rio",
    "аккумулятор 62Ah Kia Rio",
    "аккумулятор Varta Kia Rio",
    "аккумулятор Bosch Kia Rio",
    "аккумулятор Mutlu Kia Rio",
    "аккумулятор АКОМ Kia Rio",
    "аккумулятор Solite Kia Rio",
    "размер аккумулятора Kia Rio 4",
    "полярность аккумулятора Kia Rio",
    "клеммы аккумулятора Kia Rio 4",
    "обслуживание аккумулятора Kia Rio",
    "зарядка аккумулятора Kia Rio",
]

MAX_PER_QUERY = 8
PAUSE_ARTICLE = (4.0, 7.0)
PAUSE_QUERY = (5.0, 8.0)


def make_search_url(query: str) -> str:
    return f"https://www.drive2.ru/search/?text={quote(query)}"


def collect_urls_from_search(page, query: str) -> list[str]:
    url = make_search_url(query)
    print(f"\n[+] Поиск: {query}")

    try:
        page.goto(url, timeout=30000, wait_until="domcontentloaded")
        page.wait_for_timeout(5000)
    except Exception as e:
        print(f"    Ошибка: {e}")
        return []

    links = page.query_selector_all("a[href*='/l/']")
    urls = []
    seen = set()

    for link in links:
        href = link.get_attribute("href")
        if not href:
            continue
        if href.startswith("/l/"):
            href = "https://www.drive2.ru" + href
        clean = re.sub(r"[#?].*$", "", href)
        if clean not in seen and "/l/" in clean:
            seen.add(clean)
            urls.append(clean)

    print(f"    Найдено URL: {len(urls)}")
    return urls[:MAX_PER_QUERY]


def save_article_html(page, url: str, out_path: Path) -> bool:
    try:
        page.goto(url, timeout=30000, wait_until="domcontentloaded")
        page.wait_for_timeout(4000)
        html = page.content()
        out_path.write_text(html, encoding="utf-8")
        size_kb = len(html) // 1024
        print(f"      ✓ {out_path.name} ({size_kb} КБ)")
        return True
    except Exception as e:
        print(f"      ✗ Ошибка: {e}")
        return False


def main() -> None:
    raw_dir = Path("data/raw_akb")
    raw_dir.mkdir(parents=True, exist_ok=True)

    counter = 0
    all_urls_seen = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=config.USER_AGENT,
            viewport={"width": 1366, "height": 768},
            locale="ru-RU",
        )
        page = context.new_page()

        for i, query in enumerate(SEARCH_QUERIES, 1):
            print(f"\n[{i}/{len(SEARCH_QUERIES)}] {query}")
            urls = collect_urls_from_search(page, query)

            for url in urls:
                if url in all_urls_seen:
                    continue
                all_urls_seen.add(url)
                counter += 1
                filename = f"akb_{counter:03d}.html"
                out_path = raw_dir / filename

                print(f"    [{counter}] {url}")
                save_article_html(page, url, out_path)
                time.sleep(random.uniform(*PAUSE_ARTICLE))

            if i < len(SEARCH_QUERIES):
                pause = random.uniform(*PAUSE_QUERY)
                print(f"\n    Пауза {pause:.1f} сек...")
                time.sleep(pause)

        browser.close()

    print(f"\n[✓] Готово. Уникальных статей: {counter}")
    print(f"    Папка: {raw_dir}/")


if __name__ == "__main__":
    main()
