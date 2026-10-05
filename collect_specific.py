# collect_specific.py
"""Скачивает конкретные URL из my_urls.txt через Playwright."""

import random
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

import config


URLS_FILE = "my_urls.txt"
PAUSE = (4.0, 7.0)


def main() -> None:
    urls_path = Path(URLS_FILE)
    if not urls_path.exists():
        print(f"[!] Файл {URLS_FILE} не найден")
        return

    urls = [
        line.strip()
        for line in urls_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    print(f"Найдено URL: {len(urls)}\n")

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing = list(raw_dir.glob("drive2_*.html"))
    counter = 0
    if existing:
        nums = [int(re.search(r"drive2_(\d+)", f.name).group(1)) for f in existing]
        counter = max(nums)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=config.USER_AGENT,
            viewport={"width": 1366, "height": 768},
            locale="ru-RU",
        )
        page = context.new_page()

        for i, url in enumerate(urls, 1):
            counter += 1
            filename = f"drive2_{counter:03d}.html"
            out_path = raw_dir / filename

            print(f"[{i}/{len(urls)}] {url}")
            try:
                page.goto(url, timeout=30000, wait_until="domcontentloaded")
                page.wait_for_timeout(4000)
                html = page.content()
                out_path.write_text(html, encoding="utf-8")
                size_kb = len(html) // 1024
                print(f"    ✓ {filename} ({size_kb} КБ)")
            except Exception as e:
                print(f"    ✗ Ошибка: {e}")

            if i < len(urls):
                pause = random.uniform(*PAUSE)
                time.sleep(pause)

        browser.close()

    print(f"\n[✓] Готово. Скачано: {len(urls)} статей в {raw_dir}/")
    print("Теперь запусти: python parse_html.py")


if __name__ == "__main__":
    main()
