# parse_html_akb.py
"""Парсер HTML из data/raw_akb/ → JSON в data/articles_akb/."""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup


def parse_one(html_path: Path) -> dict:
    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    def meta(prop: str) -> str:
        tag = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        return tag["content"] if tag else ""

    title = meta("og:title")
    description = meta("og:description")
    published = meta("article:published_time")
    url = meta("og:url")

    if not url:
        canonical = soup.find("link", rel="canonical")
        url = canonical["href"] if canonical else ""

    body = soup.find("div", class_="c-post__body")
    post_text = body.get_text(separator="\n", strip=True) if body else ""

    comments = []
    for c in soup.find_all(class_="c-comment__text"):
        t = c.get_text(separator=" ", strip=True)
        if t and len(t) > 10:
            comments.append(t)

    full_text = post_text
    if comments:
        full_text += "\n\n=== Комментарии ===\n\n" + "\n\n---\n\n".join(comments)

    return {
        "source_file": html_path.name,
        "url": url,
        "title": title,
        "description": description,
        "date": published,
        "post_text": post_text,
        "comments": comments,
        "comments_count": len(comments),
        "text_length": len(full_text),
        "text": full_text,
    }


def main() -> None:
    raw_dir = Path("data/raw_akb")
    out_dir = Path("data/articles_akb")
    out_dir.mkdir(parents=True, exist_ok=True)

    for old in out_dir.glob("*.json"):
        old.unlink()

    files = sorted(raw_dir.glob("*.html"))
    print(f"Найдено файлов: {len(files)}\n")

    saved = 0
    total_chars = 0
    total_comments = 0

    for html_path in files:
        try:
            data = parse_one(html_path)
        except Exception as e:
            print(f"[!] {html_path.name}: ошибка — {e}")
            continue

        if data["text_length"] < 100:
            continue

        saved += 1
        total_chars += data["text_length"]
        total_comments += data["comments_count"]

        out_path = out_dir / f"akb_{saved:03d}.json"
        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if saved % 10 == 0 or saved <= 5:
            print(f"[{saved:3d}] {html_path.name:20s} | "
                  f"{data['text_length']:6d} симв. | "
                  f"comm: {data['comments_count']:3d} | "
                  f"{data['title'][:50]}")

    print(f"\n[✓] Сохранено статей: {saved}")
    print(f"    Всего символов: {total_chars:,}")
    print(f"    Всего комментариев: {total_comments:,}")


if __name__ == "__main__":
    main()
