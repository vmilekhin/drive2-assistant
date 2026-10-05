# parse_html.py
"""Парсер сохранённых HTML Drive2 → JSON (с поддержкой articles_map.json)."""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

import config


TIRE_SIZE_RE = re.compile(r"\b(\d{3})[/\s](\d{2})[\s/R]*(\d{2})\b")


def extract_tire_sizes(text: str) -> list[str]:
    sizes = []
    for m in TIRE_SIZE_RE.finditer(text):
        w, h, d = m.groups()
        sizes.append(f"{w}/{h} R{d}")
    return sorted(set(sizes))


def extract_brand(text: str) -> str | None:
    """Определяет марку авто по тексту (включая русские варианты)."""
    brand_patterns = {
        "Lada":       [r"\bLada\b", r"\bЛада\b", r"\bВАЗ\b", r"\bВеста\b", r"\bГранта\b"],
        "Kia":        [r"\bKia\b", r"\bКиа\b", r"\bРио\b"],
        "Hyundai":    [r"\bHyundai\b", r"\bХендай\b", r"\bХёндэ\b", r"\bСолярис\b", r"\bКрета\b"],
        "Toyota":     [r"\bToyota\b", r"\bТойота\b", r"\bКамри\b", r"\bКоролла\b"],
        "Volkswagen": [r"\bVolkswagen\b", r"\bФольксваген\b", r"\bVW\b", r"\bПоло\b"],
    }
    for brand, patterns in brand_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return brand
    return None

def parse_one(html_path: Path, url_map: dict) -> dict:
    """Парсит один HTML-файл, возвращает словарь (пост + комментарии)."""
    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    def meta(prop: str) -> str:
        tag = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        return tag["content"] if tag else ""

    title = meta("og:title")
    description = meta("og:description")
    published = meta("article:published_time")
    url = meta("og:url") or meta("twitter:url")

    # Если og:url нет — берём canonical
    if not url:
        canonical = soup.find("link", rel="canonical")
        url = canonical["href"] if canonical else ""

    # Если всё ещё нет — берём из карты articles_map.json
    if not url:
        url = url_map.get(html_path.name, "")

    # Основной текст поста
    body = soup.find("div", class_="c-post__body")
    post_text = body.get_text(separator="\n", strip=True) if body else ""

    # Комментарии
    comments = []
    for c in soup.find_all(class_="c-comment__text"):
        t = c.get_text(separator=" ", strip=True)
        if t and len(t) > 10:
            comments.append(t)

    # Полный текст = пост + комментарии
    full_text = post_text
    if comments:
        full_text += "\n\n=== Комментарии ===\n\n" + "\n\n---\n\n".join(comments)

    tire_sizes = extract_tire_sizes(full_text + " " + title + " " + description)
    brand = extract_brand(title + " " + full_text)

    return {
        "source_file": html_path.name,
        "url": url,
        "title": title,
        "description": description,
        "date": published,
        "brand": brand,
        "tire_sizes": tire_sizes,
        "post_text": post_text,
        "comments": comments,
        "comments_count": len(comments),
        "text_length": len(full_text),
        "text": full_text,
    }


def main() -> None:
    raw_dir = Path("data/raw")
    out_dir = Path(config.ARTICLES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем карту filename -> url (для файлов из Playwright)
    mapping_path = Path("data/articles_map.json")
    url_map = {}
    if mapping_path.exists():
        url_map = json.loads(mapping_path.read_text(encoding="utf-8"))
        print(f"Загружена карта URL: {len(url_map)} записей")

    # Очищаем старые JSON
    for old in out_dir.glob("*.json"):
        old.unlink()

    files = sorted(raw_dir.glob("*.html"))
    print(f"Найдено файлов: {len(files)}\n")

    saved = 0
    for i, html_path in enumerate(files, 1):
        try:
            data = parse_one(html_path, url_map)
        except Exception as e:
            print(f"[!] {html_path.name}: ошибка — {e}")
            continue

        # Пропускаем пустые
        if data["text_length"] < 100:
            print(f"[skip] {html_path.name}: слишком мало текста ({data['text_length']})")
            continue

        saved += 1
        out_path = out_dir / f"article_{saved:03d}.json"
        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if saved % 5 == 0 or saved <= 3:
            print(f"[{saved}] {html_path.name[:40]:40s} | "
                  f"{data['text_length']:6d} симв. | "
                  f"comm: {data['comments_count']:3d} | "
                  f"brand: {data['brand'] or '—'}")

    print(f"\n[✓] Сохранено статей: {saved}")


if __name__ == "__main__":
    main()
