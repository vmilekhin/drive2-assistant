# config.py
"""Настройки проекта Drive2-assistant."""

# User-Agent браузера — Drive2 отдаёт 402 ботам
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Паузы между запросами (секунды)
PAUSE_MIN = 3.0
PAUSE_MAX = 5.0

# Таймаут запроса
REQUEST_TIMEOUT = 20

# Пути
DATA_DIR = "data"
ARTICLES_DIR = "data/articles"
CANDIDATES_DIR = "data/candidates"

# Марки, которые нас интересуют (для фильтрации)
CAR_BRANDS = ["Lada", "Kia", "Hyundai", "Toyota", "Volkswagen"]

# Поисковые запросы (по одному на марку)
SEARCH_QUERIES = [
    "зимние шины Lada Vesta",
    "зимние шины Kia Rio",
    "зимние шины Hyundai Solaris",
    "зимние шины Toyota Camry",
    "зимние шины Volkswagen Polo",
]
