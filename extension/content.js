// content.js — добавляет кнопку AI-помощника на Drive2
(function () {
  'use strict';

  const STREAMLIT_URL = 'http://localhost:8501';
  const BUTTON_ID = 'drive2-ai-button';
  const MODAL_ID = 'drive2-ai-modal';

  // Защита от двойного внедрения
  if (document.getElementById(BUTTON_ID)) {
    return;
  }

  /**
   * Находит строку поиска на Drive2.
   * Пробуем несколько селекторов на случай изменения вёрстки.
   */
  function findSearchInput() {
    const selectors = [
      'input[type="search"]',
      'input[name="query"]',
      'input[name="q"]',
      'input[placeholder*="Поиск"]',
      'input[placeholder*="поиск"]',
      'form[action*="search"] input[type="text"]',
      '.c-search__input input',
      '[class*="search"] input[type="text"]'
    ];

    for (const sel of selectors) {
      const el = document.querySelector(sel);
      if (el && el.offsetParent !== null) {
        return el;
      }
    }
    return null;
  }

  /**
   * Создаёт кнопку AI-помощника.
   */
  function createButton() {
    const btn = document.createElement('button');
    btn.id = BUTTON_ID;
    btn.type = 'button';
    btn.innerHTML = '<span class="drive2-ai-icon">🤖</span><span class="drive2-ai-text">AI-помощник</span>';
    btn.title = 'Открыть AI-помощника по статьям Drive2';
    btn.addEventListener('click', openModal);
    return btn;
  }

  /**
   * Создаёт модальное окно с iframe Streamlit.
   */
  function createModal() {
    const modal = document.createElement('div');
    modal.id = MODAL_ID;
    modal.innerHTML = `
      <div class="drive2-ai-backdrop"></div>
      <div class="drive2-ai-content">
        <div class="drive2-ai-header">
          <div class="drive2-ai-title">
            <span>🤖</span>
            <span>AI-помощник Drive2</span>
          </div>
          <button class="drive2-ai-close" type="button" aria-label="Закрыть">✕</button>
        </div>
        <div class="drive2-ai-loading">
          <div class="drive2-ai-spinner"></div>
          <div>Загружаю AI-помощника...</div>
        </div>
        <iframe class="drive2-ai-iframe" src="about:blank" title="AI-помощник"></iframe>
      </div>
    `;

    document.body.appendChild(modal);

    // Закрытие
    modal.querySelector('.drive2-ai-backdrop').addEventListener('click', closeModal);
    modal.querySelector('.drive2-ai-close').addEventListener('click', closeModal);

    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && modal.classList.contains('active')) {
        closeModal();
      }
    });

    return modal;
  }

  /**
   * Открывает модальное окно.
   */
  function openModal() {
    let modal = document.getElementById(MODAL_ID);
    if (!modal) {
      modal = createModal();
    }

    const iframe = modal.querySelector('.drive2-ai-iframe');
    const loading = modal.querySelector('.drive2-ai-loading');

    modal.classList.add('active');
    document.body.style.overflow = 'hidden';

    // Показываем загрузку
    loading.style.display = 'flex';
    iframe.style.display = 'none';

    // Загружаем Streamlit
    iframe.src = STREAMLIT_URL;

    iframe.onload = function () {
      setTimeout(function () {
        loading.style.display = 'none';
        iframe.style.display = 'block';
      }, 500);
    };
  }

  /**
   * Закрывает модальное окно.
   */
  function closeModal() {
    const modal = document.getElementById(MODAL_ID);
    if (!modal) return;

    const iframe = modal.querySelector('.drive2-ai-iframe');

    modal.classList.remove('active');
    document.body.style.overflow = '';

    // Сбрасываем iframe, чтобы освободить ресурсы
    setTimeout(function () {
      iframe.src = 'about:blank';
    }, 300);
  }

  /**
   * Внедряет кнопку рядом со строкой поиска.
   */
  function injectButton() {
    if (document.getElementById(BUTTON_ID)) {
      return true;
    }

    const searchInput = findSearchInput();
    if (!searchInput) {
      return false;
    }

    const btn = createButton();

    // Пытаемся вставить кнопку прямо после input
    const parent = searchInput.parentNode;
    if (parent) {
      parent.insertBefore(btn, searchInput.nextSibling);
    } else {
      searchInput.insertAdjacentElement('afterend', btn);
    }

    console.log('[Drive2 AI] Кнопка добавлена на страницу');
    return true;
  }

  /**
   * Ждём, пока Drive2 отрисует поиск (может быть асинхронно).
   */
  function waitForSearch(maxAttempts = 20) {
    let attempts = 0;
    const interval = setInterval(function () {
      attempts++;
      if (injectButton() || attempts >= maxAttempts) {
        clearInterval(interval);
      }
    }, 500);
  }

  // Запуск после загрузки DOM
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', waitForSearch);
  } else {
    waitForSearch();
  }

  // Наблюдаем за изменениями — на случай SPA-навигации
  const observer = new MutationObserver(function () {
    if (!document.getElementById(BUTTON_ID)) {
      injectButton();
    }
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true
  });

})();
