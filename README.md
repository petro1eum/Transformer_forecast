# Transformer_forecast

**LLM-Optimized Time Series Transformer**

---

## 📈 Описание

Этот проект реализует современный трансформер для прогнозирования временных рядов с интеллектуальной оптимизацией гиперпараметров на базе LLM (GPT-4, Ollama и др.). Поддерживается автоматическая настройка обучения с помощью LLM (function calling, structured output), а также оптимизация под Apple Silicon (M1/M2/M3, MPS).

---

## 🚀 Основные возможности

- **Transformer для временных рядов** (PyTorch, encoder-decoder)
- **LLM-гипероптимизация**: поддержка rule-based, OpenAI GPT-4, Ollama (Qwen2.5)
- **Structured output**: чистый function calling без парсинга текста
- **Автоматический подбор learning rate, dropout, gradient clipping, early stopping**
- **MPS-ускорение** для MacBook (Apple Silicon)
- **Визуализация обучения и LLM-решений**
- **Сохранение лучших моделей и результатов**

---

## 🛠️ Установка и запуск

1. **Клонируйте репозиторий:**
   ```bash
   git clone https://github.com/yourname/Transformer_forecast.git
   cd Transformer_forecast
   ```
2. **Установите зависимости:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Создайте .env файл:**
   ```bash
   cp .env.example .env
   # Добавьте свой OPENAI_API_KEY
   ```
4. **Запустите обучение:**
   ```bash
   python LLM_transformer.py
   ```

---

## ⚙️ Структура проекта

- `LLM_transformer.py` — основной скрипт с LLM-оптимизацией
- `transformer.py` — архитектура модели и базовые классы
- `requirements.txt` — зависимости
- `.gitignore` — исключения для данных, моделей, секретов
- `llm_transformer_model.pth` — сохранённая модель (игнорируется гитом)
- `llm_optimization_results.json` — результаты оптимизации (игнорируется гитом)

---

## 📊 Пример вывода

```
[LLM] epoch=9 action=adjust_lr params={'factor': 0.5, 'lr_factor': 0.5} val_loss=0.0453
🔧 LLM: Уменьшен learning rate на 0.50x ↓ (новый LR: 0.000500)
[LLM] epoch=15 action=early_stop params={} val_loss=0.0338
🛑 LLM: Рекомендует early stopping
✅ Загружена лучшая модель с Val Loss: 0.029483
```

---

## 📚 Полезные ссылки

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Ollama (Qwen2.5)](https://ollama.com/library/qwen2.5)
- [Time Series Transformer (original paper)](https://arxiv.org/abs/2010.02803)

---

## 📝 Лицензия

MIT License 