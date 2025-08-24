# AI Challenge — CTR Prediction Baseline

Этот проект решает задачу конкурса **«Студенты. Клик или нет?»** от Сбера:  
по 22 категориальным признакам нужно предсказать вероятность клика по рекламе.

---

## 📂 Структура проекта

ai_sber/
├─ data/ # train.csv, test.csv, sample_submission.csv (в .gitignore)
├─ notebooks/ # jupyter/EDA
├─ src/ # исходный код
│ ├─ config.py
│ ├─ utils_io.py
│ ├─ train_lgbm.py
│ ├─ infer.py
├─ scripts/ # вспомогательные скрипты (bash/ps1)
├─ baseline.py # стартовый пример
├─ pyproject.toml # poetry deps
├─ poetry.lock
└─ README.md