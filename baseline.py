import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 1) Загружаем данные
train = pd.read_csv("data/ctr_train.csv", nrows=1_000_000)
test  = pd.read_csv("data/ctr_test.csv")

# 2) Признаки/цель
X_full = train.drop(columns=["click", "id"])
y_full = train["click"].astype("int8")

# 3) Приводим ВСЕ признаки к category и ВЫРАВНИВАЕМ категории train <-> test
feat_cols = X_full.columns.tolist()
for c in feat_cols:
    X_full[c] = X_full[c].astype("category")
    test[c]   = test[c].astype("category")
    all_cats = pd.Index(X_full[c].cat.categories).union(pd.Index(test[c].cat.categories))
    X_full[c] = X_full[c].cat.set_categories(all_cats)
    test[c]   = test[c].cat.set_categories(all_cats)

# 4) Делим на train/valid
X_train, X_valid, y_train, y_valid = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

# 5) Datasets для LightGBM
cat_features = feat_cols  # можно просто список имён
dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features, free_raw_data=False)
dvalid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_features, free_raw_data=False)

# 6) Параметры
params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.1,
    "num_leaves": 64,
    "verbose": -1,
}

# 7) Обучение (callbacks для early stopping)
model = lgb.train(
    params,
    dtrain,
    valid_sets=[dvalid],
    num_boost_round=200,
    callbacks=[
        lgb.early_stopping(stopping_rounds=20, verbose=False),
        lgb.log_evaluation(period=10),
    ],
)

# 8) Валидация
y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
print("AUC на валидации:", roc_auc_score(y_valid, y_pred))

# 9) Предсказание на test и сабмит
test_pred = model.predict(test[feat_cols], num_iteration=model.best_iteration)
pd.DataFrame({"id": test["id"], "click": test_pred}).to_csv("submission.csv", index=False)
print("Файл submission.csv сохранён!")
#done