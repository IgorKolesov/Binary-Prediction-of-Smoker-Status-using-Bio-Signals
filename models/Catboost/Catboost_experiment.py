import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    data = pd.read_csv('../../dataset/train.csv')
    data.dropna()
    target = 'smoking'
    cols = list(data.drop(columns=['id', target]).columns)

    cat_cols = []
    num_cols = []
    for col in cols:
        if len(data[col].value_counts()) < 30:
            cat_cols.append(col)

            if data[col].dtype == 'float64':
                data[col] = (data[col] * 10).astype(int)

            category_counts = data[col].value_counts()
            categories_to_remove = category_counts[category_counts < 150].index
            data = data[~data[col].isin(categories_to_remove)]

            print(data[col].value_counts())
            print('*' * 20)
        else:
            num_cols.append(col)
            mean = data[col].mean()
            std = data[col].std()

            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std

            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

    X_train, X_test, y_train, y_test = train_test_split(data[cols], data[target], test_size=0.2, random_state=42)

    model = CatBoostClassifier(iterations=1000, depth=8, learning_rate=0.05, l2_leaf_reg=5, eval_metric='AUC', verbose=False)
    model.fit(data[cols], data[target], cat_features=cat_cols)

    y_pred = model.predict(X_test)

    test = pd.read_csv('../../dataset/test.csv')

    test['eyesight(left)'] = (test['eyesight(left)'] * 10).astype(int)
    test['eyesight(right)'] = (test['eyesight(right)'] * 10).astype(int)
    test['serum creatinine'] = (test['serum creatinine'] * 10).astype(int)

    submission = pd.read_csv('../../dataset/sample_submission.csv')

    submission['smoking'] = model.predict_proba(test[cols])
    submission.to_csv('./catboost_best_proba.csv', index=False)