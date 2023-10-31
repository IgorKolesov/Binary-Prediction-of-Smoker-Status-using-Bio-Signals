import pandas as pd
from catboost import CatBoostClassifier

if __name__ == '__main__':
    data = pd.read_csv('../../../dataset/train.csv')

    X = data.drop('smoking', axis=1)
    y = data['smoking']

    clf = CatBoostClassifier(iterations=1000, depth=8, learning_rate=0.05, l2_leaf_reg=5, eval_metric='AUC', verbose=False)
    clf.fit(X, y)

    test = pd.read_csv('../../../dataset/test.csv')
    submission = pd.read_csv('../../../dataset/sample_submission.csv')

    submission['smoking'] = clf.predict_proba(test)[:, 1]
    submission.to_csv('./catboost_best.csv', index=False)