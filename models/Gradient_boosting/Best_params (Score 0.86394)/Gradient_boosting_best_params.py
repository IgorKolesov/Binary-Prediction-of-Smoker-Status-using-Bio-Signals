import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

if __name__ == '__main__':
    data = pd.read_csv('../../../dataset/train.csv')

    X = data.drop('smoking', axis=1)
    y = data['smoking']

    gbc = GradientBoostingClassifier(n_estimators=150, learning_rate=0.2, max_depth=3)
    gbc.fit(X, y)

    test = pd.read_csv('../../../dataset/test.csv')
    submission = pd.read_csv('../../../dataset/sample_submission.csv')

    submission['smoking'] = gbc.predict_proba(test)[:, 1]
    submission.to_csv('./gradient_boosting_best.csv', index=False)