import pandas as pd
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    data = pd.read_csv('../Data_scale/scaled_train.csv')

    X = data.drop('smoking', axis=1)
    y = data['smoking']

    lr = LogisticRegression()
    lr.fit(X, y)

    test = pd.read_csv('../Data_scale/scaled_test.csv')
    submission = pd.read_csv('../../../dataset/sample_submission.csv')

    submission['smoking'] = lr.predict_proba(test)[:, 1]
    submission.to_csv('./Log_regression.csv', index=False)