import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    data = pd.read_csv('../../../dataset/train.csv')

    X = data.drop('smoking', axis=1)
    y = data['smoking']

    xgbc = XGBClassifier(learning_rate=0.05,
                         max_depth=6,
                         n_estimators=1000,
                         reg_lambda=5,
                         min_child_weight=10,
                         gamma=0.01,
                         subsample=0.66,
                         colsample_bytree=0.5,
                         colsample_bylevel=1,
                         objective='binary:logistic',
                         eval_metric='auc',
                         reg_alpha=0,
                         )
    xgbc.fit(X, y)

    test = pd.read_csv('../../../dataset/test.csv')
    submission = pd.read_csv('../../../dataset/sample_submission.csv')

    submission['smoking'] = xgbc.predict_proba(test)[:, 1]
    submission.to_csv('./xgboost_best.csv', index=False)