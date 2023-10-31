import pandas as pd
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    data = pd.read_csv('../../../dataset/train.csv')

    X = data.drop('smoking', axis=1)
    y = data['smoking']

    rfc = RandomForestClassifier(random_state=42,
                                 max_features='sqrt',  # sqrt(len(cols))
                                 min_impurity_decrease=0.001,
                                 criterion='gini',
                                 bootstrap=True,
                                 min_samples_leaf=10,
                                 min_samples_split=15,
                                 verbose=1,
                                 max_depth=6,
                                 n_estimators=100,
                                 )
    rfc.fit(X, y)

    test = pd.read_csv('../../../dataset/test.csv')
    submission = pd.read_csv('../../../dataset/sample_submission.csv')

    submission['smoking'] = rfc.predict_proba(test)[:, 1]
    submission.to_csv('./random_forest_best.csv', index=False)