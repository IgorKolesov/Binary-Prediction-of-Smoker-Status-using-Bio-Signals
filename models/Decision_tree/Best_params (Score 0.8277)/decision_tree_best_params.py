import pandas as pd
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    data = pd.read_csv('../../../dataset/train.csv')

    X = data.drop('smoking', axis=1)
    y = data['smoking']

    dtc = DecisionTreeClassifier(random_state=42,
                                 min_impurity_decrease=0.001,
                                 criterion='entropy',
                                 max_depth=5,
                                 min_samples_leaf=2,
                                 min_samples_split=5
                                 )
    dtc.fit(X, y)

    test = pd.read_csv('../../../dataset/test.csv')
    submission = pd.read_csv('../../../dataset/sample_submission.csv')

    submission['smoking'] = dtc.predict_proba(test)[:, 1]
    submission.to_csv('./decision_tree_best.csv', index=False)