import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

if __name__ == '__main__':
    data = pd.read_csv('../../dataset/train.csv')

    cols = list(data.drop(columns=['id', 'smoking']).columns)
    target = 'smoking'

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(data[cols], data[target])

    test = pd.read_csv('../../dataset/test.csv')
    submission = pd.read_csv('../../dataset/sample_submission.csv')

    submission['smoking'] = model.predict(test[cols])

    submission.to_csv('./GB_trees.csv', index=False)