import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    data = pd.read_csv('../../dataset/train.csv')

    cols = list(data.drop(columns=['id', 'smoking']).columns)
    target = 'smoking'
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data[cols])

    model = LogisticRegression()
    model.fit(scaled_data, data[target])

    test = pd.read_csv('../../dataset/test.csv')
    submission = pd.read_csv('../../dataset/sample_submission.csv')

    scaled_test = scaler.transform(test[cols])

    submission['smoking'] = model.predict(scaled_test)

    submission.to_csv('./GB_trees.csv', index=False)