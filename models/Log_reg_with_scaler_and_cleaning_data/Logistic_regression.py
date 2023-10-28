import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    data = pd.read_csv('../../dataset/train.csv')

    cols = list(data.drop(columns=['id', 'smoking']).columns)
    target = 'smoking'

    filtered_data = data
    for col in cols:
        mean = data[col].mean()
        std = data[col].std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        filtered_data = filtered_data[(filtered_data[col] >= lower_bound) & (filtered_data[col] <= upper_bound)]

    print(len(data))
    print(len(filtered_data))

    # scaler = StandardScaler()
    #
    # scaled_data = scaler.fit_transform(data[cols])
    #
    # model = LogisticRegression()
    # model.fit(scaled_data, data[target])
    #
    # test = pd.read_csv('../../dataset/test.csv')
    # submission = pd.read_csv('../../dataset/sample_submission.csv')
    #
    # scaled_test = scaler.transform(test[cols])
    #
    # submission['smoking'] = model.predict(scaled_test)
    #
    # submission.to_csv('./GB_trees.csv', index=False)