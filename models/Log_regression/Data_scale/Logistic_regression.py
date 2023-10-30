import pandas as pd
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    data = pd.read_csv('../../../dataset/train.csv')

    X = data.drop('smoking', axis=1)
    y = data['smoking']

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)
    scaled_df = pd.DataFrame(scaled_data, columns=X.columns)
    scaled_df['smoking'] = y
    scaled_df.to_csv('./scaled_train.csv', index=False)

    test = pd.read_csv('../../../dataset/test.csv')
    scaled_data = scaler.fit_transform(test)
    scaled_df = pd.DataFrame(scaled_data, columns=test.columns)
    scaled_df.to_csv('./scaled_test.csv', index=False)
