import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    data = pd.read_csv('../../../dataset/train.csv')

    X = data.drop('smoking', axis=1)
    y = data['smoking']

    param_grid = {
        'iterations': [100, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [1, 3, 5]
    }

    clf = CatBoostClassifier()
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='roc_auc')

    grid_search.fit(X, y)

    best_parameters = grid_search.best_params_
    print(best_parameters)

    with open('./best_params.txt', 'w') as file_out:
        for key, value in best_parameters.items():
            file_out.write(f'{key} = {value}\n')
