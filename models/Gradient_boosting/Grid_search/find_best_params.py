import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    data = pd.read_csv('../../../dataset/train.csv')

    X = data.drop('smoking', axis=1)
    y = data['smoking']

    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.1, 0.2],
        'max_depth': [3, 4, 5],
    }

    gbc = GradientBoostingClassifier()
    grid_search = GridSearchCV(gbc, param_grid, cv=5, scoring='roc_auc')

    grid_search.fit(X, y)

    best_parameters = grid_search.best_params_
    print(best_parameters)

    with open('./best_params.txt', 'w') as file_out:
        for key, value in best_parameters.items():
            file_out.write(f'{key} = {value}\n')
