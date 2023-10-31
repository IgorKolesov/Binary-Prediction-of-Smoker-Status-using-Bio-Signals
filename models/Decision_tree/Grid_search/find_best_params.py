import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    data = pd.read_csv('../../../dataset/train.csv')

    X = data.drop('smoking', axis=1)
    y = data['smoking']

    param_grid = {
        'max_depth': [3, 5, 7],
        'criterion': ['gini', 'entropy'],
        'min_samples_leaf': [2, 5, 10],
        'min_samples_split': [5, 10, 15],
    }

    dtc = DecisionTreeClassifier(random_state=42,
                                 min_impurity_decrease=0.001,
                                 )

    grid_search = GridSearchCV(dtc, param_grid, cv=5, scoring='roc_auc')

    grid_search.fit(X, y)

    best_parameters = grid_search.best_params_
    print(best_parameters)

    with open('./best_params.txt', 'w') as file_out:
        for key, value in best_parameters.items():
            file_out.write(f'{key} = {value}\n')
