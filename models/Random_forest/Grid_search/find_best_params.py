import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    data = pd.read_csv('../../../dataset/train.csv')

    X = data.drop('smoking', axis=1)
    y = data['smoking']

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [4, 5, 6],
    }

    rfc = RandomForestClassifier(random_state=42,
                                 max_features='auto',  # sqrt(len(cols))
                                 min_impurity_decrease=0.001,
                                 criterion='gini',
                                 bootstrap=True,
                                 min_samples_leaf=10,
                                 min_samples_split=15,
                                 verbose=1,
                                 )
    grid_search = GridSearchCV(rfc, param_grid, cv=5, scoring='roc_auc')

    grid_search.fit(X, y)

    best_parameters = grid_search.best_params_
    print(best_parameters)

    with open('./best_params.txt', 'w') as file_out:
        for key, value in best_parameters.items():
            file_out.write(f'{key} = {value}\n')
