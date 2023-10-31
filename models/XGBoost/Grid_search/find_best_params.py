import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    data = pd.read_csv('../../../dataset/train.csv')

    X = data.drop('smoking', axis=1)
    y = data['smoking']

    param_grid = {
        'n_estimators': [100, 500, 1000],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 4, 6],
        'reg_lambda': [0.5, 1, 5]
    }

    xgbc = XGBClassifier(min_child_weight=10,
                         gamma=0.01,
                         subsample=0.66,
                         colsample_bytree=0.5,
                         colsample_bylevel=1,
                         objective='binary:logistic',
                         eval_metric='auc',
                         reg_alpha=0,
                         )

    # params = xgbc.get_xgb_params()
    # print(params)

    grid_search = GridSearchCV(xgbc, param_grid, cv=5, scoring='roc_auc', verbose=2)
    grid_search.fit(X, y)

    best_parameters = grid_search.best_params_
    print(best_parameters)

    with open('./best_params.txt', 'w') as file_out:
        for key, value in best_parameters.items():
            file_out.write(f'{key} = {value}\n')
