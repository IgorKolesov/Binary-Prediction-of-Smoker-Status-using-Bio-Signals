import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    data = pd.read_csv('../../dataset/train.csv')
    data.dropna()
    cols = list(data.drop(columns=['id', 'smoking']).columns)
    target = 'smoking'

    model = GradientBoostingClassifier()

    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.1, 0.2],
        'max_depth': [3, 4, 5],
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')

    grid_search.fit(data[cols], data[target])

    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    # Best Parameters: {'learning_rate': 0.2, 'max_depth': 5, 'n_estimators': 150}
    # Best Score: 0.8661647517002002

    test = pd.read_csv('../../dataset/test.csv')
    submission = pd.read_csv('../../dataset/sample_submission.csv')

    submission['smoking'] = grid_search.predict(test[cols])

    submission.to_csv('./GB_trees.csv', index=False)