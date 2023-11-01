import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

if __name__ == '__main__':
    data = pd.read_csv('../engineering/featured_train.csv')

    X = data.drop(['id', 'smoking'], axis=1)
    y = data['smoking']
    print(*X.columns, sep='\n')

    # Создайте тепловую карту для визуализации корреляций
    correlation_matrix = X.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation matrix')
    plt.savefig(f'./correlation.png')

    X = X.drop([
        'weight(kg)',
        'waist(cm)',
        'eyesight(left)',
        'eyesight(right)',
        'Cholesterol',
        'LDL',
        'Gtp',
        'hearing(left)',
        'hearing(right)',
        'systolic',
        'Urine protein',
        'AST',
    ], axis=1)

    base_models = [
        ('cat_boost_classifier',
         CatBoostClassifier(iterations=1000, depth=8, learning_rate=0.05, l2_leaf_reg=5, eval_metric='AUC',
                            verbose=False)),
        ('decision_tree_classifier',
         DecisionTreeClassifier(random_state=42, min_impurity_decrease=0.001, criterion='entropy', max_depth=5,
                                min_samples_leaf=2, min_samples_split=5)),
        ('random_forest',
         RandomForestClassifier(random_state=42, max_features='sqrt', min_impurity_decrease=0.001, criterion='gini',
                                bootstrap=True, min_samples_leaf=10, min_samples_split=15, verbose=1, max_depth=6,
                                n_estimators=100)),
        ('gradient_boosting', GradientBoostingClassifier(n_estimators=150, learning_rate=0.2, max_depth=3)),
        ('xgboost_classifier',
         XGBClassifier(learning_rate=0.05, max_depth=6, n_estimators=1000, reg_lambda=5, min_child_weight=10,
                       gamma=0.01, subsample=0.66, colsample_bytree=0.5, colsample_bylevel=1,
                       objective='binary:logistic', eval_metric='auc', reg_alpha=0))
    ]

    final_model = LogisticRegression()
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=final_model)

    stacking_model.fit(X, y)

    test = pd.read_csv('../engineering/featured_test.csv')
    test = test.drop([
        'weight(kg)',
        'waist(cm)',
        'eyesight(left)',
        'eyesight(right)',
        'Cholesterol',
        'LDL',
        'Gtp',
        'hearing(left)',
        'hearing(right)',
        'systolic',
        'Urine protein',
        'AST',
    ], axis=1)

    submission = pd.read_csv('../../dataset/sample_submission.csv')

    submission['smoking'] = stacking_model.predict_proba(test.drop('id', axis=1))[:, 1]
    submission.to_csv('./featuredAndSelected_stacking_prediction.csv', index=False)

