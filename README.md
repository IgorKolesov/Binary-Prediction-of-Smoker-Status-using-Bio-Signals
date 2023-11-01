<h1>Binary prediction of smoker status using bio-signals</h1>

<h2>About Dataset</h2>

<p>Smoking has been proven to negatively affect health in a multitude of ways. Smoking has been found to harm nearly every organ of the body, cause many diseases, as well as reducing the life expectancy of smokers in general. As of 2018, smoking has been considered the leading cause of preventable morbidity and mortality in the world, continuing to plague the world’s overall health.</p>

<p>According to a World Health Organization report, the number of deaths caused by smoking will reach 10 million by 2030.</p>

<p>Evidence-based treatment for assistance in smoking cessation had been proposed and promoted. However, only less than one third of the participants could achieve the goal of abstinence. Many physicians found counseling for smoking cessation ineffective and time-consuming, and did not routinely do so in daily practice. To overcome this problem, several factors had been proposed to identify smokers who had a better chance of quitting, including the level of nicotine dependence, exhaled carbon monoxide (CO) concentration, cigarette amount per day, the age at smoking initiation, previous quit attempts, marital status, emotional distress, temperament and impulsivity scores, and the motivation to stop smoking. However, individual use of these factors for prediction could lead to conflicting results that were not straightforward enough for the physicians and patients to interpret and apply. Providing a prediction model might be a favorable way to understand the chance of quitting smoking for each individual smoker. Health outcome prediction models had been developed using methods of machine learning over recent years.</p>

<p>A group of scientists are working on predictive models with smoking status as the prediction target. Your task is to help them create a machine learning model to identify the smoking status of an individual using bio-signals.</p>

<h2>Dataset Description</h2>
<ul>
    <li>age : 5-years gap</li>
    <li>height (cm)</li>
    <li>weight (kg)</li>
    <li>waist (cm) : Waist circumference length</li>
    <li>eyesight (left)</li>
    <li>eyesight (right)</li>
    <li>hearing (left)</li>
    <li>hearing (right)</li>
    <li>systolic : Blood pressure</li>
    <li>relaxation : Blood pressure</li>
    <li>fasting blood sugar</li>
    <li>Cholesterol : total</li>
    <li>triglyceride</li>
    <li>HDL : cholesterol type</li>
    <li>LDL : cholesterol type</li>
    <li>hemoglobin</li>
    <li>Urine protein</li>
    <li>serum creatinine</li>
    <li>AST : glutamic oxaloacetic transaminase type</li>
    <li>ALT : glutamic oxaloacetic transaminase type</li>
    <li>Gtp : γ-GTP</li>
    <li>dental caries</li>
    <li>smoking</li>
</ul>

<p>You can find dataset <a href="/dataset">here</a></p>

<h2>Solving the problem</h2>

<h3>Data analysis</h3>
<p>First of all, we need to analyze the data. To do this, I created plots for each column - histogram, boxplot, density plot, and added column descriptions to them. I saved all the plots as .png images in the <a href="/data_analysis/graphics">graphics directory</a>.</p>

<p>Here is an example of such a graphic:</p>
<img src="/data_analysis/graphics/weight(kg).png">

<h3>Models</h3>
<p>After that, I started training various models. For each model, I tuned the best parameters using GridSearchCV. I used the ROC AUC curve as the evaluation metric, as it works well for binary classification.</p>

<p>As a result, I trained the following models:</p>
<ol>
  <li><a href="models/Log_regression">Logistic Regression with data scaller</a></li>
  <li><a href="models/Decision_tree">Decision Tree</a>. Used params:
    <ul>
  <li>min_impurity_decrease = 0.001</li>
  <li>criterion = 'entropy'</li>
  <li>max_depth = 5</li>
  <li>min_samples_leaf = 2</li>
  <li>min_samples_split = 5</li>
</ul>
  </li>
  <li><a href="models/Random_forest">Random Forest</a>. Used params:
      <ul>
  <li>max_features = 'sqrt'</li>
  <li>min_impurity_decrease = 0.001</li>
  <li>criterion = 'gini'</li>
  <li>bootstrap = True</li>
  <li>min_samples_leaf = 10</li>
  <li>min_samples_split = 15</li>
  <li>verbose = 1</li>
  <li>max_depth = 6</li>
  <li>n_estimators = 100</li>
</ul>

  </li>
  <li><a href="models/Gradient_boosting">Gradient Boosting</a>. Used params:
      <ul>
          <li>n_estimators = 150</li>
          <li>learning_rate = 0.2</li>
          <li>max_depth = 3</li>
      </ul>
  </li>
  <li><a href="models/Catboost">Catboost</a>. Used params:
      <ul>
  <li>iterations = 1000</li>
  <li>depth = 8</li>
  <li>learning_rate = 0.05</li>
  <li>l2_leaf_reg = 5</li>
  <li>eval_metric = 'AUC'</li>
</ul>
  </li>
  <li><a href="models/XGBoost">XGBoost</a>. Used params:
      <ul>
  <li>learning_rate = 0.05</li>
  <li>max_depth = 6</li>
  <li>n_estimators = 1000</li>
  <li>reg_lambda = 5</li>
  <li>min_child_weight = 10</li>
  <li>gamma = 0.01</li>
  <li>subsample = 0.66</li>
  <li>colsample_bytree = 0.5</li>
  <li>colsample_bylevel = 1</li>
  <li>objective = 'binary:logistic'</li>
  <li>eval_metric = 'auc'</li>
  <li>reg_alpha = 0</li>
</ul>
  </li>
  <li><a href="models/Ensambles">Ensemble of multiple models (stacking)</a>. Used models:
      <ul>
          <li>CatBoostClassifier</li>
          <li>DecisionTreeClassifier</li>
          <li>RandomForestClassifier</li>
          <li>GradientBoostingClassifier</li>
          <li>XGBClassifier</li>
          <li><b>final model = LogisticRegression<b></li>
      </ul>
  </li>
</ol>

<p>Here is an example of searching for the best parameters using GridSearchCV for the XGBoost model</p>

```python
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


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
                     reg_alpha=0)

grid_search = GridSearchCV(xgbc, param_grid, cv=5, scoring='roc_auc', verbose=2)
grid_search.fit(X, y)

best_parameters = grid_search.best_params_
print(best_parameters)
```

<h3>What about score?</h3>

<p>For each model, I generated prediction results for the test dataset in a .csv file and uploaded it to the Kaggle platform, where the competition took place. Here is the score I achieved:</p>
<ul>
  <li><a href="models/Log_regression">Logistic Regression with data scaller</a>. Score = 0.83827</li>
  <li><a href="models/Decision_tree">Decision Tree</a>. Score = 0.8277</li>
  <li><a href="models/Random_forest">Random Forest</a>. Score = 0.82973</li>
  <li><a href="models/Gradient_boosting">Gradient Boosting</a>. Score = 0.86394</li>
  <li><a href="models/Catboost">Catboost</a>. Score = 0.87007</li>
  <li><a href="models/XGBoost">XGBoost</a>. Score = 0.87158</li>
  <li><a href="models/Ensambles">Ensemble of multiple models (stacking)</a>. Score = 0.87209</li>
</ul>

<p>As we can observe, boosting methods perform significantly better than decision tree, random forests, and linear regression. However, stacking all these models enables us to make predictions even slightly more accurately.</p>

<h3>Feature engineering and feature selection</h3>
<p>I attempted to <a href="/feature_engineering_with_best_model/engineering">add new features</a> to the original dataset. These features incorporated various relationships between attributes that have real-life and medical applications. This features are:</p>

```python
def new_features(df: pd.DataFrame):
    df['BMI'] = df['weight(kg)'] / (df['height(cm)'] * df['height(cm)']) * 10000
    df['mean_eyesight'] = (df['eyesight(left)'] + df['eyesight(right)']) / 2
    df['mean_hearing'] = (df['hearing(left)'] + df['hearing(right)']) / 2
    df['blood_pressure'] = df['systolic'] / df['relaxation']
    df['total_cholesterol'] = df['Cholesterol'] + df['HDL'] + df['LDL']
    df['Cholesterol/Triglyceride_ratio'] = df['Cholesterol'] / df['triglyceride']
    df['AST/ALT_ratio'] = df['AST'] / df['ALT']
    df['Liver_function'] = df['AST'] + df['ALT'] + df['Gtp']
```

<p>Then I <a href="/feature_engineering_with_best_model/selection (Score 0.86852)">plot a correlation matrix</a> and removed non-target features with high correlation among themselves, as well as features that had approximately zero correlation with the target variable. This features are:</p>

<img src="/feature_engineering_with_best_model/selection (Score 0.86852)/correlation_without_digits.png">

```python
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
```

<p>Despite the fact that the models trained faster after removing features, the score did not increase. Instead, it decreased to 0.86852.</p>
