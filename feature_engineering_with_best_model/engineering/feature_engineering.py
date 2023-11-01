import pandas as pd


def new_features(df: pd.DataFrame):
    df['BMI'] = df['weight(kg)'] / (df['height(cm)'] * df['height(cm)']) * 10000
    df['mean_eyesight'] = (df['eyesight(left)'] + df['eyesight(right)']) / 2
    df['mean_hearing'] = (df['hearing(left)'] + df['hearing(right)']) / 2
    df['blood_pressure'] = df['systolic'] / df['relaxation']
    df['total_cholesterol'] = df['Cholesterol'] + df['HDL'] + df['LDL']
    df['Cholesterol/Triglyceride_ratio'] = df['Cholesterol'] / df['triglyceride']
    df['AST/ALT_ratio'] = df['AST'] / df['ALT']
    df['Liver_function'] = df['AST'] + df['ALT'] + df['Gtp']

    return df


if __name__ == '__main__':
    train = pd.read_csv('../../dataset/train.csv')
    test = pd.read_csv('../../dataset/test.csv')

    new_train = new_features(train)
    new_test = new_features(test)

    new_train.to_csv('./featured_train.csv')
    new_test.to_csv('./featured_test.csv')

