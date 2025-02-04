import pandas as pd
from sklearn import ensemble, model_selection
import joblib

def train_model():
    raw_data = pd.read_csv('data/raw/bike_sharing.csv')

    # Feature selection
    target = 'cnt'
    numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
    categorical_features = ['season', 'holiday', 'workingday']

    # Reference and current data split
    reference_jan11 = raw_data[(raw_data['dteday'] >= '2011-01-01') & (raw_data['dteday'] <= '2011-01-28')]

    # Train test split ONLY on reference_jan11
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        reference_jan11[numerical_features + categorical_features],
        reference_jan11[target],
        test_size=0.3,
        random_state=42
    )

    X_train.to_csv("data/processed/X_train_jan.csv", index=False)
    X_test.to_csv("data/processed/X_test_jan.csv", index=False)
    y_train.to_csv("data/processed/y_train_jan.csv", index=False)
    y_test.to_csv("data/processed/y_test_jan.csv", index=False)

    regressor = ensemble.RandomForestRegressor(random_state = 0, n_estimators = 50)

    # Model training prediciton
    regressor.fit(reference_jan11[numerical_features + categorical_features], reference_jan11[target])
    joblib.dump(regressor, 'models/rf_regressor.pkl')

    # Model training test
    regressor.fit(X_train, y_train)

 
    # Predictions
    preds_train = pd.DataFrame(regressor.predict(X_train),columns=['cnt'])
    preds_test = pd.DataFrame(regressor.predict(X_test),columns=['cnt'])
    preds_train.to_csv('data/processed/y_train_pred_jan.csv', index=False)
    preds_test.to_csv('data/processed/y_test_pred_jan.csv', index=False)

if __name__ == '__main__':
    train_model()