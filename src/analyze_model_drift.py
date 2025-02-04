import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.metric_preset import RegressionPreset
from evidently.report import Report
import pandas as pd
import joblib
import utility.evidently

def analyze_model_drift():
    target = 'cnt'
    prediction = 'prediction'
    numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
    categorical_features = ['season', 'holiday', 'workingday']
    
    raw_data = pd.read_csv('data/raw/bike_sharing.csv')
    reference_jan11 = raw_data[(raw_data['dteday'] >= '2011-01-01') & (raw_data['dteday'] <= '2011-01-28')]
    

    # Perform column mapping
    column_mapping = ColumnMapping()
    column_mapping.target = target
    column_mapping.prediction = prediction
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features

    # Generate predictions for the reference data
    regressor = joblib.load('models/rf_regressor.pkl')
    ref_prediction = regressor.predict(reference_jan11[numerical_features + categorical_features])
    reference_jan11['prediction'] = ref_prediction

    # Initialize the regression performance report with the default regression metrics preset
    regression_performance_report = Report(metrics=[
        RegressionPreset(),
    ])

    # Run the regression performance report using the reference data
    regression_performance_report.run(reference_data=None, 
                                    current_data=reference_jan11,
                                    column_mapping=column_mapping)

    return regression_performance_report

if __name__ == '__main__':
    report = analyze_model_drift()
    utility.evidently.add_report(report, '02_model_drift')
