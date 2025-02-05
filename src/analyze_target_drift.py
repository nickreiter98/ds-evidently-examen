import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.metric_preset import TargetDriftPreset
from evidently.report import Report
import pandas as pd
import joblib
import utility.evidently

def analyze_target_drift(start_date, end_date):
    target = 'cnt'
    prediction = 'prediction'
    numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
    categorical_features = ['season', 'holiday', 'workingday']
    
    raw_data = pd.read_csv('data/raw/bike_sharing.csv')
    reference_jan11 = raw_data[(raw_data['dteday'] >= '2011-01-01') & (raw_data['dteday'] <= '2011-01-28')]
    current_feb11 = raw_data[(raw_data['dteday'] >= '2011-01-29') & (raw_data['dteday'] <= '2011-02-28')]
    current_feb11 = current_feb11[(current_feb11['dteday'] >= start_date) & (current_feb11['dteday'] <= end_date)]

    # Perform column mapping
    column_mapping = ColumnMapping()
    column_mapping.target = target
    column_mapping.prediction = prediction
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features

    

    # Generate predictions for the reference data
    regressor = joblib.load('models/rf_regressor.pkl')
    reference_jan11['prediction'] = regressor.predict(reference_jan11[numerical_features + categorical_features])
    current_feb11['prediction'] = regressor.predict(current_feb11[numerical_features + categorical_features])

    # Initialize the regression performance report with the default regression metrics preset
    target_drift_report = Report(metrics=[
        TargetDriftPreset(),
    ])

    # Run the regression performance report using the reference data
    target_drift_report.run(reference_data=reference_jan11.sort_index(), 
                                    current_data=current_feb11.sort_index(),
                                    column_mapping=column_mapping)

    return target_drift_report

if __name__ == '__main__':
    report = analyze_target_drift('2011-02-15', '2011-02-21')
    utility.evidently.add_report(report, '03_target_drift')
