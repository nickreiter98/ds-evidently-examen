from evidently.pipeline.column_mapping import ColumnMapping
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.ui.workspace import Workspace
from evidently.report import Report
import pandas as pd
import joblib

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
    regressor = joblib.load('models/rf_regressor_production.pkl')
    ref_prediction = regressor.predict(reference_jan11[numerical_features + categorical_features])
    reference_jan11['prediction'] = ref_prediction

    reference_jan11.to_csv('test.csv')
    print(reference_jan11.dtypes)

    # Initialize the regression performance report with the default regression metrics preset
    regression_performance_report = Report(metrics=[
        RegressionPreset(),
    ])
    print('still working')
    # Run the regression performance report using the reference data
    regression_performance_report.run(reference_data=None, 
                                    current_data=reference_jan11,
                                    column_mapping=column_mapping)
    print('not any more')

    return regression_performance_report

def add_report_to_workspace(workspace, project_name, project_description, report):
    """
    Adds a report to an existing or new project in a workspace.
    This function will be useful to you
    """
    # Check if project already exists
    project = None
    for p in workspace.list_projects():
        if p.name == project_name:
            project = p
            break

    # Create a new project if it doesn't exist
    if project is None:
        project = workspace.create_project(project_name)
        project.description = project_description

    # Add report to the project
    print(project.id)
    print(report)
    workspace.add_report(project.id, report)


if __name__ == '__main__':
    report = analyze_model_drift()

    WORKSPACE_NAME = "datascientest-workspace"
    PROJECT_NAME = "validate_predictions"
    PROJECT_DESCRIPTION = "Evidently Dashboards"

    # Create and Add report to workspace
    workspace = Workspace.create(WORKSPACE_NAME)
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, report)
