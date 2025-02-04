from evidently.pipeline.column_mapping import ColumnMapping
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.ui.workspace import Workspace
from evidently.report import Report
import pandas as pd
import joblib

def analyze_target_drfit(start_date, end_date):
    target = 'cnt'
    prediction = 'prediction'
    numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
    categorical_features = ['season', 'holiday', 'workingday']
    
    raw_data = pd.read_csv('data/raw/bike_sharing.csv')
    reference_jan11 = raw_data[(raw_data['dteday'] >= '2011-01-01') & (raw_data['dteday'] <= '2011-01-28')]
    current_feb11 = raw_data[(raw_data['dteday'] >= '2011-01-29') & (raw_data['dteday'] <= '2011-02-28')]


    column_mapping_drift = ColumnMapping()
    column_mapping_drift.target = target
    column_mapping_drift.prediction = prediction
    column_mapping_drift.numerical_features = numerical_features
    column_mapping_drift.categorical_features = []

    data_drift_report = Report(metrics=[
        DataDriftPreset(),
    ])

    current_feb11 = current_feb11[(current_feb11['dteday'] >= start_date) & (current_feb11['dteday'] <= end_date)]

    data_drift_report.run(
        reference_data=reference_jan11,
        current_data=current_feb11,
        column_mapping=column_mapping_drift,
    )

    return data_drift_report

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
    workspace.add_report(project.id, report)
    print(f"New report added to project {project_name}")


if __name__ == '__main__':
    WORKSPACE_NAME = "datascientest-workspace"
    PROJECT_NAME = "validate_drift"
    PROJECT_DESCRIPTION = "Evidently Dashboards"
    workspace = Workspace.create(WORKSPACE_NAME)
    
    time_spans = [
        ('2011-02-01', '2011-02-07'),
        ('2011-02-08', '2011-02-14'),
        ('2011-02-15', '2011-02-21')
    ]

    for time in time_spans:
        report = analyze_target_drfit(*time)
        add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, report)


    
    