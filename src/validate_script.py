import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.report import Report
from evidently.ui.workspace import Workspace
import utilities.create_workspace



def validate_script():
    numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
    categorical_features = ['season', 'holiday', 'workingday']

    X_train = pd.read_csv('data/processed/X_train_jan.csv')
    X_test = pd.read_csv('data/processed/X_test_jan.csv')
    y_test = pd.read_csv('data/processed/y_test_jan.csv')
    y_train = pd.read_csv('data/processed/y_train_jan.csv')
    y_test_pred = pd.read_csv('data/processed/y_test_pred_jan.csv')
    y_train_pred = pd.read_csv('data/processed/y_train_pred_jan.csv')

    # Add actual target and prediction columns to the training data for later performance analysis
    X_train['target'] = y_train
    X_train['prediction'] = y_train_pred

    # Add actual target and prediction columns to the test data for later performance analysis
    X_test['target'] = y_test
    X_test['prediction'] = y_test_pred

    # Initialize the column mapping object which evidently uses to know how the data is structured
    column_mapping = ColumnMapping()

    # Map the actual target and prediction column names in the dataset for evidently
    column_mapping.target = 'target'
    column_mapping.prediction = 'prediction'

    # Specify which features are numerical and which are categorical for the evidently report
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features

    # Initialize the regression performance report with the default regression metrics preset
    regression_performance_report = Report(metrics=[
        RegressionPreset(),
    ])

    # Run the regression performance report using the training data as reference and test data as current
    # The data is sorted by index to ensure consistent ordering for the comparison
    regression_performance_report.run(reference_data=X_train.sort_index(), 
                                    current_data=X_test.sort_index(),
                                    column_mapping=column_mapping)

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
    workspace.add_report(project.id, report)
    print(f"New report added to project {project_name}")


if __name__ == '__main__':
    report = validate_script()

    WORKSPACE_NAME = "datascientest-workspace"
    PROJECT_NAME = "validate_train_test"
    PROJECT_DESCRIPTION = "Evidently Dashboards"

    # Create and Add report to workspace
    workspace = Workspace.create(WORKSPACE_NAME)
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, report)

