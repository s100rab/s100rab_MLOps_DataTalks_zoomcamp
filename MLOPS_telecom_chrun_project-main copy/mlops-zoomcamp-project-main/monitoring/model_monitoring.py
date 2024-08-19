from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, CatTargetDriftTab

def generate_data_drift_report(reference_data, production_data):
    dashboard = Dashboard(tabs=[DataDriftTab, CatTargetDriftTab])
    dashboard.calculate(reference_data, production_data)
    dashboard.save("monitoring_report.html")