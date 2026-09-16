import os
import pandas as pd
import yaml
from src.data.features import preprocess_data

try:
    from evidently.report import Report
    from evidently.presets import DataDriftPreset
    if not hasattr(Report, "save_html"):
        raise AttributeError
except (ImportError, AttributeError):
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
config_path = os.path.join(BASE_DIR, "config", "config.yaml")

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

processed_data_path = os.path.join(BASE_DIR, config["data"]["processed_path"])
predictions_jsonl_path = os.path.join(BASE_DIR, config["data"]["predictions_log_path"])
reports_save_path = os.path.join(BASE_DIR, config["data"]["drift_report_path"])

def create_report():
    if not os.path.exists(predictions_jsonl_path):
        print(f"No s'ha trobat el fitxer de prediccions a: {predictions_jsonl_path}")
        return

    # 1. Carregar les dades de referència i les de producció
    ref_df = pd.read_csv(processed_data_path)
    ref_df = preprocess_data(ref_df)
    raw_df = pd.read_json(predictions_jsonl_path, lines=True)
    if len(raw_df) == 0:
        print("El fitxer de prediccions està buit.")
        return

    curr_df = pd.DataFrame(raw_df["inputs"].tolist())
    
    # Extraure la predicció Churn (1/0) per avaluar Prediction Drift
    if "prediction" in raw_df.columns:
        curr_df["Churn"] = [
            int(p["churn"]) if isinstance(p, dict) and "churn" in p else int(p) 
            for p in raw_df["prediction"]
        ]

    # 2. Executar informe de Data & Prediction Drift
    print("Generant informe de Data Drift i Prediction Drift amb Evidently...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=curr_df)

    # 3. Guardar l'informe en un arxiu HTML interactiu
    os.makedirs(os.path.dirname(reports_save_path), exist_ok=True)
    report.save_html(reports_save_path)
    print(f"Informe de Drift desat correctament a: {reports_save_path}")
    
    rep = report.as_dict()
    drift_result = rep["metrics"][0]["result"]
    if drift_result.get("dataset_drift"):
        drifted = drift_result.get("number_of_drifted_columns", 0)
        total = drift_result.get("number_of_columns", 0)
        share = drift_result.get("share_of_drifted_columns", 0) * 100
        print(f"\n[ALERTA DRIFT] S'ha detectat Data Drift global! ({drifted}/{total} columnes afectades - {share:.1f}%).")
    else:
        print("\n[OK] No s'ha detectat Data Drift significatiu a les dades de producció.")

if __name__ == "__main__":
    create_report()