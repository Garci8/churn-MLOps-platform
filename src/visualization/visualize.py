import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.pipeline import Pipeline


# ==============================================================================
# GENERACIÓN DEL GRÁFICO DE IMPORTANCIA DE VARIABLES SHAP
# ==============================================================================

def generate_shap_summary(best_pipeline: Pipeline, best_model_name: str, X_val: pd.DataFrame, models_dir: str) -> None:
    """Genera y guarda el gráfico de importancia de variables SHAP para el mejor modelo.

    Args:
        best_pipeline (Pipeline): Pipeline del mejor modelo entrenado.
        best_model_name (str): Nombre del modelo ('logreg', 'rf', 'xgboost').
        X_val (pd.DataFrame): DataFrame de validación (antes de preprocesar).
        models_dir (str): Directorio donde guardar el gráfico.
    """
    try:
        print("\nGenerando explicación de variables con SHAP...")
        preprocess_step = best_pipeline.named_steps['preprocess']
        model_step = best_pipeline.named_steps['model']
        
        # Preprocesar los datos de validación
        X_val_processed = preprocess_step.transform(X_val)
        feature_names = preprocess_step.get_feature_names_out()
        X_val_processed_df = pd.DataFrame(X_val_processed, columns=feature_names)
        
        # Seleccionar el explicador adecuado
        if best_model_name == "logreg":
            explainer = shap.LinearExplainer(model_step, X_val_processed_df)
        else:
            explainer = shap.TreeExplainer(model_step)
            
        shap_values = explainer.shap_values(X_val_processed_df)
        
        # Extraer los valores SHAP de la clase positiva (Churn = 1)
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1]
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                shap_values_to_plot = shap_values[:, :, 1]
            else:
                shap_values_to_plot = shap_values
        else:
            shap_values_to_plot = shap_values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_to_plot, X_val_processed_df, show=False)
        plt.tight_layout()
        
        os.makedirs(models_dir, exist_ok=True)
        plot_path = os.path.join(models_dir, "shap_summary.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Gráfico de importancia SHAP guardado en: {plot_path}")
    except Exception as e:
        print(f"Error al generar SHAP summary: {e}")
