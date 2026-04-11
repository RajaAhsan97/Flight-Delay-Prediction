import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow import MlflowClient

import pandas as pd

# ================================================================
# MLflow Configuration
# ================================================================
MLFLOW_EXPERIMENT_NAME = "Flight_Delay_Classification"
MLFLOW_TRACKING_URI    = "mlruns"   # local folder — change to remote URI if needed
MLFLOW_MODEL_NAME      = "FlightDelayClassifier"  # Name in Model Registry

# ===== Helper functions =====
def train_models(X_train, y_train, X_test, y_test):
    """
    Train multiple models and return the best one (based on RMSE)
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, 
                                                 class_weight='balanced', 
                                                 random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=150, 
                                               max_depth=12, 
                                               class_weight='balanced', 
                                               random_state=42, 
                                               n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=200, 
                                max_depth=6, 
                                learning_rate=0.1, 
                                subsample=0.8, 
                                colsample_bytree=0.8,
                                eval_metric='logloss', 
                                random_state=42, 
                                n_jobs=-1)
    }

    # Hyperparameters dict — logged per model
    hyperparams = {
        "LogisticRegression": {
            "max_iter"     : 1000,
            "class_weight" : "balanced",
            "random_state" : 42,
        },
        "RandomForest": {
            "n_estimators" : 150,
            "max_depth"    : 12,
            "class_weight" : "balanced",
            "random_state" : 42,
            "n_jobs"       : -1,
        },
        "XGBoost": {
            "n_estimators"    : 200,
            "max_depth"       : 6,
            "learning_rate"   : 0.1,
            "subsample"       : 0.8,
            "colsample_bytree": 0.8,
            "eval_metric"     : "logloss",
            "random_state"    : 42,
        },
    }


    best_model = None
    best_F1_score = 0
    model_scores = {}


    for name, model in models.items():
        print(f"Training {name}...")


        # ── MLflow run per model ──────────────────────────
        with mlflow.start_run(run_name=name, nested=True):

            if name == "LogisticRegression":
                # Scale for Logistic Regression
                scaler_clf = StandardScaler()
                X_train = scaler_clf.fit_transform(X_train)
                X_test  = scaler_clf.transform(X_test)

            # train model
            model.fit(X_train, y_train)

            # prediction on test data (unseen)
            y_pred = model.predict(X_test)
            y_prob  = model.predict_proba(X_test)[:, 1]
        
            
            Accuracy = accuracy_score(y_test, y_pred)
            Precision = precision_score(y_test, y_pred)
            Recall = recall_score(y_test, y_pred)
            F1_score = f1_score(y_test, y_pred)
            ROC_AUC_score = roc_auc_score(y_test, y_prob)

            metrics = {
                'Accuracy' : Accuracy,
                'Precision': Precision,
                'Recall'   : Recall,
                'F1'       : F1_score,
                'ROC_AUC'  : ROC_AUC_score,
            }   


            # ── Log to MLflow ──
            mlflow.set_tag("model_name", name)
            mlflow.set_tag("task", "classification")

            # Parameters
            mlflow.log_params(hyperparams[name])
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size",  len(X_test))
            mlflow.log_param("features",   X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0]))

            # Metrics
            mlflow.log_metrics(metrics)
     
            # Model artifact
            if name == "XGBoost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"  {name} → Accuracy: {metrics['Accuracy']:.4f} | "
                  f"Precision: {metrics['Precision']:.4f} | "
                  f"Recall: {metrics['Recall']:.4f} | "
                  f"F1: {metrics['F1']:.4f} | "
                  f"ROC-AUC: {metrics['ROC_AUC']:.4f}")

            model_scores[name] = metrics

        if F1_score > best_F1_score:
            best_F1_score = F1_score
            best_model = model
            best_model_name = name
            best_scaler = scaler_clf if name == "LogisticRegression" else None

    print(f"\nBest model: {best_model_name} with F1_score: {best_F1_score:.4f}")

    # if name == "LogisticRegression":
    #     return best_model, best_model_name, model_scores, scaler_clf    
    # else:
    #     return best_model, best_model_name, model_scores, None

    return best_model, best_model_name, model_scores, best_scaler


# ================================================================
# Register Best Model in MLflow Model Registry
# ================================================================
def register_best_model(run_id, model_name, best_metrics):
    """
    Register the best model artifact from a run into the
    MLflow Model Registry, then transition it to 'Staging'.

    Stages:
      None      → freshly registered version
      Staging   → validated, ready for QA / integration testing
      Production→ promote manually after QA passes
    """
    client = MlflowClient()

    # ── 1. Build artifact URI ──────────────────────────────────
    model_uri = f"runs:/{run_id}/best_model"

    # ── 2. Register model (creates registry entry if new,
    #       or adds a new version if name already exists) ────────
    print(f"\n📦 Registering model → '{model_name}' ...")
    registered = mlflow.register_model(
        model_uri  = model_uri,
        name       = model_name,
    )

    version = registered.version
    print(f"   Registered version : {version}")

    # ── 3. Add descriptive tags & description ──────────────────
    client.update_model_version(
        name        = model_name,
        version     = version,
        description = (
            f"Flight Delay Classifier — best model from training run {run_id}. "
            f"F1={best_metrics['F1']:.4f}  ROC-AUC={best_metrics['ROC_AUC']:.4f}"
        ),
    )
    client.set_model_version_tag(model_name, version, "f1",      f"{best_metrics['F1']:.4f}")
    client.set_model_version_tag(model_name, version, "roc_auc", f"{best_metrics['ROC_AUC']:.4f}")
    client.set_model_version_tag(model_name, version, "accuracy",f"{best_metrics['Accuracy']:.4f}")

    # ── 4. Transition to Staging ───────────────────────────────
    client.transition_model_version_stage(
        name    = model_name,
        version = version,
        stage   = "Staging",
        archive_existing_versions = True,   # archive any old Staging version
    )
    print(f"   Stage              : None → Staging")
    print(f"   Model URI          : {model_uri}")
    print(f"\n Model Registry entry:")
    print(f"   Name    : {model_name}")
    print(f"   Version : {version}")
    print(f"   Stage   : Staging")
    print(f"\n   To promote to Production run:")
    print(f"   client.transition_model_version_stage(")
    print(f"       name='{model_name}', version='{version}', stage='Production')")

    return version


def save_model(model, model_name, label_encoders, scaler=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.join(script_dir, "..", "..", "models")
    output_dir = os.path.abspath(output_dir)

    #os.makedirs(output_dir, exist_ok=True)nhjhjj

    encoder_path = os.path.join(output_dir, "label_encoders.pkl")
    joblib.dump(label_encoders, encoder_path)

    model_path = os.path.join(output_dir, f"{model_name}_clf.pkl")
    joblib.dump({"model": model, "model_name": model_name}, model_path)
    if scaler:
        scaler_path = os.path .join(output_dir, "best_clf_scaler.pkl")
        joblib.dump(scaler, scaler_path)
    print(f"Saved best model to {model_path}")

    # joblib.dump(label_encoders, f"../../models/label_encoders.pkl")

    # # --- Best Classifier ---
    # joblib.dump(model, f"../../models/{model_name}_clf.pkl")
    # if scaler:
    #     joblib.dump(scaler, f"../../models/best_clf_scaler.pkl")

    # print(f"Saved best model...")


# ===== Main Pipeline =====
if __name__ == "__main__":
    #from src.etl.extract import load_data
    from src.etl.extract import download_data
    from src.etl.transform import clean_data
    from src.etl.load import save_data
    
    # extract
    #raw_df = load_data("data/raw/T_ONTIME_REPORTING.csv")
    raw_df = download_data(2023, 1)
    # transform
    clean_df, label_encoders = clean_data(raw_df)    
    # load
    save_data(clean_df, "T_ONTIME_REPORTING_cleaned.csv")

    clean_df = ''
    clean_df = pd.read_csv("data/cleaned/T_ONTIME_REPORTING_cleaned.csv")
    print("Data loaded from path -------------------------------------------")

    CLF_FEATURES = [
        'OP_UNIQUE_CARRIER_ENC',
        'ORIGIN_ENC',
        'DEST_ENC',
        'ROUTE_ENC',
        'DISTANCE',
        'DIST_BUCKET_ENC',
        'TAXI_OUT',
        'TAXI_IN',
        'DAY_OF_WEEK',
        'DAY_OF_MONTH',
        'IS_WEEKEND',
        'CARRIER_DELAY',
        'WEATHER_DELAY',
        'DEP_DELAY', 
    ]
    CLF_TARGET = 'IS_DELAYED'

    X_clf = clean_df[CLF_FEATURES]
    y_clf = clean_df[CLF_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    # ── MLflow Setup ─────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # ── Parent Run (wraps all child model runs) ───────────────────
    with mlflow.start_run(run_name="Best_Model_Selection") as parent_run:

        mlflow.set_tag("pipeline_stage", "training")
        mlflow.set_tag("dataset",        "T_ONTIME_REPORTING.csv")

        # Dataset-level params logged on parent
        mlflow.log_param("total_samples",  len(clean_df))
        mlflow.log_param("train_samples",  len(X_train))
        mlflow.log_param("test_samples",   len(X_test))
        mlflow.log_param("n_features",     len(CLF_FEATURES))
        mlflow.log_param("target_column",  CLF_TARGET)
        mlflow.log_param("features",       CLF_FEATURES)
        
        # ── Train all models (each logs its own child run) ────────
        best_model, best_model_name, all_scores, scaler = train_models(
            X_train, y_train, X_test, y_test
        )

        # ── Log best model summary on parent run ──────────────────
        best_metrics = all_scores[best_model_name]
        mlflow.set_tag("best_model", best_model_name)
        mlflow.log_param("best_model_name",   best_model_name)
        mlflow.log_metric("best_accuracy",    best_metrics["Accuracy"])
        mlflow.log_metric("best_precision",   best_metrics["Precision"])
        mlflow.log_metric("best_recall",      best_metrics["Recall"])
        mlflow.log_metric("best_f1",          best_metrics["F1"])
        mlflow.log_metric("best_roc_auc",     best_metrics["ROC_AUC"])

        # ── Save best model artifact on parent run ────────────────
        if best_model_name == "XGBoost":
            mlflow.xgboost.log_model(best_model, artifact_path="best_model")
        else:
            mlflow.sklearn.log_model(best_model, artifact_path="best_model")

        # Log label encoders as artifact
        encoder_path = "models/label_encoders.pkl"
        if os.path.exists(encoder_path):
            mlflow.log_artifact(encoder_path, artifact_path="encoders")

        print(f"\n MLflow parent run ID : {parent_run.info.run_id}")
        print(f"   Experiment           : {MLFLOW_EXPERIMENT_NAME}")
        print(f"   Tracking URI         : {MLFLOW_TRACKING_URI}")


        # Save the best model
        if scaler:
            save_model(best_model, best_model_name, label_encoders, scaler)
        else:
            save_model(best_model, best_model_name, label_encoders)

        # ── Register best model AFTER parent run closes ───────────────
    registered_version = register_best_model(
        run_id      = parent_run.info.run_id,
        model_name  = MLFLOW_MODEL_NAME,
        best_metrics= best_metrics,
    )


# run this code using
# python -m src.model.train 

# install mlflow

# after training run
#  mlflow ui --backend-store-uri mlruns
