from src.models.anomaly_detector import AMLAnomalyDetector


def run_production_stream_pipeline():
    detector = AMLAnomalyDetector()

    try:
        # Connect to central registry rather than local joblib
        detector.load("aml_streaming_model")
        print("Successfully loaded AML model from MLflow Registry.")
    except Exception as e:
        print(f"Failed to load model from registry: {e}")


if __name__ == "__main__":
    run_production_stream_pipeline()
