
import joblib
import pandas as pd

MAIN_MODEL_PATH = "edupulse_dropout_model.joblib"

main_pipeline = joblib.load(MAIN_MODEL_PATH)


def predict_dropout(student_data):
    """
    Predict dropout probability and risk level
    using the main Lightweight Logistic Regression model.
    """

    features = main_pipeline["features"]
    preprocessor = main_pipeline["preprocessor"]
    model = main_pipeline["model"]
    thresholds = main_pipeline["risk_thresholds"]

    input_df = pd.DataFrame([student_data])

    missing_features = [
        feature
        for feature in features
        if feature not in input_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features for this model: {missing_features}"
        )

    input_df = input_df[features]

    X_input = preprocessor.transform(input_df)

    probability = float(
        model.predict_proba(X_input)[0, 1]
    )

    if probability >= thresholds["high"]:
        risk = "High Risk"
    elif probability >= thresholds["medium"]:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    return {
        "dropout_probability": probability,
        "risk_level": risk
    }
