from data_collection import data_ingestion
from feature_generateion import generate_features
from training import data_split, trainging_pipeline_run


def run_pipeline(SAMPLE_DATA):
    df, restaurants_ids = data_ingestion(SAMPLE_DATA)
    df = generate_features(df, restaurants_ids)

    X_train, X_test, y_train, y_test = data_split(df)

    best_model = trainging_pipeline_run(X_train, y_train)
