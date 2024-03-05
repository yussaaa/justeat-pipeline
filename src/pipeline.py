from model.data_collection import data_ingestion
from model.feature_generation import generate_features
from model.training import data_split, trainging_pipeline_run


def run_pipeline(SAMPLE_DATA):
    df, restaurants_ids = data_ingestion(SAMPLE_DATA)
    df = generate_features(df, restaurants_ids)

    X_train, X_test, y_train, y_test = data_split(df)

    best_model = trainging_pipeline_run(X_train, y_train)


if __name__ == "__main__":
    run_pipeline("data/final_dataset.csv")
