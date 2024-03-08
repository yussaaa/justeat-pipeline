from model.data_collection import data_ingestion
from model.feature_generation import generate_features
from model.training import data_split, trainging_pipeline_run

from google.cloud import storage
from joblib import dump, load

from google.oauth2 import service_account


def run_pipeline(SAMPLE_DATA):
    df, restaurants_ids = data_ingestion(SAMPLE_DATA)
    df = generate_features(df, restaurants_ids)

    X_train, X_test, y_train, y_test = data_split(df)

    best_model = trainging_pipeline_run(X_train, y_train)

    dump(best_model, "./models/justeat_regrssor.joblib")

    upload_blob(
        "cloud-ai-platform-4de6a256-deb4-4f4f-8b9f-8595e0183ea1",
        "./models/justeat_regrssor.joblib",
        "models/justeat_regrss.joblib",
    )


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client(project="citric-program-331111")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print("File {} uploaded to {}.".format(source_file_name, destination_blob_name))


if __name__ == "__main__":
    run_pipeline("data/final_dataset.csv")
