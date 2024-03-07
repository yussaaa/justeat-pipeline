from io import StringIO
import pandas as pd
import numpy as np

import kfp.dsl as dsl

# from kfp.dsl import Output, Artifact
from kfp.dsl import Input, Output, Dataset, Artifact, component
from typing import NamedTuple

from google.oauth2 import service_account
from google.cloud import storage


credentials = service_account.Credentials.from_service_account_file(
    "/Users/yusali/Downloads/default-compute-service-account.json"
)


def read_from_gcs(blob_path: str) -> str:
    client = storage.Client(credentials=credentials)
    bucket_name = "cloud-ai-platform-4de6a256-deb4-4f4f-8b9f-8595e0183ea1"

    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_path)
    csv_string = blob.download_as_text()

    return StringIO(csv_string)


@component
def data_ingestion(
    DATA_path: str,
    output_data: Dataset,
    output_metadata: Artifact,
) -> NamedTuple("Outputs", [("dataframe_shape", str), ("num_restaurants", int)]):

    from io import StringIO
    import pandas as pd
    import numpy as np
    from google.cloud import storage
    import json

    def read_from_gcs(blob_path: str) -> str:
        client = storage.Client()
        bucket_name, blob_name = blob_path.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        csv_string = blob.download_as_text()
        return StringIO(csv_string)

    df = pd.read_csv(read_from_gcs(DATA_path))
    # df = pd.read_csv(DATA_path)
    df.dropna(axis=0, inplace=True)

    # unique restaurants
    restaurants_ids = {}
    list_restaurants_ids = []
    for a, b in zip(df.restaurant_lat, df.restaurant_lon):
        id = "{}_{}".format(a, b)
        restaurants_ids[id] = {"lat": a, "lon": b}
    for i, key in enumerate(restaurants_ids.keys()):
        restaurants_ids[key]["id"] = i

    # labeling of restaurants
    df["restaurant_id"] = [
        restaurants_ids["{}_{}".format(a, b)]["id"]
        for a, b in zip(df.restaurant_lat, df.restaurant_lon)
    ]

    # Save processed data and metadata
    df.to_csv(output_data.path, index=False)
    with open(output_metadata.path, "w") as f:
        json.dump(restaurants_ids, f)

    return (str(df.shape), len(restaurants_ids))


if __name__ == "__main__":
    df, restaurants_ids = data_ingestion(
        DATA_path="final_dataset.csv",
        output_data="../data/test/df.csv",
        output_metadata="../data/test/restaurants_ids.json",
    )
    print(df.shape)
