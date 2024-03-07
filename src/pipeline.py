from model.data_collection import data_ingestion
from model.feature_generation import generate_features
from model.training import data_split, training_pipeline_run

import kfp.dsl as dsl
from kfp import compiler
from kfp.dsl import Artifact, Input, Model, Output

from google.cloud import aiplatform as aiplatform
from google_cloud_pipeline_components.v1.batch_predict_job import ModelBatchPredictOp
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp
from google_cloud_pipeline_components.experimental.evaluation import (
    ModelEvaluationClassificationOp,
    ModelImportEvaluationOp,
)

import logging
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M")


@dsl.pipeline(
    name="Food delivery pipeline",
    description="A pipeline that processes food delivery data.",
    pipeline_root=PIPELINE_ROOT,
)
def run_pipeline(SAMPLE_DATA):
    from model.data_collection import data_ingestion
    from model.feature_generation import generate_features
    from model.training import pipeline_run_step

    df, restaurants_ids = data_ingestion(SAMPLE_DATA)
    df = generate_features(df, restaurants_ids)

    X_train, X_test, y_train, y_test = data_split(df)

    best_model = training_pipeline_run(X_train, y_train)


if __name__ == "__main__":
    run_pipeline("data/final_dataset.csv")


# Compile and run the pipeline
aiplatform.init(project=PROJECT_ID, location=REGION, encryption_spec_key_name=KEY_ID)

logging.getLogger().setLevel(logging.INFO)
logging.info(
    f"Init with project {PROJECT_ID} in region {REGION}. Pipeline root: {PIPELINE_ROOT}"
)

FORMAT = ".json"

logging.info(f"Compiling pipeline to {PIPELINE_NAME + FORMAT}")
compiler.Compiler().compile(
    pipeline_func=run_pipeline, package_path=PIPELINE_NAME + FORMAT
)

run = aiplatform.PipelineJob(
    project=PROJECT_ID,
    location=REGION,
    display_name=PIPELINE_NAME,
    template_path=PIPELINE_NAME + FORMAT,
    job_id=f"{PIPELINE_NAME}-{TIMESTAMP}",
    pipeline_root=PIPELINE_ROOT,
    enable_caching=caching,
)

run.submit(service_account=SERVICE_ACCOUNT, network=NETWORK)


# # Import model and convert to Artifact
# @dsl.component(base_image=IMAGE)
# def get_unmanaged_model(model: Input[Model], unmanaged_model: Output[Artifact]):
#     unmanaged_model.metadata = model.metadata
#     unmanaged_model.uri = "/".join(
#         model.uri.split("/")[:-1]
#     )  # remove filename after last / - send dir rather than file


# This can be used to test the online endpoint:
#
# {
#    "instances": [
#      [1.18998913145894,-0.563413492993846,0.129352538697985,-0.302175771438239,-0.927677605983222,-0.784678753251055,-0.443713590138326,-0.0956435854887243,-0.648897198590765,0.0499810894390051,0.358011190903553,-0.445067055832097,-0.0982544178676521,-1.28002825726001,0.304411501372465,0.733464325722348,1.71246876228603,-1.78636925309304,0.163898890406551,0.180489467655959,0.0091417811964457,-0.074443134391428,-0.0011569207049818,0.327529344882462,0.332585093864499,-0.298508896918417,0.0256419259293034,0.0496775221663426,80.52]
# ]
# }
