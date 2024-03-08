from google.cloud import aiplatform
from typing import Optional, Sequence, Dict
from google.cloud.aiplatform import explain

PROJECT_NAME = "citric-program-331111"
JOB_NAME = "justeat_regressor_training"
CUSTOM_CONTAINER_IMAGE_URI = "gcr.io/citric-program-331111/justeat-pipeline"
BUCKET_NAME = "cloud-ai-platform-4de6a256-deb4-4f4f-8b9f-8595e0183ea1"

# https://cloud.google.com/vertex-ai/docs/model-registry/import-model#aiplatform_upload_model_sample-python_vertex_ai_sdk


def upload_model_sample(
    project: str,
    location: str,
    display_name: str,
    serving_container_image_uri: str,
    artifact_uri: Optional[str] = None,
    serving_container_predict_route: Optional[str] = None,
    serving_container_health_route: Optional[str] = None,
    description: Optional[str] = None,
    serving_container_command: Optional[Sequence[str]] = None,
    serving_container_args: Optional[Sequence[str]] = None,
    serving_container_environment_variables: Optional[Dict[str, str]] = None,
    serving_container_ports: Optional[Sequence[int]] = None,
    instance_schema_uri: Optional[str] = None,
    parameters_schema_uri: Optional[str] = None,
    prediction_schema_uri: Optional[str] = None,
    explanation_metadata: Optional[explain.ExplanationMetadata] = None,
    explanation_parameters: Optional[explain.ExplanationParameters] = None,
    sync: bool = True,
):

    aiplatform.init(project=project, location=location)

    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
        serving_container_predict_route=serving_container_predict_route,
        serving_container_health_route=serving_container_health_route,
        instance_schema_uri=instance_schema_uri,
        parameters_schema_uri=parameters_schema_uri,
        prediction_schema_uri=prediction_schema_uri,
        description=description,
        serving_container_command=serving_container_command,
        serving_container_args=serving_container_args,
        serving_container_environment_variables=serving_container_environment_variables,
        serving_container_ports=serving_container_ports,
        explanation_metadata=explanation_metadata,
        explanation_parameters=explanation_parameters,
        sync=sync,
    )

    model.wait()

    print(model.display_name)
    print(model.resource_name)
    return model


if __name__ == "__main__":
    upload_model_sample(
        project=PROJECT_NAME,
        location="us-central1",
        display_name="justeat-regressor",
        artifact_uri="gs://cloud-ai-platform-4de6a256-deb4-4f4f-8b9f-8595e0183ea1/models/",
        serving_container_image_uri="gcr.io/citric-program-331111/justeat-pipeline",
        description="A regression model to predict restaurant orders_busyness_by_h3_hour",
    )
