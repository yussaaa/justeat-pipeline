from google.cloud import aiplatform
from typing import Optional, Sequence, Dict, Tuple
from google.cloud.aiplatform import explain

PROJECT_NAME = "citric-program-331111"

# https://cloud.google.com/vertex-ai/docs/general/deployment#api


def create_endpoint_sample(
    project: str,
    display_name: str,
    location: str,
):
    aiplatform.init(project=project, location=location)

    endpoint = aiplatform.Endpoint.create(
        display_name=display_name,
        project=project,
        location=location,
    )

    print(endpoint.display_name)
    print(endpoint.resource_name)
    return endpoint


def deploy_model_with_dedicated_resources_sample(
    project,
    location,
    model_name: str,
    machine_type: str,
    endpoint: Optional[aiplatform.Endpoint] = None,
    deployed_model_display_name: Optional[str] = None,
    traffic_percentage: Optional[int] = 0,
    traffic_split: Optional[Dict[str, int]] = None,
    min_replica_count: int = 1,
    max_replica_count: int = 1,
    accelerator_type: Optional[str] = None,
    accelerator_count: Optional[int] = None,
    explanation_metadata: Optional[explain.ExplanationMetadata] = None,
    explanation_parameters: Optional[explain.ExplanationParameters] = None,
    metadata: Optional[Sequence[Tuple[str, str]]] = (),
    sync: bool = True,
):
    """
    model_name: A fully-qualified model resource name or model ID.
          Example: "projects/123/locations/us-central1/models/456" or
          "456" when project and location are initialized or passed.
    """

    aiplatform.init(project=project, location=location)

    model = aiplatform.Model(model_name=model_name)

    # The explanation_metadata and explanation_parameters should only be
    # provided for a custom trained model and not an AutoML model.
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=deployed_model_display_name,
        traffic_percentage=traffic_percentage,
        traffic_split=traffic_split,
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        explanation_metadata=explanation_metadata,
        explanation_parameters=explanation_parameters,
        metadata=metadata,
        sync=sync,
    )

    model.wait()

    print(model.display_name)
    print(model.resource_name)
    return model


if __name__ == "__main__":
    endpoint = create_endpoint_sample(
        project=PROJECT_NAME,
        display_name="justeat-regressor-endpoint",
        location="us-central1",
    )

    deploy_model_with_dedicated_resources_sample(
        project=PROJECT_NAME,
        location="us-central1",
        model_name="45600045738754048",
        machine_type="n1-standard-2",
        endpoint=endpoint,
    )
