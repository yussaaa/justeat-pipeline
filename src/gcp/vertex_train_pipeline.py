from google.cloud import aiplatform as vertexai

PROJECT_NAME = "citric-program-331111"
JOB_NAME = "justeat_regressor_training"
CUSTOM_CONTAINER_IMAGE_URI = "gcr.io/citric-program-331111/justeat-pipeline"
BUCKET_NAME = "cloud-ai-platform-4de6a256-deb4-4f4f-8b9f-8595e0183ea1"


def main():
    """This is the sdk version of model training. It's an alternative to the gcloud command line tool.
        gcloud ai custom-jobs create \
            --region=${{env.REGION}} \
            --display-name=${{env.JOB_NAME}} \
            --config=vertex_run.yaml
    """
    vertexai.init(project=PROJECT_NAME, staging_bucket=BUCKET_NAME)

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-8",
            },
            "replica_count": 1,
            "container_spec": {"image_uri": CUSTOM_CONTAINER_IMAGE_URI},
        }
    ]
    print(worker_pool_specs)

    job = vertexai.CustomJob(
        display_name=f"{JOB_NAME}", worker_pool_specs=worker_pool_specs
    )

    job.run(
        sync=False,
    )


if __name__ == "__main__":
    main()
