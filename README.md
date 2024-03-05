# justeattakeaway-pipeline

## Problem description:

You are presented with a dataset containing the courier locations captured during food collection at restaurants during a time interval. This data is used to generate a group of features with the intention of modeling the busyness of a certain geographical region. The team was tasked with developing a model to predict this, and the data scientist has produced a working Proof of Concept (PoC). Now, as an ML Engineer, you are tasked with productionizing this PoC.
This notebook contains the data scientist’s code to collect and create geo-location features to describe the busyness of regions (defined as h3 hexagons), and then train an ML model. The data scientist rushed to produce the PoC notebook, so the code is not well structured for a production application. As an ML engineer, your task is to:

1. Define a structured ML pipeline project.
2. Refactor the notebook into separate files to produce an executable ML pipeline using software engineering best practices and object-oriented programming appropriately.

## Expected outcome:

A structured ML pipeline project in a Git repo that you will be expected to talk us through and explain your design choices at the next stage.
It should at least contain:

- [x] Scripts for each step
- [x] Training and prediction pipelines
- [x] Configurations file/s
- [x] Dependency management
- [ ] CI/CD

### Hints:

We suggest containerization with Docker, using GCS for storage, Vertex AI to execute the pipeline, and GitHub Actions for CI/CD. But if you feel more comfortable with other tools that is ok.

Consider creating files for each step, for example, `data_collection.py`, `feature_generation.py`, `training.py`, and `prediction.py`, in addition to pipeline and config files to connect and execute the pipeline. Some features might be poorly implemented or not be in use. Your focus as an ML Engineer is refactoring the notebook into a structure project, but you can highlight any implementation issues you spot.
