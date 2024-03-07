## https://github.com/rafaello9472/c4ds/blob/main/Run%20custom%20training%20job%20with%20custom%20container%20in%20Vertex%20AI/Dockerfile

# 1. Base image
FROM python:3.9-buster

# 2. Specify directory where all subsequent instructions are run
WORKDIR /root

# 3. Copy files
COPY train.py /root/train.py
COPY requirements.txt /root/requirements.txt

# 4. Install dependencies
RUN pip install -r /root/requirements.txt

# 5. Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "train.py"]


