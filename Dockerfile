FROM python:3.10
WORKDIR /root

# Update pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

COPY . .

RUN pip install pip install --no-cache-dir -r requirements.txt

CMD ["python", "./src/pipeline.py"]