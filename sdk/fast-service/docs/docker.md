# Docker

To deploy a service as a docker container.

1. setup a Dockerfile

```dockerfile
FROM python:3.12

COPY fast-service /workspace/fast-service
RUN cd /workspace/fast-service \
    && pip install -r requirements.txt \
    && pip install -e .
CMD cd /workspace/fast-service/example/fastapi/ \
    && python example_computing_service.py
```

2. build an image based Dockerfile
```shell
docker build -t service:v1 .
```

3. run the container
```shell
docker run -d --name service-1 -p 7001:8001 service:v1
```