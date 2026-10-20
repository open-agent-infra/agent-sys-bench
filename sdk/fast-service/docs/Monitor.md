# Monitor

All the monitor components are opensource software. We directly use related opensource docker image to run these components.

There are 4 components:
1. Prometheus: monitor data storage
2. Grafana: monitor data visualization
3. cAdvisor: docker container resource monitor
4. Nvidia dcgm-exporter: docker container GPU usage monitor

When the access to a host is protected, use an ssh proxy like this:

```shell
ssh -L 8080:remote_server:3306 user@remote_server
```

Keep the connection until you finish monitoring.

## Prometheus

1. pull the latest image
```shell
docker pull prom/prometheus
```

2. run a container and let it listen given port
```shell
docker run \
    --name prometheus \
    -p 9090:9090 \
    -v ./prometheus.yml:/etc/prometheus/prometheus.yml \
    -v prometheus-data:/prometheus \
    prom/prometheus
```

3. visit the port to see if prometheus has been set up successfully

## Grafana

1. pull the latest image

```shell
docker pull grafana/grafana-enterprise
```

2. create a volume to store grafana data


```shell
docker volume create grafana-storage
```

3. run a container and let it listen given port
```shell
docker run -d -p 3000:3000 --name=grafana \
  --volume grafana-storage:/var/lib/grafana \
  grafana/grafana-enterprise
```

4. visit the port to see if grafana has been set up successfully


## cAdvisor

1. pull the latest image

```shell
docker pull gcr.io/cadvisor/cadvisor
```
2. run a container and let it listen given port
```shell
docker run \
  --volume=/:/rootfs:ro \
  --volume=/var/run:/var/run:ro \
  --volume=/sys:/sys:ro \
  --volume=/var/lib/docker/:/var/lib/docker:ro \
  --volume=/dev/disk/:/dev/disk:ro \
  --publish=8080:8080 \
  --detach=true \
  --name=cadvisor \
  --privileged \
  --device=/dev/kmsg \
  gcr.io/cadvisor/cadvisor:latest
```

3. visit the ```ip:port/docker``` to see if cAdvisor has been set up successfully
4. configure the grafana dashboard with importId 11277

## DCGM

1. pull the latest image

```shell
docker pull nvidia/dcgm-exporter
```
2. run a container and let it listen given port
```shell
docker run -d --gpus all -p 9400:9400 nvidia/dcgm-exporter:latest
```

3. visit the ```ip:port/metrics``` to see if cAdvisor has been set up successfully
4. configure the grafana dashboard with importId 12239

## Deployment integration

The `benchmark/deployment/<setting>/docker-compose.yaml` files bring this
stack up automatically (cAdvisor + Prometheus + Grafana) for each agent run.
To match the named-volume usage above and avoid bind-mount permission
friction (Docker creates bind-mount dirs as root; Prometheus/Grafana run
non-root), the **writable data dirs use named volumes**:

- `prometheus-data:/prometheus` (TSDB)
- `grafana-data:/var/lib/grafana` (Grafana state)

Read-only config stays on bind-mounts so it can be checked into the repo:
`./monitoring/prometheus.yml`, `./monitoring/grafana/provisioning/`, and
`./monitoring/grafana/dashboards/`. Data in the named volumes survives
`docker compose down` (use `down -v` to discard). To export a portable
TSDB snapshot, use the Prometheus admin snapshot API against the running
container.


