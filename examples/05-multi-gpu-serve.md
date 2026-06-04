# Multi-GPU Serve Pattern

This guide explains how to scale `tether serve` across multiple GPUs on a single node (e.g., a desktop with 4x RTX 4090s or an AWS `g5.12xlarge` with 4x A10Gs).

## The Architecture

Tether VLA binds to a single GPU per process by design (for deterministic TensorRT execution and maximum throughput). To utilize multiple GPUs, we use a **process-per-GPU** pattern and route traffic through a lightweight reverse proxy (like NGINX or HAProxy).

```
                            ┌──► tether serve (CUDA_VISIBLE_DEVICES=0, :8001)
                            │
Client ──► NGINX (:8000) ───┼──► tether serve (CUDA_VISIBLE_DEVICES=1, :8002)
          (Round Robin)     │
                            └──► tether serve (CUDA_VISIBLE_DEVICES=2, :8003)
```

## Option 1: Docker Compose (Recommended)

The cleanest way to manage this is via `docker-compose`. This ensures each worker gets exactly one isolated GPU and auto-restarts on crashes.

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  # The Load Balancer
  nginx:
    image: nginx:alpine
    ports:
      - "8000:8000"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - worker_0
      - worker_1

  # GPU 0 Worker
  worker_0:
    image: ghcr.io/fastcrest/tether:latest
    command: tether serve /exports/model --host 0.0.0.0 --port 8001
    volumes:
      - ./my_export:/exports/model:ro
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

  # GPU 1 Worker
  worker_1:
    image: ghcr.io/fastcrest/tether:latest
    command: tether serve /exports/model --host 0.0.0.0 --port 8002
    volumes:
      - ./my_export:/exports/model:ro
    environment:
      - CUDA_VISIBLE_DEVICES=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
```

And the accompanying `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream reflex_backend {
        # Least connections is better than round-robin for inference workloads
        least_conn; 
        server worker_0:8001;
        server worker_1:8002;
    }

    server {
        listen 8000;

        location / {
            proxy_pass http://reflex_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            
            # Critical: Timeout settings for inference
            proxy_read_timeout 60s;
            proxy_connect_timeout 5s;
        }
    }
}
```

Run it with:
```bash
docker-compose up -d
```

## Option 2: Bare Metal (systemd / bash)

If you aren't using Docker, you can run multiple instances directly using the `CUDA_VISIBLE_DEVICES` environment variable.

```bash
# Start worker 0 on GPU 0, binding to port 8001
CUDA_VISIBLE_DEVICES=0 tether serve ./my_export --host 127.0.0.1 --port 8001 &

# Start worker 1 on GPU 1, binding to port 8002
CUDA_VISIBLE_DEVICES=1 tether serve ./my_export --host 127.0.0.1 --port 8002 &
```

Then point your proxy (NGINX, HAProxy, or an Envoy sidecar) to `localhost:8001` and `localhost:8002`.

## Handling Warmup

Tether `serve` takes 30-60 seconds to build the TensorRT engine on its first boot (target floor: < 90s). During this time, the `/health` endpoint returns `HTTP 503 Service Unavailable`. Subsequent boots on the same hardware hit the engine cache and start in seconds.

If you are using a standard load balancer, **ensure health checks are enabled and pointing to `/health`**. The load balancer will automatically hold traffic and keep the worker out of the active pool until the engine is built and `/health` returns `HTTP 200 OK`.

## Shared TRT Engine Caching

If all GPUs are the identical architecture (e.g., all A10Gs), they can share the same TensorRT engine cache (`.trt_cache`).
If your node has mixed GPUs (e.g., an RTX 4090 and an RTX 3090), you **must** separate the export directories or disable engine caching, as TensorRT engines are hard-tied to the specific GPU SM architecture they were compiled on.

## When to use this vs `--policy-a` / `--policy-b`

This pattern (process-per-GPU + load balancer) is for **horizontal scale-out**: serving the same model on N GPUs to handle more requests per second. Each worker is identical and statelessly load-balanced.

`tether serve --policy-a ./v1/ --policy-b ./v2/ --split 80` is a different pattern for **A/B testing two different models** on a single GPU with sticky-per-episode routing (same `episode_id` always lands on the same policy shard, preserving prefix cache and RTC state within an episode). Use this when you want to roll a new policy to a percentage of your fleet without breaking cache locality.

The two patterns compose: you can run `--policy-a/--policy-b` A/B serve on each per-GPU worker if you need both horizontal scale and policy A/B testing on the same fleet.
