# Fleet telemetry

When you deploy Tether one process per robot, `--robot-id` gives each process a human-readable identity that Prometheus + Grafana can pivot on. You see per-robot p99, per-robot error rates, per-robot safety violations — all in one dashboard.

Zero cost when you're not using it. Single-robot deploys see no extra cardinality.

## Quick start

```bash
# On robot A:
tether serve ./my-export/ --robot-id warehouse-01 --port 8000

# On robot B:
tether serve ./my-export/ --robot-id warehouse-02 --port 8000

# On robot C:
tether serve ./my-export/ --robot-id arm-prototype-alpha --port 8000
```

Your Prometheus config scrapes each instance. In Grafana, import [`dashboards/reflex_fleet.json`](../dashboards/reflex_fleet.json) and select one or more robots from the dropdown.

## How it works

Each `tether serve` process exports a single info-style gauge:

```
reflex_robot_info{robot_id="warehouse-01",embodiment="franka",model_id="pi0-libero"} 1
```

Grafana joins hot metrics to this gauge via `instance`:

```promql
histogram_quantile(0.99,
  sum by (le, instance) (rate(reflex_act_latency_seconds_bucket[5m]))
) * on (instance) group_left(robot_id) reflex_robot_info
```

Result: `robot_id` appears as a label on p99 latency even though the underlying histogram doesn't carry it. Cardinality stays flat (one series per process on `reflex_robot_info`, not one per request on every histogram).

## Why not put robot_id as a label on every metric?

Cardinality. A fleet of 1,000 robots × 3 embodiments × 6 models × N metrics = hundreds of thousands of series. Prometheus handles that but pays memory for it, and most per-label slicing an operator actually wants (per-robot, not per-(robot × embodiment)) is available from the info-metric join.

Net: we keep the existing label set tight (embodiment, model_id, violation_kind, etc.) and let operators opt into per-robot slicing via `--robot-id` + the info join.

## Endpoints that expose robot_id

Every `tether serve` process surfaces the robot_id via:

- `GET /health` — `"robot_id": "warehouse-01"` in the JSON body
- `GET /config` — same key
- `GET /metrics` — `reflex_robot_info{robot_id="warehouse-01",...}` (when set)

When `--robot-id` is unset, `robot_id` is `""` on `/health` and `/config`, and no `reflex_robot_info` series is emitted.

## Alerting on a single robot

A typical Prometheus rule that pages for a specific robot's p99 breach:

```yaml
- alert: ReflexRobotLatencyHigh
  expr: |
    histogram_quantile(0.99,
      sum by (le, instance) (rate(reflex_act_latency_seconds_bucket[5m]))
    ) * on (instance) group_left(robot_id) reflex_robot_info{robot_id="warehouse-01"} > 0.2
  for: 3m
  labels: { severity: page, robot_id: warehouse-01 }
  annotations:
    summary: "Tether on {{ $labels.robot_id }} over p99=200ms"
```

Drop the `robot_id=` filter to alert on any robot in the fleet.

## Deployment patterns

### One process per robot (recommended)

```bash
# systemd unit per robot
ExecStart=/usr/local/bin/tether serve /opt/tether/export \
    --robot-id %H \
    --port 8000 \
    --slo p99=150ms
```

Using the hostname macro (`%H` in systemd) gives each robot its own identity without hand-editing units.

### Central aggregator

Don't. Tether does per-process inference; a central aggregator adds network latency that violates the real-time invariant. Instead, scrape each robot's `/metrics` from a central Prometheus and render one dashboard against the aggregate.

## What's not shipped yet

- **TelemetryEvent streaming** — structured event bus to a customer collector (Phase 2). Today's telemetry is pull-based via Prometheus scrape.
- **Per-robot resource metrics** — GPU memory, CPU load, disk I/O. Node-exporter on each robot host is the right cohabiting pattern; we don't duplicate it.
