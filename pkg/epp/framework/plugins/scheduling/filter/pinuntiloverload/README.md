# Pin Until Overload Filter Plugin

**Type:** `pin-until-overload-filter`

A benchmarking filter that creates a load imbalance on purpose. It sends every request to one endpoint until that endpoint is overloaded, then steps aside so the rest of the scheduling profile decides as usual.

## What it does

- **Armed (from startup):** keeps only the target endpoint. On each request it reads the target's model-server waiting queue size. When the queue is at or above `waitingThreshold`, a hold timer starts; a request that sees the queue below the threshold resets it.
- **Released:** once the queue has stayed at or above the threshold for `holdSeconds`, the filter returns its input unchanged for the rest of the process lifetime. It logs the release once, with the target's waiting queue size and KV cache usage.

When the target is not among the candidates, all candidates pass through and the filter stays armed. The release is held in memory, so an EPP restart re-arms the filter.

Place it before the filters whose behavior under imbalance is being measured, for example before `prefix-cache-affinity-filter`.

## Configuration

| Parameter          | Required | Description |
|--------------------|----------|-------------|
| `waitingThreshold` | yes      | Target waiting queue size that starts the hold timer. Must be positive. |
| `holdSeconds`      | no       | How long the queue must stay at or above the threshold before release. Default `0` (release on the first request that sees it). |
| `targetEndpoint`   | no       | Endpoint name to pin to. Default: the first candidate by name on the first request. |

**Configuration Example:**
```yaml
plugins:
  - type: pin-until-overload-filter
    parameters:
      waitingThreshold: 20
      holdSeconds: 30
  - type: prefix-cache-affinity-filter
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: pin-until-overload-filter
      - pluginRef: prefix-cache-affinity-filter
```
