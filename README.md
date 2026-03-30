---
title: Gbill Invoice Processing Environment Server
emoji: 🧾
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - finops
---

# Gbill Invoice Processing Environment

`gbill_env` is an OpenEnv environment for invoice triage. The agent must inspect pending bills and decide whether to approve them, reject them, or finish the task.

## Task Levels

- `easy`: approve all pending invoices.
- `medium`: reject invoices over $500 that do not include a PO.
- `hard`: apply the medium rule and reject duplicate invoices based on matching vendor and amount.

## Wire Types

### Action

`GbillAction` contains:

- `action_type`: one of `approve`, `reject`, or `finish`
- `invoice_id`: invoice identifier for approve/reject actions

### Observation

`GbillObservation` contains:

- `invoices`: current invoice list and statuses
- `account_balance`: remaining balance after approvals
- `task_description`: task-specific policy instructions
- `last_feedback`: result of the previous action
- `reward`: numeric reward from the latest step
- `done`: whether the episode is complete

## Quick Start

Install dependencies from the environment directory:

```bash
cd gbill_env
uv sync
```

Run the server locally:

```bash
uv run --project . server --port 8000
```

Interact with it through the typed client:

```python
from gbill_env import GbillAction, GbillEnv

with GbillEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_level="medium")
    print(result.observation.task_description)
    print(result.observation.invoices)

    result = env.step(GbillAction(action_type="approve", invoice_id="inv-001"))
    print(result.observation.last_feedback)

    result = env.step(GbillAction(action_type="finish"))
    print(result.reward, result.done)
```

## Docker Build

```bash
cd gbill_env
docker build -t gbill_env-env:latest -f server/Dockerfile .
```

## OpenEnv Notes

This environment follows the current OpenEnv template shape:

- `models.py` keeps OpenEnv-compatible `Action` and `Observation` subclasses.
- `openenv.yaml` stays in the minimal manifest format expected by the latest template.
- `server/app.py` uses `create_app(...)` to expose reset, step, state, schema, and websocket endpoints.

## Root Inference Script

The repository root contains `inference.py`, which can run all three task levels against a local server using an OpenAI-compatible model endpoint.
