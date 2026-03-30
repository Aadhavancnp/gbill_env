# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client for the Gbill invoice-processing environment."""

from typing import Any

try:
    from openenv.core.env_client import EnvClient
except ImportError:  # pragma: no cover - backward compatibility
    from openenv.core import EnvClient

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import GbillAction, GbillObservation


class GbillEnv(EnvClient[GbillAction, GbillObservation, State]):
    """Typed client for the Gbill OpenEnv server."""

    def _step_payload(self, action: GbillAction) -> dict[str, Any]:
        payload: dict[str, Any] = {"action_type": action.action_type}
        if action.invoice_id is not None:
            payload["invoice_id"] = action.invoice_id
        return payload

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[GbillObservation]:
        obs_data = payload.get("observation", {})
        reward = payload.get("reward", obs_data.get("reward"))
        done = payload.get("done", obs_data.get("done", False))
        observation = GbillObservation(
            invoices=obs_data.get("invoices", []),
            account_balance=obs_data.get("account_balance", 0.0),
            task_description=obs_data.get("task_description", ""),
            last_feedback=obs_data.get("last_feedback", ""),
            done=done,
            reward=reward,
            metadata=obs_data.get("metadata") or {},
        )
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
