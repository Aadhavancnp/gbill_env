# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Invoice-processing environment implementation for OpenEnv."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except ImportError:  # pragma: no cover - local fallback when openenv isn't installed
    class State(BaseModel):
        """Fallback state model for local development without openenv."""

        episode_id: str
        step_count: int = 0

    class Environment:
        """Fallback interface for local development without openenv."""

        SUPPORTS_CONCURRENT_SESSIONS: bool = True

try:
    from ..models import GbillAction, GbillObservation, Invoice
except ImportError:  # pragma: no cover - standalone imports
    from models import GbillAction, GbillObservation, Invoice


class GbillEnvironment(Environment):
    """Environment where an agent processes invoices under simple policies."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    VALID_LEVELS = {"easy", "medium", "hard"}

    def __init__(self, task_level: str = "easy"):
        self.task_level = task_level if task_level in self.VALID_LEVELS else "easy"
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.invoices: list[Invoice] = []
        self.balance = 1000.0
        self.last_feedback = "System ready. Process the pending bills."
        self.task_desc = ""
        self.reset(task_level=self.task_level)

    def _generate_invoices(self) -> list[Invoice]:
        invoices = [
            Invoice(id="inv-001", amount=50.0, vendor="Software Sub", has_po=True),
            Invoice(id="inv-002", amount=75.0, vendor="Office Supplies", has_po=False),
        ]
        if self.task_level in {"medium", "hard"}:
            invoices.append(
                Invoice(
                    id="inv-003",
                    amount=600.0,
                    vendor="Consulting",
                    has_po=False,
                )
            )
        if self.task_level == "hard":
            invoices.append(
                Invoice(
                    id="inv-001-dup",
                    amount=50.0,
                    vendor="Software Sub",
                    has_po=True,
                )
            )
        return invoices

    def _set_task_description(self) -> None:
        if self.task_level == "easy":
            self.task_desc = "Task: Approve all pending bills. They are all valid."
        elif self.task_level == "medium":
            self.task_desc = (
                "Task: Approve valid bills. Reject any bill over $500 if it lacks a PO."
            )
        else:
            self.task_desc = (
                "Task: Approve valid bills. Reject bills over $500 without a PO. "
                "Also reject duplicate bills based on vendor and amount."
            )

    def _build_observation(
        self,
        *,
        reward: float = 0.0,
        done: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> GbillObservation:
        return GbillObservation(
            invoices=self.invoices,
            account_balance=self.balance,
            task_description=self.task_desc,
            last_feedback=self.last_feedback,
            reward=reward,
            done=done,
            metadata=metadata or {"task_level": self.task_level},
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> GbillObservation:
        requested_level = kwargs.get("task_level", self.task_level)
        self.task_level = requested_level if requested_level in self.VALID_LEVELS else "easy"
        self.invoices = self._generate_invoices()
        self.balance = 1000.0
        self.last_feedback = "System ready. Process the pending bills."
        self._set_task_description()
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        return self._build_observation(
            reward=0.0,
            done=False,
            metadata={"task_level": self.task_level, "seed": seed},
        )

    def step(
        self,
        action: GbillAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> GbillObservation:
        del timeout_s, kwargs
        self._state.step_count += 1

        if action.action_type == "finish":
            final_score = self._grade_task()
            self.last_feedback = f"Task completed. Final score: {final_score:.2f}"
            return self._build_observation(reward=final_score, done=True)

        if not action.invoice_id:
            self.last_feedback = "Error: invoice_id is required for approve/reject actions."
            return self._build_observation(reward=-0.1, done=False)

        target = next((inv for inv in self.invoices if inv.id == action.invoice_id), None)
        if target is None:
            self.last_feedback = f"Error: Invoice {action.invoice_id} not found."
            return self._build_observation(reward=-0.1, done=False)

        if target.status != "pending":
            self.last_feedback = f"Error: Invoice {target.id} is already {target.status}."
            return self._build_observation(reward=-0.1, done=False)

        if action.action_type == "approve":
            target.status = "approved"
            self.balance -= target.amount
            self.last_feedback = f"Approved {target.id}. New balance: ${self.balance:.2f}"
            return self._build_observation(reward=0.1, done=False)

        if action.action_type == "reject":
            target.status = "rejected"
            self.last_feedback = f"Rejected {target.id}."
            return self._build_observation(reward=0.1, done=False)

        self.last_feedback = f"Error: Unsupported action '{action.action_type}'."
        return self._build_observation(reward=-0.1, done=False)

    def _grade_task(self) -> float:
        """Return a final task score between 0.0 and 1.0."""
        statuses = {invoice.id: invoice.status for invoice in self.invoices}

        if self.task_level == "easy":
            if (
                statuses.get("inv-001") == "approved"
                and statuses.get("inv-002") == "approved"
            ):
                return 1.0
            return 0.0

        if self.task_level == "medium":
            if (
                statuses.get("inv-001") == "approved"
                and statuses.get("inv-002") == "approved"
            ):
                if statuses.get("inv-003") == "rejected":
                    return 1.0
                return 0.5
            return 0.0

        score = 0.0
        if statuses.get("inv-001") == "approved" and statuses.get("inv-002") == "approved":
            score += 0.33
        if statuses.get("inv-003") == "rejected":
            score += 0.33
        if statuses.get("inv-001-dup") == "rejected":
            score += 0.34
        return min(1.0, score)

    @property
    def state(self) -> State:
        return self._state
