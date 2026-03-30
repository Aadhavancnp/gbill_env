# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the Gbill invoice-processing environment."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action as OpenEnvAction
    from openenv.core.env_server.types import Observation as OpenEnvObservation
except ImportError:  # pragma: no cover - local fallback when openenv isn't installed
    class OpenEnvAction(BaseModel):
        """Fallback action base for local development without openenv."""

    class OpenEnvObservation(BaseModel):
        """Fallback observation base for local development without openenv."""

        done: bool = False
        reward: Optional[float] = None
        metadata: dict[str, Any] = Field(default_factory=dict)


class Invoice(BaseModel):
    """Invoice entry presented to the agent."""

    id: str = Field(..., description="Unique invoice identifier")
    amount: float = Field(..., ge=0, description="Invoice amount in USD")
    vendor: str = Field(..., min_length=1, description="Vendor name")
    has_po: bool = Field(..., description="Whether a purchase order is attached")
    status: Literal["pending", "approved", "rejected"] = Field(
        default="pending",
        description="Current invoice decision state",
    )


class GbillAction(OpenEnvAction):
    """Action to approve, reject, or finish the invoice-processing task."""

    action_type: Literal["approve", "reject", "finish"] = Field(
        ...,
        description="Choose 'approve', 'reject', or 'finish'",
    )
    invoice_id: Optional[str] = Field(
        default=None,
        description="Invoice ID to act on. Omit this when action_type='finish'.",
    )


class GbillObservation(OpenEnvObservation):
    """Observation containing invoice state and task guidance."""

    invoices: list[Invoice] = Field(default_factory=list, description="Current invoices")
    account_balance: float = Field(..., description="Remaining account balance in USD")
    task_description: str = Field(..., description="Policy instructions for the current task")
    last_feedback: str = Field(..., description="Result of the previous action")


# Friendly aliases that mirror the user-provided wire names while preserving
# the canonical package exports used by the client/server.
Action = GbillAction
Observation = GbillObservation
