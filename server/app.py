# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Gbill invoice-processing environment."""

try:
    from openenv.core.env_server import create_app
except ImportError:  # pragma: no cover - compatibility with older layouts
    try:
        from openenv.core.env_server.http_server import create_app
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "openenv is required for the web interface. Install dependencies with "
            "'uv sync' from the gbill_env directory."
        ) from exc

try:
    from ..models import GbillAction, GbillObservation
    from .gbill_env_environment import GbillEnvironment
except ImportError:  # pragma: no cover - direct module execution
    from models import GbillAction, GbillObservation
    from server.gbill_env_environment import GbillEnvironment


app = create_app(
    GbillEnvironment,
    GbillAction,
    GbillObservation,
    env_name="gbill_env",
    max_concurrent_envs=4,
)


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
