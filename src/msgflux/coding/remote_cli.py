"""Run Vulcano as a frontend of the shared local Agent service."""

import argparse
import asyncio
from pathlib import Path

from msgflux.coding.config import load_config
from msgflux.coding.remote_host import RemoteCodingHost
from msgflux.runtime.service.local import connect_local_service


def _parser():
    parser = argparse.ArgumentParser(description="Vulcano: remote coding frontend")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--state-dir", type=Path, default=Path.home() / ".msgflux")
    parser.add_argument("--profile", help="Backend coding profile")
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--thread", help="Resume this thread using its saved workspace")
    parser.add_argument(
        "--factory",
        default="msgflux.coding.remote_backend:create_service",
        help="Trusted backend service factory (MODULE:CALLABLE)",
    )
    return parser


async def _run(args):
    from msgflux.coding.tui import CodingApp  # noqa: PLC0415

    state_dir = args.state_dir.expanduser().resolve()
    config = load_config(state_dir / "config.toml")
    profile = args.profile or config.default_profile
    agent_id = f"{profile}:read-only" if args.read_only else profile

    async def connect():
        return await connect_local_service(
            args.factory, runtime_dir=state_dir / "runtime", cwd=args.workspace
        )

    client = await connect()
    host = RemoteCodingHost(
        client,
        agent_id=agent_id,
        workspace=args.workspace,
        require_agent_id=args.read_only or args.profile is not None,
        connector=connect,
    )
    try:
        session, _ = await host.select(args.thread)
        app = CodingApp(
            session,
            host=host,
            workspace=session.workspace_root or host.workspace,
            observe_immediately=not host.is_new,
        )
        await app.run_async()
    finally:
        await host.aclose()
        await client.aclose()


def main():
    args = _parser().parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
