# Agentic MCP lifecycle contract

The reusable implementation and canonical contract have moved to
[agentic-mcp-foundation](https://github.com/GeorgeQLe/agentic-mcp-foundation).
See its [lifecycle contract](https://github.com/GeorgeQLe/agentic-mcp-foundation/blob/main/docs/lifecycle-contract.md)
for job states, idempotency, cancellation, process ownership, discovery, launcher validation,
publication recovery and conformance evidence.

Local-vision depends on an exact Git commit in `pyproject.toml`, also recorded in `uv.lock`.
It does not vendor a second copy. Registry dependencies remain hash-locked during installation;
the Git dependency is built separately from its immutable commit pin.

The foundation owns the generic supervisor and lifecycle primitives. Local-vision owns allowed
media roots, snapshots and hashes, Ollama provisioning, model downloads, capability readiness,
vision probes and workload qualification. FormSeed has not been migrated.

Generate another working adapter from a local-vision checkout:

```bash
uv sync
uv run agentic-mcp scaffold receipt-demo --output /tmp/receipt-demo
cd /tmp/receipt-demo
uv sync
uv run python -m receipt_demo
```

The starter demonstrates durable jobs, reconnect recovery and cancellation. Its domain is echo;
backend-specific readiness, installation, probes and standalone distribution remain explicit
adapter integrations. The generated project resolves the foundation's Git tag into its own lockfile.
