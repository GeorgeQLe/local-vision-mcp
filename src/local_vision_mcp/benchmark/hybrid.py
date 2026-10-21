"""Execute the conservative local-shadow/native route; never accept unqualified local answers."""

import hashlib
import json
import time
from pathlib import Path

from .providers import Provider
from .routing import recommend, reject_observation


class ShadowProvider:
    def __init__(self, model, root, timeout, stop):
        self.policy = json.loads(Path(model["routing_policy"]).read_text())
        # Validate a fail-closed policy before launching any provider.
        recommend(self.policy, task="", kind="image", model_digest="")
        self.native = Provider({**model, "perception": "native"}, root, timeout, stop)
        try:
            self.local = Provider(
                {
                    "name": "local shadow",
                    "backend": "ollama",
                    "model": model["local_model"],
                    "url": model.get("url", "http://127.0.0.1:11434"),
                    "ollama_options": model.get("ollama_options", {}),
                },
                root,
                timeout,
                stop,
            )
        except BaseException:
            self.native.close()
            raise
        self.digest = self.local.metadata["runtime"]["model"]["digest"]
        self.metadata = {
            "native": self.native.metadata,
            "local": self.local.metadata,
            "policy": self.policy,
            "policy_sha256": hashlib.sha256(Path(model["routing_policy"]).read_bytes()).hexdigest(),
        }

    def analyze(self, case):
        decision = recommend(
            self.policy,
            task=case.get("routing_task"),
            kind=case["kind"],
            model_digest=self.digest,
            allow_shadow=True,
        )
        observation, seconds = None, 0.0
        if decision["local_shadow"]:
            start = time.monotonic()
            try:
                observation = self.local.analyze(case)
            except Exception as exc:  # noqa: BLE001 — local failure must not suppress native work
                observation = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            seconds = time.monotonic() - start
        # Native receives the original pixels/question; shadow text cannot alter the answer.
        native_start = time.monotonic()
        try:
            result = self.native.analyze(case)
        except Exception as exc:  # noqa: BLE001 — retain local work even if native fails
            result = {"answer": "", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}

        return {
            **result,
            "native_seconds": time.monotonic() - native_start,
            "routing_decision": decision,
            "local_observation": observation,
            "local_seconds": seconds,
            "accepted_local": False,
            "local_rejection": reject_observation(
                observation, case.get("request", {}).get("response_schema", {}).get("required", [])
            )
            if observation is not None
            else None,
            "native_called": True,
        }

    def close(self):
        self.local.close()
        self.native.close()
