
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel


class CheckpointedAgent:
    """
    Wrap an existing Strands agent.

    If a valid checkpoint exists for the current prompt,
    return the saved Pydantic output.

    Otherwise execute the agent and save its result.
    """

    def __init__(
        self,
        name: str,
        agent,
        output_model: type[BaseModel],
        checkpoint_dir: str | Path,
        version: str = "v1",
        refresh: bool = False,
    ):
        self.name = name
        self.agent = agent
        self.output_model = output_model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.version = version
        self.refresh = refresh

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def __call__(self, prompt: str):
        # Different prompts produce different checkpoint keys.
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "name": self.name,
                    "version": self.version,
                    "output_model": self.output_model.__name__,
                    "prompt": prompt,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]

        checkpoint_path = (
            self.checkpoint_dir
            / f"{self.name}_{fingerprint}.json"
        )

        # Load existing result
        if checkpoint_path.exists() and not self.refresh:
            try:
                output = self.output_model.model_validate_json(
                    checkpoint_path.read_text(
                        encoding="utf-8"
                    )
                )

                print(
                    f"[CHECKPOINT] Loaded {self.name} "
                    f"from {checkpoint_path.name}"
                )

                return SimpleNamespace(
                    structured_output=output
                )

            except (ValueError, OSError) as exc:
                print(
                    f"[CHECKPOINT] Could not load {self.name}: "
                    f"{exc}"
                )
                print("[CHECKPOINT] Running agent again.")

        # Execute the real agent
        print(f"[LIVE] Running {self.name}")

        result = self.agent(prompt)
        output = result.structured_output

        if not isinstance(output, self.output_model):
            raise TypeError(
                f"{self.name} returned "
                f"{type(output).__name__}; expected "
                f"{self.output_model.__name__}"
            )

        # Save only successful, schema-valid outputs.
        # Temporary file + replace avoids partial JSON files.
        temporary_path = checkpoint_path.with_suffix(".tmp")

        temporary_path.write_text(
            output.model_dump_json(indent=2),
            encoding="utf-8",
        )

        temporary_path.replace(checkpoint_path)

        print(
            f"[CHECKPOINT] Saved {self.name} "
            f"to {checkpoint_path.name}"
        )

        return result
