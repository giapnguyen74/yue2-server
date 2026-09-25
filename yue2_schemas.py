"""Request bodies for the JSON tasks. Semantic validation (SongRequest, Sampling) happens in the handler."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class SubmitRequest(BaseModel):
    """POST /jobs body. `generate` and `plan` take the YuE2 SongRequest fields; `decode` takes a source job."""
    model_config = ConfigDict(extra="forbid")

    task: Literal["generate", "plan", "decode"] = "generate"
    style: str | None = None
    tags: str | None = None            # alias of style, as in the YuE2 CLI
    lyrics: str | None = None
    cot: Literal["full", "melody", "off"] = "full"
    seed: int | None = None            # random when omitted; echoed back in the status view
    abc: str | None = None
    cfg_scale: float | None = None
    id: str = "song"
    abc_sampling: dict[str, Any] | None = None
    semantic_sampling: dict[str, Any] | None = None
    # decode only
    source_job: str | None = None
    vae: Literal["standard", "legacy"] = "standard"
