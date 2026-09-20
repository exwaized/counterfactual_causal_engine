"""Shared result contract every module (uplift, ope, geolift) returns.

Keeping one shape across modules means results from very different methods
(a meta-learner CATE, an OPE value estimate, a synthetic-control lift) can be
logged, compared, and plotted the same way.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EffectEstimate:
    method: str
    point_estimate: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    std_error: float | None = None
    n: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    def __repr__(self) -> str:
        ci = ""
        if self.ci_lower is not None and self.ci_upper is not None:
            ci = f" [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        return f"EffectEstimate({self.method}: {self.point_estimate:.4f}{ci}, n={self.n})"
