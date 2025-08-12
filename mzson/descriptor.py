from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np


@dataclass
class FrontierDescriptor:
    frontier_id: int
    image_rgb: np.ndarray
    depth: np.ndarray | None
    geom: Dict[str, Any]
    text_descs: List[str]  # stage에 따라 GPT/LLAVA로 생성된 텍스트 N개
    meta: Dict[str, Any]


class DescriptorManager:
    def __init__(self) -> None:
        self._store: Dict[int, List[FrontierDescriptor]] = {}

    def update(self, frontier_id: int, new_descs: List[FrontierDescriptor]) -> None:
        self._store.setdefault(frontier_id, [])
        self._store[frontier_id].extend(new_descs)

    def get_all(self, frontier_id: int) -> List[FrontierDescriptor]:
        return self._store.get(frontier_id, [])


