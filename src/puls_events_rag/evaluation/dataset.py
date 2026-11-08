"""Jeu de test annoté ``eval/dataset.jsonl`` (#13) : schéma et chargement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

DATASET_PATH = Path("eval/dataset.jsonl")
Category = Literal["theme", "lieu", "date", "sans_reponse", "hors_sujet", "hors_zone"]
ANSWERABLE: frozenset[str] = frozenset({"theme", "lieu", "date"})


class EvalItem(BaseModel):
    """Une question annotée : référence rédigée et événements attendus."""

    id: str = Field(pattern=r"^[a-z0-9-]+$")
    category: Category
    question: str = Field(min_length=3)
    reference: str = Field(min_length=3)
    expected_uids: list[str] = Field(default_factory=list)
    expected_total: int = Field(ge=0)

    @property
    def answerable(self) -> bool:
        return self.category in ANSWERABLE

    @model_validator(mode="after")
    def _coherent(self) -> EvalItem:
        if self.answerable and not self.expected_uids:
            raise ValueError(f"{self.id} : question avec réponse sans événement attendu")
        if not self.answerable and self.expected_uids:
            raise ValueError(f"{self.id} : question sans réponse avec événements attendus")
        if len(self.expected_uids) > self.expected_total:
            raise ValueError(f"{self.id} : expected_total incohérent")
        return self


def load_dataset(path: Path = DATASET_PATH) -> list[EvalItem]:
    with path.open(encoding="utf-8") as handle:
        items = [EvalItem.model_validate(json.loads(line)) for line in handle if line.strip()]
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("identifiants de questions en double")
    return items
