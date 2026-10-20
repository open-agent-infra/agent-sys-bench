from typing import Union, TypedDict, Sequence

Vector = Union[Sequence[float], Sequence[int]]


class KnwoledgeBaseEntity(TypedDict):
    id: int
    vector: Vector
    text: str
    subject: str
