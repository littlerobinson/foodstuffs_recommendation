from pydantic import BaseModel
from typing import Union


class TargetProductModel(BaseModel):
    code: Union[int, float]
    top_n: Union[int, float]
    allergen: str
