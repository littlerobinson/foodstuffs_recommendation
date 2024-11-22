from pydantic import BaseModel
from typing import Union

class TargetProductModelImage(BaseModel):
    code: Union[int, float]
    top_n: Union[int, float]