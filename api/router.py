from fastapi import APIRouter, Response

import handler

from models.target_product_model import TargetProductModel

router = APIRouter(
    prefix="/product",
    responses={404: {"description": "Not found"}},
)


@router.get("/test")
async def test():
    """
    test route
    """
    return {"message": "Test API route work 🎉"}


@router.post("/find_similar_products_text")
async def find_similar_products_text(data: TargetProductModel):
    """
    Search for similar products in the same cluster, avoiding those containing a specific allergen.
    """
    code = data.code
    top_n = data.top_n
    allergen = data.allergen

    response = await handler.find_similar_products_text(code, allergen, top_n)
    return Response(content=response, media_type="application/json")
