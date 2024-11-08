from fastapi import APIRouter, HTTPException, Response

import json

router = APIRouter(
    prefix="/fsr",
    responses={404: {"description": "Not found"}},
)


@router.get("/")
async def root():
    """
    Root endpoint to check the status of the foodstuffs recommendation API.

    Returns:
        dict: A message indicating the API is working.
    """
    return {"message": "Hello foodstuffs recommendation 🎉"}
