from fastapi import FastAPI

import router

tags_metadata = [
    {"name": "data", "description": "Show data"},
    {"name": "machine-learning", "description": "Prediction Endpoint."},
]

app = FastAPI(
    title="🪐 Foodstuff Recommendation API",
    description="API for Foodstuff recommendation",
    version="0.1",
    contact={
        "url": "https://github.com/littlerobinson",
    },
    openapi_tags=tags_metadata,
)

app.include_router(router.router)


@app.get("/", tags=["data"])
async def root():
    return {"message": "Hello foodstuffs recommendation 🎉"}
