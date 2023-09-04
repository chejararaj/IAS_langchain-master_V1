from fastapi import FastAPI
from app.langchain.v1.api import router as langchain_v1

PREFIX_V1 = "/langchain"
TAGS_METADATA = [
    {
        "name": "Langchain Services",
        "description": """Langchain Services""",
    },
]

app = FastAPI(
    title="Langchain Services",
    openapi_tags=TAGS_METADATA,
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)
# async def catch_exceptions_middleware(request: Request, call_next):
#     try:
#         return await call_next(request)
#     except Exception as e:
#         logging.getLogger().error(str(e))
#         raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

# app.middleware('http')(catch_exceptions_middleware)

app.include_router(langchain_v1, prefix=PREFIX_V1)
