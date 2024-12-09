from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services.recommender import recommend_cards_by_category, get_available_categories
from services.formatter import extract_and_format_benefits_with_llm_batch

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendRequest(BaseModel):
    category: str
    top_n: int = 5

@app.get("/healthz")
def health_check():
    return {"status": "ok"}

@app.get("/categories")
def get_categories():
    try:
        categories = get_available_categories()
        return {"available_categories": categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch categories: {str(e)}")

@app.post("/recommend")
def recommend_cards(request: RecommendRequest):
    try:
        recommendation_result = recommend_cards_by_category(
            category=request.category, top_n=request.top_n
        )

        return recommendation_result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# FastAPI 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
