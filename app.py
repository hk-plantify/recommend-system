from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.recommender import recommend_cards_by_category, get_available_categories
from services.formatter import extract_and_format_benefits_with_llm_batch

app = FastAPI()

# 요청 데이터 모델 정의
class RecommendRequest(BaseModel):
    category: str
    top_n: int = 5  # 기본값 5 설정

@app.get("/categories")
def get_categories():
    categories = get_available_categories()
    return {"available_categories": categories}

@app.post("/recommend")
def recommend_cards(request: RecommendRequest):
    try:
        top_card, other_recommendations = recommend_cards_by_category(
            request.category, request.top_n
        )

        all_cards = [top_card] + list(other_recommendations.to_dict(orient="records"))

        formatted_cards = extract_and_format_benefits_with_llm_batch(all_cards, request.category)

        return {
            "formatted_cards": formatted_cards
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# FastAPI 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
