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
    """
    사용 가능한 카테고리를 반환.
    """
    categories = get_available_categories()
    return {"available_categories": categories}

@app.post("/recommend/")
def recommend_cards(request: RecommendRequest):
    """
    추천 카드와 혜택 정보를 반환.
    """
    try:
        # 추천 카드 가져오기
        top_card, other_recommendations = recommend_cards_by_category(
            request.category, request.top_n
        )

        # 모든 카드 정보를 합침
        all_cards = [top_card] + list(other_recommendations.to_dict(orient="records"))

        # LLM 호출로 카드 정보를 JSON 포맷으로 변환
        formatted_cards = extract_and_format_benefits_with_llm_batch(all_cards, request.category)

        # 결과 반환
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
