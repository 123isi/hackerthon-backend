from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict
from database import SessionLocal, engine
from models import Persona, Quest, Content, Base
from sqlalchemy.orm import Session
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
import re

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 모든 origin 허용
    allow_credentials=True,     # 인증 정보 포함 허용 (주의: allow_origins=["*"] 와 함께 사용 시 제한 있음)
    allow_methods=["*"],        # 모든 메서드 허용
    allow_headers=["*"],        # 모든 헤더 허용
)


Base.metadata.create_all(bind=engine)

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# 모델
class SurveyItem(BaseModel):
    number: int
    question: str
    answer: str

def analyze_survey_answers(survey: list[dict]) -> str:
    content = (
        "사용자의 설문 응답을 기반으로 아래 형식에 맞춰 분석해줘.\n\n"
        "1. 강점 3가지를 '강점명: 설명' 형식으로 작성해줘.\n"
        "2. 약점 3가지를 '약점명: 설명' 형식으로 작성해줘.\n"
        "3. 보완하면 좋을 키워드 5개를 키워드명만 리스트로 제공해줘 (설명 없이).\n\n"
        "다음은 사용자의 응답이야:\n"
    )

    for item in survey:
        content += f"{item['number']}. {item['question']}\n→ {item['answer']}\n\n"

    content += """
결과는 다음 JSON 형식으로만 응답해줘. 설명 없이 결과만 출력해. 반드시 하나의 리스트 안에 하나의 객체만 포함해야 해.

[
  {
    "strong": [
      "강점명1: 설명1",
      "강점명2: 설명2",
      "강점명3: 설명3"
    ],
    "weakness": [
      "약점명1: 설명1",
      "약점명2: 설명2",
      "약점명3: 설명3"
    ],
    "keyword": [
      "키워드1",
      "키워드2",
      "키워드3",
      "키워드4",
      "키워드5"
    ]
  }
]
"""

    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
    response = model.generate_content(content)
    return response.text.strip()


def parse_result(text: str) -> dict:
    cleaned = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"error": "Gemini 응답 파싱 실패", "raw": text}

@app.post("/api/survey")
def submit_survey(survey: List[SurveyItem]):
    survey_data = [item.dict() for item in survey]
    result_text = analyze_survey_answers(survey_data)
    parsed = parse_result(result_text)

    if not parsed or not isinstance(parsed, list) or "strong" not in parsed[0]:
        raise HTTPException(status_code=400, detail="분석 결과가 유용하지 않습니다.")

    persona_data = parsed[0]
    keyword_list = persona_data.get("keyword", [])
    keyword_str = ", ".join(keyword_list)
    strong_text = "\n".join(persona_data["strong"])
    weakness_text = "\n".join(persona_data["weakness"])

    db: Session = SessionLocal()
    try:
        new_persona = Persona(strong=strong_text, weakness=weakness_text, keyword=keyword_str)
        db.add(new_persona)
        db.commit()
        db.refresh(new_persona)
    finally:
        db.close()

    return {"message": "분석 완료 및 저장 성공", "result": parsed}

# 페르소나 설명 생성
def generate_persona_description(strong: str, keywords: List[str]) -> str:
    prompt = f"""
당신은 감성적인 작가입니다.

다음은 사용자의 성격 정보입니다.
강점: {strong}
보완 키워드: {", ".join(keywords)}

이 정보를 바탕으로 하나의 통합된 페르소나 설명을 작성해주세요.
- 4~6문장으로 구성된 한 문단
- 인간적인 서술
- 키워드들을 설명에 자연스럽게 포함
"""
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    return response.text.strip()

@app.post("/api/survey/key")
def create_final_persona(keywords: List[str]):
    db: Session = SessionLocal()
    persona = db.query(Persona).order_by(Persona.id.desc()).first()
    if not persona:
        raise HTTPException(status_code=404, detail="먼저 퍼소나 분석을 진행해주세요.")

    keyword_str = ", ".join(keywords)
    description = generate_persona_description(persona.strong, keywords)

    content = Content(description=description, keyword=keyword_str)
    db.add(content)
    db.commit()
    db.refresh(content)
    db.close()

    return {"message": "최종 페르소나 저장 완료", "data": {"description": description, "keywords": keyword_str}}
def generate_quests_from_keywords(keywords: List[str]) -> List[str]:
    prompt = f"""
너는 라이프코치이자 작가야.
사용자가 아래 키워드를 실천할 수 있도록 구체적인 액션 기반 퀘스트를 만들어줘.

조건:
- 키워드: {", ".join(keywords)}
- 각 키워드당 1개의 퀘스트 생성
- 퀘스트는 간단 명료한 실천문장
- 비슷한 퀘스트가 있다면 다른 표현이나 방식으로 바꿔줘
- JSON 배열로 응답: ["~하기", "~시도해보기" 등]

설명 없이 결과만 JSON으로 줘.
"""

    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
    response = model.generate_content(prompt)

    raw_text = response.text.strip()
    print("🔍 Gemini 응답:", raw_text)  # 로그 찍기

    try:
        return json.loads(raw_text)
    except Exception as e:
        print("❌ JSON 파싱 실패:", e)
        return []



@app.get("/api/questions")
def generate_and_save_quests():
    db: Session = SessionLocal()

    latest_content = db.query(Content).order_by(Content.id.desc()).first()
    if not latest_content or not latest_content.keyword:
        db.close()
        raise HTTPException(status_code=404, detail="contents에 키워드가 없습니다.")

    keywords = [kw.strip() for kw in latest_content.keyword.split(",")]

    # 기존 퀘스트 조회
    existing_quests = db.query(Quest).all()

    # 하나라도 NOT이 있으면 새로 만들지 말고 기존 것 반환
    if any(q.state == "NOT" for q in existing_quests):
        result = [{"id": q.id, "question": q.mission_text, "state": q.state} for q in existing_quests]
        db.close()
        return result

    # 모든 퀘스트가 SUCCESS인 경우 → 새 퀘스트 생성
    existing_texts = set(q.mission_text for q in existing_quests)

    new_quests = generate_quests_from_keywords(keywords)
    if not new_quests:
        db.close()
        raise HTTPException(status_code=400, detail="퀘스트 생성 실패")

    saved = []
    for quest_text in new_quests:
        if quest_text not in existing_texts:  # 중복 제거
            quest = Quest(mission_text=quest_text, state="NOT")
            db.add(quest)
            saved.append({"question": quest_text, "state": "NOT"})

    db.commit()

    # 전체 퀘스트 반환
    all_quests = db.query(Quest).all()
    result = [{"id": q.id, "question": q.mission_text, "state": q.state} for q in all_quests]

    db.close()
    return result


from typing import Dict
from fastapi import Body

@app.patch("/api/questions")
def update_quests_state(update_data: Dict[str, str] = Body(...)):
    db = SessionLocal()
    updated = []

    for quest_id_str, state in update_data.items():
        if state not in ["SUCCESS", "NOT"]:
            continue

        try:
            quest_id = int(quest_id_str)
        except ValueError:
            continue

        quest = db.query(Quest).filter(Quest.id == quest_id).first()
        if quest:
            quest.state = state
            updated.append({"id": quest_id, "state": state})

    db.commit()
    db.close()

    return {
        "message": "퀘스트 상태 업데이트 완료",
        "updated": updated
    }
