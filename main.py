from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from database import SessionLocal
from models import Persona, Keyword, PersonaKeyword, Quest, Content
from sqlalchemy.orm import Session
import os
from dotenv import load_dotenv
import google.generativeai as genai

from models import Base
from database import engine


Base.metadata.create_all(bind=engine)

app = FastAPI()
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
class SurveyItem(BaseModel):
    number: int
    question: str
    answer: str


def analyze_survey_answers(survey: list[dict]) -> str:
    content = "사용자의 설문 응답을 기반으로 강점, 단점, 그리고 보완되면 좋을 성향 키워드를 도출해줘.\n\n"
    content += "다음은 사용자의 응답이야:\n"

    for item in survey:
        content += f"{item['number']}. {item['question']}\n→ {item['answer']}\n\n"

    content += """
결과는 다음 JSON 형식으로만 응답해줘. 설명 없이 결과만 출력해. 반드시 하나의 리스트 안에 하나의 객체만 포함해야 해.

[
  {
    "strong": "사용자의 강점을 한 문장으로",
    "weakness": "사용자의 단점을 한 문장으로",
    "keyword": ["보완하면 좋을 키워드1", "키워드2", "키워드3", "키워드4", "키워드5"]
  }
]
"""

    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
    response = model.generate_content(content)
    return response.text.strip()


import json
import re


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
        raise HTTPException(status_code=400, detail="분석 결과가 유효하지 않습니다.")

    persona_data = parsed[0]
    keyword_list = persona_data.get("keyword", [])
    keyword_str = ", ".join(keyword_list) if isinstance(keyword_list, list) else str(keyword_list)

    db: Session = SessionLocal()
    try:
        new_persona = Persona(
            strong=persona_data["strong"],
            weakness=persona_data["weakness"],
            keyword=keyword_str
        )
        db.add(new_persona)
        db.commit()
        db.refresh(new_persona)
    finally:
        db.close()

    return {
        "message": "분석 완료 및 저장 성공",
        "result": parsed
    }

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

    content = Content(
        description=description,
        keyword=keyword_str
    )
    db.add(content)
    db.commit()
    db.refresh(content)
    db.close()

    return {
        "message": "최종 페르소나 저장 완료",
        "data": {
            "description": description,
            "keywords": keyword_str
        }
    }


def generate_quests_from_keywords(keywords: List[str]) -> List[str]:
    prompt = f"""
너는 라이프코치이자 작가야.
사용자가 아래 키워드를 실천할 수 있도록 구체적인 액션 기반 퀘스트를 만들어줘.

조건:
- 키워드: {", ".join(keywords)}
- 각 키워드당 1개의 퀘스트 생성
- 퀘스트는 간단 명료한 실천문장 (예: "하루에 한 번 감사일기 쓰기")
- JSON 배열로 응답: ["~하기", "~시도해보기", "~실천하기" 등]

설명 없이 결과만 JSON으로 줘.
"""

    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
    response = model.generate_content(prompt)

    try:
        return json.loads(response.text.strip())
    except:
        return []


@app.get("/api/questions")
def generate_and_save_quests():
    db: Session = SessionLocal()
    latest_content = db.query(Content).order_by(Content.id.desc()).first()
    if not latest_content or not latest_content.keyword:
        db.close()
        raise HTTPException(status_code=404, detail="contents에 키워드가 없습니다.")
    keywords = [kw.strip() for kw in latest_content.keyword.split(",")]
    existing_quests = db.query(Quest).all()
    if existing_quests:
        result = [{"id": q.id, "question": q.mission_text, "state": q.state} for q in existing_quests]
        db.close()
        return result
    quests = generate_quests_from_keywords(keywords)
    if not quests:
        db.close()
        raise HTTPException(status_code=400, detail="퀘스트 생성 실패")

    saved = []
    for quest_text in quests:
        quest = Quest(mission_text=quest_text, state="NOT")
        db.add(quest)
        saved.append({
            "question": quest_text,
            "state": "NOT"
        })
    db.commit()
    db.close()

    return saved

from fastapi import Body

from fastapi import Body
from typing import Dict

@app.patch("/api/questions")
def update_quests_state(update_data: Dict[int, str] = Body(...)):
    db = SessionLocal()
    updated = []

    for quest_id, state in update_data.items():
        if state not in ["SUCCESS", "NOT"]:
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
