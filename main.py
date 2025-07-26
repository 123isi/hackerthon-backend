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
origins = [
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # origin 리스트
    allow_credentials=True,
    allow_methods=["*"],             # 모든 HTTP 메서드 허용
    allow_headers=["*"],             # 모든 헤더 허용
)

Base.metadata.create_all(bind=engine)

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
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
import re
import json

def generate_quests_from_keywords(keywords: List[str]) -> List[str]:
    prompt = f"""
너는 라이프코치이자 작가야.
아래 키워드를 참고해서 사용자가 실천할 수 있는 퀘스트를 총 10개 만들어줘.

조건:
- 키워드: {", ".join(keywords)}
- 총 10개의 퀘스트 생성
- 퀘스트는 간단 명료한 실천 문장
- 비슷한 내용은 표현 다르게 해줘
- 반드시 JSON 배열로만 응답해줘 (설명 없이, ```json 같은 것도 절대 붙이지 마)

예시:
[
  "하루 10분 명상하기",
  "물 충분히 마시기",
  "잠들기 전 스트레칭하기"
]
"""

    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
    response = model.generate_content(prompt)

    raw_text = response.text.strip()
    print("🔍 Gemini 응답:", raw_text)

    # 1. 마크다운 제거
    cleaned = re.sub(r"^```json", "", raw_text)
    cleaned = re.sub(r"```$", "", cleaned).strip()

    # 2. JSON 배열 추출
    try:
        json_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not json_match:
            raise ValueError("JSON 배열을 찾을 수 없음")
        json_text = json_match.group(0)
        return json.loads(json_text)
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
    new_quests = generate_quests_from_keywords(keywords)

    if not new_quests:
        db.close()
        raise HTTPException(status_code=400, detail="퀘스트 생성 실패")

    saved = []
    for quest_text in new_quests[:10]:  # 최대 10개만 저장
        quest = Quest(mission_text=quest_text, state="NOT")
        db.add(quest)
        db.flush()  # ID 생성
        saved.append({"id": quest.id, "question": quest_text, "state": "NOT"})

    db.commit()
    db.close()
    return saved




from typing import Dict
from fastapi import Body

@app.patch("/api/questions")
def update_quests_state(update: Dict[str, str] = Body(...)):
    db = SessionLocal()
    updated = []

    for quest_id_str, state in update.items():
        if state.upper() not in ["SUCCESS", "NOT"]:
            continue

        try:
            quest_id = int(quest_id_str)
        except ValueError:
            continue

        quest = db.query(Quest).filter(Quest.id == quest_id).first()
        if quest:
            quest.state = state.upper()
            updated.append({"id": quest_id, "state": state.upper()})

    db.commit()
    db.close()

    return {
        "message": "퀘스트 상태 업데이트 완료",
        "updated": updated
    }


@app.post("/api/conversation/init")
def init_conversation():
    db = SessionLocal()
    persona = db.query(Persona).order_by(Persona.id.desc()).first()
    if not persona:
        raise HTTPException(status_code=404, detail="분석된 페르소나가 없습니다.")

    system_prompt = f"""
너는 이름이 '추구미'인 감성 조력자야.
사용자의 성향은 다음과 같아:

강점:
{persona.strong}

약점:
{persona.weakness}

보완 키워드:
{persona.keyword}

너는 따뜻하고 공감하는 말투로 대화해줘. 너무 길지 않게 2~4문장 정도로 답해.
질문에 따라 조언이나 위로, 동기부여를 해줘.
"""

    return {"system_prompt": system_prompt}


class ChatMessage(BaseModel):
    history: List[Dict[str, str]]  # [{"role": "user", "content": "..."}, {"role": "assistant", ...}]
    new_message: str

from pydantic import BaseModel

class ChatMessage(BaseModel):
    history: List[Dict[str, str]]  # [{"role": "user", "content": "..."}, ...]
    new_message: str
def convert_gpt_to_gemini(messages: List[Dict[str, str]]) -> List[Dict]:
    gemini_history = []
    system_prompt = """
    너는 사용자의 감정에 공감하고 위로해주는 대화 상대야.
    - 따뜻하고 친근한 말투를 사용해.
    - 과하게 분석하거나 설명하지 말고, 감정 중심으로 응답해.
    - 아래와 같은 포맷은 절대 사용하지 마세요:
      - 마크다운 형식 (예: **굵은 글자**, 리스트(-, *, •), / 등)
      - 이모지
      - HTML 태그
    - 포맷팅 없이 자연스러운 문장으로만 대답해주세요.
    - 문단으로 구성된 따뜻한 응답을 해줘.
    """
    gemini_history.append({
        "role": "user",
        "parts": [system_prompt.strip()]
    })

    # 기존 메시지 추가
    for msg in messages:
        gemini_history.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [msg["content"]]
        })

    return gemini_history

@app.post("/api/conversation/chat")
def chat_with_persona(chat: ChatMessage):
    gpt_format = chat.history + [{"role": "user", "content": chat.new_message}]
    gemini_format = convert_gpt_to_gemini(gpt_format)
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
    chat_session = model.start_chat(history=gemini_format)
    response = chat_session.send_message(chat.new_message)

    return {"reply": response.text.strip()}

