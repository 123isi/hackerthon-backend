from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()



class Persona(Base):
    __tablename__ = "personas"

    id = Column(Integer, primary_key=True, index=True)
    strong = Column(Text, nullable=False)
    weakness = Column(Text, nullable=False)
    keyword = Column(Text, nullable=True)



class Keyword(Base):
    __tablename__ = "keywords"

    id = Column(Integer, primary_key=True, index=True)
    keyword = Column(String(50), nullable=False)

    personas = relationship("PersonaKeyword", back_populates="keyword")


class PersonaKeyword(Base):
    __tablename__ = "persona_keywords"

    id = Column(Integer, primary_key=True, index=True)
    persona_id = Column(Integer, ForeignKey("personas.id"))
    keyword_id = Column(Integer, ForeignKey("keywords.id"))

    persona = relationship("Persona", backref="persona_keywords")
    keyword = relationship("Keyword", back_populates="personas")


class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)
    question_text = Column(Text, nullable=False)
    option_1 = Column(String(100), nullable=False)
    option_2 = Column(String(100), nullable=False)
    option_3 = Column(String(100), nullable=False)


class Answer(Base):
    __tablename__ = "answers"

    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"))
    selected_option = Column(Integer, nullable=False)  # 1, 2, or 3
    created_at = Column(DateTime, default=datetime.utcnow)


class Quest(Base):
    __tablename__ = "quests"

    id = Column(Integer, primary_key=True, index=True)
    mission_text = Column(String(200), nullable=False)
    state = Column(String(20), nullable=False, default="NOT")  # NOT / SUCCESS
class Content(Base):
    __tablename__ = "contents"
    id = Column(Integer, primary_key=True, index=True)
    description = Column(Text, nullable=False)
    keyword = Column(Text, nullable=True)