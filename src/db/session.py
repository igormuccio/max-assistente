import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

senha = os.getenv("DB_PASSWORD")
engine = create_engine(f"postgresql+psycopg2://postgres:{senha}@localhost:5432/max_db")

def obter_session():
    return Session(engine)