import os
from dotenv import load_dotenv

load_dotenv()

GROQ_KEY = os.getenv('GROQ')
SERPER_API_KEY = os.getenv('SERPER')

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_USER = os.getenv('USERNAME')
DB_PASSWORD = os.getenv('PASSWORD')
DB_NAME = os.getenv('DB_NAME')
OPEN_AI_API_KEY = os.getenv('OPEN_AI')