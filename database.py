from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME


# postgresql://USER:PASSWORD@INTERNAL_HOST:PORT/DATABASE
SQLALCHEMY_DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_async_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=AsyncSession,  # Ensure the session is asynchronous
    expire_on_commit=False,  # Prevent SQLAlchemy from expiring objects after commit
)

Base = declarative_base()


async def get_db() -> AsyncSession: # type: ignore
    async with SessionLocal() as db:
        try:
            yield db
        except Exception as e:
            print(f"Error; database.py : {e}")
        finally:
            await db.close()