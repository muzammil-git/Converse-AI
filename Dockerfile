FROM python:3.11.8
WORKDIR /app
COPY pyproject.toml poetry.lock /app/
RUN pip install poetry
RUN poetry config virtualenvs.in-project true
RUN poetry install --no-root --only main
COPY . /app
RUN echo "Force rebuild at $(date)"
EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0"]

