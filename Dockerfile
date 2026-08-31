FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY services ./services
COPY packages ./packages
COPY data ./data
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

EXPOSE 8100
# Override command for the Runtime: uvicorn services.agent_runtime.app.main:app --host 0.0.0.0 --port 8101
CMD ["uvicorn", "services.core_api.app.main:app", "--host", "0.0.0.0", "--port", "8100"]
