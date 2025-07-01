FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow==1.30.0

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlruns/mlflow.db", "--default-artifact-root", "/app/mlartifacts"]
