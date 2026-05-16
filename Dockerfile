FROM python:3.12

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip

RUN pip install fastapi uvicorn python-multipart pillow torch torchvision --no-cache-dir

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

