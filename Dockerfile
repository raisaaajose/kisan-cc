FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .

RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY . .

CMD ["python","src/train.py"]