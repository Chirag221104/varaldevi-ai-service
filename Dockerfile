FROM python:3.10

WORKDIR /app

# Install system dependencies for TensorFlow
RUN apt-get update && apt-get install -y build-essential python3-dev

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 6000

CMD ["python", "server.py"]
