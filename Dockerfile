# 1. Force the exact, stable version of Python (Fixes TensorFlow)
FROM python:3.10-slim

# 2. Force the installation of the missing Linux math library (Fixes LightGBM)
RUN apt-get update && apt-get install -y libgomp1

# 3. Set up the working folder
WORKDIR /app

# 4. Copy the libraries list and install them safely
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your code and models into the cloud computer
COPY . .

# 6. Turn the server on using Railway's specific port
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
