# 1. Use Python 3.9 (Stable and small)
FROM python:3.9-slim

# 2. Set the working directory
WORKDIR /app

# 3. Copy requirements first (to cache installations)
COPY requirements.txt .

# 4. Install libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your files
COPY . .

# 6. Open the port Hugging Face expects (7860)
EXPOSE 7860

# 7. Start the server on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]