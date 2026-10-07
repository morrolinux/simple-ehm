FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*
RUN mkdir /simple_ehm
WORKDIR /simple_ehm
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python", "./simple_ehm-runnable.py"]
CMD ["--help"]
