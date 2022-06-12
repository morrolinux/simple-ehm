FROM python:3.9-slim
RUN apt update
RUN apt install libgl1-mesa-glx -y
RUN apt install libglib2.0-0 -y
RUN apt install ffmpeg -y
RUN mkdir /simple_ehm
WORKDIR /simple_ehm
ADD . .
RUN pip install -r requirements.txt
