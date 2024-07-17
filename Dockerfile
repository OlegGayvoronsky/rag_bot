FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Установка необходимых зависимостей
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl

# Добавление PPA для Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa

# Обновление списка пакетов и установка Python 3.11
RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3.11-dev

# Установка pip для Python 3.11
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.11 get-pip.py && rm get-pip.py

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --upgrade -r /code/requirements.txt

COPY ./app/ /code

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "main.py"]