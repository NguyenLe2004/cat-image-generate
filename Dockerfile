FROM python:3

WORKDIR /src

RUN pip install -U pip
RUN pip install -U torch
RUN pip install -U torchvision
RUN pip install -U numpy
RUN pip install -U matplotlib

COPY utils ./utils
COPY generate.py ./generate.py
COPY last_model.pth ./last_model.pth

CMD ["python3", "generate.py"]