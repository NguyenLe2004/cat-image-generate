FROM python:3

WORKDIR /src

RUN pip install -U pip
RUN pip install -U torch
RUN pip install -U torchvision
RUN pip install -U numpy
RUN pip install -U matplotlib

COPY generate.py ./generate.py
COPY cat_face_generate_model.pt ./cat_face_generate_model.pt

CMD ["python3", "generate.py"]