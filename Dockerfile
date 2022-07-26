FROM python:3.9

ARG VERSION

LABEL ai.org.version=$VERSION

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY app/* /app/

ENTRYPOINT ["python"]

CMD ["app.py"]
