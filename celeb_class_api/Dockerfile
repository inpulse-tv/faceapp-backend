# pull official base image
FROM python:3.8

# copy content
WORKDIR /app
COPY . .

# install dependencies
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
RUN pip install -r requirements.txt

# start celebrity classification api
ENTRYPOINT ["./gunicorn_celeb_class.sh"]
