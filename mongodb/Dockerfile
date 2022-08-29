# pull official base image
FROM mongo:latest

# copy the different collections and the sh file
RUN mkdir -p ./volume
COPY ./celebrities.json ./volume/celebrities.json
COPY ./Images.json ./volume/Images.json
COPY ./mongodb.sh ./mongodb.sh

# run the script
ENTRYPOINT ["./mongodb.sh"]
