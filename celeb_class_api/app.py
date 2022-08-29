from base64 import b64encode
import cv2
import pickle
import numpy as np
from PIL import Image
from facenet_model import InceptionResNetV2
from functions import get_embeddings, most_similar_photo
from flask import Flask, request, jsonify
from bson import ObjectId
import pymongo
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
# face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def allowed_file(filename):
    """
    Verify if the file has the allowed extension.
    :param filename: the file's name.
    :return: True if the file has the allowed extension, False otherwise.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def hello():
    return "Celebrity Classification API !!!"


@app.route('/imageCelebrity', methods=['GET'])
def celebrites_images():
    """
    This function allows to get 20 images urls from mongodb database named "ingeniance3"
    using "test-mongo" container.
    :return: json output containing a list of image urls of 20 celebrities.
    """
    # connect to test-mongo container
    client = pymongo.MongoClient("mongodb://test-mongo:27017/")

    # access to database
    db = client["ingeniance3"]

    # access to collections
    collection1 = db["celebrities"]
    collection2 = db["Images"]

    # get a list of 20 url images
    pipline = [{"$sample": {"size": 20}}]
    celebrities = collection1.aggregate(pipline)
    list_image_celebrity = []
    for document in celebrities:
        image_celebrity = {}
        image_celebrity["celebrityName"] = document['name']
        image = collection2.find({"_id": ObjectId(document['images'][0])})
        for img in image:
            image_celebrity['image'] = img['url']
            list_image_celebrity.append(image_celebrity)

    return jsonify(
        number=20,
        list=list_image_celebrity
    )


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """
    This function allow to identify the celebrity the most similar to the user.
    :return: json output containing the name of the "celebrity twin" and his image in a binary format.
    """
    if request.method == 'POST':
        # retrieve the user's image
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file and not allowed_file(file.filename):
            return "File not allowed"
        image = Image.open(file)
        image = np.asarray(image, dtype=np.float32)

        # extract the face from the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.array(gray, dtype='uint8')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        try:
            (x, y, w, h) = faces[0]
            image = image[y:y + h, x:x + w]
        except:
            return "Face not detected"

        # resize and standardize the image
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (160, 160))
        image = image.astype('float32')
        mean, std = image.mean(), image.std()
        image = (image - mean) / std

        # embed the image using FaceNet
        model = InceptionResNetV2()
        model.load_weights('facenet_keras_weights.h5')
        clas = pickle.load(open('model.sav', 'rb'))
        embed = get_embeddings(model, image)

        # classify the embedding using SVM
        celeb = clas.predict(embed).tolist()

        # get the most similar photo to the user from the celebrity's images
        ans, path = most_similar_photo(celeb[0], embed, model)

        filename = path
        with open(filename, 'rb') as fh:
            return jsonify(
                filemeta=celeb[0],
                filedata=b64encode(fh.read()).decode()
            )


if __name__ == '__main__':
    DEBUG = False
    PORT = 5002
    HOST = '0.0.0.0'
    app.run(host=HOST, port=PORT, debug=DEBUG)
