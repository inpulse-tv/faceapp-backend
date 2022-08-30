# faceapp-backend
BackEnd of the FaceApp application developed as part of a study on facial recognition and Deep Fake.

## Table of contents
* [Celebrity classification API](#celebrity-classification-api)
* [Face morphing API](#face-morphing-api)
* [Face manipulation API](#face-manipulation-api)
* [Mongodb](#mongodb)
* [Docker-compose](#docker-compose)


## Celebrity classification API
Python version > 3.8.

This API return the most similar celebrity to the user. It uses FaceNet model and a saved SVM classifier. You can refer to the "celeb_class_api" directory if you want to test. You can also retrain the model on your own dataset using the notebook "facenet.ipynb" in the "notebooks" directory. 

In addition, You can create Docker image using the Dockerfile in the "celeb_class_api" directory.  You can use these commands:

```
$ cd celeb_class_api
$ sudo docker build -t celeb_class_api:2.0 .
$ sudo docker run -it --rm -p 5002:5002 -v ../../../mnt/d/big_dataset/Train:/app/volume -e NUMBER_WORKERS=10 -e NUMBER_THREADS=4 celeb_class_api:2.0
```
After training the model on your own dataset, replace the path "../../../mnt/d/big_dataset/Train" with the path of your dataset.

## Face morphing API
Python version = 3.7 or 3.6

By employing the SimSwap model, this API manages to merge two faces. To use it, you can refer to the "simswap_api" directory. You should at first clone the [SimSwap github](https://github.com/neuralchen/SimSwap). Next, you must prepare the evironment and install the different pre-trained models. All the steps are given in [this link](https://github.com/neuralchen/SimSwap/blob/main/docs/guidance/preparation.md). Finally, add the file given in the "simswap_api" folder and install requirements. You can either start the API through the "uwsfi_simswap.sh" script.

If you find problems with the dependicies, you can run the API as a Docker image through the following commands:

```
$ cd simswap_api
$ sudo docker build -t simswap:1.0 .
$ sudo docker run -it --rm -p 5001:5001 -e NUMBER_PROCESSES=10 -e NUMBER_THREADS=1 simswap:1.0
```
## Face manipulation API
Python version = 3.7

This API generates face manipulation videos using [First Order Model for Image Animation github](https://github.com/AliaksandrSiarohin/first-order-model). After cloning the previous github, you must download checkpoints using this [link](https://drive.google.com/drive/folders/1PyQJmkdCsAkOYwUyaj_l-l0as-iLDgeH). In this project, we used the checkpoint nammed "vox-cpk.pth.tar". Then, install all the requirements as well as face-allignement library as follows:
```
$ git clone https://github.com/1adrianb/face-alignment
$ cd face-alignment
$ pip install -r requirements.txt
$ python setup.py install
```

You can also launch the API using Docker as follows:

```
$ cd fom_api
$ sudo docker build fom_api_url:4.0 .
$ docker run -it --rm --gpus all -p 5000:5000 -e NUMBER_WORKERS=10 -e NUMBER_THREADS=2 fom_api_url:4.0
```

## Mongodb
The Dockerfile inside the "mongodb" folder aims to create a database called "ingeniance" that contains all the image urls that will be needed in the face manipulation task. It is important that this data was not used to train the classification model. You can create this docker image as follows:

```
$ cd mongodb
$ sudo docker build -t mongo:latest .
```

## Docker-compose

In order to run this docker file that start the entire application at one, you must at first build att the previous images as well as the image of the app front-end given in this [github](https://github.com/inpulse-tv/webcam). In addition, it is important to mention that this docker-compose must be executed inside the "webcam" folder created after clonning the font-end [github](https://github.com/inpulse-tv/webcam). Therefore, you must copy it to the "webcam" folder.

The commands for running docker-compose are:

```
$ cd webcam
$ sudo docker-compose up
```

## 