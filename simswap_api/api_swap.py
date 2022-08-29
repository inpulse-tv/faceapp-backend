import os
from base64 import b64encode
import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.videoswap import video_swap
from flask import Flask, request, jsonify

import warnings
warnings.filterwarnings("ignore")


def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0


transformer = transforms.Compose([
        transforms.ToTensor(),
    ])

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

detransformer = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
    ])

opt = TestOptions().parse()

app = Flask(__name__)


@app.route("/")
def hello():
    return "SimSwap API !!!"


@app.route("/swap_image", methods=["POST"])
def get_image():
    """
    This function swap faces given two images.
    :return: json output containing the generated image in a binary format.
    """

    # retrieve the two images
    size = (224, 224)
    file_a = request.files['file_a']
    file_b = request.files['file_b']

    # swap faces using SimSwap model
    opt = TestOptions().parse()
    torch.nn.Module.dump_patches = True
    model = create_model(opt)
    model.eval()

    with torch.no_grad():
        img_a = Image.open(file_a)
        img_a = np.asarray(img_a, dtype=np.float32)
        gray = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray = np.array(gray, dtype='uint8')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        try:
            (x, y, w, h) = faces[0]
            img_a = img_a[y:y + h, x:x + w]
        except:
            return "Face not detected"

        img_a = Image.fromarray(np.uint8(img_a)).convert('RGB')
        img_a = img_a.resize(size)
        img_a = transformer_Arcface(img_a)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        img_b = Image.open(file_b).convert('RGB')
        img_b = img_b.resize(size)
        img_b = transformer(img_b)
        img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])

        # convert numpy to tensor
        img_id = img_id.cuda()
        img_att = img_att.cuda()

        # create latent id
        img_id_downsample = F.interpolate(img_id, size=(112, 112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = latend_id.detach().to('cpu')
        latend_id = latend_id/np.linalg.norm(latend_id, axis=1, keepdims=True)
        latend_id = latend_id.to('cuda')

        ############## Forward Pass ######################
        img_fake = model(img_id, img_att, latend_id, latend_id, True)

        for i in range(img_id.shape[0]):
            if i == 0:
                row1 = img_id[i]
                row2 = img_att[i]
                row3 = img_fake[i]
            else:
                row1 = torch.cat([row1, img_id[i]], dim=2)
                row2 = torch.cat([row2, img_att[i]], dim=2)
                row3 = torch.cat([row3, img_fake[i]], dim=2)

        # full = torch.cat([row1, row2, row3], dim=1).detach()
        full = row3.detach()
        full = full.permute(1, 2, 0)
        output = full.to('cpu')
        output = np.array(output)
        output = output[..., ::-1]
        output = output*255

        cv2.imwrite(opt.output_path + 'result.jpg', output)

        with open(os.path.abspath(opt.output_path + 'result.jpg'), 'rb') as fh:
            return jsonify(
                filedata=b64encode(fh.read()).decode()
            )


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


@app.route("/swap_video", methods=["POST"])
def get_video():
    """
    Given the user's video and the celebrity's image, this function can generate DeepFake
    video.
    NB: you need GPU to use this function.
    :return: json output containing the generated video in a binary format.
    """

    # retrieve the image and the video
    image = request.files['image']
    video = request.files['video']
    video.save('swap_video.mp4')
    video = os.path.abspath('swap_video.mp4')
    opt = TestOptions().parse()
    opt.output_path = "./output/resultat.mp4"
    crop_size = opt.crop_size

    # generate DeepFake video using SimSwap model
    torch.nn.Module.dump_patches = True
    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'
    model = create_model(opt)
    model.eval()

    face_detect = Face_detect_crop(name='antelope', root='./insightface_func/models')
    face_detect.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640, 640), mode=mode)
    with torch.no_grad():
        pic_a = image
        img_a_whole = Image.open(pic_a).convert('RGB')
        img_a_whole = np.array(img_a_whole)
        img_a_align_crop, _ = face_detect.get(img_a_whole, crop_size)
        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
        img_id = img_id.to('cpu')

        # create latent id
        img_id_downsample = F.interpolate(img_id, size=(112, 112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)

        video_swap(video, latend_id, model, face_detect, opt.output_path, temp_results_dir=opt.temp_path,
                   no_simswaplogo=opt.no_simswaplogo, use_mask=opt.use_mask, crop_size=crop_size)

        os.remove("swap_video.mp4")

        with open(os.path.abspath(opt.output_path), 'rb') as fh:
            return jsonify(
                filedata=b64encode(fh.read()).decode()
            )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
