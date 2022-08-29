from urllib.request import urlopen
import imageio
import os
import numpy as np
from PIL import Image
from skimage.transform import resize
from demo import load_checkpoints
from demo import make_animation
from skimage import img_as_ubyte
from flask import Flask, request, send_file, jsonify
from base64 import b64encode
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)


@app.route("/")
def hello():
    return "FOM API !!!"


@app.route("/image_animation", methods=["POST"])
def image_animation():
    """
    This function generates the image animation video using the "First Order Model for Image Animation" github.
    The request takes an image url of the celebrity and a video file of the user
    :return:  Face manipulation video in a binary format
    """

    # retrieve image and video
    image = request.form['imageurl']
    video = request.files['video']
    video.save('fom_video.mp4')
    video = os.path.abspath('fom_video.mp4')
    source_image = Image.open(urlopen(image)).convert('RGB')
    source_image = np.asarray(source_image)
    reader = imageio.get_reader(video)

    # resize image and video to 256x256
    source_image = resize(source_image, (256, 256))[..., :3]
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

    # make predictions
    generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml', checkpoint_path='./vox-cpk.pth.tar',
                                              cpu=False)
    predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)

    # save resulting video
    imageio.mimsave('./test/generated.mp4', [img_as_ubyte(frame) for frame in predictions], fps=fps)

    # remove the input video
    os.remove("fom_video.mp4")

    # return the generated video in a binary format
    with open(os.path.abspath('./test/generated.mp4'), 'rb') as fh:
        return jsonify(
            filedata=b64encode(fh.read()).decode()
        )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
