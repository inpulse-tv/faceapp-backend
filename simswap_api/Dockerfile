# pull official base image
FROM continuumio/anaconda3

# set working directory
WORKDIR /home/SimSwap

# install dependencies
RUN apt-get update &&\
    apt-get install ffmpeg libsm6 libxext6  -y
SHELL ["/bin/bash", "-c"]
RUN	conda create -n simswap python=3.6 && source ~/.bashrc && conda activate simswap && \
    conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly -c pytorch -y && \
	pip install --ignore-installed imageio && \
	pip install insightface==0.2.1 onnxruntime moviepy
RUN source ~/.bashrc && conda activate simswap && pip install flask && pip install gunicorn &&\
    conda install -c conda-forge uwsgi

# copy the content
COPY . .

# use CPU instead of GPU
RUN sed -i 's/base/simswap/' ~/.bashrc
RUN sed -i "s/if len(self.opt.gpu_ids)/if torch.cuda.is_available() and len(self.opt.gpu_ids)/g" /home/SimSwap/options/base_options.py && sed -i "s/device = torch.device('cuda:0')/torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')/g" /home/SimSwap/models/fs_model.py && sed -i "s/torch.load(netArc_checkpoint)/torch.load(netArc_checkpoint) if torch.cuda.is_available() else torch.load(netArc_checkpoint, map_location=torch.device('cpu'))/g" /home/SimSwap/models/fs_model.py && \
find /home/SimSwap -type f -exec sed -i "s/net.load_state_dict(torch.load(save_pth))/net.load_state_dict(torch.load(save_pth)) if torch.cuda.is_available() else net.load_state_dict(torch.load(save_pth, map_location=torch.device('cpu')))/g" {} \; && find /home/SimSwap -type f -exec sed -i "s/.cuda()/.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))/g" {} \; && find /home/SimSwap -type f -exec sed -i "s/.to('cuda')/.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))/g" {} \; && sed -i "s/torch.device(\"cuda:0\")/torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')/g" /home/SimSwap/models/fs_model.py

# start api
ENTRYPOINT ["./uwsgi_simswap.sh"]
