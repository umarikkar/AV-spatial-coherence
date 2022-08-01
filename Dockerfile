#FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime
#FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
FROM pytorch/pytorch:latest

RUN apt-get update -y # Good practice, update the package database.

RUN apt-get install openssh-server sssd -y # If you want to interactively connect to the container.
RUN mkdir /var/run/sshd

RUN apt-get install -y 	libopenblas-dev # Install the BLAS.

RUN pip install pandas
RUN pip install sklearn
RUN pip install scipy
RUN apt-get update
RUN apt-get install -y libsndfile1
RUN pip install tqdm
RUN pip install librosa
RUN pip install matplotlib
RUN pip install Pillow
RUN pip install python_speech_features 
RUN pip install h5py
RUN pip install soundfile

RUN apt-get autoremove -y && apt-get autoclean -y # Good practice, to keep the Docker image as small as possible.
