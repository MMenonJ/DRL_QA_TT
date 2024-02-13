FROM nvcr.io/nvidia/pytorch:21.10-py3
RUN pip install --upgrade pip setuptools
RUN pip install farm-haystack[all-gpu]==1.18.1
RUN pip install stable-baselines3[extra]
RUN pip install gym
RUN pip install opencv-python==4.5.5.64
RUN pip install ujson