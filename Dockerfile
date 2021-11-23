FROM pytorch/pytorch:latest
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY
ENV KAGGLE_USERNAME=$KAGGLE_USERNAME
ENV KAGGLE_KEY=$KAGGLE_KEY
COPY requirements.txt ./
RUN apt update
RUN apt install -y git unzip make git-lfs neovim wget
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt
RUN pip install --no-cache-dir kaggle
RUN touch .env
COPY . .
RUN touch ~/.no_auto_tmux
RUN make get_dataset
