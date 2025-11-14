FROM python:3.10-slim-bookworm

RUN apt-get update && apt install -y openssh-server zip vim  \
    # Clean up to keep the image small
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /var/run/sshd

WORKDIR /data/app
COPY . .

EXPOSE 22

CMD ["./entrypoint.sh"]
