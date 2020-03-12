FROM nvcr.io/nvidia/tensorrt:19.09-py3
#FROM pinctada

ARG env

WORKDIR /workspace/

# Copy code.
COPY . /workspace/

RUN if [ "$env" = "live" ] ; then cp ./conf/live/setting.ini  ./conf/  ; else cp ./conf/dev/setting.ini  ./conf/ ; fi

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y make apt-utils cron

COPY ./deploy/clean_job /etc/cron.d/clean-cron
RUN chmod 0644 /etc/cron.d/clean-cron
RUN crontab /etc/cron.d/clean-cron

RUN cd /workspace/  &&  make depe

# EXPOSE 12242/tcp

ENV envname ${env}
ENTRYPOINT python2 /workspace/src/pinctada_server.py
