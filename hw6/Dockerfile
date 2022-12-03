FROM python:3.10.6

# workdir is remote
WORKDIR $APP_HOME

# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
     mariadb-client \
     mariadb-server \
  && rm -rf /var/lib/apt/lists/*

# Copy over code
COPY ./baseball.sql baseball.sql
COPY ./hw2.sql hw2.sql
COPY ./script.sh script.sh

# Run app
# /bin/sh running permission
# /hw6/script.sh remote
CMD /bin/sh script.sh
