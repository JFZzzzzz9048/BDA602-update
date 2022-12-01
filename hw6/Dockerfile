FROM python:3.10.6

WORKDIR /hw6

# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
    mariadb-client \
  && rm -rf /var/lib/apt/lists/*

# Copy over code
COPY /hw2.sql .
COPY /script.sh .


# Run app
CMD /script.sh
