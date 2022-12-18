FROM python:3.10.6

# workdir is remote
WORKDIR /final_project

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
COPY ./build_model.py build_model.py
COPY ./cat_correlation.py cat_correlation.py
COPY ./data_process.py data_process.py
COPY ./difference_mean.py difference_mean.py
COPY ./final.py final.py
COPY ./plots.py plots.py
COPY ./final.sql final.sql
COPY ./script.sh script.sh


# Get necessary python libraries
COPY ./requirements.txt requirements.txt
RUN pip3 install --compile --no-cache-dir -r requirements.txt


RUN chmod +x script.sh

# Run app
# /bin/sh running permission
# /final_project/script.sh remote
CMD /bin/sh script.sh