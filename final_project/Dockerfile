FROM python:3.10.6

# workdir is remote
RUN mkdir /final_project
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
COPY ./baseball.sql /final_project/baseball.sql
COPY ./build_model.py /final_project/build_model.py
COPY ./cat_correlation.py /final_project/cat_correlation.py
COPY ./data_process.py /final_project/data_process.py
COPY ./difference_mean.py /final_project/difference_mean.py
COPY ./final.py /final_project/final.py
COPY ./plots.py /final_project/plots.py
COPY ./final.sql /final_project/final.sql
COPY ./script.sh /final_project/script.sh


# Get necessary python libraries
COPY ./requirements.txt /final_project/requirements.txt
RUN pip3 install --compile --no-cache-dir -r /final_project/requirements.txt


RUN chmod +x /final_project/script.sh

# Run app
# /bin/sh running permission
# /final_project/script.sh remote
CMD /bin/sh /final_project/script.sh