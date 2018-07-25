FROM ubuntu:16.04

ENV PATH /opt/conda/bin:$PATH

# Install base packages.
RUN apt-get update && apt-get install -y unzip curl bzip2

# Install Anaconda.
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh \
    && curl -o ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh

# Download weights
RUN curl -o data_folder_nlg.zip https://s3.eu-central-1.amazonaws.com/nlp-machine-learning-data/data_folder_nlg.zip \
    && mkdir -p data_folder/ \
    && unzip data_folder_nlg.zip -d data_folder/ \
    && rm data_folder_nlg.zip

# Use python 3.6
RUN conda install python=3.6

COPY data_loader/ data_loader/
COPY helper/ helper/
COPY metrics/ metrics/
COPY model/ model/
COPY res/ res/
COPY infer.py infer.py
COPY yahoo_main.py yahoo_main.py
COPY requirements.txt requirements.txt
COPY run.sh run.sh

RUN pip install -r requirements.txt

CMD ["/bin/bash", "run.sh"]