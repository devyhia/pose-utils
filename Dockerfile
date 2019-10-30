FROM nvidia/cuda:9.0-cudnn7-devel

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda2-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

RUN conda create -n py36 python=3.6
RUN echo "conda activate py36" >> ~/.bashrc

RUN conda install -n py36 matplotlib
RUN conda install -n py36 scikit-learn
RUN conda install -n py36 pillow
RUN conda install -n py36 ipython
RUN conda install -n py36 Cython  # Matplotlib requires Cython
RUN /opt/conda/envs/py36/bin/pip install pdbpp

# Install Vim for on-the-fly edits

# Install jupyter
RUN conda install -n py36 jupyter

RUN conda install -n py36 keras==2.1.6
RUN conda install -n py36 pydot
RUN conda install -n py36 -c menpo opencv3

# Install ml-cli
RUN /opt/conda/envs/py36/bin/pip install ml-cli

# Install Apple Core ML Tools
RUN /opt/conda/envs/py36/bin/pip install coremltools

# Install Visualization Tools
#RUN apt-get install python-pydot python-pydot-ng graphviz
RUN apt install -y libgtk2.0-dev

RUN /opt/conda/envs/py36/bin/pip install tqdm

 RUN conda install -n py36 tensorflow-gpu==1.5.0

## Fix problem with matplotlib verison in conda
##RUN pip uninstall -y matplotlib
##RUN python -m pip install --upgrade pip
##RUN pip install matplotlib

RUN apt install -y vim

# Link cuda binaries for python c++ libraries that need cuda (theano, tensorflow, ...)
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN apt-get update
RUN apt-get install -y graphviz

# Visualization
RUN /opt/conda/envs/py36/bin/pip install visdom

# NLP
RUN /opt/conda/envs/py36/bin/pip install nltk
RUN /opt/conda/envs/py36/bin/pip install mxnet-cu100
RUN /opt/conda/envs/py36/bin/pip install bert-embedding

WORKDIR /usr/local/src/code
