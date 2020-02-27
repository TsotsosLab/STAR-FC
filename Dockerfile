FROM nvidia/cuda:8.0-cudnn5-devel

ENV STAR_FC_ROOT=/opt/STAR_FC
WORKDIR $STAR_FC_ROOT

## this is necessary for linking against cuda libraries ##
ENV PYTHONPATH=$STAR_FC_ROOT/contrib/caffe/python/:$STAR_FC_ROOT/contrib/OpenSALICON/:$PYTHONPATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:/usr/local/lib/x86_64-linux-gnu:/usr/local/lib/i386-linux-gnu:${LD_LIBRARY_PATH}
ENV PATH /usr/local/nvidia/bin:${PATH}

## install dependencies ##

RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential \
autoconf \
automake \
libtool \
cmake \
curl \
libgtk2.0-dev \
pkg-config \
libavcodec-dev \
libavformat-dev \
libswscale-dev \
python3-pip \
python3-dev \
python3-setuptools \
qtdeclarative5-dev \
libleveldb-dev \
libsnappy-dev \
libhdf5-serial-dev \
libgtest-dev \
libfftw3-dev \
libboost-all-dev \
libgoogle-glog-dev \
libgtest-dev \
libmatio-dev \
libatlas-base-dev \
liblmdb-dev \
libxext-dev \
libx11-dev \
x11proto-gl-dev \
dbus \
wget \
unzip \
dh-autoreconf \
git && rm -rf /var/lib/apt/lists/*

WORKDIR $STAR_FC_ROOT

## compile OpenCV 3 with CUDA support (necessary to run AIM fast) ##
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/3.3.0.zip && unzip opencv.zip && mkdir opencv-3.3.0/build && rm -rf opencv.zip

WORKDIR $STAR_FC_ROOT/opencv-3.3.0/build

RUN cmake -DCMAKE_BUILD_TYPE=RELEASE \
   -DWITH_GTK=OFF -DWITH_GTK_2_X=OFF -DWITH_QT=ON \
   -DBUILD_JPEG=ON \
   -DCMAKE_INSTALL_PREFIX=/usr/local \
   -DINSTALL_PYTHON_EXAMPLES=OFF \
   -DINSTALL_C_EXAMPLES=OFF  \
   -DWITH_CUDA=ON \
   -DBUILD_EXAMPLES=OFF .. && make -j8 && make install

RUN pip3 install matplotlib==2.2.3 scikit-image==0.15 numpy

## install protobuf
RUN pip3 install --user --upgrade protobuf==3.1.0
RUN wget -O protobuf.zip https://github.com/google/protobuf/archive/v3.1.0.zip && \
       unzip protobuf.zip && cd protobuf-3.1.0 && ./autogen.sh && ./configure && make && make install && ldconfig && rm -rf protobuf.zip

## compile caffe ##
RUN wget -O caffe.zip https://github.com/BVLC/caffe/archive/rc4.zip && unzip caffe.zip && \
        mkdir -p $STAR_FC_ROOT/contrib/caffe && mv caffe-rc4/* $STAR_FC_ROOT/contrib/caffe/

COPY Makefile.config $STAR_FC_ROOT/contrib/caffe/

WORKDIR $STAR_FC_ROOT/contrib/caffe
RUN protoc --version && protoc src/caffe/proto/caffe.proto --cpp_out=. && mkdir include/caffe/proto && \
mv src/caffe/proto/caffe.pb.h include/caffe/proto

RUN make -j8 && make pycaffe

## Add BMS, VOCUS2 and SALICON which are used to compute peripheral and central saliency
## and inih lirary for parsing .INI configuration files
WORKDIR $STAR_FC_ROOT/contrib/
RUN wget http://cs-people.bu.edu/jmzhang/BMS/BMS_v2-mex.zip && unzip BMS_v2-mex.zip -d BMS_v2
RUN git clone https://github.com/GeeeG/VOCUS2.git VOCUS2
RUN git clone https://github.com/CLT29/OpenSALICON.git OpenSALICON
RUN git clone https://github.com/benhoyt/inih.git inih

WORKDIR $STAR_FC_ROOT/contrib/OpenSALICON
RUN wget http://www.cs.pitt.edu/%7Echris/files/2016/model_files.tgz && tar -zxvf model_files.tgz && cp model_files/salicon_osie.caffemodel . &&  rm -rf model_files && rm model_files.tgz

WORKDIR $STAR_FC_ROOT
## Compile STAR-FC
COPY CMakeLists.txt $STAR_FC_ROOT
COPY ./src $STAR_FC_ROOT/src

RUN cmake -DWITH_SALICON=ON -DCMAKE_BUILD_TYPE=Debug . && make
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64/stubs/:/usr/local/lib/x86_64-linux-gnu:/usr/local/lib/i386-linux-gnu:${LD_LIBRARY_PATH}
