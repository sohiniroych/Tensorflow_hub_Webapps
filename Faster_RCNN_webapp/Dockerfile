FROM python:3.7
MAINTAINER Sohini Roychowdhury <roych@uw.edu>

# Install build utilities
RUN apt-get update && \
    apt-get install -y protobuf-compiler python3-pil python3-lxml python3-pip python3-dev git && \
    apt-get -y upgrade

#Install Object Detection dependencies
RUN python3 -m pip install Flask==1.1.1 WTForms==2.2.1 Flask_WTF==0.14.2 Werkzeug==0.16.0 tensorflow==2.0.0

# Install Object Detection API library
RUN git clone --depth 1 https://github.com/tensorflow/models/ && \
    cd models/research && \
    protoc object_detection/protos/*.proto --python_out=.
    
RUN cp object_detection/packages/tf2/setup.py . && \
    pip3 install .

RUN git clone https://github.com/sohiniroych/Tensorflow_hub_Webapps.git && \
    chmod u+x Faster_RCNN_webapp/app.py
    
RUN python3 Faster_RCNN_webapp/app.py

    
# expose ports
EXPOSE 8080

#Command
#CMD ["python3", "/opt/object_detection_app_p1/app.py", "serve"]
