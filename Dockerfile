FROM python:3.10.14-slim

RUN mkdir /pasteur

# Install pastml
RUN cd /usr/local/ && pip3 install --no-cache-dir pastml==1.9.51

# The entrypoint runs pastml with command line arguments
ENTRYPOINT ["pastml"]
