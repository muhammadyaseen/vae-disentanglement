FROM nvcr.io/nvidia/pytorch:22.01-py3

# Update packages
RUN apt-get update && apt-get install -y --no-install-recommends && rm -rf /var/lib/apt/lists/

# Install packages that were in my conda environment
WORKDIR /tmp
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install disentanglement_lib
# Custom cache invalidation - because of new commits
#ARG CACHEBUST=1
RUN git clone https://github.com/muhammadyaseen/disentanglement_lib
#COPY ../disentanglement_lib ./
WORKDIR /tmp/disentanglement_lib
RUN pip install --no-cache-dir .[tf_gpu]

# Patched Visdom 
WORKDIR /tmp
RUN git clone https://github.com/muhammadyaseen/visdom.git
WORKDIR /tmp/visdom
RUN pip install --no-cache-dir -e .

# Default folder
WORKDIR /workspace

CMD ["echo", "Welcome!"]



