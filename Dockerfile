FROM nvcr.io/partners/gridai/pytorch-lightning:v1.4.0

# Update packages
RUN apt-get update && apt-get install -y --no-install-recommends &&  rm -rf /var/lib/apt/lists/

# Install packages that were in my conda environment
WORKDIR /tmp
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install disentanglement_lib
RUN git clone https://github.com/muhammadyaseen/disentanglement_lib
WORKDIR ./disentanglement_lib
RUN pip install .[tf_gpu]

# Default folder
WORKDIR /workspace

CMD ["echo", "Welcome!"]



