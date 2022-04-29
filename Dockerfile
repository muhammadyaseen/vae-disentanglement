FROM nvcr.io/partners/gridai/pytorch-lightning:v1.4.0

# Update packages
RUN apt-get update && apt-get install -y --no-install-recommends &&  rm -rf /var/lib/apt/lists/

WORKDIR /tmp

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /workspace

CMD ["echo", "Welcome!"]



