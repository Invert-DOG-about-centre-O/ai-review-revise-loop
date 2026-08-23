import os, certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
from huggingface_hub import hf_hub_download
p = hf_hub_download("gpt2", "config.json")
print(p)
