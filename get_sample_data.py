from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
	
# Download the entire repository
snapshot_download(    
    token=HF_TOKEN,
	repo_id="cadene/droid",    
	repo_type="dataset",    
	local_dir="rlds_from_hf_data"
)

