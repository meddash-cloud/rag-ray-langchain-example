import os
import ray
raydocs_root="/tmp/raydocs"
num_cpus=4
num_gpus=0
from pathlib import Path

# Directories
EFS_DIR = Path("/home/jovyan/work/raydocs")
ROOT_DIR = Path(__file__).parent.parent.absolute()

# Mappings
EMBEDDING_DIMENSIONS = {
            "thenlper/gte-base": 768,
                "thenlper/gte-large": 1024,
                    "BAAI/bge-large-en": 1024,
                        "text-embedding-ada-002": 1536,
                        }
MAX_CONTEXT_LENGTHS = {
            "gpt-4": 8192,
                "gpt-3.5-turbo": 4096,
                    "gpt-3.5-turbo-16k": 16384,
                        "meta-llama/Llama-2-7b-chat-hf": 4096,
                            "meta-llama/Llama-2-13b-chat-hf": 4096,
                                "meta-llama/Llama-2-70b-chat-hf": 4096,
                                }





import sys; sys.path.append("..")
import warnings; warnings.filterwarnings("ignore")
from dotenv import load_dotenv; load_dotenv()


DB_CONNECTION_STRING=""
OPENAI_API_BASE=""
OPENAI_API_KEY = "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"


ray.shutdown()
ray_context = ray.init(ignore_reinit_error=True,num_cpus=num_cpus, num_gpus=num_gpus,runtime_env={
        "env_vars": {
                    "OPENAI_API_BASE": OPENAI_API_BASE,
                            "OPENAI_API_KEY": OPENAI_API_KEY, 
                                    # "ANYSCALE_API_BASE": os.environ["ANYSCALE_API_BASE"],
                                            # "ANYSCALE_API_KEY": os.environ["ANYSCALE_API_KEY"],
                                                    "DB_CONNECTION_STRING": DB_CONNECTION_STRING,
                                                        },
            "working_dir": str(ROOT_DIR),
            })

print(ray_context)

from rag.config import EMBEDDING_DIMENSIONS, MAX_CONTEXT_LENGTHS

from pathlib import Path
from rag.config import EFS_DIR

EFS_DIR="/tmp/raydocs"

import os
import ray
# Ray dataset
DOCS_DIR = Path(EFS_DIR, "docs.ray.io/en/master/")
ds = ray.data.from_items([{"path": path} for path in DOCS_DIR.rglob("*.html") if not path.is_dir()])
print(f"{ds.count()} documents")



import matplotlib.pyplot as plt
from rag.data import extract_sections



sample_html_fp = Path(EFS_DIR, "docs.ray.io/en/master/rllib/rllib-env.html")
extract_sections({"path": sample_html_fp})[0]



sections_ds = ds.flat_map(extract_sections)
sections_ds.count()



section_lengths = []
for section in sections_ds.take_all():
    section_lengths.append(len(section["text"]))







from functools import partial
from langchain.text_splitter import RecursiveCharacterTextSplitter



# Text splitter
chunk_size = 300
chunk_overlap = 50
text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
                chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                        length_function=len)







# Chunk a sample section
sample_section = sections_ds.take(1)[0]
chunks = text_splitter.create_documents(
            texts=[sample_section["text"]], 
                metadatas=[{"source": sample_section["source"]}])
print("=============================")
print("chunk[0] content:")
print (chunks[0])
print("=============================")









def chunk_section(section, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len)
    chunks = text_splitter.create_documents(
        texts=[section["text"]], 
        metadatas=[{"source": section["source"]}])
    return [{"text": chunk.page_content, "source": chunk.metadata["source"]} for chunk in chunks]



# Scale chunking
chunks_ds = sections_ds.flat_map(partial(
        chunk_section, 
            chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap))
print(f"{chunks_ds.count()} chunks")
chunks_ds.show(1)




from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import numpy as np
from ray.data import ActorPoolStrategy





class EmbedChunks:
    def __init__(self, model_name):
        if model_name == "text-embedding-ada-002":
            self.embedding_model = OpenAIEmbeddings(
                model=model_name,
                openai_api_base=os.environ["OPENAI_API_BASE"],
                openai_api_key=os.environ["OPENAI_API_KEY"])
        else:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"device": "cpu", "batch_size": 100})
    
    def __call__(self, batch):
        print("\ncall EmbedChunks\n")
        embeddings = self.embedding_model.embed_documents(batch["text"])
        return {"text": batch["text"], "source": batch["source"], "embeddings": embeddings}


# Embed chunks
embedding_model_name = "thenlper/gte-base"
embedded_chunks = chunks_ds.map_batches(
            EmbedChunks,
                fn_constructor_kwargs={"model_name": embedding_model_name},
                    batch_size=100, 
                        num_gpus=num_gpus,
                            compute=ActorPoolStrategy(size=4))




# Sample
sample = embedded_chunks.take(1)
print ("embedding size:", len(sample[0]["embeddings"]))
print (sample[0]["text"])
