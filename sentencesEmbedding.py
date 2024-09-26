from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Embedding:
    def __init__(self, chunkList):
        self.chunkList = chunkList    
    
    def EmbedModel(self):        
        embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu") # choose the device to load the model to (note: GPU will often be *much* faster than CPU)
        embedding_model.to(device)

        for item in self.chunkList:
            item["embedding"] = embedding_model.encode(item["sentence_chunk"])
            
        text_chunks = [item["sentence_chunk"] for item in self.chunkList]
        
        # Embed all texts in batches
        text_chunk_embeddings = embedding_model.encode(text_chunks,
                                                    batch_size=32,
                                                    convert_to_tensor=True) # optional to return embeddings as tensor instead of array

        embedding_df = pd.DataFrame(self.chunkList)
        
        # RAG
        # Convert texts and embedding df to list of dicts
        pages_and_chunks = embedding_df.to_dict(orient="records")

        # Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
        embeddings = torch.tensor(np.array(embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)
        
        return embeddings, pages_and_chunks
