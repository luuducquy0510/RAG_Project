from getSentence import querySentence
from chunkSplit import GetChunk
from sentencesEmbedding import Embedding
import textwrap
from sentence_transformers import SentenceTransformer, util
from time import perf_counter as timer
import torch


file_path = 'database/Neural_Network.pdf'

if __name__ == "__main__":
    # extract sentences into a list
    text = querySentence(file_path)
    list_of_sentence = text.extractSentence()

    # split doc into chunks
    chunks = GetChunk(list_of_sentence)
    chunk = chunks.splitChunks()
    
    # embedding chunks
    embedded = Embedding(chunk)
    embeddedChunk, pages_and_chunks = embedded.EmbedModel()
    
    def print_wrapped(text, wrap_length=80):
        wrapped_text = textwrap.fill(text, wrap_length)
        print(wrapped_text)
    
    # query
    query = input('Input something to find: ')
    print(f"Query: {query}")

    # 2. Embed the query to the same numerical space as the text examples
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu") # choose the device to load the model to (note: GPU will often be *much* faster than CPU)
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    # 3. Get similarity scores with the dot product (we'll time this for fun)
    start_time = timer()
    dot_scores = util.dot_score(a=query_embedding, b=embeddedChunk)[0]
    end_time = timer()

    print(f"Time take to get scores on {len(embeddedChunk)} embeddings: {end_time-start_time:.5f} seconds.")

    # 4. Get the top-k results ( 5)
    top_results_dot_product = torch.topk(dot_scores, k=3)
    top_results_dot_product

    print("Results:")
    # Loop through zipped together scores and indicies from torch.topk
    for score, idx in zip(top_results_dot_product[0], top_results_dot_product[1]):
        print(f"Score: {score:.4f}")
        # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
        print("Text:")
        print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
        # Print the page number too so we can reference the textbook further (and check the results)
        print(f"Page number: {pages_and_chunks[idx]['page_number']}")
        print("\n")