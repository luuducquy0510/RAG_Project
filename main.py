from getSentence import querySentence
from chunkSplit import GetChunk
import pandas as pd

file_path = 'database/Neural_Network.pdf'

if __name__ == "__main__":
    # extract sentences into a list
    text = querySentence(file_path)
    list_of_sentence = text.extractSentence()

    # split doc into chunks
    chunks = GetChunk(list_of_sentence)
    chunk = chunks.splitChunks()

    print(chunk)
    # df = pd.DataFrame(chunks)
    # minimize chunk 
    # min_token_length = 10
    # for row in df[df['chunk_token_count']<= min_token_length].sample(1, replace = True).iterrows():
    #     print(f'Chunk token count: {row[1]["chunk_token_count"]} , Text: {row[1]["sentence_chunk"]}')

    # pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
