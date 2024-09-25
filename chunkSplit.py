import re

num_chunk_size = 30
def split_list(input_list:list[str],
               slice_size: int=num_chunk_size) -> list[list[str]]:
  return [input_list[i: i + slice_size] for i in range(0, len(input_list), slice_size)]



class GetChunk:
    def __init__(self, listSentence) -> None:
        self.listSentence = listSentence

    def splitChunks(self):    
        for item in self.listSentence:
            item['sentences_chunks'] = split_list(item['sentences'],num_chunk_size)
            item['num_chunks'] = len(item['sentences_chunks'])

        page_and_chunks = []
        for item in self.listSentence:
            for sentence_chunk in item['sentences_chunks']:
                chunk_dict = {}
                chunk_dict['page_number'] = item['page_number']


                joined_sentences_chunk = ''.join(sentence_chunk).replace('  ', ' ').strip()
                joined_sentences_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentences_chunk) # add space before capitalize words

                chunk_dict['sentence_chunk'] = joined_sentences_chunk
                # stats of chunk
                chunk_dict['chunk_char_count'] = len(joined_sentences_chunk)
                chunk_dict['chunk_word_count'] = len([word for word in joined_sentences_chunk.split(' ')])
                chunk_dict['chunk_token_count'] = len(joined_sentences_chunk) / 4

                page_and_chunks.append(chunk_dict)
        
        return page_and_chunks