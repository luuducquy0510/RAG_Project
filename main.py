from file_extraction import file_extract
from nlp import model




file_path='database/Neural_Network.pdf'


# Extract text from document
file_extrator = file_extract(file_path)
pages_and_texts = file_extrator.open_and_read_pdf()

for item in pages_and_texts:
    nlp = model(item['text'])
    item['sentences'] = list(nlp.NLP().sents)
    # change all item in sentences to string
    item['sentences'] = [str(sentence) for sentence in item['sentences']]
    # count the sentences

    item['pages_sentence_count_spacy'] = len(item['sentences'])

