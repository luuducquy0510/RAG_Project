from file_extraction import file_extract
from nlp import model

class querySentence:
    def __init__(self, filepath) -> None:
        self.filepath = filepath

    def extractSentence(self):
        file_path=self.filepath

        # Extract text from document
        file_extrator = file_extract(file_path)
        pages_and_texts = file_extrator.open_and_read_pdf()
        sentenceList = []
        for item in pages_and_texts:
            nlp = model(item['text'])
            item['sentences'] = list(nlp.NLP().sents)
            # change all item in sentences to string
            item['sentences'] = [str(sentence) for sentence in item['sentences']]
            # count the sentences

            item['pages_sentence_count_spacy'] = len(item['sentences'])
            sentenceList.append(item)
        return sentenceList