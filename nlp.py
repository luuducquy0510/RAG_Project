from spacy.lang.en import English


class model:
    def __init__(self, text) -> None:
        self.text = text
    
    
    def NLP(self):
        nlp = English()

        # add a sentencizer pipeline
        nlp.add_pipe('sentencizer')
        return nlp(self.text)