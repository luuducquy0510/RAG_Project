import fitz

class file_extract:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path



    def text_formatter(self, text : str) -> str:
        cleaned_text = text.replace('\n', ' ').strip()

        return cleaned_text

    def open_and_read_pdf(self)->list[dict]:
        doc = fitz.open(self.pdf_path)
        pages_and_texts = []
        for page_number, page in enumerate(doc):
            text = page.get_text()
            text = self.text_formatter(text)
            pages_and_texts.append({'page_number': page_number,
                                'page_char_count': len(text),
                                'page_word_count': len(text.split(' ')),
                                'page_sentence_count_raw': len(text.split('.')),
                                'page_token_count': len(text) / 4,
                                'text': text})
        return pages_and_texts
    

