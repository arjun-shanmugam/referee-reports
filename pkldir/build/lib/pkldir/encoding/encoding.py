import textract 
import base64

from docx import Document    


class Base64Encoded:
    filename: str
    doctype: str
    data: bytes

    def __init__(self, filename):
        if filename[-4:] == ".txt":
            self.doctype = 'txt'
            with open(filename, "rb") as f:
                text = f.read()

        elif filename[-3:] == ".md":
            self.doctype = 'md'
            with open(filename, "rb") as f:
                text = f.read()

        elif filename[-5:] == ".docx":
            self.doctype = "docx"
            with open(filename, "rb") as f:
                doc = Document(f)
                text = "".join([para.text for para in doc.paragraphs])
                text = text.encode()

        elif filename[-4:] == ".pdf":
            self.doctype = "pdf"
            text = textract.process(filename) 
            
        else:
            msg = "The file " + filename + " is not one of our supported file types (e.g., docx, pdf, txt)." 
            raise ValueError(msg)

        self.data = base64.b64encode(text)
        self.filename = filename

    def decode(self):
        return base64.b64decode(self.data)