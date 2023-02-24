import unittest
from preprocessing import *

class TestPreprocessing(unittest.TestCase):
    
    def test_remove_stopwords(self):
        text = "this is a sample text that includes some stop words such as the, and, etc."
        expected_output = "sample text includes stop words like , , etc ."
        self.assertEqual(remove_stopwords(text), expected_output)
    
    def test_perform_lemmatization(self):
        text = "running played plays"
        expected_output = "running played play"
        self.assertEqual(perform_lemmatization(text), expected_output)
    
    def test_perform_stemming(self):
        text = "running played plays"
        expected_output = "run play play"
        self.assertEqual(perform_stemming(text), expected_output)
    
    def test_preprocess_text(self):
        text = "This is a sample text. It includes some stop words, and it has words in different tenses (e.g. playing, played)."
        expected_output = "thi sampl text . includ stop word , word differ tens ( e.g. play , play ) ."
        self.assertEqual(preprocess_text(text), expected_output)

if __name__ == '__main__':
    unittest.main()
