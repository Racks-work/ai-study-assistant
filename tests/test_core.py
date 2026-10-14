import unittest
from study_assistant.core import Chunk, chunk_text, extract_document, retrieve, make_quiz, offline_answer

class CoreTests(unittest.TestCase):
    def test_chunk_overlap_and_limit(self):
        chunks = chunk_text('hello world ' * 200, 'notes.txt', 'document', size=100, overlap=20)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(c.text) <= 100 for c in chunks))
    def test_utf8_text(self):
        chunks = extract_document('a.txt', 'Neural networks use weights.'.encode())
        self.assertEqual(chunks[0].source, 'a.txt')
    def test_invalid_file(self):
        with self.assertRaises(ValueError):
            extract_document('malware.exe', b'bad')
    def test_retrieval(self):
        chunks = [Chunk('a.txt', 'document', 'Photosynthesis uses sunlight and chlorophyll.'), Chunk('b.txt', 'document', 'Database indexes speed up SQL queries.')]
        found = retrieve('What is chlorophyll in photosynthesis?', chunks)
        self.assertEqual(found[0][0].source, 'a.txt')
    def test_no_hallucinated_answer(self):
        self.assertIn('could not find', offline_answer('unknown', []).lower())
    def test_quiz(self):
        chunks = [Chunk('a.txt', 'document', 'Supervised learning algorithms use labeled training examples to predict outcomes accurately.')]
        quiz = make_quiz(chunks)
        self.assertEqual(len(quiz), 1)
        self.assertIn('________', quiz[0]['question'])
        self.assertTrue(quiz[0]['answer'])

if __name__ == '__main__':
    unittest.main()
