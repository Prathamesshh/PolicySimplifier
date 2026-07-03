import unittest

from src.document_processor import DocumentProcessingError, chunk_text, normalize_input


class NormalizeInputTests(unittest.TestCase):
    def test_text_input_returns_text(self):
        value = normalize_input(file_bytes=None, raw_text="Hello policy", input_type="text")
        self.assertEqual(value, "Hello policy")

    def test_text_input_requires_non_empty_text(self):
        with self.assertRaises(DocumentProcessingError):
            normalize_input(file_bytes=None, raw_text="   ", input_type="text")

    def test_unknown_input_type_raises(self):
        with self.assertRaises(DocumentProcessingError):
            normalize_input(file_bytes=None, raw_text="x", input_type="unknown")  # type: ignore[arg-type]


class ChunkTextTests(unittest.TestCase):
    def test_chunk_text_adds_metadata(self):
        text = ("Policy clause. " * 400).strip()
        docs = chunk_text(text, source_name="policy.pdf")

        self.assertGreater(len(docs), 1)
        self.assertEqual(docs[0].metadata["source"], "policy.pdf")
        self.assertEqual(docs[0].metadata["chunk_id"], 0)
        self.assertEqual(docs[1].metadata["chunk_id"], 1)
        self.assertTrue(all(d.page_content.strip() for d in docs))


if __name__ == "__main__":
    unittest.main()
