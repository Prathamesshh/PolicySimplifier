import unittest
from unittest.mock import Mock, patch

from langchain.docstore.document import Document

from src.rag_pipeline import Reranker, answer_question, rewrite_query


class RewriteQueryTests(unittest.TestCase):
    def test_returns_original_question_when_no_history(self):
        llm = Mock()
        question = "What is the leave policy?"
        result = rewrite_query(llm=llm, question=question, chat_history=[])
        self.assertEqual(result, question)
        llm.assert_not_called()


class RerankerTests(unittest.TestCase):
    def test_rerank_orders_by_model_score(self):
        reranker = Reranker.__new__(Reranker)
        reranker._model = Mock()
        reranker._model.predict.return_value = [0.2, 0.9, 0.5]

        docs = [
            Document(page_content="low", metadata={"chunk_id": 1}),
            Document(page_content="high", metadata={"chunk_id": 2}),
            Document(page_content="mid", metadata={"chunk_id": 3}),
        ]

        top = reranker.rerank(query="q", documents=docs, top_k=2)
        self.assertEqual([d.page_content for d in top], ["high", "mid"])


class AnswerQuestionTests(unittest.TestCase):
    def test_answer_question_returns_answer_and_citations(self):
        retriever = Mock()
        docs = [
            Document(page_content="Clause A", metadata={"source": "doc.pdf", "chunk_id": 3}),
            Document(page_content="Clause B", metadata={"source": "doc.pdf", "chunk_id": 4}),
        ]
        retriever.invoke.return_value = docs

        reranker = Mock()
        reranker.rerank.return_value = docs[:1]

        fake_chain = Mock()
        fake_chain.invoke.return_value = Mock(content="Answer from context")
        fake_prompt = Mock()
        fake_prompt.__or__ = Mock(return_value=fake_chain)
        llm = Mock()

        with patch("src.rag_pipeline.rewrite_query", return_value="standalone"), patch(
            "src.rag_pipeline._ANSWER_PROMPT", fake_prompt
        ):
            result = answer_question(
                llm=llm,
                reranker=reranker,
                ensemble_retriever=retriever,
                question="original question",
                chat_history=[],
            )

        self.assertEqual(result["answer"], "Answer from context")
        self.assertEqual(result["standalone_query"], "standalone")
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["citations"][0]["source"], "doc.pdf")
        self.assertEqual(result["citations"][0]["chunk_id"], 3)


if __name__ == "__main__":
    unittest.main()
