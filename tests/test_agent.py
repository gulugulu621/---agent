import json
import tempfile
import unittest
from pathlib import Path

from src.agent import DATA, MockPolicy, ask, validate_finish
from src.retrieval import Corpus


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.corpus = Corpus(DATA)

    def run_case(self, question, **kwargs):
        return ask(question, corpus=self.corpus, policy=MockPolicy(), **kwargs)

    def test_clear_answer_and_quote(self):
        result = self.run_case("报销需要哪些材料")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["citations"][0]["doc_id"], "D03")
        self.assertIn(result["citations"][0]["quote"], self.corpus.read("D03", "C1")["text"])

    def test_conflict(self):
        result = self.run_case("差旅报销期限多久")
        self.assertEqual(result["status"], "conflict")
        self.assertEqual({x["doc_id"] for x in result["citations"]}, {"D01", "D02"})
        self.assertEqual(sum(x["action"] == "search" for x in result["trace"]), 2)

    def test_out_of_domain(self):
        self.assertEqual(self.run_case("量子芯片的制造工艺是什么")["status"], "insufficient")

    def test_paraphrase(self):
        self.assertEqual(self.run_case("出差费用申请要带什么票据和单子")["status"], "answered")

    def test_context_followup(self):
        result = self.run_case("那要交什么材料", history=[{"role": "user", "content": "差旅报销怎么办"}])
        self.assertEqual(result["status"], "answered")

    def test_tool_failure(self):
        class Broken(Corpus):
            def search(self, query, top_k=5):
                raise OSError("unavailable")
        result = ask("报销材料", corpus=Broken(DATA), policy=MockPolicy())
        self.assertEqual(result["status"], "tool_error")
        self.assertFalse(result["citations"])

    def test_changed_document_changes_evidence(self):
        docs = json.loads(DATA.read_text(encoding="utf-8"))
        docs = [d for d in docs if d["doc_id"] != "D02"]
        docs[0]["chunks"][0]["text"] = "差旅费用应在出差结束后45天内提交报销申请。"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "docs.json"
            path.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
            result = ask("差旅报销期限多久", corpus=Corpus(path), policy=MockPolicy())
        self.assertEqual(result["status"], "answered")
        self.assertIn("45天", result["answer"])

    def test_unread_quote_rejected_and_limit(self):
        self.assertEqual(validate_finish({"status": "answered", "answer": "15天", "citations": [{"doc_id": "D02", "chunk_id": "C1", "quote": "15天"}]}, [])["status"], "insufficient")
        self.assertEqual(self.run_case("报销期限多久", max_steps=1)["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
