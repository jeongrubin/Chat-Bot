import unittest
from pathlib import Path

from recommender import ProgramRetriever, build_fallback_answer, load_programs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "programs.json"


class ProgramRetrieverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.programs = load_programs(DATA_PATH)
        cls.retriever = ProgramRetriever(cls.programs)

    def test_load_programs(self) -> None:
        self.assertTrue(self.programs)
        self.assertTrue(all(program.get("제목") for program in self.programs))

    def test_search_finds_ncs_program(self) -> None:
        results = self.retriever.search("공기업 NCS 필기 준비", top_k=3)
        self.assertTrue(results)
        self.assertTrue(any("NCS" in result["제목"] for result in results))

    def test_fallback_answer_is_grounded(self) -> None:
        results = self.retriever.search("면접 취업 상담", top_k=1)
        answer = build_fallback_answer(results)
        self.assertIn(results[0]["제목"], answer)

    def test_empty_query_returns_no_results(self) -> None:
        self.assertEqual(self.retriever.search("   "), [])


if __name__ == "__main__":
    unittest.main()
