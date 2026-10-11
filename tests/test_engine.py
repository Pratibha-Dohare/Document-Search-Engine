import unittest
from search_engine import Document, SearchEngine, tokenize

class SearchEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = SearchEngine([
            Document("1", "Python search", "inverted index python heap"),
            Document("2", "Machine learning", "python machine learning"),
            Document("3", "Database", "sql database indexing"),
        ])

    def test_tokenize(self):
        self.assertEqual(tokenize("The QUICK, fox!"), ["quick", "fox"])

    def test_inverted_index(self):
        self.assertEqual(set(self.engine.postings["python"]), {"1", "2"})

    def test_and_matching(self):
        self.assertEqual([r.document.doc_id for r in self.engine.search("python heap", mode="AND")], ["1"])

    def test_or_matching(self):
        self.assertEqual(len(self.engine.search("python sql", mode="OR")), 3)

    def test_and_unknown_term(self):
        self.assertEqual(self.engine.search("python nonexistentword", mode="AND"), [])

    def test_top_k(self):
        self.assertEqual(len(self.engine.search("python", top_k=1)), 1)

    def test_unknown_query(self):
        self.assertEqual(self.engine.search("qwertynotindexed"), [])

    def test_empty_document(self):
        self.assertEqual(SearchEngine([Document("e", "", "")]).search("any"), [])

    def test_duplicate_ids(self):
        with self.assertRaises(ValueError):
            SearchEngine([Document("x", "one", "foo"), Document("x", "two", "bar")])

if __name__ == "__main__":
    unittest.main()
