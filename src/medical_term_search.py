import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TERMINOLOGY_FILE = (
    BASE_DIR.parent / "data" / "terminology" / "terminology_v1.json"
)


class MedicalTermSearch:
    def __init__(self, terminology_file=None):
        self.terminology_file = (
            Path(terminology_file)
            if terminology_file is not None
            else DEFAULT_TERMINOLOGY_FILE
        )
        self.concepts = self._load_terminology()
        self.search_index = self._build_search_index()

    def _load_terminology(self):
        with open(self.terminology_file, "r", encoding="utf-8") as file:
            return json.load(file)

    def _normalize(self, text):
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _build_search_index(self):
        index = {}

        for concept in self.concepts:
            concept_id = concept["concept_id"]

            terms = [concept["preferred_term"]]
            terms.extend(concept["synonyms"])

            for term in terms:
                normalized_term = self._normalize(term)

                if normalized_term not in index:
                    index[normalized_term] = []

                index[normalized_term].append(concept_id)

        return index
    def search(self, query):
        query = self._normalize(query)

        results = []

        for concept in self.concepts:
            preferred_term = self._normalize(concept["preferred_term"])
            synonyms = [
                self._normalize(synonym)
                for synonym in concept["synonyms"]
            ]

            score = 0
            match_type = None

            # 1. Exact preferred-term match
            if query == preferred_term:
                score = 100
                match_type = "exact_preferred"

            # 2. Exact synonym match
            elif query in synonyms:
                score = 90
                match_type = "exact_synonym"

            # 3. Prefix match
            elif preferred_term.startswith(query):
                score = 70
                match_type = "prefix_preferred"

            elif any(synonym.startswith(query) for synonym in synonyms):
                score = 60
                match_type = "prefix_synonym"

            # 4. Keyword match
            else:
                query_words = set(query.split())

                all_terms = [preferred_term] + synonyms

                for term in all_terms:
                    term_words = set(term.split())

                    if query_words.issubset(term_words):
                        score = 40
                        match_type = "keyword"
                        break

            if score > 0:
                results.append({
                    "concept_id": concept["concept_id"],
                    "preferred_term": concept["preferred_term"],
                    "definition": concept["definition"],
                    "score": score,
                    "match_type": match_type
                })

        results.sort(key=lambda result: result["score"], reverse=True)

        return results


if __name__ == "__main__":
    search = MedicalTermSearch()

    print("Loaded concepts:", len(search.concepts))
    print("Indexed terms:", len(search.search_index))

    test_queries = [
        "dyspnea",
        "orthop",
        "breathing",
        "fever",
        "chest pain",
        "chest",
        "short of breath",
        "coughing up blood",
        "phlegm",
        "heart racing",
        "dizzy",
        "throwing up",
    ]

    for query in test_queries:
        print(f"\nSearch: {query}")

        results = search.search(query)

        for result in results[:5]:
            print(
                f"- {result['preferred_term']} "
                f"({result['score']}, {result['match_type']})"
            )
