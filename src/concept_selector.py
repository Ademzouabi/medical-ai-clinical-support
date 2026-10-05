from medical_term_search import MedicalTermSearch


class ConceptSelector:
    def __init__(self, search_engine):
        self.search_engine = search_engine

    def select(self, query):
        results = self.search_engine.search(query)

        if not results:
            print("\nNo matching medical concepts found.")
            return None

        print(f"\nSearch: {query}")
        print("\nMatching concepts:")

        for index, result in enumerate(results, start=1):
            print(
                f"{index}. {result['preferred_term']} "
                f"({result['score']}, {result['match_type']})"
            )

        while True:
            choice = input("\nSelect a concept number (0 to cancel): ").strip()

            if choice == "0":
                return None

            if not choice.isdigit():
                print("Please enter a valid number.")
                continue

            choice = int(choice)

            if 1 <= choice <= len(results):
                selected = results[choice - 1]

                print(
                    f"\nSelected: {selected['preferred_term']}"
                )
                print(
                    f"Concept ID: {selected['concept_id']}"
                )

                return selected["concept_id"]

            print(
                f"Please enter a number between 0 and {len(results)}."
            )


if __name__ == "__main__":
    search_engine = MedicalTermSearch()
    selector = ConceptSelector(search_engine)

    while True:
        query = input(
            "\nEnter a medical term "
            "(or 'exit' to quit): "
        ).strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Please enter a medical term.")
            continue

        concept_id = selector.select(query)

        if concept_id:
            print(f"\nCanonical concept: {concept_id}")