from networkx import MultiDiGraph
from project.utils.rpq import rpq
from pyformlang.regular_expression import Regex


class TestsForRpq:
    def test_works_as_expected(self):
        graph = MultiDiGraph()
        graph.add_edges_from(
            [
                (0, 1, {"label": "C"}),
                (1, 2, {"label": "A"}),
                (2, 3, {"label": "T"}),
                (3, 4, {"label": "S"}),
            ]
        )
        regex = Regex("C.A.T.S")
        assert rpq(regex, graph, {0}, {4}) == {(0, 4)}

    def test_empty_graph(self):
        assert rpq(Regex("Put your text here"), MultiDiGraph()) == set()

    def test_empty_regex(self):
        graph = MultiDiGraph()
        graph.add_edges_from(
            [
                (0, 1, {"label": "C"}),
                (1, 2, {"label": "A"}),
                (2, 3, {"label": "T"}),
                (3, 4, {"label": "S"}),
            ]
        )
        assert rpq(Regex(""), graph) == set()
