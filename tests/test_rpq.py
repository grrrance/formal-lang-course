from networkx import MultiDiGraph
from project.utils.rpq import tensors_rpq
from project.utils.rpq import bfs_rpq
from pyformlang.regular_expression import Regex


class TestsForTensorsRpq:
    graph = MultiDiGraph()
    graph.add_edges_from(
        [
            (0, 1, {"label": "C"}),
            (1, 2, {"label": "A"}),
            (2, 3, {"label": "T"}),
            (3, 4, {"label": "S"}),
        ]
    )

    def test_works_as_expected(self):
        regex = Regex("C.A.T.S")
        assert tensors_rpq(regex, self.graph, {0}, {4}) == {(0, 4)}

    def test_empty_graph(self):
        assert tensors_rpq(Regex("Put your text here"), MultiDiGraph()) == set()

    def test_empty_regex(self):
        assert tensors_rpq(Regex(""), self.graph) == set()


class TestsForBfsRpq:
    graph = MultiDiGraph()
    graph.add_edges_from(
        [
            (1, 2, {"label": "b"}),
            (2, 0, {"label": "a"}),
            (0, 1, {"label": "a"}),
            (0, 3, {"label": "b"}),
            (3, 0, {"label": "b"}),
        ]
    )
    regex = Regex("a.b*")

    def test_works_as_expected_including_all(self):
        result = bfs_rpq(self.regex, self.graph, {0}, {2}, True)
        assert result == {2}

    def test_works_as_expected_separately(self):
        result = bfs_rpq(self.regex, self.graph, {0}, {2}, False)
        assert result == {0: {2}}

    def test_empty_graph(self):
        graph = MultiDiGraph()
        regex = Regex("Put your text here")
        assert bfs_rpq(regex, graph, is_include_all=False) == {}
        assert bfs_rpq(regex, graph, is_include_all=True) == set()

    def test_empty_regex(self):
        assert bfs_rpq(Regex(""), self.graph, is_include_all=False) == {}
        assert bfs_rpq(Regex(""), self.graph, is_include_all=True) == set()
