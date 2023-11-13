from networkx import MultiDiGraph
from pyformlang.cfg import CFG

from project.utils.hellings import cfpq
from tests.test_cases.tests_utils import get_path_to_test_case


class TestsForCFPQ:
    @staticmethod
    def create_graph(edges: set) -> MultiDiGraph:
        graph = MultiDiGraph()
        graph.add_edges_from(
            list(map(lambda edge: (edge[0], edge[2], {"label": edge[1]}), edges))
        )
        return graph

    def test_cfpq_not_empty(self):
        graph = self.create_graph({(0, "a", 1), (1, "a", 2), (2, "b", 3), (3, "b", 4)})
        cfg = get_path_to_test_case("cfpq_test.txt")
        res = cfpq(cfg, graph, {0}, {4})

        assert res == set()

    def test_cfpq_empty(self):
        graph = self.create_graph({(0, "a", 1), (1, "a", 2), (2, "b", 3), (3, "b", 4)})
        cfg = CFG()
        res = cfpq(cfg, graph, {0}, {4})
        assert res == set()

        cfg = get_path_to_test_case("cfpq_test.txt")
        graph = MultiDiGraph()
        res = cfpq(cfg, graph, {0}, {4})
        assert res == set()
