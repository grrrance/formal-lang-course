from networkx import MultiDiGraph
from pyformlang.cfg import CFG, Variable

from project.utils.cfpq import cfpq
from project.utils.cfpq import CFPQAlgorithm
from tests.test_cases.tests_utils import get_path_to_test_case


class TestsForCFPQ:
    algorithms = [CFPQAlgorithm.MATRIX, CFPQAlgorithm.HELLINGS, CFPQAlgorithm.TENSOR]

    @staticmethod
    def create_graph(edges: set) -> MultiDiGraph:
        graph = MultiDiGraph()
        graph.add_edges_from(
            list(map(lambda edge: (edge[0], edge[2], {"label": edge[1]}), edges))
        )
        return graph

    @staticmethod
    def assertion(
        cfg: CFG | str,
        graph: MultiDiGraph | str,
        expected_result: set,
        start_nodes: set = None,
        final_nodes: set = None,
        start_symbol: Variable = Variable("S"),
    ):
        for algo in TestsForCFPQ.algorithms:
            res = cfpq(cfg, graph, start_nodes, final_nodes, start_symbol, algo)
            assert res == expected_result

    def test_cfpq_not_empty(self):
        graph = self.create_graph({(0, "a", 1), (1, "a", 2), (2, "b", 3), (3, "b", 4)})
        cfg = get_path_to_test_case("cfpq_test.txt")
        TestsForCFPQ.assertion(cfg, graph, {(0, 4)}, {0}, {4})

    def test_cfpq_empty(self):
        graph = self.create_graph({(0, "a", 1), (1, "a", 2), (2, "b", 3), (3, "b", 4)})
        cfg = CFG()
        TestsForCFPQ.assertion(cfg, graph, set(), {0}, {4})

        cfg = get_path_to_test_case("cfpq_test.txt")
        graph = MultiDiGraph()
        TestsForCFPQ.assertion(cfg, graph, set(), {0}, {4})
