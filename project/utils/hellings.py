from collections import defaultdict
from networkx import MultiDiGraph
from pyformlang.cfg import CFG, Variable, Terminal
from project.utils.cfg import create_wcnf_from_cfg, read_cfg_from_file
from project.utils.graph_lib import get_graph_by_name


def hellings_closure(cfg: CFG, graph: MultiDiGraph) -> set:
    """
    Using the Hellings algorithm, determines the reachability between all pairs of vertices for a given graph and a
    given context-free grammar

    Parameters
    ----------
    cfg: CFG
    Context-free grammar

    graph: MultiDiGraph
    A directed graph class

    Returns
    -------
    result: set
    Result is set of triples of the form (vertex, non-terminal, vertex)
    """
    wcnf = create_wcnf_from_cfg(cfg)
    eps_prods = set()
    term_prods = defaultdict(set)
    non_term_prods = defaultdict(set)

    for p in wcnf.productions:
        if len(p.body) == 0:
            eps_prods.add(p.head)
        elif len(p.body) == 1:
            term_prods[p.head].add(p.body[0])
        else:
            non_term_prods[p.head].add((p.body[0], p.body[1]))

    result = {(node, v, node) for v in eps_prods for node in graph.nodes}.union(
        {
            (i, v, j)
            for v, terms in term_prods.items()
            for i, j, label in graph.edges(data="label")
            if Terminal(label) in terms
        }
    )

    q = result.copy()

    while len(q) > 0:
        temp = set()
        i, v, j = q.pop()

        for i_res, v_res, j_res in result:
            if i == j_res:
                for head, body in non_term_prods.items():
                    if (i_res, head, j) not in result and (v_res, v) in body:
                        q.add((i_res, head, j))
                        temp.add((i_res, head, j))

            if j == i_res:
                for head, body in non_term_prods.items():
                    if (i, head, j_res) not in result and (v, v_res) in body:
                        q.add((i, head, j_res))
                        temp.add((i, head, j_res))

        result = result.union(temp)

    return result


def cfpq(
    cfg: CFG | str,
    graph: MultiDiGraph | str,
    start_nodes: set = None,
    final_nodes: set = None,
    start_symbol: Variable = Variable("S"),
) -> set:
    """
    Solves the reachability problem for a graph, a grammar, a given set of starting and final vertices, and a given
    non-terminal using the Hellings algorithm

    Parameters
    ----------
    cfg: CFG | str
    Context-free grammar or path to the file where the CFG is described

    graph: MultiDiGraph | str
    A directed graph class or name of the graph from the dataset

    start_nodes: set | None
    This is set of start nodes of the graph

    final_nodes: set | None
    This is set of final nodes of the graph

    start_symbol: Variable
    Start symbol of CFG

    Returns
    -------
    result: set
    Result is a set of pairs of vertices
    """
    if isinstance(cfg, str):
        cfg = read_cfg_from_file(cfg)

    if isinstance(graph, str):
        graph = get_graph_by_name(graph)

    cfg._start_symbol = start_symbol

    if start_nodes is None:
        start_nodes = set(graph.nodes)

    if final_nodes is None:
        final_nodes = set(graph.nodes)

    result = set()
    for i, v, j in hellings_closure(cfg, graph):
        if v == start_symbol and i in start_nodes and j in final_nodes:
            result.add((i, j))

    return result
