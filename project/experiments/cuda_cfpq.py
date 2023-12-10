from collections import defaultdict
from enum import Enum
from typing import AbstractSet

import pycubool
from networkx import MultiDiGraph
from pyformlang.cfg import CFG, Variable, Terminal, Production
from pyformlang.finite_automaton import EpsilonNFA

from project.utils.cfg import create_wcnf_from_cfg, read_cfg_from_file
from project.utils.graph_lib import get_graph_by_name
from project.utils.ecfg import ECFG
from project.utils.rsm import RSM
from project.experiments.cuda_matrix_automata import CuBoolMatrixAutomata


def _convert_wcnf_prods(prods: AbstractSet[Production]) -> (set, dict, dict):
    """
    Converts productions of context-free grammar in wcnf

    Parameters
    ----------
    prods: AbstractSet[Production]
    Productions of context-free grammar in wcnf

    Returns
    -------
    result: (set, dict, dict)
    Result is a triple of a set of non-terminals, a dictionary where
    the key is a non-terminal and the value is a terminal, a dictionary where the key is a non-terminal and the
    value is a pair of non-terminals
    """
    eps_prods = set()
    term_prods = defaultdict(set)
    non_term_prods = defaultdict(set)

    for p in prods:
        if len(p.body) == 0:
            eps_prods.add(p.head)
        elif len(p.body) == 1:
            term_prods[p.head].add(p.body[0])
        else:
            non_term_prods[p.head].add((p.body[0], p.body[1]))
    return eps_prods, term_prods, non_term_prods


def cubool_hellings_closure(cfg: CFG, graph: MultiDiGraph) -> set:
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
    eps_prods, term_prods, non_term_prods = _convert_wcnf_prods(wcnf.productions)

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


def cubool_matrix_closure(wcnf: CFG, graph: MultiDiGraph) -> set:
    """
    Using the Matrix algorithm, determines the reachability between all pairs of vertices for a given graph and a
    given context-free grammar

    Parameters
    ----------
    wcnf: CFG
    Context-free grammar

    graph: MultiDiGraph
    A directed graph class

    Returns
    -------
    result: set
    Result is set of triples of the form (vertex, non-terminal, vertex)
    """
    wcnf = create_wcnf_from_cfg(wcnf)
    eps_prods, term_prods, non_term_prods = _convert_wcnf_prods(wcnf.productions)
    nodes = list(graph.nodes)
    indexes_nodes = {node: i for i, node in enumerate(nodes)}
    n = graph.number_of_nodes()

    matrices = {var: pycubool.Matrix.empty((n, n)) for var in wcnf.variables}

    for i in range(n):
        for var in eps_prods:
            matrices[var][i, i] = True

    for u, v, label in graph.edges(data="label"):
        i, j = indexes_nodes[u], indexes_nodes[v]
        for var, vars in term_prods.items():
            if Terminal(label) in vars:
                matrices[var][i, j] = True

    while True:
        changed = False
        for head, body in non_term_prods.items():
            for vars in body:
                old_nnz = matrices[head].nvals
                matrices[vars[0]].mxm(
                    matrices[vars[1]], out=matrices[head], accumulate=True
                )
                new_nnz = matrices[head].nvals
                changed |= old_nnz != new_nnz

        if not changed:
            break

    return set(
        (nodes[i], v, nodes[j]) for v, mat in matrices.items() for i, j in mat.to_list()
    )


def cubool_tensor_closure(cfg: CFG, graph: MultiDiGraph) -> set:
    """
    Using the Tensor algorithm, determines the reachability between all pairs of vertices for a given graph and a
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
    ecfg = ECFG.from_cfg(cfg)
    rsm = RSM.from_ecfg(ecfg).minimize()
    rsm_mat = CuBoolMatrixAutomata.create_matrix_from_rsm(rsm)
    rsm_states = {i: state for state, i in rsm_mat.indexes.items()}

    graph_mat = CuBoolMatrixAutomata.create_matrix_from_fa(
        EpsilonNFA.from_networkx(graph)
    )
    n = graph_mat.count_states
    graph_states = {i: st for st, i in graph_mat.indexes.items()}

    for var in cfg.get_nullable_symbols():
        if var.value not in graph_mat.adjacency_matrices:
            graph_mat.adjacency_matrices[var.value] = pycubool.Matrix.empty((n, n))
        for i in range(n):
            graph_mat.adjacency_matrices[var.value][i, i] = True

    prev_nnz = 0

    while True:
        tc = rsm_mat.intersect(graph_mat).transitive_closure()

        if tc.nvals == prev_nnz:
            break

        prev_nnz = tc.nvals

        for i, j in tc.to_list():
            cfg_i, cfg_j = i // n, j // n
            graph_i, graph_j = i % n, j % n

            state_source = rsm_states[cfg_i]
            state_target = rsm_states[cfg_j]

            var, _ = state_source.value

            if (
                state_source in rsm_mat.start_states
                and state_target in rsm_mat.final_states
            ):
                if var not in graph_mat.adjacency_matrices:
                    graph_mat.adjacency_matrices[var] = pycubool.Matrix.empty((n, n))
                graph_mat.adjacency_matrices[var][graph_i, graph_j] = True

    return {
        (graph_states[graph_i], var, graph_states[graph_j])
        for var, mat in graph_mat.adjacency_matrices.items()
        for graph_i, graph_j in mat.to_list()
    }


class CuBoolCFPQAlgorithm(Enum):
    """
    Class that represents an enumeration of algorithms that solves the CFPQ problem

    Values
    ----------
    HELLINGS : CuBoolCFPQAlgorithm
    Hellings algorithm

    MATRIX : CuBoolCFPQAlgorithm
    Matrix algorithm

    TENSOR: CuBoolCFPQAlgorithm
    Tensor algorithm
    """

    HELLINGS = cubool_hellings_closure
    MATRIX = cubool_matrix_closure
    TENSOR = cubool_tensor_closure


def cfpq(
    cfg: CFG | str,
    graph: MultiDiGraph | str,
    start_nodes: set = None,
    final_nodes: set = None,
    start_symbol: Variable = Variable("S"),
    algorithm: CuBoolCFPQAlgorithm = CuBoolCFPQAlgorithm.HELLINGS,
) -> set:
    """
    Solves the reachability problem for a graph, a grammar, a given set of starting and final vertices, and a given
    non-terminal using the given algorithm

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

    algorithm: CuBoolCFPQAlgorithm
    Algorithm that solves the problem of reachability between all pairs of vertices for a
    given graph and a given context-free grammar

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
    for i, v, j in algorithm(cfg, graph):
        if v == start_symbol and i in start_nodes and j in final_nodes:
            result.add((i, j))

    return result
