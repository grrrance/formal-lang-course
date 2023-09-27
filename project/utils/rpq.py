from networkx import MultiDiGraph
from pyformlang.regular_expression import Regex
from typing import Set
from project.utils.matrix_automata import MatrixAutomata
from project.utils.automata_lib import create_nfa, create_minimum_dfa


def rpq(
    regex: Regex,
    graph: MultiDiGraph,
    start_states: Set[int] = None,
    final_states: Set[int] = None,
) -> set:
    """
    Computes Regular Path Querying from given graph language and regular expression language

    Parameters
    ----------
    regex: Regex
    Regular expression

    graph: MultiDiGraph
    A directed graph class from dataset that can store multi edges

    start_states: Set[int]
    Set of start states

    final_states: Set[int]
    Set of final states

    Returns
    -------
    res: Set[Tuple]
    Regular path query
    """
    result = set()
    graph_matrix = MatrixAutomata.create_matrix_from_fa(
        create_nfa(graph, start_states, final_states)
    )
    regex_matrix = MatrixAutomata.create_matrix_from_fa(create_minimum_dfa(regex))
    intersected_matrix = graph_matrix.intersect(regex_matrix)
    tc = intersected_matrix.transitive_closure()
    for start in intersected_matrix.start_states:
        for final in intersected_matrix.final_states:
            if tc[start, final]:
                result.add(
                    (
                        start // regex_matrix.count_states,
                        final // regex_matrix.count_states,
                    )
                )

    return result
