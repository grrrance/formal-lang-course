from networkx import MultiDiGraph
from pyformlang.regular_expression import Regex
from typing import Set
from project.experiments.cuda_matrix_automata import CuBoolMatrixAutomata
from project.utils.automata_lib import create_nfa, create_minimum_dfa


def cubool_tensors_rpq(
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
    graph_matrix = CuBoolMatrixAutomata.create_matrix_from_fa(
        create_nfa(graph, start_states, final_states)
    )
    regex_matrix = CuBoolMatrixAutomata.create_matrix_from_fa(create_minimum_dfa(regex))
    intersected_matrix = graph_matrix.intersect(regex_matrix)
    tc = intersected_matrix.transitive_closure()
    inter_states = {i: state for state, i in intersected_matrix.indexes.items()}
    for source_i, target_i in tc.to_list():
        state_source, state_target = inter_states[source_i], inter_states[target_i]
        if (
            state_source in intersected_matrix.start_states
            and state_target in intersected_matrix.final_states
        ):
            result.add((state_source[0].value, state_target[0].value))

    return result
