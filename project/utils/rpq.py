from networkx import MultiDiGraph
from pyformlang.regular_expression import Regex
from typing import Set
from project.utils.matrix_automata import MatrixAutomata
from project.utils.automata_lib import create_nfa, create_minimum_dfa


def tensors_rpq(
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
    inter_states = {i: state for state, i in intersected_matrix.indexes.items()}
    for source_i, target_i in zip(*tc.nonzero()):
        state_source, state_target = inter_states[source_i], inter_states[target_i]
        if (
            state_source in intersected_matrix.start_states
            and state_target in intersected_matrix.final_states
        ):
            result.add((state_source[0].value, state_target[0].value))

    return result


def bfs_rpq(
    regex: Regex,
    graph: MultiDiGraph,
    start_states: Set[int] = None,
    final_states: Set[int] = None,
    is_include_all: bool = True,
):
    """
    Computes regular reachability queries on graphs with regular constraints for multiple starting states.
    Computes a set of reachable states for a set of start states if is_include_all is True, or a dictionary,
    where the key is the start state and the value is the set of reachable states for it otherwise

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

    is_include_all: bool
    Flag indicating reachability from all starting states simultaneously

    Returns
    -------
    result: set | dict
    A set of reachable states for a set of start states or a dictionary,
    where the key is the start state and the value is the set of reachable states for it
    """
    graph_matrix = MatrixAutomata.create_matrix_from_fa(
        create_nfa(graph, start_states, final_states)
    )
    regex_matrix = MatrixAutomata.create_matrix_from_fa(create_minimum_dfa(regex))

    return graph_matrix.sync_bfs(regex_matrix, is_include_all)
