from typing import Set

from networkx import MultiDiGraph
from pyformlang.regular_expression import Regex
from pyformlang.finite_automaton import DeterministicFiniteAutomaton
from pyformlang.finite_automaton import EpsilonNFA, State, Symbol, Epsilon


def create_minimum_dfa(regex: Regex) -> DeterministicFiniteAutomaton:
    """
    Constructs a minimal DFA given a regular expression

    Parameters
    ----------
    regex: Regex
    Regular expression

    Returns
    -------
    dfa: DeterministicFiniteAutomaton
    Deterministic automata
    """
    return regex.to_epsilon_nfa().to_deterministic().minimize()


def create_nfa(
    graph: MultiDiGraph, start_states: Set[int] = None, final_states: Set[int] = None
) -> EpsilonNFA:
    """
    Constructs a non-deterministic finite automaton from a graph. It is possible to specify starting and final
    vertices. If they are not specified, then all vertices are considered to be starting and final.

    Parameters
    ----------
    graph: MultiDiGraph
    A directed graph class from dataset that can store multi edges

    start_states: Set[int]
    Set of start states

    final_states: Set[int]
    Set of final states

    Returns
    -------
    nfa: EpsilonNFA
    A non-deterministic automaton where epsilon transitions are allowed
    """
    states = set(state for state in graph.nodes)
    nfa = EpsilonNFA()
    start_states = states if start_states is None else start_states
    final_states = states if final_states is None else final_states

    for state in start_states:
        nfa.add_start_state(State(state))

    for state in final_states:
        nfa.add_final_state(State(state))

    for source_node, target_node, attributes_of_edge in graph.edges(data=True):
        nfa.add_transition(
            State(source_node),
            Epsilon()
            if "label" not in attributes_of_edge
            else Symbol(attributes_of_edge["label"]),
            State(target_node),
        )

    return nfa
