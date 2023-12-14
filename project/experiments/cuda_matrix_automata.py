from collections import defaultdict
from typing import Dict, List
import pycubool
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State
from project.utils.rsm import RSM


class CuBoolMatrixAutomata:
    """
    Representation of FA as a PyCuBool Matrix

    Attributes
    ----------
    start_states: set
    Start states of FA

    final_states: set
    Final states of FA

    adjacency_matrices: dict
    Dictionary of pycubool matrices.
    Keys are FA labels

    indexes: dict
    Dictionary of correspondence of states and indices in matrices.
    Keys are states

    count_states: int
    Number of states
    """

    def __init__(self):
        self.start_states = set()
        self.final_states = set()
        self.adjacency_matrices = {}
        self.indexes = {}
        self.count_states = 0

    @classmethod
    def create_matrix_from_fa(
        cls, nfa: NondeterministicFiniteAutomaton
    ) -> "CuBoolMatrixAutomata":
        """
        Transforms FA into CuBoolMatrixAutomata

        Parameters
        ----------
        nfa: NondeterministicFiniteAutomaton
        NFA to transform

        Returns
        -------
        obj: CuBoolMatrixAutomata
        CuBoolMatrixAutomata object from NFA
        """
        obj = cls()
        obj.start_states = nfa.start_states
        obj.final_states = nfa.final_states
        matrices = {}
        obj.count_states = len(nfa.states)
        obj.indexes = {state: i for i, state in enumerate(nfa.states)}
        nfa_transitions = nfa.to_dict()

        for symbol in nfa.symbols:
            matrices[symbol] = pycubool.Matrix.empty(
                (obj.count_states, obj.count_states)
            )

        for state_source, transition in nfa_transitions.items():
            for symbol, states_target in transition.items():
                if not isinstance(states_target, set):
                    states_target = {states_target}
                for state_target in states_target:
                    matrices[symbol][
                        obj.indexes[state_source], obj.indexes[state_target]
                    ] = True

        obj.adjacency_matrices = matrices

        return obj

    @classmethod
    def create_matrix_from_rsm(cls, rsm: RSM) -> "CuBoolMatrixAutomata":
        """
        Transforms RSM into CuBoolMatrixAutomata

        Parameters
        ----------
        rsm: RSM
        Recursive state machine to transform

        Returns
        -------
        obj: CuBoolMatrixAutomata
        Representation of a CuBoolMatrixAutomata class object from NFA
        """
        obj = cls()
        states, start_states, final_states = set(), set(), set()

        for var, automata in rsm.boxes.items():
            for state in automata.states:
                st = State((var, state.value))
                states.add(st)
                if state in automata.start_states:
                    start_states.add(st)
                if state in automata.final_states:
                    final_states.add(st)

        obj.start_states = start_states
        obj.final_states = final_states
        obj.count_states = len(states)
        indexes = {
            state: i for i, state in enumerate(sorted(states, key=lambda s: s.value[1]))
        }
        obj.indexes = indexes

        matrices = defaultdict(
            lambda: pycubool.Matrix.empty((obj.count_states, obj.count_states))
        )

        for var, automata in rsm.boxes.items():
            for state_source, transition in automata.to_dict().items():
                for symbol, states_target in transition.items():
                    if not isinstance(states_target, set):
                        states_target = {states_target}
                    for state_target in states_target:
                        matrices[symbol.value][
                            indexes[State((var, state_source.value))],
                            indexes[State((var, state_target.value))],
                        ] = True

        obj.adjacency_matrices = matrices

        return obj

    def intersect(self, other_matrix: "CuBoolMatrixAutomata") -> "CuBoolMatrixAutomata":
        """
        Computes intersection of self boolean matrix with other

        Parameters
        ----------
        other_matrix: CuBoolMatrixAutomata
        Other pycubool matrix

        Returns
        -------
        intersection: CuBoolMatrixAutomata
        Intersection of two pycubool matrices
        """
        intersection = CuBoolMatrixAutomata()
        symbols = (
            self.adjacency_matrices.keys() & other_matrix.adjacency_matrices.keys()
        )

        intersection.count_states = self.count_states * other_matrix.count_states
        for symbol in symbols:
            intersection.adjacency_matrices[symbol] = self.adjacency_matrices[
                symbol
            ].kronecker(other_matrix.adjacency_matrices[symbol])

        for state_1, i_1 in self.indexes.items():
            for state_2, i_2 in other_matrix.indexes.items():
                state = (state_1, state_2)
                index = i_1 * len(other_matrix.indexes) + i_2
                intersection.indexes[state] = index
                if (
                    state_1 in self.start_states
                    and state_2 in other_matrix.start_states
                ):
                    intersection.start_states.add(state)
                if (
                    state_1 in self.final_states
                    and state_2 in other_matrix.final_states
                ):
                    intersection.final_states.add(state)

        return intersection

    def transitive_closure(self) -> pycubool.Matrix:
        """
        Computes transitive closure of pycubool matrices

        Returns
        -------
        tc: pycubool.Matrix
        Transitive closure of pycubool matrices
        """
        if len(self.adjacency_matrices) == 0:
            return pycubool.Matrix.empty((1, 1))

        n = list(self.adjacency_matrices.values())[0].shape
        tc = pycubool.Matrix.empty(n)
        for mat in self.adjacency_matrices.values():
            tc.ewiseadd(mat, out=tc)
        prev_nnz = tc.nvals
        curr_nnz = 0

        while prev_nnz != curr_nnz:
            tc.mxm(tc, out=tc, accumulate=True)
            prev_nnz, curr_nnz = curr_nnz, tc.nvals

        return tc