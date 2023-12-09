from collections import defaultdict
from typing import List, Dict

from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State
from scipy import sparse
from scipy.sparse import (
    dok_matrix,
    kron,
    block_diag,
    lil_matrix,
    lil_array,
    csr_matrix,
    vstack,
)

from project.utils.rsm import RSM


class MatrixAutomata:
    """
    Representation of FA as a Boolean Matrix

    Attributes
    ----------
    start_states: set
    Start states of FA

    final_states: set
    Final states of FA

    adjacency_matrices: dict
    Dictionary of boolean matrices.
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
    ) -> "MatrixAutomata":
        """
        Transforms FA into MatrixAutomata

        Parameters
        ----------
        nfa: NondeterministicFiniteAutomaton
        NFA to transform

        Returns
        -------
        obj: MatrixAutomata
        MatrixAutomata object from NFA
        """
        obj = cls()
        obj.start_states = nfa.start_states
        obj.final_states = nfa.final_states
        matrices = {}
        obj.count_states = len(nfa.states)
        obj.indexes = {state: i for i, state in enumerate(nfa.states)}
        nfa_transitions = nfa.to_dict()

        for symbol in nfa.symbols:
            matrices[symbol] = dok_matrix(
                (obj.count_states, obj.count_states), dtype=bool
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
    def create_matrix_from_rsm(cls, rsm: RSM) -> "MatrixAutomata":
        """
        Transforms RSM into MatrixAutomata

        Parameters
        ----------
        rsm: RSM
        Recursive state machine to transform

        Returns
        -------
        obj: MatrixAutomata
        Representation of a MatrixAutomata class object from NFA
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
            lambda: sparse.dok_matrix((obj.count_states, obj.count_states), dtype=bool)
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

    def create_fa_from_matrix(self) -> NondeterministicFiniteAutomaton:
        """
        Transforms MatrixAutomata into NFA

        Returns
        -------
        nfa: NondeterministicFiniteAutomaton
        A non-deterministic automaton
        """
        nfa = NondeterministicFiniteAutomaton()

        states = {i: state for state, i in self.indexes.items()}

        for state in self.start_states:
            nfa.add_start_state(state)

        for state in self.final_states:
            nfa.add_final_state(state)

        for symbol in self.adjacency_matrices.keys():
            matrix = self.adjacency_matrices[symbol].toarray()
            for i in range(self.count_states):
                for j in range(self.count_states):
                    if matrix[i][j]:
                        nfa.add_transition(
                            states[i],
                            symbol,
                            states[j],
                        )

        return nfa

    def intersect(self, other_matrix: "MatrixAutomata") -> "MatrixAutomata":
        """
        Computes intersection of self boolean matrix with other

        Parameters
        ----------
        other_matrix: MatrixAutomata
        Other boolean matrix

        Returns
        -------
        intersection: MatrixAutomata
        Intersection of two boolean matrices
        """
        intersection = MatrixAutomata()
        symbols = (
            self.adjacency_matrices.keys() & other_matrix.adjacency_matrices.keys()
        )

        intersection.count_states = self.count_states * other_matrix.count_states
        for symbol in symbols:
            intersection.adjacency_matrices[symbol] = kron(
                self.adjacency_matrices[symbol],
                other_matrix.adjacency_matrices[symbol],
                format="dok",
            )

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

    def transitive_closure(self) -> dok_matrix:
        """
        Computes transitive closure of boolean matrices

        Returns
        -------
        tc: dok_matrix
        Transitive closure of boolean matrices
        """
        if len(self.adjacency_matrices) == 0:
            return dok_matrix((0, 0), dtype=bool)

        tc = sum(
            self.adjacency_matrices.values(),
            start=dok_matrix((self.count_states, self.count_states), dtype=bool),
        )
        prev_nnz = tc.nnz
        curr_nnz = 0

        while prev_nnz != curr_nnz:
            tc += tc @ tc
            prev_nnz, curr_nnz = curr_nnz, tc.nnz

        return tc

    def sync_bfs(
        self, other_matrix: "MatrixAutomata", is_include_all: bool
    ) -> set | dict:
        """
        Computes a set of reachable states for a set of start states if is_include_all is True, or a dictionary,
        where the key is the start state and the value is the set of reachable states for it otherwise

        Parameters
        ----------
        other_matrix: MatrixAutomata
        Other boolean matrix

        is_include_all: bool
        Flag indicating reachability from all starting states simultaneously

        Returns
        -------
        result: set | dict
        A set of reachable states for a set of start states or a dictionary,
        where the key is the start state and the value is the set of reachable states for it
        """
        result = set() if is_include_all else {}
        directed_sum = self._direct_sum(other_matrix)

        self_states = {i: state for state, i in self.indexes.items()}
        other_states = {i: state for state, i in other_matrix.indexes.items()}
        start_states = {i: state for i, state in enumerate(self.start_states)}

        front = self._init_front(other_matrix, is_include_all, start_states)
        visited = front.copy()

        while True:
            prev_visited_nnz = visited.nnz
            for matrix in directed_sum.adjacency_matrices.values():
                produced_front = visited @ matrix
                visited += MatrixAutomata._normalize_front(other_matrix, produced_front)

            if visited.nnz == prev_visited_nnz:
                break

        for i, j in zip(*visited.nonzero()):
            if (
                j >= other_matrix.count_states
                and other_states[i % other_matrix.count_states]
                in other_matrix.final_states
            ):
                if self_states[j - other_matrix.count_states] in self.final_states:
                    if is_include_all:
                        result.add(self_states[j - other_matrix.count_states].value)
                    else:
                        if start_states[i // self.count_states].value in result:
                            result[start_states[i // self.count_states].value].add(
                                self_states[j - other_matrix.count_states].value
                            )
                        else:
                            result[start_states[i // self.count_states].value] = {
                                self_states[j - other_matrix.count_states].value
                            }

        return result

    def _direct_sum(self, other_matrix: "MatrixAutomata") -> "MatrixAutomata":
        """
        Computes a block diagonal matrix to synchronize a breadth-first traversal across two adjacency matrices

        Parameters
        ----------
        other_matrix: MatrixAutomata
        Other boolean matrix

        Returns
        -------
        direct_sum: MatrixAutomata
        A block diagonal matrix
        """
        directed_sum = MatrixAutomata()
        directed_sum.start_states = self.start_states | other_matrix.start_states
        directed_sum.final_states = self.final_states | other_matrix.final_states
        directed_sum.count_states = self.count_states + other_matrix.count_states
        directed_sum.adjacency_matrices = {
            label: block_diag(
                (
                    other_matrix.adjacency_matrices[label],
                    self.adjacency_matrices[label],
                ),
                format="csr",
            )
            for label in self.adjacency_matrices.keys()
            & other_matrix.adjacency_matrices.keys()
        }

        directed_sum.indexes = {
            **self.indexes,
            **{
                state: len(self.indexes) + i
                for state, i in other_matrix.indexes.items()
            },
        }

        return directed_sum

    def _init_front(
        self,
        other_matrix: "MatrixAutomata",
        is_include_all: bool,
        start_states: Dict[int, State],
    ) -> csr_matrix:
        """
        Computes the initial front to bypass bfs

        Parameters
        ----------
        other_matrix: MatrixAutomata
        Other boolean matrix

        is_include_all: bool
        Flag indicating reachability from all starting states simultaneously

        start_states: Dict[int, State]
        Dictionary, where the key is the index and the value is the starting state

        Returns
        -------
        front: csr_matrix
        Initial front to bypass bfs
        """

        def create_front_with_right_part(row: lil_array) -> lil_matrix:
            front = lil_matrix(
                (
                    other_matrix.count_states,
                    other_matrix.count_states + self.count_states,
                )
            )
            for state, i in other_matrix.indexes.items():
                front[i, i] = True
                if state in other_matrix.start_states:
                    front[i, other_matrix.count_states :] = row
            return front

        def create_fronts() -> List[lil_matrix]:
            states = {i: state for state, i in self.indexes.items()}
            if is_include_all:
                return [
                    create_front_with_right_part(
                        lil_array(
                            [
                                states[i] in self.start_states
                                for i in range(len(self.indexes))
                            ]
                        )
                    )
                ]
            else:
                return [
                    create_front_with_right_part(
                        lil_array(
                            [
                                states[i] == start_states[start_i]
                                for i in range(len(self.indexes))
                            ]
                        )
                    )
                    for start_i in range(len(start_states))
                ]

        fronts = create_fronts()

        if len(fronts) > 0:
            return csr_matrix(vstack(fronts))
        else:
            return csr_matrix(
                (
                    other_matrix.count_states,
                    other_matrix.count_states + self.count_states,
                )
            )

    @staticmethod
    def _normalize_front(
        other_matrix: "MatrixAutomata", produced_front: csr_matrix
    ) -> csr_matrix:
        """
        Computes the normalized front by produced one

        Parameters
        ----------
        other_matrix: MatrixAutomata
        Other boolean matrix

        produced_front: csr_matrix
        Produced front by bypass

        Returns
        -------
        front: csr_matrix
        Normalized front
        """
        normalized_front = lil_array(produced_front.shape)

        for i, j in zip(*produced_front.nonzero()):
            if j < other_matrix.count_states:
                row = produced_front.getrow(i).tolil()[[0], other_matrix.count_states :]

                if row.nnz > 0:
                    shift_row = (
                        i // other_matrix.count_states * other_matrix.count_states
                    )
                    normalized_front[shift_row + j, j] = True
                    normalized_front[
                        [shift_row + j], other_matrix.count_states :
                    ] += row

        return normalized_front.tocsr()
