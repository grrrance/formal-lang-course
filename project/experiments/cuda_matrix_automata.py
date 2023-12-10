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
            return pycubool.Matrix.empty((0, 0))

        n = list(self.adjacency_matrices.values())[0].shape
        tc = pycubool.Matrix.empty((n, n))
        for mat in self.adjacency_matrices.values():
            tc.ewiseadd(mat, out=tc)
        prev_nnz = tc.nvals
        curr_nnz = 0

        while prev_nnz != curr_nnz:
            tc.mxm(tc, out=tc, accumulate=True)
            prev_nnz, curr_nnz = curr_nnz, tc.nvals

        return tc

    def sync_bfs(
        self, other_matrix: "CuBoolMatrixAutomata", is_include_all: bool
    ) -> set | dict:
        """
        Computes a set of reachable states for a set of start states if is_include_all is True, or a dictionary,
        where the key is the start state and the value is the set of reachable states for it otherwise

        Parameters
        ----------
        other_matrix: CuBoolMatrixAutomata
        Other pycubool matrix

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
        visited = pycubool.Matrix.empty(front.shape)

        while True:
            prev_visited_nnz = visited.nvals
            for matrix in directed_sum.adjacency_matrices.values():
                if front is None:
                    produced_front = visited.mxm(matrix)
                else:
                    produced_front = front.mxm(matrix)

                visited = visited.ewiseadd(
                    CuBoolMatrixAutomata._normalize_front(other_matrix, produced_front)
                )

            front = None

            if visited.nvals == prev_visited_nnz:
                break

        for i, j in visited.to_list():
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

    def _direct_sum(
        self, other_matrix: "CuBoolMatrixAutomata"
    ) -> "CuBoolMatrixAutomata":
        """
        Computes a block diagonal matrix to synchronize a breadth-first traversal across two adjacency matrices

        Parameters
        ----------
        other_matrix: CuBoolMatrixAutomata
        Other pycubool matrix

        Returns
        -------
        direct_sum: CuBoolMatrixAutomata
        A block diagonal matrix
        """
        directed_sum = CuBoolMatrixAutomata()
        directed_sum.start_states = self.start_states | other_matrix.start_states
        directed_sum.final_states = self.final_states | other_matrix.final_states
        directed_sum.count_states = self.count_states + other_matrix.count_states
        for label in (
            self.adjacency_matrices.keys() & other_matrix.adjacency_matrices.keys()
        ):
            directed_mat = pycubool.Matrix.empty(
                (directed_sum.count_states, directed_sum.count_states)
            )
            for i, j in self.adjacency_matrices[label]:
                directed_mat[i, j] = True

            for i, j in other_matrix.adjacency_matrices[label]:
                directed_mat[self.count_states + i, self.count_states + j] = True
            directed_sum.adjacency_matrices[label] = directed_mat

        return directed_sum

    def _init_front(
        self,
        other_matrix: "CuBoolMatrixAutomata",
        is_include_all: bool,
        start_states: Dict[int, State],
    ) -> pycubool.Matrix:
        """
        Computes the initial front to bypass bfs

        Parameters
        ----------
        other_matrix: CuBoolMatrixAutomata
        Other pycubool matrix

        is_include_all: bool
        Flag indicating reachability from all starting states simultaneously

        start_states: Dict[int, State]
        Dictionary, where the key is the index and the value is the starting state

        Returns
        -------
        front: CuBoolMatrixAutomata
        Initial front to bypass bfs
        """

        def create_front_with_right_part(start_indexes: List[int]) -> pycubool.Matrix:
            front = pycubool.Matrix.empty(
                (
                    other_matrix.count_states,
                    other_matrix.count_states + self.count_states,
                )
            )

            for state, i in other_matrix.indexes.items():
                front[i, i] = True
                if state in other_matrix.start_states:
                    for j in start_indexes:
                        front[i, other_matrix.count_states + j] = True
            return front

        def create_fronts() -> List[pycubool.Matrix]:
            states = {i: state for state, i in self.indexes.items()}
            if is_include_all:
                return [
                    create_front_with_right_part(
                        [
                            i
                            for i in range(len(self.indexes))
                            if states[i] in self.start_states
                        ]
                    )
                ]
            else:
                return [
                    create_front_with_right_part([start_i])
                    for start_i in range(len(start_states))
                ]

        fronts = create_fronts()

        if len(fronts) > 0:
            res = pycubool.Matrix.empty(
                (
                    len(fronts) * other_matrix.count_states,
                    other_matrix.count_states + self.count_states,
                )
            )
            vstack_helper = pycubool.Matrix.empty((len(fronts), 1))
            for i, front in enumerate(fronts):
                vstack_helper.build(rows={i}, cols={0})
                res = res.ewiseadd(vstack_helper.kronecker(front))
            return res
        else:
            return pycubool.Matrix.empty(
                (
                    other_matrix.count_states,
                    other_matrix.count_states + self.count_states,
                )
            )

    @staticmethod
    def _normalize_front(
        other_matrix: "CuBoolMatrixAutomata", produced_front: pycubool.Matrix
    ) -> pycubool.Matrix:
        """
        Computes the normalized front by produced one

        Parameters
        ----------
        other_matrix: CuBoolMatrixAutomata
        Other pycubool matrix

        produced_front: CuBoolMatrixAutomata
        Produced front by bypass

        Returns
        -------
        front: CuBoolMatrixAutomata
        Normalized front
        """
        normalized_front = pycubool.Matrix.empty(produced_front.shape)

        for i, j in produced_front.to_list():
            if j < other_matrix.count_states:
                right_part = produced_front[i : i + 1, other_matrix.count_states :]

                if right_part.nvals > 0:
                    shift_row = (
                        i // other_matrix.count_states * other_matrix.count_states
                    )
                    normalized_front[shift_row + j, j] = True
                    for _, r_j in right_part:
                        normalized_front[
                            shift_row + j, other_matrix.count_states + r_j
                        ] += True

        return normalized_front
