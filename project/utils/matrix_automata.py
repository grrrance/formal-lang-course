from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State
from scipy.sparse import dok_matrix, kron


class MatrixAutomata:
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

    def create_fa_from_matrix(self) -> NondeterministicFiniteAutomaton:
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
                state = index = i_1 * len(other_matrix.indexes) + i_2
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
