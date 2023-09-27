from project.utils.matrix_automata import MatrixAutomata
from project.utils.graph_lib import get_graph_by_name
from project.utils.automata_lib import create_nfa
from pyformlang.finite_automaton import State, NondeterministicFiniteAutomaton


class TestsForMatrixAutomata:
    start_states = {State(0), State(1)}
    final_states = {State(2), State(3)}
    nfa = create_nfa(get_graph_by_name("travel"), start_states, final_states)

    def test_create_matrix_from_fa(self):
        matrix = MatrixAutomata.create_matrix_from_fa(self.nfa)
        assert matrix.start_states == self.start_states
        assert matrix.final_states == self.final_states
        assert matrix.count_states == len(self.nfa.states)
        assert len(matrix.adjacency_matrices) == len(self.nfa.symbols)

    def test_create_fa_from_matrix(self):
        matrix = MatrixAutomata.create_matrix_from_fa(self.nfa)
        nfa = matrix.create_fa_from_matrix()
        assert nfa.is_equivalent_to(self.nfa)

    def test_intersect(self):
        nfa_1 = NondeterministicFiniteAutomaton()
        nfa_1.add_start_state(State(0))
        nfa_1.add_start_state(State(1))
        nfa_1.add_final_state(State(2))
        nfa_1.add_final_state(State(1))
        nfa_1.add_transitions(
            [
                (0, "meow", 1),
                (0, "nya", 2),
            ]
        )

        nfa_2 = NondeterministicFiniteAutomaton()
        nfa_2.add_start_state(State(0))
        nfa_2.add_start_state(State(2))
        nfa_2.add_final_state(State(0))
        nfa_2.add_final_state(State(1))
        nfa_2.add_transitions(
            [
                (0, "nya", 1),
                (0, "not meow", 2),
            ]
        )

        matrix = MatrixAutomata.create_matrix_from_fa(nfa_1).intersect(
            MatrixAutomata.create_matrix_from_fa(nfa_2)
        )
        assert matrix.adjacency_matrices.keys() == {"nya"}
        assert matrix.count_states == 9
        assert len(matrix.start_states) == 4
        assert len(matrix.final_states) == 4

    def test_transitive_closure(self):
        nfa = NondeterministicFiniteAutomaton()
        nfa.add_transitions(
            [
                (0, "meow", 1),
                (0, "meow", 2),
            ]
        )
        matrix = MatrixAutomata.create_matrix_from_fa(nfa)
        tc = matrix.transitive_closure()
        assert tc.sum() == 2
