from project.utils.automata_lib import *
from project.utils.graph_lib import *


class TestsForCreateMinimumDfa:
    def test_empty(self):
        dfa = create_minimum_dfa(Regex(""))
        assert dfa.is_empty()
        assert dfa.is_deterministic()

    def test_concatenation(self):
        def assert_concatenation(dfa: DeterministicFiniteAutomaton):
            assert dfa.symbols == {"a", "b"}
            assert dfa.accepts(["a", "b"])
            assert not dfa.accepts(["a b"])
            assert not dfa.accepts(["ab"])
            assert dfa.is_deterministic()

        assert_concatenation(create_minimum_dfa(Regex("a b")))
        assert_concatenation(create_minimum_dfa(Regex("a.b")))

    def test_union(self):
        def assert_union(dfa: DeterministicFiniteAutomaton):
            assert dfa.symbols == {"abc", "d"}
            assert dfa.accepts(["abc"])
            assert dfa.accepts(["d"])
            assert not dfa.accepts(["abcd"])
            assert not dfa.accepts(["abc", "d"])
            assert dfa.is_deterministic()

        assert_union(create_minimum_dfa(Regex("abc|d")))
        assert_union(create_minimum_dfa(Regex("abc+d")))

    def test_kleene(self):
        dfa = create_minimum_dfa(Regex("a*"))
        assert dfa.accepts(["a"])
        assert dfa.accepts(["a", "a"])
        assert dfa.accepts([])
        assert not dfa.accepts(["aa"])
        assert dfa.is_deterministic()

    def test_epsilon(self):
        def assert_epsilon(dfa: DeterministicFiniteAutomaton):
            assert dfa.accepts([])
            assert not dfa.is_empty()
            assert not dfa.accepts(["a"])
            assert dfa.is_deterministic()

        assert_epsilon(create_minimum_dfa(Regex("$")))
        assert_epsilon(create_minimum_dfa(Regex("epsilon")))


def assert_combinations_states(graph, start_states, final_states, info):
    def assert_nfa_graph(start_states_option, final_states_option):
        nfa = create_nfa(graph, start_states_option, final_states_option)
        len_start_states = (
            info[0] if start_states_option is None else len(start_states_option)
        )
        len_final_states = (
            info[0] if final_states_option is None else len(final_states_option)
        )
        assert info[0] == len(nfa.states)
        assert len_start_states == len(nfa.start_states)
        assert len_final_states == len(nfa.final_states)
        assert info[1] == nfa.symbols
        assert not nfa.is_deterministic()

    argument_states = [{1, 3}, {2, 3}]

    for i in range(4):
        start = start_states if i in argument_states[0] else None
        final = final_states if i in argument_states[1] else None
        assert_nfa_graph(start, final)


class TestsForCreateNfa:
    start_states = {State(0), State(1)}
    final_states = {State(2), State(3)}

    def test_nfa_with_graph_by_name(self):
        graph = get_graph_by_name("travel")
        info = get_info_graph("travel")
        info = info[0], info[2]
        assert_combinations_states(graph, self.start_states, self.final_states, info)

    def test_nfa_with_two_cycle_graph(self):
        first = 2
        second = 2
        labels = ("Some", "None")
        info = first + second + 1, set(labels)
        graph = create_two_cycle_graph(first, second, labels)
        assert_combinations_states(graph, self.start_states, self.final_states, info)
