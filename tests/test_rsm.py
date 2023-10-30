from typing import Dict
from pyformlang.finite_automaton import DeterministicFiniteAutomaton
from tests.test_cases.tests_utils import get_path_to_test_case
from project.utils.ecfg import ECFG
from pyformlang.cfg import Variable
from pyformlang.regular_expression import Regex
from project.utils.rsm import RSM


class TestsForRSM:
    prods_ecfg = {
        Variable("S"): Regex("(a|(((b)*.S)|Z))"),
        Variable("Z"): Regex("(z|a)"),
    }

    @staticmethod
    def check_rsm_eq_prods(
        actual: Dict[Variable, DeterministicFiniteAutomaton],
        expected: Dict[Variable, Regex],
    ):
        for k, v in actual.items():
            assert k in expected
            expected_dfa = expected[k].to_epsilon_nfa().to_deterministic()
            assert v.is_equivalent_to(expected_dfa)

    def test_create_from_ecfg(self):
        ecfg = ECFG.from_file(get_path_to_test_case("ecfg_test.txt"))
        rsm = RSM.from_ecfg(ecfg)
        assert rsm.start_symbol == Variable("S")
        TestsForRSM.check_rsm_eq_prods(rsm.boxes, self.prods_ecfg)
