from typing import Dict

from project.utils.cfg import read_cfg_from_file
from tests.test_cases.tests_utils import get_path_to_test_case
from project.utils.ecfg import ECFG
from pyformlang.cfg import Variable
from pyformlang.regular_expression import Regex
from project.utils.automata_lib import create_minimum_dfa


class TestsForECFGCreation:
    prods_cfg = {
        Variable("S"): Regex("((B.(C.(S.D)))|s)"),
        Variable("B"): Regex("b"),
        Variable("C"): Regex("c"),
        Variable("D"): Regex("$"),
        Variable("A"): Regex("a"),
    }

    prods_ecfg = {
        Variable("S"): Regex("(a|(((b)*.S)|Z))"),
        Variable("Z"): Regex("(z|a)"),
    }

    @staticmethod
    def check_ecfg_eq_prods(
        actual: Dict[Variable, Regex], expected: Dict[Variable, Regex]
    ):
        for k, v in actual.items():
            actual_dfa = create_minimum_dfa(v)
            assert k in expected
            expected_dfa = create_minimum_dfa(expected[k])
            assert actual_dfa.is_equivalent_to(expected_dfa)

    def test_create_from_cfg(self):
        cfg = read_cfg_from_file(get_path_to_test_case("cfg_test.txt"))
        ecfg = ECFG.from_cfg(cfg)
        assert ecfg.start_symbol == Variable("S")
        TestsForECFGCreation.check_ecfg_eq_prods(ecfg.productions, self.prods_cfg)

    def test_create_from_file(self):
        ecfg = ECFG.from_file(get_path_to_test_case("ecfg_test.txt"))
        assert ecfg.start_symbol == Variable("S")
        TestsForECFGCreation.check_ecfg_eq_prods(ecfg.productions, self.prods_ecfg)
