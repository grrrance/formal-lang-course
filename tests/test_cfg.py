import os

import pytest

from project.utils.cfg import read_cfg_from_file
from project.utils.cfg import create_wcnf_from_cfg
from pyformlang.cfg import Variable, Production, Terminal


def get_path_to_test_case():
    directory = os.path.join("tests", "test_cases")
    return os.path.join(directory, "cfg_test.txt")


class TestsForReadingCFG:
    prods = {
        Production(Variable("S"), [Terminal("s")]),
        Production(Variable("C"), [Terminal("c")]),
        Production(Variable("D"), []),
        Production(
            Variable("S"), [Variable("B"), Variable("C"), Variable("S"), Variable("D")]
        ),
        Production(Variable("B"), [Terminal("b")]),
        Production(Variable("A"), [Terminal("a")]),
    }

    def test_for_incorrect_path(self):
        with pytest.raises(FileNotFoundError):
            read_cfg_from_file("somepath")

    def test_for_correct_path(self):
        cfg = read_cfg_from_file(get_path_to_test_case())
        assert cfg.start_symbol == Variable("S")
        for i in self.prods:
            assert i in cfg.productions


class TestsForCreationWCNF:
    def test_weakness_cnf(self):
        cfg = read_cfg_from_file(get_path_to_test_case())
        cfg = create_wcnf_from_cfg(cfg)
        assert not cfg.is_empty()
        assert Variable("A") not in cfg.variables
        assert Terminal("a") not in cfg.terminals
        assert Production(Variable("D"), []) in cfg.productions

        for prod in cfg.productions:
            assert len(prod.body) <= 2
