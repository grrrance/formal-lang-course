from dataclasses import dataclass
from typing import Dict

from pyformlang.finite_automaton import DeterministicFiniteAutomaton

from project.utils.ecfg import ECFG
from pyformlang.cfg import Variable


@dataclass
class RSM:
    start_symbol: Variable
    boxes: Dict[Variable, DeterministicFiniteAutomaton]

    @staticmethod
    def from_ecfg(ecfg: "ECFG") -> "RSM":
        return RSM(
            ecfg.start_symbol,
            {
                var: regex.to_epsilon_nfa().to_deterministic()
                for var, regex in ecfg.productions.items()
            },
        )

    def minimize(self) -> "RSM":
        return RSM(
            self.start_symbol,
            {var: automata.minimize() for var, automata in self.boxes.items()},
        )
