from dataclasses import dataclass
from typing import Dict

from pyformlang.finite_automaton import DeterministicFiniteAutomaton

from project.utils.ecfg import ECFG
from pyformlang.cfg import Variable


@dataclass
class RSM:
    """
    Representation of recursively finite state machine

    Attributes
    ----------
    start_symbol: Variable
    Start non-terminal symbol

    boxes: Dict[Variable, Regex] A dictionary of productions, where the keys are non-terminals and the values are
    deterministic finite state machines
    """

    start_symbol: Variable
    boxes: Dict[Variable, DeterministicFiniteAutomaton]

    @staticmethod
    def from_ecfg(ecfg: "ECFG") -> "RSM":
        """
        Creates RSM from a ECFG

        Parameters
        ----------
        ecfg: ECFG
        Representation of Extended Context-Free Grammars

        Returns
        -------
        RSM: "RSM"
        Representation of Extended Context-Free Grammars
        """
        return RSM(
            ecfg.start_symbol,
            {
                var: regex.to_epsilon_nfa().to_deterministic()
                for var, regex in ecfg.productions.items()
            },
        )

    def minimize(self) -> "RSM":
        """
        Minimizes RSM

        Returns
        -------
        RSM: "RSM"
        Representation of Extended Context-Free Grammars
        """
        return RSM(
            self.start_symbol,
            {var: automata.minimize() for var, automata in self.boxes.items()},
        )
