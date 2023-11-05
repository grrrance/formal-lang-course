from typing import Dict
from pyformlang.regular_expression import Regex

from pyformlang.cfg import Variable, CFG, Epsilon
from dataclasses import dataclass


@dataclass
class ECFG:
    """
    Representation of Extended Context-Free Grammars

    Attributes
    ----------
    start_symbol: Variable
    Start non-terminal symbol

    productions: Dict[Variable, Regex]
    A dictionary of productions, where the keys are non-terminals and the values are regular expressions
    """

    start_symbol: Variable
    productions: Dict[Variable, Regex]

    @staticmethod
    def from_text(text: str, start_symbol: Variable = Variable("S")) -> "ECFG":
        """
        Creates ECFG from a text string

        Parameters
        ----------
        text: str
        Description of the grammar in the string

        start_symbol: Variable = Variable("S")
        Start non-terminal symbol

        Returns
        -------
        ecfg: "ECFG"
        Representation of Extended Context-Free Grammars

        Raises
        ----------
        SyntaxError
        If an incorrect rule was read

        ValueError
        If more than one rule was read for a particular non-terminal
        """
        prods = {}
        rules = [rule for rule in text.split("\n") if rule != ""]
        for rule in rules:
            separated_rule = rule.split("->")
            if len(separated_rule) != 2:
                raise SyntaxError("expected rule separated by ->")

            separated_rule[0] = separated_rule[0].strip()
            separated_rule[1] = separated_rule[1].strip()

            if separated_rule[0] in prods:
                raise ValueError("there is more than one rule for a given non-terminal")

            prods[Variable(separated_rule[0])] = Regex(separated_rule[1])

        return ECFG(start_symbol, prods)

    @staticmethod
    def from_file(path: str, start_symbol: Variable = Variable("S")) -> "ECFG":
        """
        Creates ECFG from a file

        Parameters
        ----------
        path: str
        String path to the file

        start_symbol: Variable = Variable("S")
        Start non-terminal symbol

        Returns
        -------
        ecfg: "ECFG"
        Representation of Extended Context-Free Grammars

        Raises
        ----------
        SyntaxError
        If an incorrect rule was read

        ValueError
        If more than one rule was read for a particular non-terminal
        """
        with open(path) as file:
            return ECFG.from_text(file.read(), start_symbol)

    @staticmethod
    def from_cfg(cfg: CFG) -> "ECFG":
        """
        Creates ECFG from a CFG

        Parameters
        ----------
        cfg: CFG
        Context-free grammar

        Returns
        -------
        ecfg: "ECFG"
        Representation of Extended Context-Free Grammars
        """
        prods = {}
        for prod in cfg.productions:
            regex = Regex(
                " ".join(
                    "$" if isinstance(var, Epsilon) else var.value for var in prod.body
                )
                if len(prod.body) > 0
                else "$"
            )
            prods[prod.head] = (
                regex if prod.head not in prods else prods[prod.head].union(regex)
            )

        return ECFG(
            cfg.start_symbol if cfg.start_symbol is not None else Variable("S"), prods
        )
