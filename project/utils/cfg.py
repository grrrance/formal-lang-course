from pyformlang.cfg import CFG


def read_cfg_from_file(path: str) -> CFG:
    """
    Reads context-free grammar from a file

    Parameters
    ----------
    path: str
    Path to the file

    Returns
    -------
    cfg: CFG
    Context-free grammar
    """
    with open(path) as file:
        return CFG.from_text(file.read())


def create_wcnf_from_cfg(cfg: CFG) -> CFG:
    """
    Converts a context-free grammar into Chomskyan weakened normal form

    Parameters
    ----------
    cfg: CFG
    Context-free grammar

    Returns
    -------
    cfg: CFG
    Chomskyan weakened normal form
    """
    cfg = cfg.eliminate_unit_productions().remove_useless_symbols()
    productions = cfg._decompose_productions(
        cfg._get_productions_with_only_single_terminals()
    )
    return CFG(start_symbol=cfg.start_symbol, productions=set(productions))
