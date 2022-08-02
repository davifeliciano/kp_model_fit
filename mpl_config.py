def xtick_label_formatter(value: float, tick_number: int) -> str:
    if value == -1.0:
        return r"$\Gamma$"
    if value == 0.0:
        return r"$K$"
    if value == 1.0:
        return r"$M$"
    return ""


latex_preamble_entries = [
    r"\usepackage[sc]{mathpazo}",
    r"\usepackage{siunitx}",
    r"\usepackage{chemformula}",
]

latex_preamble = " ".join(latex_preamble_entries)
