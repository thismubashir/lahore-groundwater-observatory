"""Tiny Markdown table helper (avoids an extra dependency)."""

import pandas as pd


def md_table(df: pd.DataFrame, cols=None, max_rows=80, digits=3) -> str:
    df = df if cols is None else df[cols]
    df = df.head(max_rows).copy()
    if df.empty:
        return "_none_\n"
    for c in df.select_dtypes("number"):
        df[c] = df[c].round(digits)

    def fmt(v):
        s = str(v)
        return "" if s in ("nan", "None") else s.replace("|", "/").replace("\n", " ")

    head = "| " + " | ".join(map(str, df.columns)) + " |\n"
    sep = "|" + "---|" * len(df.columns) + "\n"
    body = "".join("| " + " | ".join(fmt(v) for v in r) + " |\n" for r in df.itertuples(index=False))
    return head + sep + body
