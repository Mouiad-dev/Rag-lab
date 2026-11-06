"""Offline retrieval fixture; all non-ASCII text is escaped in source."""

FACT_EN = "The tidal observatory at Port Sallow records sea levels every 17 minutes."
FACT_AR = (
    "\u062a\u062d\u0641\u0638 \u0645\u0643\u062a\u0628\u0629 \u0627\u0644\u0648\u0627\u062d\u0629 "
    "\u062b\u0644\u0627\u062b\u0629 \u0622\u0644\u0627\u0641 \u0648\u0645\u0626\u062a\u064a "
    "\u0645\u062e\u0637\u0648\u0637\u0629 \u0641\u064a \u0642\u0628\u0648\u0647\u0627 "
    "\u0627\u0644\u0634\u0645\u0627\u0644\u064a."
)
DISTRACTORS = (
    "A bakery on Mill Lane bakes rye bread before dawn.",
    "The mountain railway climbs nine hundred metres in an hour.",
    "Members of the chess club meet on Tuesday evenings.",
    "Volunteers repaint the wooden bridge every spring.",
    "A choir rehearses in the old library hall.",
    "Farmers bring apples and honey to the Saturday market.",
    "The cycling team trains on the coastal road.",
    "Children fly kites in the park when the wind is strong.",
    "\u064a\u0632\u0631\u0639 \u0627\u0644\u0645\u0632\u0627\u0631\u0639\u0648\u0646 \u0627\u0644\u062a\u0645\u0631 "
    "\u0641\u064a \u0627\u0644\u062c\u0646\u0648\u0628 \u0643\u0644 \u0639\u0627\u0645.",
    "\u064a\u0641\u062a\u062d \u0627\u0644\u0633\u0648\u0642 \u0627\u0644\u0642\u062f\u064a\u0645 "
    "\u0623\u0628\u0648\u0627\u0628\u0647 \u0641\u064a \u0627\u0644\u0635\u0628\u0627\u062d "
    "\u0627\u0644\u0628\u0627\u0643\u0631.",
    "\u064a\u062a\u062f\u0631\u0628 \u0627\u0644\u0641\u0631\u064a\u0642 \u0639\u0644\u0649 "
    "\u0643\u0631\u0629 \u0627\u0644\u0642\u062f\u0645 \u0645\u0633\u0627\u0621 \u0643\u0644 \u062e\u0645\u064a\u0633.",
    "\u062a\u0645\u0631 \u0627\u0644\u0642\u0637\u0627\u0631\u0627\u062a \u0627\u0644\u0633\u0631\u064a\u0639\u0629 "
    "\u0639\u0628\u0631 \u0627\u0644\u0648\u0627\u062f\u064a \u0645\u0631\u062a\u064a\u0646 "
    "\u064a\u0648\u0645\u064a\u0627.",
    "\u064a\u0637\u0628\u062e \u0627\u0644\u062c\u064a\u0631\u0627\u0646 \u0627\u0644\u0623\u0631\u0632 "
    "\u0645\u0639 \u0627\u0644\u0639\u062f\u0633 \u0641\u064a \u0627\u0644\u0634\u062a\u0627\u0621.",
    "\u062a\u063a\u0646\u064a \u0627\u0644\u062c\u0648\u0642\u0629 \u0641\u064a \u0627\u0644\u0633\u0627\u062d\u0629 "
    "\u062e\u0644\u0627\u0644 \u0639\u0637\u0644\u0629 \u0627\u0644\u0631\u0628\u064a\u0639.",
)
MUSEUM = (
    "A maritime museum preserves maps and journals from many generations. "
    + "Visitors study harbour charts in quiet rooms. " * 6
    + "Curators catalogue donated ship logs each winter. " * 6
)
MARKDOWN = "# Foundry field notes\n\n" + "\n\n".join((FACT_EN, FACT_AR, *DISTRACTORS, MUSEUM))
QUERY_EN = "How often does the Port Sallow observatory record sea levels?"
QUERY_AR = (
    "\u0643\u0645 \u0645\u062e\u0637\u0648\u0637\u0629 \u062a\u062d\u0641\u0638 "
    "\u0645\u0643\u062a\u0628\u0629 \u0627\u0644\u0648\u0627\u062d\u0629\u061f"
)
TEST_CHUNK_SIZE = 120
TEST_OVERLAP = 20
