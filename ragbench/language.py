"""A small, transparent guess whether a short text is German or English.

The prompt-injection data has no language column. This counts common
function words of each language (and umlauts or ß for German). It is a
heuristic for splitting the results, not a language detector; a text with
no clear signal is "unknown".
"""

import re

# Words that are common in both languages ("was", "an", "also", "die") are left out.
GERMAN = set(
    "der das und ist nicht ich du sie wir ihr ein eine einen einem einer zu mit auf für von den dem des "
    "sich es im sind wie wer warum wo bitte vergiss vergessen alle alles oder aber auch nur noch "
    "kann können soll sollst mir mich dich dir uns euch jetzt dann wenn als über unter nach bei aus".split()
)
ENGLISH = set(
    "the and is are were not you your we they a to with on for of from this that these those it "
    "what who why where how please forget all or but only can could should would me my now then "
    "if as about under after at out ignore".split()
)
WORD = re.compile(r"[a-zäöüß]+")


def guess_language(text: str) -> str:
    words = WORD.findall(text.lower())
    german = sum(w in GERMAN for w in words) + 2 * len(re.findall(r"[äöüß]", text.lower()))
    english = sum(w in ENGLISH for w in words)
    if german == english:
        return "unknown"
    return "de" if german > english else "en"
