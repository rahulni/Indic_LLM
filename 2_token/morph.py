"""Phase-C experimental helpers: Dravidian suffix splitting + Brahmic transliteration.

Both are OFF unless enabled in the train_all PROFILE. They are intentionally small
and conservative -- morphology rules are error-prone, so we keep a short curated
suffix list and only split when a comfortable stem remains.
"""
import regex as _re
import unicodedata

# --- Dravidian suffixes (longest first). Approximate, curated by hand. ---------
# Telugu vibhakti (case), plural, postpositions and common derivational endings.
_TE_SUFFIXES = [
    "గురించి", "ద్వారా", "కోసం", "వరకు", "నుండి", "వల్ల", "లోని", "యొక్క",
    "లను", "లకు", "లో", "కి", "కు", "తో", "గా", "పై", "చే", "అనే",
    "ను", "ని", "లు", "ల", "వి", "డు", "మ", "పు",
]
# Kannada case/plural/postpositions.
_KN_SUFFIXES = [
    "ದಲ್ಲಿ", "ಯನ್ನು", "ವನ್ನು", "ಗಾಗಿ", "ಇಂದ", "ಗಳು", "ಗಳ", "ಗೆ", "ಯಲ್ಲಿ",
    "ವರೆಗೆ", "ದ", "ವು", "ನ", "ಯ", "ಕ್ಕೆ",
]
_SUFFIXES = sorted(set(_TE_SUFFIXES + _KN_SUFFIXES), key=len, reverse=True)

_MIN_STEM = 2  # keep at least this many grapheme clusters in the stem


def _clusters(s):
    return _re.findall(r"\X", s)


def split_suffixes(word: str):
    """Strip at most one recognised suffix; return [stem, suffix] or [word]."""
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) > len(suf):
            stem = word[: -len(suf)]
            if len(_clusters(stem)) >= _MIN_STEM:
                return [stem, suf]
    return [word]


# --- Brahmic -> shared script (Devanagari pivot) so scripts share BPE tokens ---
try:
    from indic_transliteration import sanscript
    _SANS_OK = True
except Exception:  # noqa: BLE001
    _SANS_OK = False

# Unicode block starts for quick script detection
_BLOCKS = [
    (0x0C00, 0x0C7F, "TELUGU"),
    (0x0C80, 0x0CFF, "KANNADA"),
    (0x0900, 0x097F, "DEVANAGARI"),
]


def _detect_scheme(word: str):
    for ch in word:
        cp = ord(ch)
        for lo, hi, name in _BLOCKS:
            if lo <= cp <= hi:
                return name
    return None


def transliterate(word: str, target: str = "DEVANAGARI") -> str:
    """Map a Brahmic word to a shared script so cross-script tokens can merge."""
    if not _SANS_OK:
        return word
    src = _detect_scheme(word)
    if not src or src == target:
        return word
    try:
        out = sanscript.transliterate(word, getattr(sanscript, src), getattr(sanscript, target))
        return unicodedata.normalize("NFC", out)
    except Exception:  # noqa: BLE001
        return word
