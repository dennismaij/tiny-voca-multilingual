"""Pronunciation Controls, Custom Lexicon & Pre-Normalization for tiny-voca.

Provides a 3-tier pronunciation pipeline:
1. Inline text markup (<sub alias="...">, [visual](alias: spoken), <phoneme ph="...">)
2. Language-aware pre-normalization (currencies, numbers, unspaced acronyms)
3. Lexicon overrides (orthographic Say-As aliases and direct phonemic/IPA injection)
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Tier 1 & 3: Overrides & Lexicon Registries
# ---------------------------------------------------------------------------

# Phonemic / IPA Overrides: (phone, stress) lists bypassing rule-based G2P
DEFAULT_WORD_OVERRIDES: Dict[str, Dict[str, List[Tuple[str, int]]]] = {
    "en-gb": {
        # Troublesome UK English geographical names & pronunciations
        "edinburgh": [("ɛ", 1), ("d", 0), ("ɪ", 0), ("n", 0), ("b", 0), ("ɹ", 0), ("ə", 0)],
        "worcestershire": [("w", 0), ("ʊ", 1), ("s", 0), ("t", 0), ("ə", 0), ("ʃ", 0), ("ə", 0)],
        "worcester": [("w", 0), ("ʊ", 1), ("s", 0), ("t", 0), ("ə", 0)],
        "leicester": [("l", 0), ("ɛ", 1), ("s", 0), ("t", 0), ("ə", 0)],
        "gloucester": [("ɡ", 0), ("l", 0), ("ɒ", 1), ("s", 0), ("t", 0), ("ə", 0)],
        "gloucestershire": [("ɡ", 0), ("l", 0), ("ɒ", 1), ("s", 0), ("t", 0), ("ə", 0), ("ʃ", 0), ("ə", 0)],
        # Tech terminology & acronyms
        "sql": [("s", 0), ("iː", 1), ("k", 0), ("w", 0), ("əl", 0)],
        "saas": [("s", 0), ("æ", 1), ("s", 0)],
        "oauth": [("əʊ", 1), ("ɔː", 0), ("θ", 0)],
        "fastapi": [("f", 0), ("ɑː", 1), ("s", 0), ("t", 0), ("eɪ", 1), ("p", 0), ("iː", 0)],
        "edge": [("ɛ", 1), ("dʒ", 0)],
    },
    "en-us": {
        "edinburgh": [("ɛ", 1), ("d", 0), ("ɪ", 0), ("n", 0), ("b", 0), ("ɚ", 0), ("oʊ", 0)],
        "worcestershire": [("w", 0), ("ʊ", 1), ("s", 0), ("t", 0), ("ɚ", 0), ("ʃ", 0), ("ɚ", 0)],
        "worcester": [("w", 0), ("ʊ", 1), ("s", 0), ("t", 0), ("ɚ", 0)],
        "leicester": [("l", 0), ("ɛ", 1), ("s", 0), ("t", 0), ("ɚ", 0)],
        "gloucester": [("ɡ", 0), ("l", 0), ("ɑː", 1), ("s", 0), ("t", 0), ("ɚ", 0)],
        "sql": [("s", 0), ("iː", 1), ("k", 0), ("w", 0), ("əl", 0)],
        "saas": [("s", 0), ("æ", 1), ("s", 0)],
        "oauth": [("oʊ", 1), ("ɔː", 0), ("θ", 0)],
        "fastapi": [("f", 0), ("æ", 1), ("s", 0), ("t", 0), ("eɪ", 1), ("p", 0), ("iː", 0)],
        "edge": [("ɛ", 1), ("dʒ", 0)],
    },
    "nl": {
        # Latinate -eren verbs & participles (stress on penult syllable)
        "opererend": [("oː", 0), ("p", 0), ("ə", 0), ("r", 0), ("eː", 1), ("r", 0), ("ə", 0), ("n", 0), ("t", 0)],
        "opereren": [("oː", 0), ("p", 0), ("ə", 0), ("r", 0), ("eː", 1), ("r", 0), ("ə", 0), ("n", 0)],
        "opereert": [("oː", 0), ("p", 0), ("ə", 0), ("r", 0), ("eː", 1), ("r", 0), ("t", 0)],
        "geopereerd": [("ɣ", 0), ("ə", 0), ("oː", 0), ("p", 0), ("ə", 0), ("r", 0), ("eː", 1), ("r", 0), ("t", 0)],
        "concept": [("k", 0), ("ɔ", 0), ("n", 0), ("s", 0), ("ɛ", 1), ("p", 0), ("t", 0)],
        "concepten": [("k", 0), ("ɔ", 0), ("n", 0), ("s", 0), ("ɛ", 1), ("p", 0), ("t", 0), ("ə", 0), ("n", 0)],
        "complex": [("k", 0), ("ɔ", 0), ("m", 0), ("p", 0), ("l", 0), ("ɛ", 1), ("k", 0), ("s", 0)],
        "complexe": [("k", 0), ("ɔ", 0), ("m", 0), ("p", 0), ("l", 0), ("ɛ", 1), ("k", 0), ("s", 0), ("ə", 0)],
        "meedogenloze": [("m", 0), ("eː", 0), ("d", 0), ("oː", 1), ("ɣ", 0), ("ə", 0), ("n", 0), ("l", 0), ("oː", 0), ("z", 0), ("ə", 0)],
        "meedogenloos": [("m", 0), ("eː", 0), ("d", 0), ("oː", 1), ("ɣ", 0), ("ə", 0), ("n", 0), ("l", 0), ("oː", 0), ("s", 0)],
        "uiteindelijke": [("œy", 0), ("t", 0), ("ɛɪ", 1), ("n", 0), ("d", 0), ("ə", 0), ("l", 0), ("ə", 0), ("k", 0), ("ə", 0)],
        "uiteindelijk": [("œy", 0), ("t", 0), ("ɛɪ", 1), ("n", 0), ("d", 0), ("ə", 0), ("l", 0), ("ə", 0), ("k", 0)],
        # Tech & loanwords
        "edge": [("ɛ", 1), ("dʒ", 0)],
        "cases": [("k", 1), ("eː", 0), ("z", 0), ("ə", 0), ("s", 0)],
        "case": [("k", 1), ("eː", 0), ("s", 0)],
        "drag": [("d", 1), ("r", 0), ("ɛ", 0), ("k", 0)],
        "happy": [("h", 1), ("ɛ", 0), ("p", 0), ("i", 0)],
        "path": [("p", 1), ("aː", 0), ("t", 0)],
        "stresstests": [("s", 0), ("t", 0), ("r", 0), ("ɛ", 1), ("s", 0), ("t", 0), ("ɛ", 0), ("s", 0), ("t", 0), ("s", 0)],
        "stresstest": [("s", 0), ("t", 0), ("r", 0), ("ɛ", 1), ("s", 0), ("t", 0), ("ɛ", 0), ("s", 0), ("t", 0)],
        "occlusion": [("ə", 0), ("k", 0), ("l", 0), ("uː", 1), ("ʒ", 0), ("ə", 0), ("n", 0)],
        "umbris": [("ɵ", 1), ("m", 0), ("b", 0), ("r", 0), ("ɪ", 0), ("s", 0)],
        "orbious": [("ɔ", 1), ("r", 0), ("b", 0), ("i", 0), ("j", 0), ("ə", 0), ("s", 0)],
        "zinsser": [("z", 1), ("ɪ", 0), ("n", 0), ("s", 0), ("ə", 0), ("r", 0)],
        "faalmodi": [("f", 1), ("aː", 0), ("l", 0), ("m", 0), ("oː", 0), ("d", 0), ("i", 0)],
        "risicoanalyse": [("r", 1), ("i", 0), ("z", 0), ("i", 0), ("k", 0), ("oː", 0), ("aː", 0), ("n", 0), ("aː", 0), ("l", 0), ("i", 0), ("z", 0), ("ə", 0)],
        "foutanalyse": [("f", 1), ("ʌʊ", 0), ("t", 0), ("aː", 0), ("n", 0), ("aː", 0), ("l", 0), ("i", 0), ("z", 0), ("ə", 0)],
        "online": [("ɔ", 1), ("n", 0), ("l", 0), ("aɪ", 1), ("n", 0)],
        "tools": [("t", 1), ("uː", 0), ("l", 0), ("s", 0)],
        "tool": [("t", 1), ("uː", 0), ("l", 0)],
        "prompt": [("p", 0), ("r", 0), ("ɔ", 1), ("m", 0), ("t", 0)],
        "prompts": [("p", 0), ("r", 0), ("ɔ", 1), ("m", 0), ("t", 0), ("s", 0)],
        "dashboard": [("d", 1), ("ɛ", 0), ("ʃ", 0), ("b", 0), ("ɔː", 0), ("r", 0), ("t", 0)],
    },
}

# Orthographic Aliases / "Say-As" (Text Substitution before G2P)
DEFAULT_ORTHOGRAPHIC_ALIASES: Dict[str, Dict[str, str]] = {
    "en-gb": {
        "SQL": "sequel",
        "API": "A P I",
        "APIs": "A P Is",
        "SaaS": "sass",
        "OAuth": "O Auth",
        "FastAPI": "Fast A P I",
        "GDPR": "G D P R",
        "TTS": "T T S",
        "UI": "U I",
        "UX": "U X",
        "Leicester": "Lester",
        "Edinburgh": "Edin-bruh",
        "Worcester": "Worster",
        "Worcestershire": "Wostershire",
        "Gloucester": "Gloster",
        "Gloucestershire": "Glostershire",
    },
    "en-us": {
        "SQL": "sequel",
        "API": "A P I",
        "APIs": "A P Is",
        "SaaS": "sass",
        "OAuth": "O Auth",
        "FastAPI": "Fast A P I",
        "GDPR": "G D P R",
        "TTS": "T T S",
        "UI": "U I",
        "UX": "U X",
        "Leicester": "Lester",
        "Edinburgh": "Edin-burg",
        "Worcester": "Wooster",
        "Worcestershire": "Woostershire",
    },
    "nl": {
        "SQL": "sequel",
        "API": "A P I",
        "SaaS": "sass",
        "OAuth": "O Auth",
        "GDPR": "G D P R",
        "AVG": "A V G",
        "TTS": "T T S",
        "UI": "U I",
        "UX": "U X",
        "AI": "A I",
        "ZZP": "Z Z P",
        "BTW": "B T W",
    },
}

# Runtime Custom Overrides state
_custom_word_overrides: Dict[str, Dict[str, List[Tuple[str, int]]]] = {"en-gb": {}, "en-us": {}, "nl": {}}
_custom_orthographic_aliases: Dict[str, Dict[str, str]] = {"en-gb": {}, "en-us": {}, "nl": {}}

# Session-level phoneme token injection registry (for inline <phoneme ph="..."> tags)
_inline_phoneme_cache: Dict[str, List[Tuple[str, int]]] = {}


def register_word_override(word: str, units: List[Union[Tuple[str, int], List[Union[str, int]]]], lang: Optional[str] = None):
    """Register a custom phonemic override: word -> [(phone, stress), ...]."""
    target_lang = (lang or "en-gb").lower()
    if target_lang not in _custom_word_overrides:
        _custom_word_overrides[target_lang] = {}
    
    clean_units = [(str(u[0]), int(u[1])) for u in units]
    _custom_word_overrides[target_lang][word.lower().strip()] = clean_units


def register_alias(term: str, replacement: str, lang: Optional[str] = None):
    """Register an orthographic alias (Say-As) replacement."""
    target_lang = (lang or "en-gb").lower()
    if target_lang not in _custom_orthographic_aliases:
        _custom_orthographic_aliases[target_lang] = {}
    _custom_orthographic_aliases[target_lang][term] = replacement


def load_pronunciation_dictionary(dict_or_path: Union[dict, str, os.PathLike], lang: Optional[str] = None):
    """Load a custom pronunciation dictionary from JSON or python dict."""
    if isinstance(dict_or_path, (str, os.PathLike)):
        with open(dict_or_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = dict_or_path

    target_lang = (lang or "en-gb").lower()

    # Supports either top-level or categorized { "aliases": {...}, "phonemes": {...} }
    if "aliases" in data:
        for k, v in data["aliases"].items():
            register_alias(k, v, target_lang)
    if "phonemes" in data:
        for k, v in data["phonemes"].items():
            register_word_override(k, v, target_lang)
    
    # If flat dictionary
    for k, v in data.items():
        if k in ("aliases", "phonemes"):
            continue
        if isinstance(v, str):
            register_alias(k, v, target_lang)
        elif isinstance(v, list):
            register_word_override(k, v, target_lang)


def clear_custom_overrides(lang: Optional[str] = None):
    """Clear custom overrides."""
    if lang:
        _custom_word_overrides[lang.lower()] = {}
        _custom_orthographic_aliases[lang.lower()] = {}
    else:
        for l in _custom_word_overrides:
            _custom_word_overrides[l] = {}
            _custom_orthographic_aliases[l] = {}
    _inline_phoneme_cache.clear()


def get_active_word_overrides(lang: str) -> Dict[str, List[Tuple[str, int]]]:
    """Retrieve combined default and custom word overrides for language."""
    merged = dict(DEFAULT_WORD_OVERRIDES.get(lang.lower(), {}))
    merged.update(_custom_word_overrides.get(lang.lower(), {}))
    # Also attach any active inline <phoneme> placeholders
    merged.update(_inline_phoneme_cache)
    return merged


def get_active_aliases(lang: str) -> Dict[str, str]:
    """Retrieve combined default and custom orthographic aliases."""
    merged = dict(DEFAULT_ORTHOGRAPHIC_ALIASES.get(lang.lower(), {}))
    merged.update(_custom_orthographic_aliases.get(lang.lower(), {}))
    return merged


# ---------------------------------------------------------------------------
# Tier 2: Inline Text Markup / SSML Parser
# ---------------------------------------------------------------------------

_SUB_TAG_RE = re.compile(r'<sub\s+alias=["\']([^"\']+)["\']>(.*?)</sub>', re.IGNORECASE)
_MARKDOWN_ALIAS_RE = re.compile(r'\[([^\]]+)\]\(alias:\s*([^)]+)\)', re.IGNORECASE)
_PHONEME_TAG_RE = re.compile(r'<phoneme\s+ph=["\']([^"\']+)["\']>(.*?)</phoneme>', re.IGNORECASE)


def parse_inline_phonemes(ph_string: str) -> List[Tuple[str, int]]:
    """Parse space-separated IPA symbols with optional stress marks (ˈ primary, ˌ secondary)."""
    units = []
    # Tokenize space-separated phones or IPA characters
    tokens = [t.strip() for t in ph_string.split() if t.strip()]
    for tok in tokens:
        stress = 0
        phone = tok
        if "ˈ" in phone:
            stress = 1
            phone = phone.replace("ˈ", "")
        elif "ˌ" in phone:
            stress = 2
            phone = phone.replace("ˌ", "")
        if phone:
            units.append((phone, stress))
    return units


def parse_inline_ssml(text: str, lang: str = "en-gb") -> str:
    """Preprocesses inline SSML and markdown markup, resolving aliases and caching direct phonemes."""
    # 1. Alias tags: <sub alias="...">visual</sub> -> alias
    text = _SUB_TAG_RE.sub(r' \1 ', text)

    # 2. Markdown format: [visual](alias: spoken) -> spoken
    text = _MARKDOWN_ALIAS_RE.sub(r' \2 ', text)

    # 3. Phonetic tag: <phoneme ph="...">visual</phoneme>
    def _replace_phoneme(match):
        ph_str = match.group(1).strip()
        parsed_units = parse_inline_phonemes(ph_str)
        if not parsed_units:
            return match.group(2)
        # Generate stable placeholder token
        placeholder = f"__ph_{len(_inline_phoneme_cache)}_{abs(hash(ph_str)) % 100000}__"
        _inline_phoneme_cache[placeholder.lower()] = parsed_units
        return f" {placeholder} "

    text = _PHONEME_TAG_RE.sub(_replace_phoneme, text)
    return text


# ---------------------------------------------------------------------------
# Tier 3: Language & Number Pre-Normalization
# ---------------------------------------------------------------------------

_UNITS_EN = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
    17: "seventeen", 18: "eighteen", 19: "nineteen"
}
_TENS_EN = {
    2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"
}

_UNITS_NL = {
    0: "nul", 1: "een", 2: "twee", 3: "drie", 4: "vier", 5: "vijf", 6: "zes", 7: "zeven", 8: "acht", 9: "negen",
    10: "tien", 11: "elf", 12: "twaalf", 13: "dertien", 14: "veertien", 15: "vijftien", 16: "zestien",
    17: "zeventien", 18: "achttien", 19: "negentien"
}
_TENS_NL = {
    2: "twintig", 3: "dertig", 4: "veertig", 5: "vijftig", 6: "zestig", 7: "zeventig", 8: "tachtig", 9: "negentig"
}


def number_to_words_en(n: int) -> str:
    """Convert integer up to 999,999 to English words."""
    if n < 0:
        return "minus " + number_to_words_en(-n)
    if n < 20:
        return _UNITS_EN[n]
    if n < 100:
        tens = _TENS_EN[n // 10]
        rem = n % 10
        return f"{tens}-{_UNITS_EN[rem]}" if rem else tens
    if n < 1000:
        hundreds = _UNITS_EN[n // 100] + " hundred"
        rem = n % 100
        return f"{hundreds} and {number_to_words_en(rem)}" if rem else hundreds
    if n < 1000000:
        thousands = number_to_words_en(n // 1000) + " thousand"
        rem = n % 1000
        if rem == 0:
            return thousands
        if rem < 100:
            return f"{thousands} and {number_to_words_en(rem)}"
        return f"{thousands} {number_to_words_en(rem)}"
    return str(n)


def number_to_words_nl(n: int) -> str:
    """Convert integer up to 999,999 to Dutch words."""
    if n < 0:
        return "min " + number_to_words_nl(-n)
    if n < 20:
        return _UNITS_NL[n]
    if n < 100:
        rem = n % 10
        tens = _TENS_NL[n // 10]
        if rem == 0:
            return tens
        conj = "ën" if rem in (2, 3) else "en"
        return f"{_UNITS_NL[rem]}{conj}{tens}"
    if n < 1000:
        h = n // 100
        h_str = ("honderd" if h == 1 else f"{_UNITS_NL[h]}honderd")
        rem = n % 100
        return f"{h_str} {number_to_words_nl(rem)}" if rem else h_str
    if n < 1000000:
        th = n // 1000
        th_str = ("duizend" if th == 1 else f"{number_to_words_nl(th)}duizend")
        rem = n % 1000
        return f"{th_str} {number_to_words_nl(rem)}" if rem else th_str
    return str(n)


def normalize_currencies_and_numbers(text: str, lang: str = "en-gb") -> str:
    """Normalizes currency symbols (£, $, €) and standalone numbers into spoken words."""
    is_dutch = lang.lower() == "nl"
    is_uk = lang.lower() == "en-gb"

    # 1. Currency conversions
    # British Pounds: £4.50 -> four pounds fifty, £10 -> ten pounds, 50p -> fifty pence
    def _replace_pounds(m):
        whole = int(m.group(1))
        dec = m.group(2)
        whole_str = number_to_words_en(whole) + (" pound" if whole == 1 else " pounds")
        if dec:
            cents = int(dec[1:])
            if cents == 0:
                return whole_str
            return f"{whole_str} {number_to_words_en(cents)}"
        return whole_str

    text = re.sub(r'£(\d+)(\.\d{2})?', _replace_pounds, text)
    text = re.sub(r'\b(\d+)p\b', lambda m: f"{number_to_words_en(int(m.group(1)))} pence", text)

    # US Dollars: $12.99 -> twelve dollars ninety-nine cents, $5 -> five dollars
    def _replace_dollars(m):
        whole = int(m.group(1))
        dec = m.group(2)
        whole_str = number_to_words_en(whole) + (" dollar" if whole == 1 else " dollars")
        if dec:
            cents = int(dec[1:])
            if cents == 0:
                return whole_str
            return f"{whole_str} {number_to_words_en(cents)} cents"
        return whole_str

    text = re.sub(r'\$(\d+)(\.\d{2})?', _replace_dollars, text)

    # Euros: €15.50 or €15,50
    def _replace_euros(m):
        whole = int(m.group(1))
        dec = m.group(2)
        if is_dutch:
            whole_str = number_to_words_nl(whole) + " euro"
            if dec:
                cents = int(dec[1:])
                return f"{whole_str} {number_to_words_nl(cents)}" if cents else whole_str
            return whole_str
        else:
            whole_str = number_to_words_en(whole) + (" euro" if whole == 1 else " euros")
            if dec:
                cents = int(dec[1:])
                return f"{whole_str} {number_to_words_en(cents)}" if cents else whole_str
            return whole_str

    text = re.sub(r'€(\d+)([.,]\d{2})?', _replace_euros, text)
    return text


def normalize_acronyms_and_aliases(text: str, lang: str = "en-gb") -> str:
    """Replaces registered orthographic aliases and letter-spaces unspaced all-caps acronyms."""
    aliases = get_active_aliases(lang)

    # Apply known orthographic aliases first (case-sensitive or insensitive lookup)
    for term, spoken in aliases.items():
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE if term.islower() else 0)
        text = pattern.sub(spoken, text)

    # Space out remaining unspaced all-caps acronyms of 2-5 letters (e.g. GDPR -> G D P R, TTS -> T T S)
    # Exclude common uppercase words like I, A, OK, TV if handled
    def _space_acronym(m):
        word = m.group(0)
        # Avoid spacing internal placeholders like __ph_...__
        if word.startswith("__") or word.lower() in ("a", "i", "ok"):
            return word
        return " ".join(list(word))

    text = re.sub(r'\b[A-Z]{2,5}\b', _space_acronym, text)
    return text


def preprocess_text_for_pronunciation(text: str, lang: str = "en-gb") -> str:
    """Complete 3-tier pre-normalization pipeline before tokenization/G2P."""
    if not text or not text.strip():
        return ""
    # Tier 1: Inline SSML / Markdown tags
    text = parse_inline_ssml(text, lang)
    # Tier 2: Currency & Numbers
    text = normalize_currencies_and_numbers(text, lang)
    # Tier 3: Acronyms & Orthographic Aliases
    text = normalize_acronyms_and_aliases(text, lang)
    return re.sub(r'\s+', ' ', text).strip()
