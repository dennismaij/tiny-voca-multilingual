"""English front end: text -> phoneme tokens with a stress feature.

The same code runs during annotation, training-set preparation and inference,
so tokens can never drift between them. Two programs can produce the phonemes:

* ``espeak`` (front-end version 3, the default): eSpeak NG, as described below.
  The current checkpoints and annotations were made with it.
* ``own`` (front-end version 4): the separate ``dizione`` package, a dictionary
  built from CMUdict, a small model for unknown words and rules. No eSpeak, so
  the whole system can be shipped under a permissive licence. Same 76 symbols.

Choose with ``set_frontend("own")`` or ``TINY_VOCA_FRONTEND=own``. Everything
below about tokens holds for both; the rest of this note is the eSpeak path.

* eSpeak NG (``en-us``) is the only pronunciation source. It is an external
  rule-based program: no neural G2P and no downloaded weights.
* A token is one eSpeak phoneme exactly as eSpeak segments it (``--sep``), for
  example ``t``, ``aɪ``, ``ɑːɹ`` or ``n̩``. This is also the label inventory of
  the phoneme recogniser used for offline forced alignment, so one aligned
  label corresponds to exactly one token.
* Lexical stress is a parallel feature (0 none, 1 primary, 2 secondary), not a
  token, so it never has to be given audio frames.
* Punctuation, word boundaries and the utterance edges are explicit tokens that
  may last zero frames. They carry pauses when the recording has them.

Versions 1 and 2 encoded IPA as UTF-8 bytes and piped text to eSpeak without a
final newline; eSpeak NG 1.51 then drops the last character of the input, so
the final phonemes of every clause were wrong. Version 3 terminates the input
and has a regression test for it.
"""

from dataclasses import dataclass
import hashlib
import json
import os
import re
import shutil
import subprocess
import unicodedata

FRONTEND_VERSION = 3
SEPARATOR = "_"
FRONTENDS = ("espeak", "own")
FRONTEND_VARIABLE = "TINY_VOCA_FRONTEND"
_frontend = os.environ.get(FRONTEND_VARIABLE, "espeak")
if _frontend not in FRONTENDS:
    raise ValueError(f"{FRONTEND_VARIABLE} must be one of {FRONTENDS}, not {_frontend!r}")

PAD, BOS, EOS, SP, UNK = "<pad>", "<bos>", "<eos>", "<sp>", "<unk>"
SPECIALS = (PAD, BOS, EOS, SP, UNK)
PUNCTUATION = (",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…")
STRESS_MARKS = {"ˈ": 1, "ˌ": 2}

LANGUAGE_CONFIGS = {
    "en-us": {
        "voice": "en-us",
        "consonants": ("p", "b", "t", "d", "k", "ɡ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
                       "m", "n", "ŋ", "l", "ɹ", "j", "w", "tʃ", "dʒ", "ɾ", "ʔ"),
        "vowels": ("ə", "ɐ", "ɚ", "ᵻ", "ɪ", "i", "iː", "ɛ", "æ", "ʌ", "ɑː", "ɔ", "ɔː", "ʊ", "uː", "ɜː",
                   "eɪ", "aɪ", "ɔɪ", "oʊ", "aʊ", "iə", "oː",
                   "ɑːɹ", "ɔːɹ", "oːɹ", "ɪɹ", "ɛɹ", "ʊɹ", "aɪɚ", "aɪə", "əl", "n̩"),
        "voiceless": ("p", "t", "k", "f", "θ", "s", "ʃ", "h", "tʃ", "ʔ"),
        "fallback": {
            "r": ("ɹ",), "ɻ": ("ɹ",), "ʁ": ("ɹ",), "x": ("k",), "ç": ("h",), "ɬ": ("l",), "ɫ": ("l",), "ɲ": ("n", "j"),
            "ts": ("t", "s"), "dz": ("d", "z"), "β": ("v",), "ɣ": ("ɡ",), "ʎ": ("l", "j"),
            "a": ("æ",), "aː": ("ɑː",), "ɑ": ("ɑː",), "ɒ": ("ɑː",), "ɑ̃": ("ɑː",), "ɔ̃": ("ɔː",), "ɛ̃": ("æ",),
            "e": ("eɪ",), "eː": ("eɪ",), "o": ("oʊ",), "u": ("uː",), "y": ("uː",), "yː": ("uː",),
            "ø": ("ɜː",), "øː": ("ɜː",), "œ": ("ɜː",), "ɜ": ("ɜː",), "ɛː": ("ɛ",), "ɪː": ("ɪ",),
            "əʊ": ("oʊ",), "eə": ("ɛɹ",), "ɪə": ("iə",), "ʊə": ("ʊɹ",), "aʊɚ": ("aʊ", "ɚ"), "l̩": ("əl",), "m̩": ("ə", "m"),
        },
        "abbreviations": {"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc"},
    },
    "en-gb": {
        "voice": "en-gb",
        "consonants": ("p", "b", "t", "d", "k", "ɡ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
                       "m", "n", "ŋ", "l", "ɹ", "j", "w", "tʃ", "dʒ", "ɾ", "ʔ"),
        "vowels": ("ə", "ɐ", "ɪ", "i", "iː", "ɛ", "æ", "ʌ", "ɑː", "ɒ", "ɔ", "ɔː", "ʊ", "uː", "ɜː",
                   "eɪ", "aɪ", "ɔɪ", "əʊ", "aʊ", "ɪə", "iə", "eə", "ʊə", "aɪə", "aʊə", "oː", "əl", "n̩"),
        "voiceless": ("p", "t", "k", "f", "θ", "s", "ʃ", "h", "tʃ", "ʔ"),
        "fallback": {
            "r": ("ɹ",), "ɻ": ("ɹ",), "ʁ": ("ɹ",), "x": ("k",), "ç": ("h",), "ɬ": ("l",), "ɫ": ("l",), "ɲ": ("n", "j"),
            "ts": ("t", "s"), "dz": ("d", "z"), "β": ("v",), "ɣ": ("ɡ",), "ʎ": ("l", "j"),
            "a": ("æ",), "aː": ("ɑː",), "ɑ": ("ɑː",), "ɒ": ("ɑː",), "ɑ̃": ("ɑː",), "ɔ̃": ("ɔː",), "ɛ̃": ("æ",),
            "e": ("eɪ",), "eː": ("eɪ",), "o": ("əʊ",), "oʊ": ("əʊ",), "u": ("uː",), "y": ("uː",), "yː": ("uː",),
            "ø": ("ɜː",), "øː": ("ɜː",), "œ": ("ɜː",), "ɜ": ("ɜː",), "ɛː": ("ɛ",), "ɪː": ("ɪ",),
            "ɚ": ("ə",), "ɑːɹ": ("ɑː",), "ɔːɹ": ("ɔː",), "oːɹ": ("ɔː",), "ɪɹ": ("ɪə",), "ɛɹ": ("eə",),
            "ʊɹ": ("ʊə",), "aʊɚ": ("aʊə",), "aɪɚ": ("aɪə",), "l̩": ("əl",), "m̩": ("ə", "m"),
        },
        "abbreviations": {"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc"},
    },
    "nl": {
        "voice": "nl",
        "consonants": ("p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z", "ʃ", "ʒ", "x", "ɣ", "h",
                       "m", "n", "ŋ", "l", "r", "j", "ʋ", "w", "ʔ", "tʃ", "dʒ"),
        "vowels": ("ə", "ɪ", "i", "iː", "eː", "ɛ", "ɛː", "ɑ", "aː", "ɔ", "ɔː", "oː", "u", "uː", "y", "yː",
                   "øː", "œː", "ɵ", "ɪː", "ɛɪ", "œy", "ʌʊ", "aʊ", "aɪ", "ɔɪ", "ui"),
        "voiceless": ("p", "t", "k", "f", "s", "ʃ", "x", "h", "ʔ", "tʃ"),
        "fallback": {
            "tʲ": ("t", "j"), "dʲ": ("d", "j"), "nʲ": ("n", "j"), "sʲ": ("s", "j"), "lʲ": ("l", "j"),
            "kʲ": ("k", "j"), "pʲ": ("p", "j"), "bʲ": ("b", "j"), "mʲ": ("m", "j"),
            "ɻ": ("r",), "ʁ": ("r",), "ɹ": ("r",), "ç": ("x",), "g": ("ɣ",),
            "ɑu": ("ʌʊ",), "ou": ("ʌʊ",), "ɛi": ("ɛɪ",), "e": ("eː",), "o": ("oː",), "a": ("ɑ",),
            "œ": ("øː",), "ts": ("t", "s"), "dz": ("d", "z"), "θ": ("s",), "ð": ("z",),
        },
        "abbreviations": {"bv", "bijv", "dr", "drs", "prof", "ds", "ir", "ing", "mr", "mevr", "dhr", "blz", "pag", "vs", "enz"},
    },
}

LANGUAGE_VARIABLE = "TINY_VOCA_LANG"
_language = os.environ.get(LANGUAGE_VARIABLE, "en-us").lower()
if _language not in LANGUAGE_CONFIGS:
    raise ValueError(f"{LANGUAGE_VARIABLE} must be one of {tuple(LANGUAGE_CONFIGS.keys())}, not {_language!r}")

PHONE_SET = set()
PHONE_IDS = set()
VOICELESS = set()
VOWEL_SET = set()
VOWEL_IDS = set()
FALLBACK = {}
ABBREVIATIONS = set()
SYMBOL_TO_ID = {}
ID_TO_SYMBOL = {}
SYMBOLS = ()
PHONES = ()
CONSONANTS = ()
VOWELS = ()
ESPEAK_VOICE = "en-us"


def set_language(lang):
    global _language, ESPEAK_VOICE, CONSONANTS, VOWELS, PHONES, SYMBOLS
    lang = lang.lower()
    if lang not in LANGUAGE_CONFIGS:
        raise ValueError(f"Language must be one of {tuple(LANGUAGE_CONFIGS.keys())}, not {lang!r}")
    _language = lang
    cfg = LANGUAGE_CONFIGS[lang]
    ESPEAK_VOICE = cfg["voice"]
    CONSONANTS = tuple(cfg["consonants"])
    VOWELS = tuple(cfg["vowels"])
    PHONES = CONSONANTS + VOWELS
    SYMBOLS = SPECIALS + PUNCTUATION + PHONES

    SYMBOL_TO_ID.clear()
    SYMBOL_TO_ID.update({symbol: index for index, symbol in enumerate(SYMBOLS)})

    ID_TO_SYMBOL.clear()
    ID_TO_SYMBOL.update({index: symbol for index, symbol in enumerate(SYMBOLS)})

    PHONE_SET.clear()
    PHONE_SET.update(PHONES)

    PHONE_IDS.clear()
    PHONE_IDS.update(SYMBOL_TO_ID[p] for p in PHONES)
    PHONE_IDS.add(SYMBOL_TO_ID[UNK])

    VOICELESS.clear()
    VOICELESS.update(cfg["voiceless"])

    VOWEL_SET.clear()
    VOWEL_SET.update(VOWELS)

    VOWEL_IDS.clear()
    VOWEL_IDS.update(SYMBOL_TO_ID[v] for v in VOWELS)

    FALLBACK.clear()
    FALLBACK.update(cfg["fallback"])

    ABBREVIATIONS.clear()
    ABBREVIATIONS.update(cfg["abbreviations"])


SUPPORTED_LANGUAGES = tuple(LANGUAGE_CONFIGS.keys())
activate_language = set_language


def language():
    return _language


active_language = language


set_language(_language)

_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "`": "'", "´": "'",
                         "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
                         "–": "—", "―": "—", "[": "(", "{": "(", "]": ")", "}": ")"})
_PUNCTUATION_CHARS = frozenset(PUNCTUATION)
ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc"}
_LANGUAGE_TAG = re.compile(r"\((?:_?[a-z0-9]_?){2,3}(?:-(?:_?[a-z0-9]_?)+)*\)")


class NothingToPronounce(ValueError):
    """The text has no word, number or letter to say: only marks, emoji and the like."""


@dataclass(frozen=True)
class Phonemized:
    symbols: tuple
    ids: tuple
    stress: tuple
    unknown: tuple  # eSpeak phonemes that had no inventory mapping

    def __len__(self):
        return len(self.ids)


def frontend():
    """Name of the program that produces the phonemes: ``"espeak"`` or ``"own"``."""
    return _frontend


def set_frontend(name):
    """Select the phonemizer for this process. Tokens of the two front ends
    must not be mixed: annotations and checkpoints record which one made them."""
    global _frontend
    if name not in FRONTENDS:
        raise ValueError(f"front end must be one of {FRONTENDS}, not {name!r}")
    _frontend = name


def _dizione():
    try:
        import dizione
    except ImportError as error:
        raise RuntimeError('the "own" front end needs the dizione package: pip install dizione') from error
    if tuple(dizione.symbols.SYMBOLS) != SYMBOLS:
        raise RuntimeError("dizione's symbol inventory differs from this model's")
    return dizione


def frontend_identity():
    if _frontend == "own":
        phonemizer = _dizione()
        return {"frontend_version": phonemizer.VERSION, "phonemizer": "dizione", "symbols": list(SYMBOLS)}
    res = {"frontend_version": FRONTEND_VERSION, "voice": ESPEAK_VOICE, "separator": SEPARATOR,
           "symbols": list(SYMBOLS), "fallback": {key: list(value) for key, value in sorted(FALLBACK.items())}}
    if _language != "en-us":
        res["language"] = _language
    return res


def frontend_runtime():
    """What an annotation run records about the exact program behind its
    tokens: the eSpeak version, or the fingerprint of the phonemizer build."""
    if _frontend == "own":
        return {"phonemizer": _dizione().identity()["fingerprint"]}
    return {"espeak_version": espeak_version()}


def frontend_fingerprint():
    return hashlib.sha256(json.dumps(frontend_identity(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    text = re.sub(r"\.{3,}", "…", text)
    text = re.sub(r"-{2,}|\s-\s", "—", text)
    return re.sub(r"\s+", " ", text).strip()


def chunks(text):
    """Yield ``(is_punctuation, piece)``. Decimals, clock times, digit groups,
    contractions, title abbreviations and initialisms stay inside the text piece
    for eSpeak; every other punctuation mark becomes its own token."""
    start = 0
    for i, char in enumerate(text):
        if char not in _PUNCTUATION_CHARS:
            continue
        previous = text[i-1] if i else ""
        following = text[i+1] if i+1 < len(text) else ""
        if char == "'" and previous.isalnum() and following.isalnum():
            continue
        if char in ".,:" and previous.isdigit() and following.isdigit():
            continue
        if char == ".":
            word = re.search(r"([A-Za-z]+)$", text[:i])
            if word and word[1].lower() in ABBREVIATIONS:
                continue
            if previous.isalpha() and following.isalpha():
                continue
        if text[start:i].strip():
            yield False, text[start:i].strip()
        yield True, char
        start = i+1
    if text[start:].strip():
        yield False, text[start:].strip()


def espeak_executable():
    executable = shutil.which("espeak-ng")
    if executable is None:
        raise RuntimeError("Install espeak-ng on the cloud machine before preprocessing/inference")
    return executable


def espeak_version():
    return subprocess.run([espeak_executable(), "--version"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()


def espeak_words(piece):
    """Phonemize one text piece. Returns words as lists of ``(phone, stress)``.

    The input must end with a newline: eSpeak NG 1.51 silently drops the last
    character of unterminated standard input.
    """
    result = subprocess.run(
        [espeak_executable(), "-q", "--ipa", "--sep=" + SEPARATOR, "-v", ESPEAK_VOICE, "--stdin"],
        input=piece.rstrip("\n") + "\n", text=True, capture_output=True, check=True, timeout=30)
    return parse_espeak(result.stdout)


def parse_espeak(output):
    words = []
    for word in _LANGUAGE_TAG.sub("", output).split():
        units = []
        for unit in word.split(SEPARATOR):
            stress = 0
            for mark, level in STRESS_MARKS.items():
                if mark in unit:
                    stress = stress or level
                    unit = unit.replace(mark, "")
            unit = "".join(c for c in unit if unicodedata.category(c) != "Cf" and c not in "()")
            if unit:
                units.append((unit, stress))
        if units:
            words.append(units)
    return words


def canonical(phone):
    """Map an eSpeak phoneme to inventory phones; ``None`` when unknown."""
    if phone in PHONE_SET:
        return (phone,)
    if phone in FALLBACK:
        return FALLBACK[phone]
    stripped = "".join(c for c in unicodedata.normalize("NFD", phone) if not unicodedata.combining(c))
    for candidate in (stripped, stripped.replace("ː", ""), stripped + "ː"):
        if candidate != phone and candidate in PHONE_SET:
            return (candidate,)
        if candidate != phone and candidate in FALLBACK:
            return FALLBACK[candidate]
    return None


def tokenize(pieces):
    """Build the token sequence from ``(is_punctuation, value)`` pieces where a
    text piece's value is the parsed eSpeak word list. Pure function (testable
    without eSpeak)."""
    symbols, stress, unknown = [BOS], [0], []
    for punctuation, value in pieces:
        if punctuation:
            if symbols[-1] != value:  # collapse repeats such as "!!"
                symbols.append(value)
                stress.append(0)
            continue
        for word in value:
            if symbols[-1] in PHONE_SET or symbols[-1] == UNK:
                symbols.append(SP)
                stress.append(0)
            for phone, level in word:
                mapped = canonical(phone)
                if mapped is None:
                    unknown.append(phone)
                    mapped = (UNK,)
                # A fallback may expand to several phones: stress goes to its first vowel.
                carrier = next((i for i, item in enumerate(mapped) if item in VOWEL_SET), 0)
                for index, item in enumerate(mapped):
                    symbols.append(item)
                    stress.append(level if index == carrier else 0)
    symbols.append(EOS)
    stress.append(0)
    if not any(s in PHONE_SET or s == UNK for s in symbols):
        raise NothingToPronounce("the text has nothing to pronounce")
    return Phonemized(tuple(symbols), tuple(SYMBOL_TO_ID[s] for s in symbols), tuple(stress), tuple(unknown))


def phonemize(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")
    if _frontend == "own":
        return tokenize(_dizione().pieces(text))
    pieces = []
    for punctuation, piece in chunks(normalize_text(text)):
        pieces.append((True, piece) if punctuation else (False, espeak_words(piece)))
    return tokenize(pieces)


def is_phone(token_ids):
    """Boolean list/tensor helper: tokens that must be realised as audio."""
    return [int(i) in PHONE_IDS for i in token_ids]
