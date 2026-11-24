"""Automated regression tests for pronunciation controls & lexicon overrides in tiny-voca.

Verifies:
1. Custom lexicon / pronunciation dictionary interface (Say-As orthographic aliases & direct IPA mappings)
2. Inline text markup / SSML parsing (<sub alias="...">, [visual](alias: spoken), <phoneme ph="...">)
3. Language-aware pre-normalization (currencies, numbers, unspaced acronyms for en-gb, en-us, nl)
4. Domain-specific and troublesome UK English words (Edinburgh, Worcestershire, SQL, OAuth, FastAPI)
"""

import unittest
from tiny_voca.pronunciation import (
    clear_custom_overrides,
    get_active_aliases,
    get_active_word_overrides,
    load_pronunciation_dictionary,
    normalize_acronyms_and_aliases,
    normalize_currencies_and_numbers,
    parse_inline_phonemes,
    parse_inline_ssml,
    preprocess_text_for_pronunciation,
    register_alias,
    register_word_override,
)
from tiny_voca import text


class TestPronunciationControls(unittest.TestCase):

    def setUp(self):
        clear_custom_overrides()
        text.activate_language("en-gb")

    def tearDown(self):
        clear_custom_overrides()

    def test_orthographic_aliases_say_as(self):
        """Verify default Say-As text substitution for tech acronyms and UK places."""
        processed = preprocess_text_for_pronunciation("We use SQL and FastAPI for our SaaS platform.", "en-gb")
        self.assertIn("sequel", processed)
        self.assertIn("Fast A P I", processed)
        self.assertIn("sass", processed)

        processed_uk = preprocess_text_for_pronunciation("They travelled from Leicester to Edinburgh.", "en-gb")
        self.assertIn("Lester", processed_uk)
        self.assertIn("Edin-bruh", processed_uk)

    def test_custom_dictionary_registration(self):
        """Allow users and projects to dynamically register custom aliases and IPA overrides."""
        # 1. Custom alias
        register_alias("Kubernetes", "Kube", "en-gb")
        res = preprocess_text_for_pronunciation("Deploy with Kubernetes today.", "en-gb")
        self.assertIn("Kube", res)

        # 2. Custom direct phonemic override
        register_word_override("customwidget", [("w", 0), ("ɪ", 1), ("dʒ", 0), ("ɪ", 0), ("t", 0)], "en-gb")
        overrides = get_active_word_overrides("en-gb")
        self.assertIn("customwidget", overrides)
        self.assertEqual(overrides["customwidget"], [("w", 0), ("ɪ", 1), ("dʒ", 0), ("ɪ", 0), ("t", 0)])

    def test_load_dictionary_from_dict(self):
        """Test loading a full pronunciation dictionary structure with aliases and phonemes."""
        data = {
            "aliases": {
                "gRPC": "G R P C",
                "Wasm": "waz-um"
            },
            "phonemes": {
                "wasm": [["w", 0], ["ɒ", 1], ["z", 0], ["ə", 0], ["m", 0]]
            }
        }
        load_pronunciation_dictionary(data, "en-gb")
        aliases = get_active_aliases("en-gb")
        self.assertEqual(aliases.get("gRPC"), "G R P C")
        self.assertEqual(aliases.get("Wasm"), "waz-um")

        overrides = get_active_word_overrides("en-gb")
        self.assertIn("wasm", overrides)
        self.assertEqual(overrides["wasm"], [("w", 0), ("ɒ", 1), ("z", 0), ("ə", 0), ("m", 0)])

    def test_inline_ssml_sub_alias_tag(self):
        """Verify inline <sub alias="...">visual</sub> tag replacement."""
        input_text = 'The <sub alias="B B C">BBC</sub> broadcast reported on <sub alias="Worster">Worcester</sub>.'
        result = preprocess_text_for_pronunciation(input_text, "en-gb")
        self.assertNotIn("<sub", result)
        self.assertNotIn("BBC", result)
        self.assertIn("B B C", result)
        self.assertIn("Worster", result)

    def test_inline_markdown_alias(self):
        """Verify inline Markdown [visual](alias: spoken replacement) parsing."""
        input_text = "Visit [Worcestershire](alias: Wostershire) and drink [SQL](alias: sequel)."
        result = preprocess_text_for_pronunciation(input_text, "en-gb")
        self.assertNotIn("[Worcestershire]", result)
        self.assertIn("Wostershire", result)
        self.assertIn("sequel", result)

    def test_inline_ssml_phoneme_tag(self):
        """Verify inline <phoneme ph="..."> phonetic tag injection."""
        input_text = 'Here is <phoneme ph="f ə ˈn ɛ t ɪ k">phonetic</phoneme> speech.'
        result = preprocess_text_for_pronunciation(input_text, "en-gb")
        self.assertNotIn("<phoneme", result)
        # Should create an internal placeholder token cached in word overrides
        overrides = get_active_word_overrides("en-gb")
        self.assertTrue(any(k.startswith("__ph_") for k in overrides.keys()))

    def test_acronym_detection_and_spacing(self):
        """Verify unspaced all-caps acronyms are spaced out for letter-by-letter pronunciation."""
        res = preprocess_text_for_pronunciation("Check GDPR compliance on the TTS UI.", "en-gb")
        self.assertIn("G D P R", res)
        self.assertIn("T T S", res)
        self.assertIn("U I", res)

    def test_currency_and_number_normalization_en_gb(self):
        """Verify British pounds, pence, and numeric conversions."""
        res = preprocess_text_for_pronunciation("That costs £4.50 or 50p in the UK.", "en-gb")
        self.assertIn("four pounds fifty", res)
        self.assertIn("fifty pence", res)

        res_large = preprocess_text_for_pronunciation("A grant of £250 was awarded.", "en-gb")
        self.assertIn("two hundred and fifty pounds", res_large)

    def test_currency_and_number_normalization_en_us(self):
        """Verify US dollars and cents conversions."""
        res = preprocess_text_for_pronunciation("The subscription is $12.99 per month.", "en-us")
        self.assertIn("twelve dollars ninety-nine cents", res)

    def test_currency_and_number_normalization_nl(self):
        """Verify Dutch Euro conversions."""
        res = preprocess_text_for_pronunciation("Het kost €15,50 in Amsterdam.", "nl")
        self.assertIn("vijftien euro vijftig", res)

    def test_troublesome_uk_and_domain_overrides(self):
        """Verify troublesome UK words and tech terms are in the default phonemic registry."""
        overrides_gb = get_active_word_overrides("en-gb")
        for term in ("edinburgh", "worcestershire", "gloucester", "sql", "oauth", "fastapi"):
            self.assertIn(term, overrides_gb, f"Missing default override for {term} in en-gb")
            units = overrides_gb[term]
            self.assertTrue(len(units) > 0)
            self.assertTrue(all(isinstance(u[0], str) and isinstance(u[1], int) for u in units))


if __name__ == "__main__":
    unittest.main()
