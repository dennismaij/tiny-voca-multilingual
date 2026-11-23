import unittest
from tiny_voca import text
from tiny_voca.config import Config


class MultilingualTests(unittest.TestCase):
    def tearDown(self):
        text.set_language("en-us")

    def test_switch_to_british_english(self):
        text.set_language("en-gb")
        self.assertEqual(text.language(), "en-gb")
        self.assertEqual(text.ESPEAK_VOICE, "en-gb")
        res = text.phonemize("Hello world, this is rather splendid weather.")
        self.assertGreater(len(res.symbols), 5)
        self.assertEqual(len(res.unknown), 0, f"Unexpected unknown phones: {res.unknown}")
        c = Config()
        self.assertEqual(c.vocab_size, len(text.SYMBOLS))

    def test_switch_to_dutch(self):
        text.set_language("nl")
        self.assertEqual(text.language(), "nl")
        self.assertEqual(text.ESPEAK_VOICE, "nl")
        # Test vocabulary with tricky Dutch sounds: ui, ij, ch, g, and diminutive -tje
        sentence = "Hallo wereld, dit is een test voor het huisje met de mooie tuin."
        res = text.phonemize(sentence)
        self.assertGreater(len(res.symbols), 5)
        self.assertEqual(len(res.unknown), 0, f"Unexpected unknown phones: {res.unknown}")
        c = Config()
        self.assertEqual(c.vocab_size, len(text.SYMBOLS))

    def test_language_switch_reverts_cleanly_to_us_english(self):
        text.set_language("nl")
        text.set_language("en-us")
        self.assertEqual(text.language(), "en-us")
        self.assertEqual(len(text.SYMBOLS), 76)
        res = text.phonemize("The quick brown fox jumps over the lazy dog.")
        self.assertEqual(len(res.unknown), 0)


if __name__ == "__main__":
    unittest.main()
