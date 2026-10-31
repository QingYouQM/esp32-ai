"""The acceptance gate has to fail the things that actually shipped broken.

Every check in accept.py maps to a defect that was observed in real output, so
each one is pinned here with the output that produced it rather than with an
invented example:

  stop    the pre-15396eb bins never emitted EOD (0/24)
  echo    long decodes collapsed into the material template ("人物：……")
  hybrid  the 10M checkpoint wrote "陈嘉" for a card that said "陈嘉庚"
  repeat  4-grams looping is why generate() bans them at all

  uv run python -m unittest discover -s tests
"""

import unittest

from research.kb.accept import story_stats

CARD = ["陈嘉庚", "叶亚来"]


class StoryStats(unittest.TestCase):
    def test_a_plain_story_passes_everything(self):
        s = story_stats("老张把船推上岸，天刚亮。他坐在石头上抽了一袋烟，然后回村。", CARD)
        self.assertFalse(s["echo"])
        self.assertFalse(s["hybrid"])
        self.assertEqual(s["prefixes"], [])
        self.assertEqual(s["repeat4"], 0.0)

    def test_material_template_echo_is_caught(self):
        self.assertTrue(story_stats("人物：陈嘉庚。时间：一九一二年。", CARD)["echo"])

    def test_a_truncated_name_counts_as_hybrid(self):
        # "陈嘉" is what b10 wrote when the card said 陈嘉庚.
        s = story_stats("陈嘉推开门，走了出去。", CARD)
        self.assertTrue(s["hybrid"])
        self.assertEqual(s["prefixes"], ["陈嘉"])

    def test_a_whole_name_is_exact_not_hybrid(self):
        s = story_stats("陈嘉庚推开门，走了出去。", CARD)
        self.assertFalse(s["hybrid"])
        self.assertGreater(s["exact"], 0)

    def test_a_shared_first_character_is_not_a_truncation(self):
        # One character in common is ordinary Chinese prose, not a copy gone wrong.
        self.assertFalse(story_stats("陈家做了一锅饭。", CARD)["hybrid"])

    def test_two_character_names_cannot_truncate(self):
        self.assertFalse(story_stats("叶亚回村了。", ["叶亚"])["hybrid"])

    def test_punctuation_does_not_inflate_length(self):
        plain = story_stats("他走了", CARD)["chars"]
        padded = story_stats("他，走。了！", CARD)["chars"]
        self.assertEqual(plain, padded)

    def test_a_loop_shows_up_as_repeated_grams(self):
        story = "他学着它们的样子" * 4
        self.assertGreater(story_stats(story, CARD)["repeat4"], 0.5)


if __name__ == "__main__":
    unittest.main()
