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

from research.kb.accept import story_errors, story_stats, story_warnings

CARD = ["陈嘉庚", "叶亚来"]
ENTRY = {
    "fields": {
        "人物": {"text": "陈嘉庚，福建同安集美人。"},
        "时间": {"text": "1921年。"},
        "地点": {"text": "福建厦门。"},
        "起因": {"text": "陈嘉庚决定兴办教育。"},
        "经过": {"text": "陈嘉庚于1921年创办厦门大学。"},
        "结果": {"text": "厦门大学成为当时全国院系最齐全的五所大学之一。"},
    },
    "acceptance": {
        "required_spans": ["1921年", "福建厦门", "创办厦门大学", "院系最齐全的五所大学之一"],
    },
}
VALID_STORY = (
    "1921年，陈嘉庚在福建厦门创办厦门大学。他先把办学所需的事情逐项安排妥当，"
    "再与众人商量校舍和课程。开学以后，学生们陆续走进课堂，校园里有了读书声。"
    "陈嘉庚仍常到学校查看，关心教学是否顺利，也提醒办事的人把每项工作做细。"
    "后来，厦门大学成为当时全国院系最齐全的五所大学之一。这段办学经历，"
    "让他长久以来的心愿有了实实在在的落点，也让更多青年获得了求知的机会。"
)


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


class HistoricalStoryGate(unittest.TestCase):
    def test_a_complete_story_passes_the_automatic_gate(self):
        self.assertEqual(story_errors(ENTRY, VALID_STORY), [])

    def test_length_is_enforced_per_story_not_by_median(self):
        errors = story_errors(ENTRY, "1921年，陈嘉庚在福建厦门创办厦门大学。")
        self.assertTrue(any(e.startswith("LENGTH:") for e in errors))

    def test_a_clean_stop_is_required(self):
        self.assertIn("STOP", story_errors(ENTRY, VALID_STORY, stopped=False))

    def test_a_missing_event_anchor_is_caught(self):
        story = VALID_STORY.replace("创办厦门大学", "筹办学校")
        errors = story_errors(ENTRY, story)
        self.assertIn("MISSING_FACT:创办厦门大学", errors)

    def test_an_undeclared_person_is_surfaced_for_review(self):
        warnings = story_warnings(ENTRY, VALID_STORY + "周强后来也来到学校。")
        self.assertTrue(any(w.startswith("UNDECLARED_PERSON:") for w in warnings))

    def test_an_undeclared_number_is_caught(self):
        errors = story_errors(ENTRY, VALID_STORY.replace("1921年", "1922年"))
        self.assertIn("UNSUPPORTED_NUMBER:1922", errors)

    def test_missing_fact_annotations_fail_closed(self):
        entry = {**ENTRY}
        entry.pop("acceptance")
        self.assertIn("FACT_SPEC", story_errors(entry, VALID_STORY))


if __name__ == "__main__":
    unittest.main()
