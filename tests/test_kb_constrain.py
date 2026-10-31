"""The name schedule is the only thing standing between 0% and 100%.

If it slips, nothing errors: the stories simply stop containing the names, and
the coverage table reports a model failure that is not the model's fault. So the
three behaviours that matter are pinned -- the first name opens the story, a
later name waits for a sentence break, and a model that will not punctuate
cannot starve the names still waiting.

  uv run python -m unittest discover -s tests
"""

import unittest

import torch

from research.kb.constrain import ForcedNames

PERIOD = 99
VOCAB = 32


def step(policy, last_token):
    """Advance the schedule once; return the tokens it still allows."""
    idx = torch.tensor([[1, last_token]])
    out = policy(idx, torch.zeros(1, VOCAB))
    return torch.isfinite(out[0]).nonzero().flatten().tolist()


class ForcedNamesTest(unittest.TestCase):

    def test_first_name_opens_the_story(self):
        p = ForcedNames([[5, 6], [7]], boundary={PERIOD}, budget=20)
        self.assertEqual(step(p, 1), [5])
        self.assertEqual(step(p, 5), [6])
        # Slot closed, and the second name is not due yet: the model is back in
        # charge of every token.
        self.assertEqual(step(p, 6), list(range(VOCAB)))

    def test_a_later_name_waits_for_a_sentence_break(self):
        p = ForcedNames([[5], [7]], boundary={PERIOD}, budget=20)
        self.assertEqual(step(p, 1), [5])
        for _ in range(9):
            self.assertEqual(len(step(p, 5)), VOCAB)
        # Past the slot's share of the budget, but mid-sentence: still no name.
        self.assertEqual(len(step(p, 4)), VOCAB)
        self.assertEqual(step(p, PERIOD), [7])

    def test_a_model_that_never_punctuates_cannot_starve_the_rest(self):
        p = ForcedNames([[5], [7]], boundary={PERIOD}, budget=20)
        step(p, 1)
        for _ in range(20):
            if step(p, 4) == [7]:
                return
        self.fail("the second name never got a slot")

    def test_a_batch_of_stories_is_refused(self):
        p = ForcedNames([[5]], boundary={PERIOD}, budget=10)
        with self.assertRaises(ValueError):
            p(torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, VOCAB))


if __name__ == "__main__":
    unittest.main()
