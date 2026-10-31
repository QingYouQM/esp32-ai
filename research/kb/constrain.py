"""Make the declared names appear by construction, not by hope.

A 3M-parameter core cannot copy a string across context: measured on 24 real
cards, the entity rate was 0% on every checkpoint, and the names it did produce
were invented ones. That is a capability limit, so no amount of prompting fixes
it -- the fix has to sit outside the model.

This is the outside. Each declared name gets a slot: the first one opens the
story, each later one lands at the first sentence break after its even share of
the token budget. Inside a slot the next token is forced; everywhere else the
distribution is untouched, so the prose between names is still the model own.

The trade is explicit. Coverage becomes 100% mechanically, which means it no
longer measures the model at all -- it measures the schedule. What it buys is a
clean split: if the stories still read as stories, the model can use a name it
was handed and the failure was only copying; if they fall apart around the
inserted names, the failure is deeper than copying.
"""

import torch

# Sentence ends. A name wants to start a sentence; starting one after a comma
# reads like a typo, and starting one mid-clause reads like a paste job.
BOUNDARIES = "\u3002\uff01\uff1f\u2026\uff1b\n"


def boundary_ids(tok):
    """Token ids that end a sentence, so the schedule can wait for one."""
    return {i for i in range(tok.get_vocab_size())
            if tok.decode([i]).endswith(tuple(BOUNDARIES))}


class ForcedNames:
    """One story at a time, every name once, at planned positions.

    Callable as the logits_processor of TinyLM.generate: it masks all but the
    forced token while a slot is open, and returns the logits untouched
    otherwise.
    """

    def __init__(self, names, boundary, budget):
        if not names:
            raise ValueError("nothing to force")
        self.names = [list(n) for n in names]
        self.boundary = set(boundary)
        self.budget = budget
        self.step = 0
        self.next_name = 0
        self.queue = []
        # Evenly spaced, so the first name cannot eat the opening and starve
        # the rest. Floor division keeps the last slot inside the budget.
        self.due = [i * budget // len(names) for i in range(len(names))]
        # A model that will not punctuate must not be allowed to swallow the
        # remaining names; after this much grace a slot opens anyway.
        self.slack = max(6, budget // (6 * len(names)))

    @property
    def fired(self):
        """How many slots have opened. Equals len(names) when the run was
        not starved by a budget too small for the schedule."""
        return self.next_name

    def __call__(self, idx, logits):
        if idx.size(0) != 1:
            raise ValueError("ForcedNames schedules a single story (batch 1)")
        if not self.queue and self.next_name < len(self.names):
            at = self.due[self.next_name]
            last = idx[0, -1].item()
            if (self.step == 0
                    or (self.step >= at and last in self.boundary)
                    or self.step >= at + self.slack):
                self.queue = list(self.names[self.next_name])
                self.next_name += 1
        if self.queue:
            mask = torch.full_like(logits, -float("inf"))
            mask[:, self.queue.pop(0)] = 0.0
            logits = mask
        self.step += 1
        return logits
