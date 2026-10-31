"""Constrain decoding so material-supported spans land in the story.

The model still writes the prose between spans. During a span slot every token
but the next required one is masked; elsewhere the distribution is untouched.
Slots are evenly spaced and normally open at sentence boundaries. EOD can be
suppressed until every span is complete, preventing an early ending from
silently starving the remaining facts.

This is a generation pipeline, not evidence that the checkpoint learned to
copy. accept.py reports free and constrained runs separately, and every span
is still checked in the decoded text.
"""

import torch

BOUNDARIES = "。！？…；\n"


def boundary_ids(tok):
    """Token ids that end a sentence, so a slot can wait for a clean boundary."""
    return {i for i in range(tok.get_vocab_size())
            if tok.decode([i]).endswith(tuple(BOUNDARIES))}


class ForcedSpans:
    """One story at a time, every token span once, at planned positions."""

    def __init__(self, spans, boundary, budget, eod_id=None):
        if not spans:
            raise ValueError("nothing to force")
        self.spans = [list(span) for span in spans]
        if any(not span for span in self.spans):
            raise ValueError("cannot force an empty span")
        self.boundary = set(boundary)
        self.budget = budget
        self.eod_id = eod_id
        self.step = 0
        self.next_span = 0
        self.queue = []
        self.due = [i * budget // len(spans) for i in range(len(spans))]
        self.slack = max(6, budget // (6 * len(spans)))

    @property
    def fired(self):
        return self.next_span

    @property
    def complete(self):
        return self.next_span == len(self.spans) and not self.queue

    def __call__(self, idx, logits):
        if idx.size(0) != 1:
            raise ValueError("ForcedSpans schedules a single story (batch 1)")
        if not self.queue and self.next_span < len(self.spans):
            at = self.due[self.next_span]
            last = idx[0, -1].item()
            if (self.step == 0
                    or (self.step >= at and last in self.boundary)
                    or self.step >= at + self.slack):
                self.queue = list(self.spans[self.next_span])
                self.next_span += 1
        if self.queue:
            mask = torch.full_like(logits, -float("inf"))
            mask[:, self.queue.pop(0)] = 0.0
            logits = mask
        elif self.eod_id is not None and not self.complete:
            logits = logits.clone()
            logits[:, self.eod_id] = -float("inf")
        self.step += 1
        return logits
