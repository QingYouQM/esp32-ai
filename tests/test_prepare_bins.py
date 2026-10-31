"""Every document has to be closed by an endoftext, in place.

This is checked on the produced bin, not on the source, because the bug it
guards against looks like a harmless one-liner: appending len(docs) copies of
endoftext to the end of the flattened array instead of one after each document.
The bins are the same length either way, training does not complain, and the
only symptom appears at sampling time -- the model never learns to stop and
answers a story with the next card's header.

  uv run python -m unittest discover -s tests
"""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
TAG = "_test_prepare_bins"
DOCS = 60


def corpus_text():
    return "".join(
        f"<|material|>\n人物：工人。\n时间：一天。\n地点：桥边。\n"
        f"起因：他要过桥。\n经过：他走到桥中间，听见声音。\n结果：他回了家。\n"
        f"<|story|>\n工人走过桥，听见水声。第{i}次。\n<|endoftext|>\n"
        for i in range(DOCS)
    )


class PrepareBinsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir = ROOT / "data" / TAG
        src = Path(tempfile.mkdtemp()) / "corpus.txt"
        src.write_text(corpus_text(), encoding="utf-8")
        r = subprocess.run(
            [sys.executable, "-m", "research.tinystories.prepare",
             "--tag", TAG, "--vocab", "512", "--corpus", str(src),
             "--seq-len", "32", "--val-fraction", "0.1",
             "--markers", "<|material|>,<|story|>"],
            cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError(f"prepare failed:\n{r.stdout}\n{r.stderr}")
        shutil.rmtree(src.parent, ignore_errors=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def bins(self):
        out = self.dir / "vocab-512"
        eot = Tokenizer.from_file(str(out / "tokenizer.json")).token_to_id("<|endoftext|>")
        for name in ("train.bin", "val.bin"):
            yield name, np.memmap(out / name, dtype=np.uint16, mode="r"), eot

    def test_documents_are_separated_where_they_end(self):
        for name, arr, eot in self.bins():
            hits = np.flatnonzero(arr == eot)
            with self.subTest(name):
                self.assertTrue(len(hits) > 0, "no endoftext at all")
                # The failure mode: one contiguous run parked at the tail.
                self.assertGreater(int(np.diff(hits).min()), 1,
                                   "two endoftext tokens are adjacent")
                self.assertLess(int(hits[0]), len(arr) // 2,
                                "the first boundary is at the very end of the bin")

    def test_no_document_is_left_open(self):
        # The split takes its documents off the end, so val holds a handful and
        # train holds the rest; the documents are ~20 tokens long here.
        for name, arr, eot in self.bins():
            hits = np.flatnonzero(arr == eot)
            with self.subTest(name):
                self.assertGreaterEqual(len(hits), 1 if name == "val.bin" else 40)
                self.assertLessEqual(len(hits), DOCS)


if __name__ == "__main__":
    unittest.main()
