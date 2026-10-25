import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import effective_dimension as ed

CONFIG = {"vocab_size": 20, "block_size": 8, "embed_dim": 8, "layers": 2, "heads": 2, "parameters": 0}


# Escritor independente do formato KATSUCK1 (o mesmo de save_params em katsu-core).
def write_checkpoint(path, tensors):
    with Path(path).open("wb") as file:
        file.write(b"KATSUCK1" + struct.pack("<Q", len(tensors)))
        for tensor in tensors:
            tensor = np.asarray(tensor, dtype="<f4")
            file.write(struct.pack("<Q", tensor.ndim) + struct.pack(f"<{tensor.ndim}Q", *tensor.shape))
            file.write(tensor.tobytes())


def model(rng, config=CONFIG):
    v, t, d, h = config["vocab_size"], config["block_size"], config["embed_dim"], config["heads"]
    dh = d // h
    tensors = [rng.normal(0, 0.02, (v, d)), rng.normal(0, 0.02, (t, d))]
    for _ in range(config["layers"]):
        tensors += [np.ones(d), np.zeros(d)]
        for head in range(h):
            q, k = rng.normal(0, 0.02, (d, dh)), rng.normal(0, 0.02, (d, dh))
            if head == 0:  # cabeça 0: Q·Kᵀ de posto 1
                q = np.outer(rng.normal(size=d), np.eye(dh)[0])
                k = np.outer(rng.normal(size=d), np.eye(dh)[0])
            tensors += [q, k, rng.normal(0, 0.02, (d, dh)), rng.normal(0, 0.02, (dh, d))]
        tensors += [np.ones(d), np.zeros(d), rng.normal(0, 0.02, (d, 4 * d)), np.zeros(4 * d),
                    rng.normal(0, 0.02, (4 * d, d)), np.zeros(d)]
    tensors += [np.ones(d), np.zeros(d), rng.normal(0, 0.02, (d, v))]
    return tensors


class SpectrumTests(unittest.TestCase):
    def test_known_spectra(self):
        full = ed.spectrum_stats(np.eye(6))
        self.assertAlmostEqual(full["erank"], 6)
        self.assertAlmostEqual(full["pr"], 6)
        self.assertEqual(full["d90"], 6)
        self.assertAlmostEqual(full["frac"], 1.0)
        one = ed.spectrum_stats(np.outer([1.0, 2.0, 3.0], [1.0, -1.0, 0.5, 2.0]))
        self.assertAlmostEqual(one["erank"], 1)
        self.assertAlmostEqual(one["pr"], 1)
        self.assertEqual((one["d90"], one["top"]), (1, 1.0))
        # Valores singulares 3 e 1: variâncias 9 e 1 -> PR = 100/82.
        two = ed.spectrum_stats(np.diag([3.0, 1.0]))
        self.assertAlmostEqual(two["pr"], 100 / 82)
        self.assertAlmostEqual(two["erank"], np.exp(-(0.75 * np.log(0.75) + 0.25 * np.log(0.25))))
        self.assertEqual(ed.spectrum_stats(np.zeros((3, 3)))["erank"], 0.0)

    def test_centering_removes_shared_offset(self):
        rows = np.tile([5.0, 5.0, 5.0], (10, 1)) + np.outer(np.arange(10.0), [1.0, 0.0, 0.0])
        self.assertAlmostEqual(ed.spectrum_stats(rows, center=True)["erank"], 1)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.run = Path(self.temp.name) / "run"
        self.run.mkdir()
        (self.run / "config.json").write_text(json.dumps(CONFIG))
        rng = np.random.default_rng(3)
        self.base = model(rng)
        trained = [t.copy() for t in self.base]
        trained[0][[1, 4, 7]] += 0.5  # o "SFT" mexeu em 3 tokens
        write_checkpoint(self.run / "model.ckpt", trained)
        write_checkpoint(self.run / "base.ckpt", self.base)

    def tearDown(self):
        self.temp.cleanup()

    def test_report_detects_rank_one_head_and_touched_tokens(self):
        report = ed.analyze(self.run, self.run / "base.ckpt")
        self.assertEqual(report["sft_touched_tokens"], 3)
        self.assertEqual(report["head_dim"], 4)
        rows = {(r["group"], r["name"]): r for r in report["rows"]}
        for layer in ("L0", "L1"):
            self.assertAlmostEqual(rows[(layer, "h0.QK")]["erank"], 1, places=5)
            self.assertAlmostEqual(rows[(layer, "h0.QK")]["frac"], 0.25, places=5)
            self.assertGreater(rows[(layer, "h1.QK")]["frac"], 0.5)  # cabeça aleatória usa quase tudo
            self.assertEqual(rows[(layer, "h0.QK")]["k"], 4)
        self.assertEqual(rows[("embeddings", "wte_touched")]["tokens"], 3)
        self.assertLess(report["summary"]["attention_head_frac_min"], 0.26)
        self.assertGreater(report["summary"]["attention_head_frac_random"], 0.5)
        self.assertIn("dominant_vs_mean_logit", rows[("output", "head")])

    def test_cli_writes_json_and_compares(self):
        out = self.run / "effdim.json"
        with patch("sys.stdout"):
            self.assertEqual(ed.main([str(self.run), "--json", str(out)]), 0)
        saved = json.loads(out.read_text())
        self.assertIsNone(saved["sft_touched_tokens"])
        text = ed.format_report(ed.analyze(self.run), compare=saved)
        self.assertIn("comparação com run", text)
        self.assertIn("atenção frac média", text)

    def test_rejects_corrupt_or_mismatched_checkpoints(self):
        data = (self.run / "model.ckpt").read_bytes()
        broken = self.run / "broken.ckpt"
        for blob in (b"NOTKATSU" + data[8:], data[:-10], data + b"\0"):
            broken.write_bytes(blob)
            with self.subTest(size=len(blob)), self.assertRaises(ValueError):
                ed.load_checkpoint(broken)
        params = ed.load_checkpoint(self.run / "model.ckpt")
        with self.assertRaises(ValueError):
            ed.named_params(params, {**CONFIG, "layers": 1})
        with self.assertRaises(ValueError):
            ed.named_params(params, {**CONFIG, "vocab_size": 21})
        missing = Path(self.temp.name) / "sem-run"
        with patch("sys.stderr"):
            self.assertEqual(ed.main([str(missing)]), 1)


if __name__ == "__main__":
    unittest.main()
