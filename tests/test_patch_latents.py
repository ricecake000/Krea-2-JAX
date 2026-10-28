"""scripts/modal_patch_latents.py build + patched-version support in BucketDataset."""

import json, re, shutil
from types import SimpleNamespace

import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
import pytest
from safetensors.numpy import load_file

from helpers import make_fake_latent_cache
from krea2_jax.data import BucketDataset
from test_data import bucket_rows

GOLDEN = {  # fingerprints of main@92b5ee5 (before patched-version support), fixed digests
    (16, (0.5, 0.3, 0.2), 0.5): "80e73388a5172ea72c8231870081fa7e17dad59b1671ba1174ed790489fc4269",
    (4096, (0.6, 0.3, 0.1), 0.25): "1768c9e5b57fc2d83fc94387f076010623f172c34b70ebc1ae8edd63ea3bef11"}
TMPL = SimpleNamespace(text_rows=32, prefix_ids=[11, 12, 13], suffix_ids=[14, 15], pad_id=0)
SPECS = {13: ("crop", "ok", (48, 48), ""), 450: ("crop", "crop_unsure", (64, 32), ""),
         600: ("caption", "ok", None, ", white border"), 77: ("remove", "ok", None, "")}
ds_of = lambda d, it=16, **kw: BucketDataset(d, TMPL, img_tokens=it, patch=2, **kw)  # noqa: E731


def edit_manifest(d, fn):
    m = json.loads((d / "latents/manifest.json").read_text())
    fn(m)
    (d / "latents/manifest.json").write_text(json.dumps(m))


def test_unpatched_fingerprint_unchanged(tmp_path):
    make_fake_latent_cache(tmp_path, bucket_rows())
    edit_manifest(tmp_path, lambda m: m.update(digests=dict(zip(m["shards"], ("a" * 64, "b" * 64)))))
    for (it, probs, ap), want in GOLDEN.items():
        ds = ds_of(tmp_path, it, caption_probs=probs, artist_prob=ap, verify_digests=False)
        assert ds.fingerprint() == want


def test_build_load_and_reject_inconsistent(tmp_path):
    pytest.importorskip("torch")
    root = tmp_path
    from PIL import Image
    import modal_patch_latents as mpl
    make_fake_latent_cache(root / "base", bucket_rows(), lat_channels=16, vae="stub")
    base = {r["id"]: r for r in pq.read_table(root / "base/metadata.parquet").to_pylist()}
    (root / "patch/images").mkdir(parents=True)
    rows = []
    for i, (action, status, bucket, sfx) in SPECS.items():
        b, (w, h) = base[i], bucket or (base[i]["bucket_w"], base[i]["bucket_h"])
        rows.append({"id": i, "action": action, "status": status, "bucket_w": w, "bucket_h": h})
        for c in ("long", "medium", "short"):
            rows[-1].update({c + "_old": b[c], c: b[c] + sfx, c + "_tok": len((b[c] + sfx).split())})
        if action == "crop":
            Image.fromarray(np.random.default_rng(i).integers(0, 256, (h, w, 3), dtype=np.uint8)).save(
                root / f"patch/images/{i}.webp", "WEBP", quality=95)
    pq.write_table(pa.Table.from_pylist(rows), root / "patch/patch.parquet")
    (root / "patch/patch.json").write_text(json.dumps({"review_id": "r1"}))
    mpl.cli(["encode", "--patch-dir", str(root / "patch"), "--vae-stub", "--batch", "2"])
    mpl.build_version(root / "base", root / "patch", root / "new", "c0" * 20, link_base_shards=True)
    man, bman = (json.loads((root / d / "latents/manifest.json").read_text()) for d in ("new", "base"))
    assert man["shards"] == bman["shards"] + ["patch-0000.safetensors"]
    assert all(man["digests"][s] == bman["digests"][s] for s in bman["shards"])
    assert man["superseded"] == {"train-0000.safetensors": [77, 450], "train-0001.safetensors": [13]}
    ds, bds = ds_of(root / "new"), ds_of(root / "base")
    assert ds.verified_shards == 3 and ds.n == 12 and 77 not in ds.ids
    assert ds.fingerprint() != bds.fingerprint()
    pos, bpos = ({i: j for j, i in enumerate(d.ids.tolist())} for d in (ds, bds))
    enc = load_file(str(root / "patch/patch_latents.safetensors"))
    for i, hw in ((13, [3, 3]), (450, [2, 4])):  # crops to 48x48 and 64x32
        assert ds.shards[ds.shard_of[pos[i]]] == "patch-0000.safetensors" and ds.hw[pos[i]].tolist() == hw
        assert ds.read_latent(pos[i]).tobytes() == enc[str(i)].tobytes()
    assert ds.caption(pos[600], 0, False) == base[600]["long"] + ", white border"
    for i in (907, 2, 1001, 600):
        assert ds.read_latent(pos[i]).tobytes() == bds.read_latent(bpos[i]).tobytes()
    for k, (edit, needle) in enumerate([
            (lambda m: m["superseded"]["train-0000.safetensors"].append(100907),
             "1 superseded ids are not in their shard (train-0000.safetensors:100907)"),
            (lambda m: m.pop("superseded"), "2 ids appear in more than one shard (13, 450)")]):
        edit_manifest(d := shutil.copytree(root / "new", root / f"bad{k}", symlinks=True), edit)
        with pytest.raises(ValueError, match=re.escape(needle)):
            ds_of(d, verify_digests=False)
