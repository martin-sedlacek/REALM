# DreamZero: camera-height override and dummy GLIGEN grounding (salvaged experiment)

Salvaged from the untracked `realm/inference.py` before its deletion (2026-08-18). That file was
the pre-split inference monolith (last common ancestor `233a38b^`, 2026-03-24) plus four
uncommitted local edits; everything else in it was either byte-identical to the
`realm/inference/` package or deliberately removed (hamster, GR00T_N16 wiring). These three hunks
existed **nowhere else in the repo or its history** and are recorded here in case the experiment
is picked up again. The live client is `realm/inference/client.py`, whose dreamzero branch
hard-codes 320x180 and *asserts* a second camera instead.

## 1. Variable DreamZero input height

`InferenceClient.__init__` took `camera_height=180` and the dreamzero branch resized with it:

```python
dz_size = (320, self.camera_height)
base_im_resized = np.array(Image.fromarray(base_im).resize(dz_size), dtype=np.uint8)
wrist_im_resized = np.array(Image.fromarray(wrist_im).resize(dz_size), dtype=np.uint8)
```

## 2. Dummy GLIGEN grounding keys

The obs dict carried pre-padded, fully-masked grounding boxes so the server's GatedBoxAttention
was a no-op:

```python
# Dummy GLIGEN grounding -- all slots masked so GatedBoxAttention is a no-op.
# Pass pre-padded format so _prepare_boxes uses them directly.
"boxes": np.zeros((16, 5), dtype=np.float32),
"box_mask": np.ones((16,), dtype=np.float32),
```

## 3. Zero-pad fallback for a missing second camera

Instead of asserting `--multi-view`, the old branch padded:

```python
# Add second camera if available, otherwise PAD WITH ZEROS to prevent crashes
if base_im_second is not None:
    obs_dict["observation/exterior_image_1_left"] = base_im_second_resized
else:
    # The server strictly requires this key.
    obs_dict["observation/exterior_image_1_left"] = np.zeros_like(base_im_resized)
```

The package's assert (`realm/inference/client.py`) replaced this deliberately -- a zero second
view silently degrades the policy, an assert makes the misconfiguration loud -- so restore the
pad only if a DreamZero variant genuinely runs single-view. Note the old branch also sent
`observation/cartesian_position` as zeros with a `# TODO: fix cartesian`; the live client sends
the real robot-frame EE pose, which supersedes that.
