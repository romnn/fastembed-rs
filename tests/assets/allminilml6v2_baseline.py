"""Print the baseline `test_allminilml6v2_match_python_counterpart` checks fastembed against.

Run it on the snapshot of `Qdrant/all-MiniLM-L6-v2-onnx` fastembed downloaded:

    uv run --no-project --with onnxruntime==1.31.0 --with tokenizers --with numpy \\
        python -I tests/assets/allminilml6v2_baseline.py <snapshot directory>
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime
from tokenizers import Tokenizer

snapshot = Path(sys.argv[1])
text = (Path(__file__).parent / "sample_text.txt").read_text()
# fastembed caps the requested 384 tokens at the tokenizer's declared `model_max_length`.
model_max_length = json.loads((snapshot / "tokenizer_config.json").read_text())["model_max_length"]
tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
tokenizer.enable_truncation(max_length=min(384, int(model_max_length)))
tokenizer.no_padding()
encoding = tokenizer.encode(text)
feeds = {
    "input_ids": np.asarray([encoding.ids], dtype=np.int64),
    "attention_mask": np.asarray([encoding.attention_mask], dtype=np.int64),
    "token_type_ids": np.asarray([encoding.type_ids], dtype=np.int64),
}
session = onnxruntime.InferenceSession(str(snapshot / "model.onnx"), providers=["CPUExecutionProvider"])
names = {value.name for value in session.get_inputs()}
hidden = session.run(None, {name: value for name, value in feeds.items() if name in names})[0]
mask = feeds["attention_mask"][:, :, None]
pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
normalized = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
print([float(f"{value:.8g}") for value in normalized[0, :10]])
