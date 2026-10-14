import torch
import torch.nn.functional as F

def padding_old(x, D, K, S):
    import numpy as np
    L = x.shape[-1]
    for _ in range(D):
        if L < K:
            L = 1
        else:
            L = 1 + np.ceil((L - K) / S)
    for _ in range(D):
        L = (L - 1) * S + K
    L = int(L)
    x = F.pad(x, (0, L - x.shape[-1]))
    return x

def padding_new(x, D, K, S):
    L = x.size(-1)
    for _ in range(D):
        # We assume L >= K for audio tasks
        L = 1 + (L - K + S - 1) // S
    for _ in range(D):
        L = (L - 1) * S + K
    pad_len = L - x.size(-1)
    x = F.pad(x, (0, pad_len))
    return x

class DummyModelOld(torch.nn.Module):
    def forward(self, x):
        return padding_old(x, 8, 4, 2)

class DummyModelNew(torch.nn.Module):
    def forward(self, x):
        return padding_new(x, 8, 4, 2)

for Model, name in [(DummyModelOld, "old"), (DummyModelNew, "new")]:
    model = Model()
    x = torch.randn(1, 1, 16000)
    torch.onnx.export(
        model, x, f"test_{name}.onnx",
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {2: "time"}, "output": {2: "time"}},
        opset_version=14
    )
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(f"test_{name}.onnx", providers=['CPUExecutionProvider'])
    try:
        out = sess.run(None, {"input": np.random.randn(1, 1, 58240).astype(np.float32)})
        print(f"{name} success: out shape {out[0].shape}")
    except Exception as e:
        print(f"{name} failed: {e}")
