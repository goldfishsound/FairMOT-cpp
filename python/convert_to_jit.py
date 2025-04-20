import importlib
import sys
from pathlib import Path
from typing import NamedTuple

import torch

sys.path.append(str(Path(__file__).resolve().parent / "FairMOT" / "src" / "lib"))
sys.path.append(str(Path(__file__).resolve().parent / "pytorch-dcnv2"))
# print(sys.path)

sys.modules["dcn_v2"] = importlib.import_module("dcn")
sys.modules["dcn_v2"].__dict__["DCN"] = sys.modules["dcn_v2"].__dict__["DCNv2"]
# print(sys.modules["dcn_v2"])
# print(sys.modules["dcn_v2"].__dict__["DCN"])

from models.networks.pose_dla_dcn import DLASeg


class DLASegOutput(NamedTuple):
    hm: torch.Tensor
    wh: torch.Tensor
    id: torch.Tensor
    reg: torch.Tensor


class DLASegCustom(DLASeg):
    def forward(self, x):
        return DLASegOutput(**super().forward(x)[-1])

def main():
    weights_dir = Path(__file__).resolve().parents[1] / "weights"
    weights_path = weights_dir / "fairmot_dla34.pth"
    converted_weights_path = weights_dir / "fairmot_dla34_jit.pth"
    scripted_model_path = weights_dir / "scripted_fairmot_dla34_jit.pth"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps")

    model = DLASegCustom(
        "dla34",
        {"hm": 1, "wh": 4, "id": 128, "reg": 2},
        pretrained=False,
        down_ratio=4,
        final_kernel=1,
        last_level=5,
        head_conv=256,
    )
    checkpoint = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    model = model.to(device)
    model.eval()

    # scripted_model = torch.jit.script(model)
    # print("Saving scripted model to: ", scripted_model_path)
    # scripted_model.save(scripted_model_path)
    
    example_input = torch.rand(1, 3, 480, 864).to(device)
    print("Tracing model")
    traced_module = torch.jit.trace(model, example_input)
    
    # print("Traced module")
    # print(traced_module)
    
    print("Saving model")
    traced_module.save(converted_weights_path)
    # print("Running model.forward")
    # output = traced_module.forward(example_input)
    # print("Traced output", output)
    # print("Model graph")
    # print(traced_module.graph)
    # print("Model code")
    # print(traced_module.code)


if __name__ == "__main__":
    main()
