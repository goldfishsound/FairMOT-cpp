import importlib
import sys
from pathlib import Path
from typing import NamedTuple
import torch
from models.networks.pose_dla_dcn import DLASeg

sys.path.append(str(Path(__file__).resolve().parent / "FairMOT" / "src" / "lib"))
sys.path.append(str(Path(__file__).resolve().parent / "pytorch-dcnv2"))
# print(sys.path)

sys.modules["dcn_v2"] = importlib.import_module("dcn")
sys.modules["dcn_v2"].__dict__["DCN"] = sys.modules["dcn_v2"].__dict__["DCNv2"]
# print(sys.modules["dcn_v2"])
# print(sys.modules["dcn_v2"].__dict__["DCN"])
class DLASegOutput(NamedTuple):
    hm: torch.Tensor
    wh: torch.Tensor
    id: torch.Tensor
    reg: torch.Tensor


class DLASegCustom(DLASeg):
    def forward(self, x):
        return DLASegOutput(**super().forward(x)[-1])


def main():
    modelname = "fairmot_dla34"
    weights_dir = Path(__file__).resolve().parents[1] / "weights"
    weights_path = weights_dir / "fairmot_dla34.pth"
    converted_weights_path = weights_dir / "fairmot_dla34_jit.pth"
    converted_weights_cpu_path = weights_dir / "fairmot_dla34_cpu_jit.pth"
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")        
        print("Using MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

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

    print("Tracing model on device: ", device)    
    example_input = torch.rand(1, 3, 480, 864).to(device)
    traced_model = torch.jit.trace(model, example_input)

    print("Tracing model on device: cpu")
    model.eval()
    model.to("cpu")
    traced_model_cpu = torch.jit.trace(model, example_input.to("cpu"))

    print("Saving model to: ", converted_weights_path)    
    traced_model.save(converted_weights_path)
    print("Saving model to: ", converted_weights_cpu_path)
    traced_model_cpu.save(converted_weights_cpu_path)


if __name__ == "__main__":
    main()
