import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from transformers import AutoModel
from transformer_perf.models.parser import TransformerGraphParser


def main():

    print("Loading model...")

    model = AutoModel.from_pretrained("bert-base-uncased")

    parser = TransformerGraphParser(model)

    dummy_input = torch.randn(1, 16, 768)

    graph = parser.parse(dummy_input)

    print("\nParsed Operations:\n")

    for node in graph:

        print("Operation:", node.op_type)
        print("Input shape:", node.input_shapes)
        print("Output shape:", node.output_shape)
        print("FLOPs:", node.flops)
        print("Memory bytes:", node.memory_bytes)
        print("-" * 40)


if __name__ == "__main__":
    main()