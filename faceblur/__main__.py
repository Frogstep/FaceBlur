import sys

import torch  # noqa: F401  (load PyTorch's CUDA DLLs before onnxruntime)

from .cli import main

sys.exit(main())
