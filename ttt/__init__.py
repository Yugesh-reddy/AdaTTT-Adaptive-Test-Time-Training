"""
Efficient TTT: Adaptive Test-Time Training for VQA.

Core package providing model components, TTT adaptation logic,
adaptive routing, dataset classes, and evaluation metrics.
"""

from ttt.models import (
    FullVQAModel,
    FusionModule,
    ConfidenceGate,
    PredictionHead,
    MaskedPatchProjection,
    RotationHead,
    load_frozen_vit,
    load_frozen_bert,
)
from ttt.ttt_loop import TTTAdapter
from ttt.gate import AdaptiveRouter
from ttt.data import VQADataset, VizWizDataset, Memotion2Dataset, build_memotion2_label_map
from ttt.metrics import (
    vqa_accuracy,
    accuracy_by_question_type,
    pareto_frontier,
    compute_gate_statistics,
    mcnemar_test,
)
from ttt.utils import (
    load_config,
    save_json,
    load_json,
    save_checkpoint,
    load_checkpoint,
    setup_logging,
    get_device,
    count_parameters,
    set_seed,
)

__version__ = "0.1.0"
