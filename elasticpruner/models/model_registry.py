# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0
# Modified by dodowon, 2026. See THIRD_PARTY_NOTICES.md for details.

from elasticpruner.models.composer_llama import ComposerMosaicLlama
from elasticpruner.models.composer_pythia import ComposerMosaicPythia
from elasticpruner.models.composer_llama3 import ComposerMosaicLlama3
# from elasticpruner.models.composer_llama_reduction import ComposerMosaicLlamaReduction

COMPOSER_MODEL_REGISTRY = {
    'mosaic_llama_125m': ComposerMosaicLlama,
    'mosaic_llama_370m': ComposerMosaicLlama,
    "mosaic_llama_1.3b": ComposerMosaicLlama,
    "mosaic_llama_3b": ComposerMosaicLlama,
    'mosaic_llama_7b': ComposerMosaicLlama,
    'mosaic_llama_13b': ComposerMosaicLlama,
    'mosaic_llama_30b': ComposerMosaicLlama,
    'mosaic_llama_65b': ComposerMosaicLlama,
    'mosaic_pythia_70m': ComposerMosaicPythia,
    'mosaic_pythia_160m': ComposerMosaicPythia,
    'mosaic_pythia_410m': ComposerMosaicPythia,
    'mosaic_pythia_1.4b': ComposerMosaicPythia,
    'mosaic_llama2_125m': ComposerMosaicLlama,
    'mosaic_llama2_370m': ComposerMosaicLlama,
    "mosaic_llama2_1.3b": ComposerMosaicLlama,
    "mosaic_llama2_3b": ComposerMosaicLlama,
    'mosaic_llama2_7b': ComposerMosaicLlama,
    'mosaic_llama2_13b': ComposerMosaicLlama,
    'mosaic_together_3b': ComposerMosaicPythia,
    'llama3_8b':ComposerMosaicLlama3,
    # 'reduction_llama_125m': ComposerMosaicLlamaReduction,
}
