# ImageNet Torchvision Quantized Models

Source: [https://docs.pytorch.org/vision/stable/models.html#quantized-models](https://docs.pytorch.org/vision/stable/models.html#quantized-models)
Torchvision version used for metadata: `0.26.0+cu126`

| Model | Original Weights | Quantized Weights | Top-1 Original | Top-1 Quant | Delta | Top-5 Original | Top-5 Quant | Size MB Original | Size MB Quant | Size Ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Inception V3 | `Inception_V3_Weights.IMAGENET1K_V1` | `Inception_V3_QuantizedWeights.IMAGENET1K_FBGEMM_V1` | 77.294 | 77.176 | -0.118 | 93.450 | 93.354 | 103.903 | 23.146 | 0.223 |
| ResNet-50 | `ResNet50_Weights.IMAGENET1K_V2` | `ResNet50_QuantizedWeights.IMAGENET1K_FBGEMM_V2` | 80.858 | 80.282 | -0.576 | 95.434 | 94.976 | 97.790 | 24.953 | 0.255 |
| ResNeXt-101 32x8d | `ResNeXt101_32X8D_Weights.IMAGENET1K_V2` | `ResNeXt101_32X8D_QuantizedWeights.IMAGENET1K_FBGEMM_V2` | 82.834 | 82.574 | -0.260 | 96.228 | 96.132 | 339.673 | 86.645 | 0.255 |
