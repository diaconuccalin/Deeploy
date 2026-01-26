# MobileNet v2 x0.35 Layer Deduplication Report

This table shows the unique layer configurations and their occurrence counts.

| Layer Name | Layer Type | Occurrences |
|------------|------------|-------------|
| 00_features_conv0_conv0_0_Conv | Conv | 1 |
| 04_features_conv1_pointwise_pointwise_0_Conv | Conv | 1 |
| 08_features_conv2_pointwise_pointwise_0_Conv | Conv | 1 |
| 12_features_conv3_pointwise_pointwise_0_Conv | Conv | 1 |
| 16_features_conv4_pointwise_pointwise_0_Conv | Conv | 1 |
| 20_features_conv5_pointwise_pointwise_0_Conv | Conv | 1 |
| 24_features_conv6_pointwise_pointwise_0_Conv | Conv | 1 |
| 32_features_conv8_pointwise_pointwise_0_Conv | Conv | 5 |
| 48_features_conv12_pointwise_pointwise_0_Conv | Conv | 1 |
| 52_features_conv13_pointwise_pointwise_0_Conv | Conv | 1 |
| 02_features_conv1_depthwise_depthwise_0_Conv | DWConv | 1 |
| 06_features_conv2_depthwise_depthwise_0_Conv | DWConv | 1 |
| 10_features_conv3_depthwise_depthwise_0_Conv | DWConv | 1 |
| 14_features_conv4_depthwise_depthwise_0_Conv | DWConv | 1 |
| 18_features_conv5_depthwise_depthwise_0_Conv | DWConv | 1 |
| 22_features_conv6_depthwise_depthwise_0_Conv | DWConv | 1 |
| 38_features_conv10_depthwise_depthwise_0_Conv | DWConv | 5 |
| 46_features_conv12_depthwise_depthwise_0_Conv | DWConv | 1 |
| 50_features_conv13_depthwise_depthwise_0_Conv | DWConv | 1 |
| 55_Flatten | Flatten | 1 |
| 56_classifier_classifier_1_Gemm | Gemm | 1 |
| 54_avgpool_ReduceMean | ReduceMean | 1 |
| 01_features_conv0_conv0_2_Relu | Relu | 2 |
| 05_features_conv1_pointwise_pointwise_2_Relu | Relu | 1 |
| 07_features_conv2_depthwise_depthwise_2_Relu | Relu | 1 |
| 11_features_conv3_depthwise_depthwise_2_Relu | Relu | 3 |
| 15_features_conv4_depthwise_depthwise_2_Relu | Relu | 1 |
| 17_features_conv4_pointwise_pointwise_2_Relu | Relu | 3 |
| 23_features_conv6_depthwise_depthwise_2_Relu | Relu | 1 |
| 39_features_conv10_depthwise_depthwise_2_Relu | Relu | 11 |
| 47_features_conv12_depthwise_depthwise_2_Relu | Relu | 1 |
| 53_features_conv13_pointwise_pointwise_2_Relu | Relu | 3 |
