# MobileNet v2 x0.35 Layer Deduplication Report

This table shows the unique layer configurations and their occurrence counts.

| Layer Name | Layer Type | Occurrences |
|------------|------------|-------------|
| 15_features_features_3_Add | Add | 1 |
| 26_features_features_5_Add | Add | 2 |
| 43_features_features_8_Add | Add | 3 |
| 66_features_features_12_Add | Add | 2 |
| 89_features_features_16_Add | Add | 2 |
| 00_features_features_0_conv_Conv | Conv | 1 |
| 04_features_features_1_conv_conv_1_Conv | Conv | 1 |
| 05_features_features_2_conv_conv_0_conv_Conv | Conv | 1 |
| 09_features_features_2_conv_conv_2_Conv | Conv | 2 |
| 10_features_features_3_conv_conv_0_conv_Conv | Conv | 2 |
| 20_features_features_4_conv_conv_2_Conv | Conv | 1 |
| 21_features_features_5_conv_conv_0_conv_Conv | Conv | 3 |
| 25_features_features_5_conv_conv_2_Conv | Conv | 2 |
| 37_features_features_7_conv_conv_2_Conv | Conv | 1 |
| 44_features_features_9_conv_conv_0_conv_Conv | Conv | 4 |
| 48_features_features_9_conv_conv_2_Conv | Conv | 3 |
| 60_features_features_11_conv_conv_2_Conv | Conv | 1 |
| 61_features_features_12_conv_conv_0_conv_Conv | Conv | 3 |
| 71_features_features_13_conv_conv_2_Conv | Conv | 2 |
| 77_features_features_14_conv_conv_2_Conv | Conv | 1 |
| 78_features_features_15_conv_conv_0_conv_Conv | Conv | 3 |
| 82_features_features_15_conv_conv_2_Conv | Conv | 2 |
| 94_features_features_17_conv_conv_2_Conv | Conv | 1 |
| 95_features_features_18_conv_Conv | Conv | 1 |
| 02_features_features_1_conv_conv_0_conv_Conv | DWConv | 1 |
| 07_features_features_2_conv_conv_1_conv_Conv | DWConv | 1 |
| 12_features_features_3_conv_conv_1_conv_Conv | DWConv | 1 |
| 18_features_features_4_conv_conv_1_conv_Conv | DWConv | 1 |
| 29_features_features_6_conv_conv_1_conv_Conv | DWConv | 2 |
| 35_features_features_7_conv_conv_1_conv_Conv | DWConv | 1 |
| 52_features_features_10_conv_conv_1_conv_Conv | DWConv | 4 |
| 69_features_features_13_conv_conv_1_conv_Conv | DWConv | 2 |
| 75_features_features_14_conv_conv_1_conv_Conv | DWConv | 1 |
| 80_features_features_15_conv_conv_1_conv_Conv | DWConv | 3 |
| 98_classifier_classifier_1_Gemm | Gemm | 1 |
| 97_ReduceMean | ReduceMean | 1 |
| 01_features_features_0_relu_Relu | Relu | 2 |
| 06_features_features_2_conv_conv_0_relu_Relu | Relu | 1 |
| 11_features_features_3_conv_conv_0_relu_Relu | Relu | 4 |
| 19_features_features_4_conv_conv_1_relu_Relu | Relu | 1 |
| 30_features_features_6_conv_conv_1_relu_Relu | Relu | 5 |
| 36_features_features_7_conv_conv_1_relu_Relu | Relu | 1 |
| 45_features_features_9_conv_conv_0_relu_Relu | Relu | 8 |
| 68_features_features_13_conv_conv_0_relu_Relu | Relu | 5 |
| 76_features_features_14_conv_conv_1_relu_Relu | Relu | 1 |
| 79_features_features_15_conv_conv_0_relu_Relu | Relu | 6 |
| 96_features_features_18_relu_Relu | Relu | 1 |
