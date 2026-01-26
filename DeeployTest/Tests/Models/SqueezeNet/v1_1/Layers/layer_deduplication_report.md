# MobileNet v2 x0.35 Layer Deduplication Report

This table shows the unique layer configurations and their occurrence counts.

| Layer Name | Layer Type | Occurrences |
|------------|------------|-------------|
| 16_features_features_4_Concat | Concat | 2 |
| 24_features_features_6_Concat | Concat | 2 |
| 39_features_features_9_Concat | Concat | 2 |
| 60_features_features_12_Concat | Concat | 2 |
| 00_features_features_0_Conv | Conv | 1 |
| 03_features_features_3_squeeze_Conv | Conv | 1 |
| 10_features_features_4_squeeze_Conv | Conv | 1 |
| 12_features_features_4_expand1x1_Conv | Conv | 2 |
| 14_features_features_4_expand3x3_Conv | Conv | 2 |
| 18_features_features_6_squeeze_Conv | Conv | 1 |
| 22_features_features_6_expand3x3_Conv | Conv | 2 |
| 25_features_features_7_squeeze_Conv | Conv | 1 |
| 27_features_features_7_expand1x1_Conv | Conv | 2 |
| 33_features_features_9_squeeze_Conv | Conv | 1 |
| 37_features_features_9_expand3x3_Conv | Conv | 2 |
| 40_features_features_10_squeeze_Conv | Conv | 1 |
| 42_features_features_10_expand1x1_Conv | Conv | 2 |
| 47_features_features_11_squeeze_Conv | Conv | 1 |
| 54_features_features_12_squeeze_Conv | Conv | 1 |
| 56_features_features_12_expand1x1_Conv | Conv | 2 |
| 58_features_features_12_expand3x3_Conv | Conv | 2 |
| 61_classifier_classifier_1_Conv | Conv | 1 |
| 02_features_features_2_MaxPool | MaxPool | 1 |
| 17_features_features_5_MaxPool | MaxPool | 1 |
| 32_features_features_8_MaxPool | MaxPool | 1 |
| 63_classifier_classifier_3_ReduceMean | ReduceMean | 1 |
| 01_features_features_1_Relu | Relu | 1 |
| 04_features_features_3_squeeze_activation_Relu | Relu | 2 |
| 06_features_features_3_expand1x1_activation_Relu | Relu | 4 |
| 21_features_features_6_expand1x1_activation_Relu | Relu | 4 |
| 26_features_features_7_squeeze_activation_Relu | Relu | 2 |
| 34_features_features_9_squeeze_activation_Relu | Relu | 2 |
| 38_features_features_9_expand3x3_activation_Relu | Relu | 4 |
| 50_features_features_11_expand1x1_activation_Relu | Relu | 4 |
| 55_features_features_12_squeeze_activation_Relu | Relu | 2 |
| 62_classifier_classifier_2_Relu | Relu | 1 |
| 64_Reshape | Reshape | 1 |
