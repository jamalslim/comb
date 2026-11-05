# Stroke-CoMB results: UCI optical digits (scikit-learn load_digits, 1797 images, 8x8 upsampled to 28x28); 1200/297/300 split, seed 2026

Held-out k-point loss: no dependence 2.792, train-vs-test floor 0.350. Two-sample AUC floor 0.479. Test digits: classifier confidence 0.93, class entropy 3.30 bits. Geometry errors are W1/sd against held-out digits; the train-vs-test floor is at most 0.13 for every descriptor.

| model | params | k-point loss | kappa 2 / 3 / pairs / path | height | width | closure | winding | turning | AUC | confidence | classes (bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CoMB, chi=4 (k-point + path), seed 0 | 3136 | 1.077 | 75% / 78% / 54% / 91% | 1.44 | 0.47 | 0.19 | 0.35 | 1.83 | 0.901 | 0.70 | 2.94 |
| CoMB, chi=4 (k-point), seed 0 | 3136 | 0.935 | 80% / 83% / 63% / 82% | 2.23 | 0.51 | 0.19 | 0.30 | 1.67 | 0.911 | 0.71 | 2.97 |
| dephased twin, chi=4 (k-point + path), seed 0 | 3136 | 1.181 | 73% / 77% / 43% / 91% | 1.66 | 0.49 | 0.22 | 0.33 | 1.73 | 0.905 | 0.71 | 2.97 |
| dephased twin, chi=4 (k-point), seed 0 | 3136 | 1.082 | 75% / 79% / 52% / 80% | 2.32 | 0.62 | 0.21 | 0.39 | 1.85 | 0.913 | 0.72 | 2.97 |
| classical HMM, chi=16 (k-point), seed 0 | 28672 | 0.409 | 99% / 99% / 94% / 88% | 1.60 | 0.44 | 0.25 | 0.24 | 0.24 | 0.813 | 0.77 | 3.19 |
| classical HMM, chi=4 (k-point + path), seed 0 | 4480 | 0.681 | 91% / 91% / 74% / 91% | 1.58 | 0.56 | 0.34 | 0.28 | 1.10 | 0.880 | 0.71 | 3.06 |
| classical HMM, chi=4 (k-point + path), seed 1 | 4480 | 0.688 | 92% / 91% / 72% / 91% | 1.53 | 0.53 | 0.31 | 0.24 | 1.08 | 0.861 | 0.71 | 3.07 |
| classical HMM, chi=4 (k-point + path), seed 2 | 4480 | 0.759 | 89% / 88% / 69% / 92% | 1.50 | 0.51 | 0.29 | 0.27 | 1.17 | 0.902 | 0.71 | 3.00 |
| classical HMM, chi=4 (k-point), seed 0 | 4480 | 0.589 | 95% / 94% / 79% / 67% | 2.19 | 0.87 | 0.59 | 0.18 | 0.80 | 0.891 | 0.74 | 3.11 |
| general coherent memory, chi=4 (any circuit, 1 env. levels; k-point + path), seed 0 | 28672 | 0.580 | 97% / 96% / 75% / 90% | 1.68 | 0.49 | 0.35 | 0.30 | 0.66 | 0.892 | 0.72 | 3.04 |
| general coherent memory, chi=4 (any circuit, 16 env. levels; k-point + path), seed 0 | 458752 | 0.577 | 97% / 96% / 75% / 90% | 1.68 | 0.49 | 0.34 | 0.31 | 0.67 | 0.888 | 0.71 | 3.05 |
| general coherent memory, chi=4 (any circuit, 16 env. levels; k-point + path), seed 1 | 458752 | 0.575 | 97% / 96% / 75% / 90% | 1.70 | 0.54 | 0.39 | 0.29 | 0.71 | 0.899 | 0.72 | 3.03 |
| general coherent memory, chi=4 (any circuit, 16 env. levels; k-point + path), seed 2 | 458752 | 0.579 | 97% / 96% / 75% / 90% | 1.70 | 0.52 | 0.37 | 0.30 | 0.68 | 0.893 | 0.71 | 3.02 |
| general memory, dephased, chi=4 (any circuit, 16 env. levels; k-point + path), seed 0 | 458752 | 0.705 | 92% / 91% / 69% / 90% | 1.60 | 0.71 | 0.40 | 0.29 | 1.05 | 0.902 | 0.70 | 3.04 |
| general memory, dephased, chi=4 (any circuit, 16 env. levels; k-point + path), seed 1 | 458752 | 0.682 | 93% / 92% / 71% / 90% | 1.66 | 0.59 | 0.35 | 0.30 | 1.07 | 0.878 | 0.70 | 3.05 |
| general memory, dephased, chi=4 (any circuit, 16 env. levels; k-point + path), seed 2 | 458752 | 0.666 | 93% / 92% / 72% / 90% | 1.62 | 0.68 | 0.40 | 0.31 | 1.02 | 0.890 | 0.70 | 3.04 |

## Mean +- sd over seeds

| model | seeds | validation loss | k-point loss | kappa 2 / 3 / pairs / path | height | turning | AUC |
|---|---|---|---|---|---|---|---|
| general coherent memory, chi=4 (any circuit, 16 env. levels; k-point + path) | 3 | 0.720 +- 0.005 | 0.577 +- 0.002 | 97 +- 0 / 96 +- 0 / 75 +- 0 / 90 +- 0 | 1.69 +- 0.01 | 0.68 +- 0.02 | 0.893 +- 0.006 |
| general memory, dephased, chi=4 (any circuit, 16 env. levels; k-point + path) | 3 | 0.840 +- 0.008 | 0.684 +- 0.020 | 93 +- 1 / 92 +- 1 / 71 +- 1 / 90 +- 0 | 1.63 +- 0.03 | 1.04 +- 0.03 | 0.890 +- 0.012 |
| classical HMM, chi=4 (k-point + path) | 3 | 0.842 +- 0.038 | 0.710 +- 0.043 | 91 +- 2 / 90 +- 2 / 72 +- 2 / 92 +- 0 | 1.54 +- 0.04 | 1.11 +- 0.05 | 0.881 +- 0.021 |
