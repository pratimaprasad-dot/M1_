"""Values digitised from the published figures of [Hu] (C. Hu et al., IEEE TCCN vol. 12, 2026), from 400-dpi crops of the
PDF (grid-line interpolation, reading error about +-0.03 bit/s/Hz). Nothing here is produced by our code."""

# Fig. 3: SAR vs P_c (dBm), N_t = N_r = 8, M = K = 4, P^max = 20 dBm, eta_min = 10 dB, perfect CSI
FIG3_PC = [0, 5, 10, 15, 20]
FIG3 = {"opt": [4.39, 6.54, 8.35, 8.77, 8.80],          # Alg1 (inset at P_c = 5 dBm: 6.5346)
        "LMMSE": [3.40, 5.46, 7.53, 8.13, 8.46],
        "ZF": [3.10, 5.26, 7.32, 7.96, 8.27],
        "MRC/MRT": [2.65, 4.70, 6.94, 7.61, 7.99]}

# Fig. 2: converged SAR of Algorithm 1 (P_c = P^max = 20 dBm) and iterations to settle (~20 per the text)
FIG2 = {5.0: 9.27, 10.0: 8.80, 15.0: 7.30}
FIG2_ITERS = 20
TABLE_IV_PF_ITERS = 23          # Table IV, (8, 8, 4, 4), tolerance 1e-4

# Fig. 9: SAR vs eta_min (dB), P_c = P^max = 20 dBm; robust curves are worst-case SAR; eps = normalised radius
FIG9_ETA = [0, 3, 6, 9, 12, 15]
FIG9 = {"perfect": [10.57, 10.25, 9.91, 9.36, 8.78, 7.62],
        "alg3_0.03": [10.45, 10.11, 9.75, 9.20, 8.62, 7.45],
        "alg3_0.05": [10.29, 9.97, 9.63, 9.09, 8.50, 7.35],
        "nonrob_0.03": [8.47, 8.07, 7.61, 7.00, 6.27, 5.10],
        "nonrob_0.05": [8.22, 7.85, 7.33, 6.70, 5.95, 4.70]}

# Table V: GLDT-CS iterations and per-iteration time (MATLAB, i7-12700) for (8,8,4,4)
TABLE_V = {0.03: (129, 3.0539), 0.05: (116, 2.9671)}

# Internal inconsistency noted: Fig. 9 perfect-CSI at eta = 10 dB interpolates to ~9.17, while Fig. 3 (P_c = 20 dBm),
# Fig. 2 and Figs. 10-11 (P^max = 20 dBm, N_r = 8) give ~8.80 at the same nominal setting.
