"""
Test for Rrs data: example_Rrs.csv

"""
from owt_pc_classification import water_type_OWT, water_type_PC
from qaa_models import QAAs_bbp
import numpy as np
import pandas as pd

def calculate_SPM(bbp_esti, Rrs, Wave, water_type):
    """
    Estimate Suspended Particulate Matter (SPM) concentration using bbp and green band Rrs.
    SPM=b_bp(lambda0)/b_bp^* (lambda0)
    Parameters
    ----------
    bbp_esti : pd.Series
        Estimated particulate backscattering coefficient at reference wavelength (e.g., 560 nm).
    Rrs : pd.DataFrame
        Remote sensing reflectance for each sample, columns as wavelength strings.
    Wavelength : array-like
        List of wavelengths corresponding to Rrs columns.
    water_type : int
        Optical water type (e.g., 1~4 from Jiang et al. 2021).

    Returns
    -------
    SPM_estimated : pd.Series
        Estimated SPM concentration for each sample (g/m³).
    """
    Wave = np.array(Wave, dtype=int)

    # Find index of 560 nm band
    idx_560 = int(np.argmin(abs(Wave - 560)))
    Rrs_560 = Rrs.iloc[:, idx_560]

    log_Rrs_560 = np.log10(Rrs_560)

    # POC/SPM values from green band Rrs
    coefs = [-0.973, -3.323]
    POC_SPM = 10 ** (coefs[0]*log_Rrs_560 + coefs[1])

    # Mixed type: bbp_star_lambda0 median values
    SPM_coefs_median = {1: 94.607, 2: 114.012, 3: 137.665, 4: 166.168}
    # Organic-dominated type: bbp_star_lambda0 lower values
    SPM_coefs_lower = {1: 153.918, 2: 200.974, 3: 234.130, 4: 299.709}
    # Mineral-dominated type: bbp_star_lambda0 upper values
    SPM_coefs_upper = {1: 54.203, 2: 73.433, 3: 90.311, 4: 107.418}

    SPM_estimated = []
    # Estimate SPM for each sample
    for i in range(len(bbp_esti)):
        poc_value = POC_SPM.iloc[i]
        if pd.isna(poc_value) or pd.isna(bbp_esti.iloc[i]):
            SPM_estimated.append(np.nan)
            continue

        if poc_value >= 0.12:
            coef = SPM_coefs_lower[water_type]
        elif 0.06 < poc_value < 0.12:
            coef = SPM_coefs_median[water_type]
        else:
            coef = SPM_coefs_upper[water_type]

        spm_value = coef * bbp_esti.iloc[i]
        SPM_estimated.append(spm_value)

    return pd.Series(SPM_estimated, index=bbp_esti.index)

df_path = r'\OWT-PC-main\example_Rrs.csv'
# Path to CSV file containing Specific IOP data.
df_Siops_path = r'\OWT-PC-main\SIOPs_KSM_pure_water.csv'

data = pd.read_csv(df_path)
# MERIS or OLCI bands
Wavelength = np.array([413, 443, 490, 510, 560, 620, 665, 681, 709, 754, 761, 779, 865])
wave_cols = [str(w) for w in Wavelength]
Rrs = data[wave_cols]

# Water type classification
data['OWTs'] = Rrs.apply(water_type_OWT, axis=1)
data['PC_Types'] = Rrs.apply(water_type_PC, axis=1)

SPM_results = []

# Process each water type
for water_type, group in data.groupby('OWTs'):
    if pd.isna(water_type):
        continue

    group_Rrs = group[wave_cols]

    # Estimate bbp and a at reference wavelength
    bbp_esti_lamda0 = QAAs_bbp(
        group_Rrs,
        Wavelength,
        water_type,
        df_Siops_path,
    )

    # Calculate SPM
    spm_esti = calculate_SPM(bbp_esti_lamda0, group_Rrs, Wavelength, water_type)
    temp_df = pd.DataFrame({'index': group.index, 'SPM_esti': spm_esti})
    SPM_results.append(temp_df)

if SPM_results:
    SPM_all = pd.concat(SPM_results).set_index('index').sort_index()
    data.loc[SPM_all.index, 'SPM_esti'] = SPM_all['SPM_esti']
else:
    data['SPM_esti'] = np.nan



