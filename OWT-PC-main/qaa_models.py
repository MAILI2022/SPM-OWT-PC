import numpy as np
import pandas as pd

def IOP_QAA_v5(Rrs, Wavelength, df_SIOPs):
    """
    Estimate absorption coefficient (a) and particulate backscattering coefficient (bbp)
    using QAA-v5 algorithm (Lee et al., 2002).

    Parameters
    ----------
    Rrs : array-like
        Remote sensing reflectance spectrum at each wavelength.
    Wavelength : array-like
        Corresponding wavelengths (nm).
    df_SIOPs_path : str
        File path to CSV file containing specific inherent optical properties of Lake Kasumigaura, Japan(SIOPs),
        including columns 'wavelength(nm)', 'aw', and 'bbw'.

    Returns
    -------
    a_df : DataFrame
        Absorption coefficient spectrum (m⁻¹), columns are wavelengths.
    bbp_df : DataFrame
        Particulate backscattering coefficient spectrum (m⁻¹), columns are wavelengths.

    """
    spec_water = pd.read_csv(df_SIOPs)
    spec_water['wavelength'] = spec_water['wavelength(nm)']

    g = np.array([0.089, 0.1245])
    h = np.array([-1.146, -1.366, -0.469])

    a = np.empty_like(Rrs)
    bbp = np.empty_like(Rrs)
    bb = np.empty_like(Rrs)

    idx_443 = np.argmin(np.abs(Wavelength - 443))
    idx_490 = np.argmin(np.abs(Wavelength - 490))
    idx_55x = np.argmin(np.abs(Wavelength - 560))
    idx_670 = np.argmin(np.abs(Wavelength - 670))

    aw = np.full(len(Wavelength), np.nan)
    bbw = np.full(len(Wavelength), np.nan)


    for i, iw in enumerate(Wavelength):
        idw = np.where(spec_water.wavelength == iw)[0]
        aw[i] = float(spec_water.loc[idw, 'aw'].iloc[0])
        bbw[i] = float(spec_water.loc[idw, 'bbw'].iloc[0])

    # Step 0 1
    rrs = Rrs / (0.52 + 1.7 * Rrs)
    u = (-g[0] + np.sqrt(g[0] ** 2 + 4 * g[1] * rrs)) / (2 * g[1])  # 计算u值

    # Step 2: a_chi=anw
    chi = np.log10((rrs.iloc[:, idx_443] + rrs.iloc[:, idx_490]) /
                   (rrs.iloc[:, idx_55x] + 5 * rrs.iloc[:, idx_670] ** 2 / rrs.iloc[:, idx_490]))
    a_chi = np.power(10, h[0] + h[1] * chi + h[2] * chi ** 2)
    a0_qaa_v5 = aw[idx_55x] + a_chi

    # Step 3: bbp0
    bbp0_qaa_v5 = (u.iloc[:, idx_55x] * a0_qaa_v5) / (1 - u.iloc[:, idx_55x]) - bbw[idx_55x]

    bbp0 = bbp0_qaa_v5
    lambda0 = Wavelength[idx_55x]
    # step4 5 6: a, bb
    eta = 2 * (1 - 1.2 * np.exp(-0.9 * rrs.iloc[:, idx_443] / rrs.iloc[:, idx_55x]))
    for iw, wb in enumerate(Wavelength):
        bbp_iw = bbp0 * (lambda0 / wb)**eta
        bbp[:, iw] = bbp_iw
        bb[:, iw] = bbp[:, iw] + bbw[iw]
        a_iw = (1 - u.iloc[:, iw]) * bb[:, iw] / u.iloc[:, iw]
        a[:, iw] = a_iw
    del bbp_iw, a_iw
    a = pd.DataFrame(a, columns=Wavelength).fillna(0)
    bbp = pd.DataFrame(bbp, columns=Wavelength).fillna(0)
    return a, bbp

def IOP_QAA_v6_665(Rrs, Wavelength, df_SIOPs):
    """
    Estimate absorption coefficient (a) and particulate backscattering coefficient (bbp)
    using QAA-v6 algorithm lambda0=665nm (IOCCG,2014).

    Parameters
    ----------
    Rrs : array-like
        Remote sensing reflectance spectrum at each wavelength.
    Wavelength : array-like
        Corresponding wavelengths (nm).
    df_SIOPs_path : str
        File path to CSV file containing specific inherent optical properties of Lake Kasumigaura, Japan(SIOPs),
        including columns 'wavelength(nm)', 'aw', and 'bbw'.

    Returns
    -------
    a_df : DataFrame
        Absorption coefficient spectrum (m⁻¹), columns are wavelengths.
    bbp_df : DataFrame
        Particulate backscattering coefficient spectrum (m⁻¹), columns are wavelengths.

    """
    spec_water = pd.read_csv(df_SIOPs)
    spec_water['wavelength'] = spec_water['wavelength(nm)']

    g = np.array([0.089, 0.1245])  # 定义数组g

    idx_443 = np.argmin(np.abs(Wavelength - 443))
    idx_490 = np.argmin(np.abs(Wavelength - 490))
    idx_55x = np.argmin(np.abs(Wavelength - 560))
    idx_665 = np.argmin(np.abs(Wavelength - 665))

    aw = np.full(len(Wavelength), np.nan)
    bbw = np.full(len(Wavelength), np.nan)
    for i, iw in enumerate(Wavelength):
        idw = np.where(spec_water.wavelength == iw)[0]
        aw[i] = float(spec_water.loc[idw, 'aw'].iloc[0])
        bbw[i] = float(spec_water.loc[idw, 'bbw'].iloc[0])

    # Step 0 1
    rrs = Rrs / (0.52 + 1.7 * Rrs)
    u = (-g[0] + np.sqrt(g[0] ** 2 + 4 * g[1] * rrs)) / (2 * g[1])

    ###---- step 2
    a0_qaa_v6 = aw[idx_665] + 0.39*(Rrs.iloc[:, idx_665] / (Rrs.iloc[:, idx_443] + Rrs.iloc[:, idx_490]))**1.14
    bbp0_qaa_v6 = (u.iloc[:, idx_665] * a0_qaa_v6) / (1 - u.iloc[:, idx_665]) - bbw[idx_665]
    ###---- step 3
    lambda0 = Wavelength[idx_665]

    ###---- step 4 & 5 & 6
    eta = 2*(1 - 1.2*np.exp(-0.9*rrs.iloc[:, idx_443] / rrs.iloc[:, idx_55x]))
    a = np.empty_like(Rrs)
    bbp = np.empty_like(Rrs)
    bb = np.empty_like(Rrs)
    for iw, wb in enumerate(Wavelength):
        bbp_iw = bbp0_qaa_v6 * (lambda0 / wb)**eta
        bbp[:, iw] = bbp_iw
        bb[:, iw] = bbp[:, iw] + bbw[iw]
        a_iw = (1 - u.iloc[:, iw]) * bb[:, iw] / u.iloc[:, iw]
        a[:, iw] = a_iw
    del bbp_iw, a_iw
    a = pd.DataFrame(a, columns=Wavelength).replace([np.inf, -np.inf], np.nan).fillna(0)
    bbp = pd.DataFrame(bbp, columns=Wavelength).replace([np.inf, -np.inf], np.nan).fillna(0)

    return a, bbp

def IOP_QAA_T754(Rrs, Wavelength, df_SIOPs):
    """
    Estimate absorption coefficient (a) and particulate backscattering coefficient (bbp)
    using QAA-Turbid algorithm (Yang et al., 2013).

    Parameters
    ----------
    Rrs : array-like
        Remote sensing reflectance spectrum at each wavelength.
    Wavelength : array-like
        Corresponding wavelengths (nm).
    df_SIOPs_path : str
        File path to CSV file containing specific inherent optical properties of Lake Kasumigaura, Japan(SIOPs),
        including columns 'wavelength(nm)', 'aw', and 'bbw'.

    Returns
    -------
    a_df : DataFrame
        Absorption coefficient spectrum (m⁻¹), columns are wavelengths.
    bbp_df : DataFrame
        Particulate backscattering coefficient spectrum (m⁻¹), columns are wavelengths.

    """

    spec_water = pd.read_csv(df_SIOPs)
    spec_water['wavelength'] = spec_water['wavelength(nm)']
    a = np.empty_like(Rrs)
    bbp = np.empty_like(Rrs)
    bb = np.empty_like(Rrs)
    ###---- settings
    g = [0.089, 0.1245]
    yy = [-372.99, 37.286, 0.84]

    idx_754 = int(np.argmin(abs(Wavelength - 754)))
    idx_780 = int(np.argmin(abs(Wavelength - 780)))

    aw = np.full([len(Wavelength)], np.nan)
    bbw = np.full([len(Wavelength)], np.nan)
    for i, iw in enumerate(Wavelength):
        idw = np.where(spec_water.wavelength == iw)[0]
        aw[i] = float(spec_water.loc[idw, 'aw'].iloc[0])
        bbw[i] = float(spec_water.loc[idw, 'bbw'].iloc[0])
  ###---- step 0 & 1
    rrs = Rrs / (0.52 + 1.7 * Rrs)
    u = (-g[0] + np.sqrt(g[0]**2 + 4*g[1]*rrs)) / (2*g[1])
    a0_value = aw[idx_754]
    a0 = np.full(Rrs.shape[0], a0_value)
    ###---- step 3
    bbp0 = (u.iloc[:, idx_754] * a0) / (1 - u.iloc[:, idx_754]) - bbw[idx_754]
    lambda0 = Wavelength[idx_754]

    ###---- step 4 & 5 & 6
    beta = np.log10(u.iloc[:, idx_754] / u.iloc[:, idx_780])
    eta = yy[0] * beta**2 + yy[1] * beta + yy[2]

    for iw, wb in enumerate(Wavelength):
        bbp_iw = bbp0 * (lambda0 / wb)**eta
        bbp[:, iw] = bbp_iw
        bbp[:, iw] = bbp_iw
        bb[:, iw] = bbp[:, iw] + bbw[iw]
        a_iw = (1 - u.iloc[:, iw]) * bb[:, iw] / u.iloc[:, iw]
        a[:, iw] = a_iw
    del bbp_iw, a_iw
    a = pd.DataFrame(a, columns=Wavelength)
    bbp = pd.DataFrame(bbp, columns=Wavelength)

    return a, bbp

def IOP_QAA_T865(Rrs, Wavelength, df_SIOPs):
    """
    Estimate absorption coefficient (a) and particulate backscattering coefficient (bbp)
    using QAA-Turbid algorithm (Jiang et al., 2021).

    Parameters
    ----------
    Rrs : array-like
        Remote sensing reflectance spectrum at each wavelength.
    Wavelength : array-like
        Corresponding wavelengths (nm).
    df_SIOPs_path : str
        File path to CSV file containing specific inherent optical properties of Lake Kasumigaura, Japan(SIOPs),
        including columns 'wavelength(nm)', 'aw', and 'bbw'.

    Returns
    -------
    a_df : DataFrame
        Absorption coefficient spectrum (m⁻¹), columns are wavelengths.
    bbp_df : DataFrame
        Particulate backscattering coefficient spectrum (m⁻¹), columns are wavelengths.

    """
    Wavelength = np.atleast_1d(Wavelength)
    spec_water = pd.read_csv(df_SIOPs)
    spec_water['wavelength'] = spec_water['wavelength(nm)']

    a = np.empty_like(Rrs)
    bbp = np.empty_like(Rrs)
    bb = np.empty_like(Rrs)
    ###---- settings
    g = [0.089, 0.1245]
    yy = [-372.99, 37.286, 0.84]

    idx_754 = int(np.argmin(abs(Wavelength - 754)))
    idx_779 = int(np.argmin(abs(Wavelength - 779)))
    idx_865 = int(np.argmin(abs(Wavelength - 865)))

    aw = np.full([len(Wavelength)], np.nan)
    bbw = np.full([len(Wavelength)], np.nan)
    for i, iw in enumerate(Wavelength):
        idw = np.where(spec_water.wavelength == iw)[0]

        aw[i] = float(spec_water.loc[idw, 'aw'].iloc[0])
        bbw[i] = float(spec_water.loc[idw, 'bbw'].iloc[0])
    ###---- step 0 & 1
    rrs = Rrs / (0.52 + 1.7 * Rrs)
    u = (-g[0] + np.sqrt(g[0]**2 + 4*g[1]*rrs)) / (2*g[1])
    ###---- step 2
    a0_value = aw[idx_865]
    a0 = np.full(Rrs.shape[0], a0_value)
    # ###---- step 3
    bbp0 = (u.iloc[:, idx_865] * a0) / (1 - u.iloc[:, idx_865]) - bbw[idx_865]
    lambda0 = Wavelength[idx_865]

    ###---- step 4 & 5 & 6
    beta = np.log10(u.iloc[:, idx_754] / u.iloc[:, idx_779])
    eta = yy[0] * beta**2 + yy[1] * beta + yy[2]

    for iw, wb in enumerate(Wavelength):
        bbp_iw = bbp0 * (lambda0 / wb)**eta
        bbp[:, iw] = bbp_iw
        bbp[:, iw] = bbp_iw
        bb[:, iw] = bbp[:, iw] + bbw[iw]
        a_iw = (1 - u.iloc[:, iw]) * bb[:, iw] / u.iloc[:, iw]
        a[:, iw] = a_iw
    del bbp_iw, a_iw
    a = pd.DataFrame(a, columns=Wavelength)
    bbp = pd.DataFrame(bbp, columns=Wavelength)
    return a, bbp

def QAAs_bbp(Rrs, Wavelength, water_type, df_Siops):
    """
    Estimate absorption coefficient (a) and particulate backscattering coefficient (bbp)
    at a reference wavelength based on water type.

    Parameters
    ----------
    Rrs : array-like
        Remote sensing reflectance spectrum.
    Wavelength : array-like
        Corresponding wavelengths (nm) of Rrs.
    water_type : int
        Optical water type classified using Jiang et al. (2021) method.
        1: clear water
        2: moderate turbidity water
        3: high turbidity water
        4: extremely turbid water
    df_Siops : DataFrame
        Specific inherent optical properties (SIOPs) dataset for inversion.

    Returns
    -------
    bbp_lamda0 : float
        Estimated particulate backscattering coefficient at reference wavelength (m⁻¹).
    a_lamda0 : float
        Estimated absorption coefficient at reference wavelength (m⁻¹).
    """
    if pd.isna(water_type):
        return np.nan, np.nan

    Wavelength = np.array(Wavelength, dtype=int)

    bbp_lamda0 = None
    a_lamda0 = None
    lamda0 = None

    if water_type == 1:
        # Clear water: QAA-v5, λ₀ = 560 nm(Lee et al., 2002)
        lamda0 = 560
        _, bbp_esti = IOP_QAA_v5(Rrs, Wavelength, df_Siops)
        bbp_lamda0 = bbp_esti[lamda0]

    elif water_type == 2:
        # Moderate turbidity: QAA-v6, λ₀ = 665 nm(IOCCG, 2014)
        lamda0 = 665
        _, bbp_esti = IOP_QAA_v6_665(Rrs, Wavelength, df_Siops)
        bbp_lamda0 = bbp_esti[lamda0]

    elif water_type == 3:
        # High turbidity: QAA-Turbid (Yang et al., 2013), λ₀ = 754 nm
        lamda0 = 754
        _, bbp_esti = IOP_QAA_T754(Rrs, Wavelength, df_Siops)
        bbp_lamda0 = bbp_esti[lamda0]

    elif water_type == 4:
        # Extremely turbid: QAA-Turbid (Jiang et al., 2021), λ₀ = 865 nm
        lamda0 = 865
        _, bbp_esti = IOP_QAA_T865(Rrs, Wavelength, df_Siops)
        bbp_lamda0 = bbp_esti[lamda0]
    if bbp_lamda0 is None:
        raise ValueError(f"Estimated IOPs do not contain wavelength {lamda0}")

    return bbp_lamda0

