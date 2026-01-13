import numpy as np

def water_type_OWT(row):
    """
    divide water into 4 types according to spectra information

        water_type : int
        Optical water type classified using Jiang et al. (2021) method.
        1: clear water
        2: moderate turbidity water
        3: high turbidity water
        4: extremely turbid water

    :param row: A single row of the data - MERIS or OLCI bands
    :return: OWT water type
    Jiang'2021 water type classification
    """

    Rrs_490, Rrs_560, Rrs_620, Rrs_754 = row['490'], row['560'], row['620'], row['754']

    if Rrs_490 >= 0:
        # SPM for Clear water
        if Rrs_490 > Rrs_560 > 0:
            water_type = 1
        else:
            # SPM for moderate turbidity water
            if Rrs_490 > Rrs_620 > 0:
                water_type = 2
            else:
                # SPM for extreme turbidity water
                if Rrs_754 > Rrs_490 > 0 and Rrs_754 > 0.01:
                    water_type = 4
                else:
                    # SPM for high turbidity water
                    if not np.isnan(Rrs_754):
                        water_type = 3
                    else:
                        water_type = None
    else:
        water_type = None
    return water_type

def water_type_PC(row):
    """
    Classify water type based on Rrs(560) and estimated POC/SPM value.
    :param row: A single row of the data - MERIS or OLCI bands
    :return: PC water type
    Jiang'2021 water type classification
    POC/SPM value from Teng et al.,2025 green band algorithm
    """
    try:
        Rrs_values = {
            '560': float(row['560']),
        }
    except (KeyError, ValueError):
        return np.nan

    # Config
    coefs = [-0.973, -3.323]
    low_thresh = 0.06
    high_thresh = 0.12

    if Rrs_values['560'] <= 0:
        return np.nan

    log_Rrs_560 = np.log10(Rrs_values['560'])

    # POC/SPM values from green band Rrs
    POC_SPM = 10 ** (coefs[0] * log_Rrs_560 + coefs[1])

    if POC_SPM >= high_thresh:
        return 'Organic-dominated'
    elif low_thresh < POC_SPM < high_thresh:
        return 'Mixed'
    elif POC_SPM <= low_thresh:
        return 'Mineral-dominated'

    return np.nan