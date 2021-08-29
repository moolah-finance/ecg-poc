def _signal_power_instant_get(psd, frequency_band):

    indices = np.logical_and(
        psd["Frequency"] >= frequency_band[0], psd["Frequency"] < frequency_band[1]
    ).values  # pylint: disable=no-member

    out = {}
    out["{:.2f}-{:.2f}Hz".format(frequency_band[0], frequency_band[1])] = np.trapz(
        y=psd["Power"][indices], x=psd["Frequency"][indices]
    )
    out = {key: np.nan if value == 0.0 else value for key, value in out.items()}
    return out

