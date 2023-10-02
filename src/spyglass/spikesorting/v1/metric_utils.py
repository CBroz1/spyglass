import spikeinterface as si
import spikeinterface.qualitymetrics as sq


def compute_isi_violation_fractions(
    waveform_extractor: si.WaveformExtractor,
    isi_threshold_ms: float = 2.0,
    min_isi_ms: float = 0.0,
):
    """Computes the fraction of interspike interval violations.

    Parameters
    ----------
    waveform_extractor: si.WaveformExtractor
        The extractor object for the recording.

    """

    # Extract the total number of spikes that violated the isi_threshold for each unit
    isi_violation_counts = sq.compute_isi_violations(
        waveform_extractor,
        isi_threshold_ms=isi_threshold_ms,
        min_isi_ms=min_isi_ms,
    ).isi_violations_count

    # Extract the total number of spikes from each unit. The number of ISIs is one less than this
    num_spikes = sq.compute_num_spikes(waveform_extractor)

    # Calculate the fraction of ISIs that are violations
    isi_viol_frac_metric = {
        str(unit_id): isi_violation_counts[unit_id] / (num_spikes[unit_id] - 1)
        for unit_id in waveform_extractor.sorting.get_unit_ids()
    }
    return isi_viol_frac_metric


def get_peak_offset(
    waveform_extractor: si.WaveformExtractor, peak_sign: str, **metric_params
):
    """Computes the shift of the waveform peak from center of window.

    Parameters
    ----------
    waveform_extractor: si.WaveformExtractor
        The extractor object for the recording.
    peak_sign: str
        The sign of the peak to compute. ('neg', 'pos', 'both')
    """
    if "peak_sign" in metric_params:
        del metric_params["peak_sign"]
    peak_offset_inds = si.get_template_extremum_channel_peak_shift(
        waveform_extractor=waveform_extractor,
        peak_sign=peak_sign,
        **metric_params,
    )
    peak_offset = {key: int(abs(val)) for key, val in peak_offset_inds.items()}
    return peak_offset


def get_peak_channel(
    waveform_extractor: si.WaveformExtractor, peak_sign: str, **metric_params
):
    """Computes the electrode_id of the channel with the extremum peak for each unit."""
    if "peak_sign" in metric_params:
        del metric_params["peak_sign"]
    peak_channel_dict = si.get_template_extremum_channel(
        waveform_extractor=waveform_extractor,
        peak_sign=peak_sign,
        **metric_params,
    )
    peak_channel = {key: int(val) for key, val in peak_channel_dict.items()}
    return peak_channel


def get_num_spikes(waveform_extractor: si.WaveformExtractor, this_unit_id: int):
    """Computes the number of spikes for each unit."""
    all_spikes = sq.compute_num_spikes(waveform_extractor)
    cluster_spikes = all_spikes[this_unit_id]
    return cluster_spikes
