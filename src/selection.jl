const default_far_thresh = 1

function mass_cut(
    df;
    m1_min=default_m1_min,
    m2_min=default_m2_min,
    m2_max=default_m2_max,
    thresh=0.5
)
    evt_df = groupby(df, :gwname)

    df = subset(
        evt_df,
        [:mass_1_source, :mass_2_source] =>
        ((m1s, m2s) ->
            sum(
                (m1s .> m1_min) .&&
                (m2s .> m2_min) .&&
                (m2s .< m2_max)
            ) / length(m1s) > thresh
        ),
        ungroup=true
    )

    # Retain only posterior samples inside the mass domain
    df = df[
        (df.mass_1_source .> m1_min) .&&
        (df.mass_2_source .> m2_min) .&&
        (df.mass_2_source .< m2_max),
        :
    ]

    return df
end


function far_cut(df; far_thresh=default_far_thresh)
    evt_df = groupby(df, :gwname)

    df = subset(
        evt_df,
        :far => fars -> all(fars .< far_thresh),
        ungroup=true
    )

    return df
end


"""
function cut_selection_table(
    df,
    df_pe;
    m1_min=default_m1_min,
    m2_min=default_m2_min,
    m2_max=default_m2_max,
    far_thresh=default_far_thresh,
    snr_thresh=10,
    thresh=0.9
)

    # Get mass samples from df_pe
    m1s = df_pe["Source_Frame_m1"]
    m2s = df_pe["Source_Frame_m2"]

    # Compute the fraction of samples satisfying:
    # m1 > m1_min and m2_min < m2 < m2_max
    keep = [
        sum(
            (m1s[:, i] .> m1_min) .&
            (m2s[:, i] .> m2_min) .&
            (m2s[:, i] .< m2_max)
        ) / size(m1s, 1) > thresh

        for i in 1:size(m1s, 2)
    ]

    # Filter rows of df accordingly
    filtered_df = DataFrame([
        row
        for (i, row) in enumerate(eachrow(df))
        if keep[i]
    ])

    return filtered_df
end
"""


function resample_selection(log_wt_fn, samples, log_pdraw, Ndraw, Nsamp)
    log_pdraw_new_unnorm = [log_wt_fn(s...) for s in samples]

    log_norm =
        logsumexp(log_pdraw_new_unnorm .- log_pdraw) -
        log(Ndraw)

    log_pdraw_new = log_pdraw_new_unnorm .- log_norm

    log_wts = log_pdraw_new .- log_pdraw
    wts = exp.(log_wts .- logsumexp(log_wts))

    inds = sample(
        1:length(samples),
        Weights(wts),
        Nsamp,
        replace=false
    )

    return samples[inds], log_pdraw_new[inds], Nsamp
end