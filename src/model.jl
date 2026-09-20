square(x) = x*x

raw"""
log_dNdm(mbh, alpha, mtr, mbhmax, sigma)

Returns the log of the black hole mass function at mbh given parameters
describing the initial-final mass relation.
"""
function log_dNdm(mbh, alpha, mtr, mbhmax, sigma)
c = 1 / (4 * (mbhmax - mtr))
m_max = 2 * mbhmax - mtr

if mbh < mbhmax
    a = mbhmax - mbh
    b = c
    x = a*a / (4*sigma*sigma)

    log_wt =
        log(sqrt(a*pi/(2*b)) / (4*sigma)) +
        log(
            besselix(-0.25, x) +
            besselix(0.25, x)
        )

    if mbh < mtr
        mlow = mbh
        mhigh = m_max + sqrt((mbhmax - mbh) / c)
        log_wt_low = 0.0
        log_wt_high = log_wt
    else
        d = sqrt((mbhmax - mbh) / c)
        mlow = m_max - d
        mhigh = m_max + d
        log_wt_low = log_wt
        log_wt_high = log_wt
    end
else
    a = mbh - mbhmax
    b = c
    x = a*a / (4*sigma*sigma)

    log_wt =
        log(sqrt(a/(b*pi)) / (4*sigma)) -
        2*x +
        log(besselkx(0.25, x))

    mlow = m_max
    mhigh = m_max
    log_wt_low = log_wt
    log_wt_high = log_wt
end

logplow = -alpha * log(mlow / m_max)
logphigh = -alpha * log(mhigh / m_max)

return logaddexp(
    log_wt_low + logplow,
    log_wt_high + logphigh
)

end

function _c(mtr, mbhmax)
1 / (4 * (mbhmax - mtr))
end

function _mini_max(mtr, mbhmax)
2 * mbhmax - mtr
end

function mrem_of_mini(mini, mtr, mbhmax)
if mini < mtr
return mini
else
c = _c(mtr, mbhmax)
m_max = _mini_max(mtr, mbhmax)

    return mbhmax - c * (mini - m_max)^2
end

end

function mini_left_of_mrem(mrem, mtr, mbhmax)
if mrem > mbhmax
return _mini_max(mtr, mbhmax)
elseif mrem < mtr
return mrem
else
c = _c(mtr, mbhmax)
m_max = _mini_max(mtr, mbhmax)
d = sqrt((mbhmax - mrem) / c)

    return m_max - d
end

end

function mini_right_of_mrem(mrem, mtr, mbhmax)
if mrem > mbhmax
return _mini_max(mtr, mbhmax)
else
c = _c(mtr, mbhmax)
m_max = _mini_max(mtr, mbhmax)
d = sqrt((mbhmax - mrem) / c)

    return m_max + d
end

end

function log_trapz(xs, log_ys)
log_dx = log.(xs[2] .- xs[1])

log_wts =
    log(0.5) .+
    log_dx .+
    logaddexp.(
        log_ys[1:end-1],
        log_ys[2:end]
    )

return logsumexp(log_wts)

end

function mini_integral_log(
mrem,
alpha,
mtr,
mbhmax,
sigma
)
m_max = _mini_max(mtr, mbhmax)

mrem_low = max(0.01, mrem - 5*sigma)
mrem_high = mrem + 5*sigma

if mrem_high < mbhmax
    mill = mini_left_of_mrem(
        mrem_low,
        mtr,
        mbhmax
    )

    milh = mini_left_of_mrem(
        mrem_high,
        mtr,
        mbhmax
    )

    mi = range(
        mill,
        stop = milh,
        length = 128
    )

    log_ys = [
        -alpha * log(m / m_max) +
        logpdf(
            Normal(
                mrem_of_mini(
                    m,
                    mtr,
                    mbhmax
                ),
                sigma
            ),
            mrem
        )
        for m in mi
    ]

    log_i1 = log_trapz(mi, log_ys)

    mirl = mini_right_of_mrem(
        mrem_high,
        mtr,
        mbhmax
    )

    mirh = mini_right_of_mrem(
        mrem_low,
        mtr,
        mbhmax
    )

    mi = range(
        mirl,
        stop = mirh,
        length = 128
    )

    log_ys = [
        -alpha * log(m / m_max) +
        logpdf(
            Normal(
                mrem_of_mini(
                    m,
                    mtr,
                    mbhmax
                ),
                sigma
            ),
            mrem
        )
        for m in mi
    ]

    log_i2 = log_trapz(mi, log_ys)

    return logaddexp(log_i1, log_i2)

elseif mrem_low < mbhmax
    ml = mini_left_of_mrem(
        mrem_low,
        mtr,
        mbhmax
    )

    mr = mini_right_of_mrem(
        mrem_low,
        mtr,
        mbhmax
    )

    mi = range(
        ml,
        stop = mr,
        length = 128
    )

    log_ys = [
        -alpha * log(m / m_max) +
        logpdf(
            Normal(
                mrem_of_mini(
                    m,
                    mtr,
                    mbhmax
                ),
                sigma
            ),
            mrem
        )
        for m in mi
    ]

    return log_trapz(mi, log_ys)

else
    mr = max(
        0.01,
        mbhmax - 5*sigma
    )

    ml = mini_left_of_mrem(
        mr,
        mtr,
        mbhmax
    )

    mr = mini_right_of_mrem(
        mr,
        mtr,
        mbhmax
    )

    mi = range(
        ml,
        stop = mr,
        length = 128
    )

    log_ys = [
        -alpha * log(m / m_max) +
        logpdf(
            Normal(
                mrem_of_mini(
                    m,
                    mtr,
                    mbhmax
                ),
                sigma
            ),
            mrem
        )
        for m in mi
    ]

    return log_trapz(mi, log_ys)
end

end

function make_log_dNdm_gridded_efficient(
alpha,
mtr,
mbhmax,
sigma;
mmin = 10.0,
mmax = 75.0
)
dm = 10*sigma / 128

ms = collect(
    mmin:dm:mmax
)

log_dNdms = [
    mini_integral_log(
        m,
        alpha,
        mtr,
        mbhmax,
        sigma
    )
    for m in ms
]

function log_dNdm_gridded(m)
    interp1d(
        m,
        ms,
        log_dNdms
    )
end

return log_dNdm_gridded

end

function make_log_dNdm_gridded(
mgrid,
alpha,
mtr,
mbhmax,
sigma;
mmin = 10.0,
mmax = 100.0
)
ms = mgrid

m_max = _mini_max(
    mtr,
    mbhmax
)

log_dNdms = [
    log_trapz(
        ms,
        [
            -alpha * log(mi / m_max) +
            logpdf(
                Normal(
                    mrem_of_mini(
                        mi,
                        mtr,
                        mbhmax
                    ),
                    sigma
                ),
                mr
            )
            for mi in ms
        ]
    )
    for mr in ms
]

function log_dNdm_gridded(m)
    interp1d(
        m,
        ms,
        log_dNdms
    )
end

return log_dNdm_gridded

end

function make_log_dNdm_peak_gridded(
mgrid,
mu_peak,
sigma_peak
)
ms = mgrid

log_dNdms = logpdf(
    Normal(
        mu_peak,
        sigma_peak
    ),
    ms
)

function log_dNdm_gridded(m)
    interp1d(
        m,
        ms,
        log_dNdms
    )
end

return log_dNdm_gridded

end

function log_dNdq(
q,
mu,
sigma
)
if !(0.0 <= q <= 1.0) || !(sigma > 0.0)
return -Inf
end

d = truncated(
    Normal(mu, sigma),
    0.0,
    1.0
)

return logpdf(d, q) - logpdf(d, 1.0)

end

raw"""
log_mdsfr(z, lambda, zp, kappa)

Returns the log of the un-normalized Madau-Dickinson-like
redshift evolution

(1+z)^lambda /
[1 + ((1+z)/(1+zp))^kappa].
"""
function log_mdsfr(
z,
lambda,
zp,
kappa
)
return (
lambda * log1p(z)
-
log1p(
((1 + z) / (1 + zp))^kappa
)
)
end

function make_log_dNdm_lm(
log_rlm,
m0,
alphalm
)
function log_dN(m)
return (
log_rlm
-
alphalm * log(m / m0)
-
log1p(exp(m - m0))
+
log(2)
)
end

return log_dN

end

function make_log_dNdm_hm(
log_r,
m0,
alpha
)
function log_dN(m)
return (
log_r
-
alpha * log(m / m0)
-
log1p(exp(-(m - m0)))
+
log(2)
)
end

return log_dN

end

"""
Three-segment broken power-law log-pdf factory.
"""
function make_combined_log_dNdm1(
alphalm1,
alphamm1,
alphahm1,
mbreakf1,
mbreaks1;
mlow = 5.0,
mhigh = 300.0
)
@assert (
mlow <
mbreakf1 <
mbreaks1 <
mhigh
)

integrate_segment(
    a,
    b,
    alpha,
    mref
) = (
    alpha == 1
    ?
    mref * log(b / a)
    :
    mref * (
        (b / mref)^(1 - alpha) -
        (a / mref)^(1 - alpha)
    ) / (1 - alpha)
)

C = (
    mbreaks1 /
    mbreakf1
)^(-alphamm1)

logC =
    -alphamm1 *
    (
        log(mbreaks1) -
        log(mbreakf1)
    )

Z1 = integrate_segment(
    mlow,
    mbreakf1,
    alphalm1,
    mbreakf1
)

Z2 = integrate_segment(
    mbreakf1,
    mbreaks1,
    alphamm1,
    mbreakf1
)

Z3 =
    C *
    integrate_segment(
        mbreaks1,
        mhigh,
        alphahm1,
        mbreaks1
    )

Z = Z1 + Z2 + Z3

@assert Z > 0

logZ = log(Z)
lmbf1 = log(mbreakf1)
lmbs1 = log(mbreaks1)

function log_dN(m)
    if m < mlow || m > mhigh
        return -Inf
    elseif m <= mbreakf1
        return (
            -alphalm1 *
            (log(m) - lmbf1)
            -
            logZ
        )
    elseif m <= mbreaks1
        return (
            -alphamm1 *
            (log(m) - lmbf1)
            -
            logZ
        )
    else
        return (
            logC
            -
            alphahm1 *
            (log(m) - lmbs1)
            -
            logZ
        )
    end
end

return log_dN

end

"""
Single broken power-law mass model.
"""
function make_combined_log_dNdm(
alphalm1,
alphahm1,
mbreak1;
mlow = 3.0,
mhigh = 300.0
)
@assert mlow < mbreak1 < mhigh

integrate_segment(
    a,
    b,
    alpha,
    mref
) = (
    alpha == 1
    ?
    mref * log(b / a)
    :
    mref * (
        (b / mref)^(1 - alpha) -
        (a / mref)^(1 - alpha)
    ) / (1 - alpha)
)

Z1 = integrate_segment(
    mlow,
    mbreak1,
    alphalm1,
    mbreak1
)

Z2 = integrate_segment(
    mbreak1,
    mhigh,
    alphahm1,
    mbreak1
)

Z = Z1 + Z2

@assert Z > 0

logZ = log(Z)
lmbreak = log(mbreak1)

function log_dN(m)
    if m < mlow || m > mhigh
        return -Inf
    elseif m <= mbreak1
        return (
            -alphalm1 *
            (log(m) - lmbreak)
            -
            logZ
        )
    else
        return (
            -alphahm1 *
            (log(m) - lmbreak)
            -
            logZ
        )
    end
end

return log_dN

end

function make_log_dNdm1_broken_powerlaw(
alphatr,
alpha,
mtr;
mlow = 5.0,
mhigh = 300.0
)
@assert mlow < mtr < mhigh

integrate_segment(a, b, slope, mref) =
    slope == 1 ?
    mref * log(b / a) :
    mref * (
        (b / mref)^(1 - slope) -
        (a / mref)^(1 - slope)
    ) / (1 - slope)

Z_low = integrate_segment(
    mlow,
    mtr,
    alphatr,
    mtr
)

Z_high = integrate_segment(
    mtr,
    mhigh,
    alpha,
    mtr
)

logZ = log(Z_low + Z_high)
log_mtr = log(mtr)

function log_dNdm1(m1)
    if m1 < mlow || m1 > mhigh
        return -Inf
    end

    if m1 < mtr
        return (
            -alphatr *
            (log(m1) - log_mtr)
            -
            logZ
        )
    else
        return (
            -alpha *
            (log(m1) - log_mtr)
            -
            logZ
        )
    end
end

return log_dNdm1

end

function log_skewnormal_unnorm(
q,
m1,
mu,
alpha_q,
sigma_q,
gamma_q
)
xi_q = (
mu / m1
)^alpha_q

z_q = (
    q - xi_q
) / sigma_q

return (
    log(2) -
    log(sigma_q) +
    logpdf(Normal(), z_q) +
    logcdf(Normal(), gamma_q * z_q)
)

end

function make_log_dNdq_piecewise_skewnormal(
mtr,
mu,
alpha_q,
sigma_q,
gamma_q;
mlow = 5.0,
mhigh = 300.0,
m2_min = 3.0,
Nm = 160,
Nq = 513
)
@assert mlow < mtr < mhigh
@assert sigma_q > 0
@assert mlow > m2_min

mnorm_grid = collect(
    range(
        mlow,
        mhigh,
        length = Nm
    )
)

logZ_grid = map(
    mnorm_grid
) do m1
    qmin = m2_min / m1

    qnorm_grid = collect(
        range(
            qmin,
            1.0,
            length = Nq
        )
    )

    log_ys = [
        log_skewnormal_unnorm(
            q,
            m1,
            mu,
            alpha_q,
            sigma_q,
            gamma_q
        )
        for q in qnorm_grid
    ]

    log_trapz(
        qnorm_grid,
        log_ys
    )
end

function log_dNdq(
    q,
    m1
)
    if !(mlow <= m1 <= mhigh)
        return -Inf
    end

    m1_q = max(
        m1,
        mtr
    )

    qmin = m2_min / m1_q

    if !(qmin <= q <= 1.0)
        return -Inf
    end

    logp = log_skewnormal_unnorm(
        q,
        m1_q,
        mu,
        alpha_q,
        sigma_q,
        gamma_q
    )

    logZ = interp1d(
        m1_q,
        mnorm_grid,
        logZ_grid
    )

    return logp - logZ
end

return log_dNdq

end

function make_log_dNdm1dqdVdt(
alphatr,
alpha,
mtr,
mu,
alpha_q,
sigma_q,
gamma_q,
lambda,
zp,
kappa;
mlow = 5.0,
mhigh = 300.0,
m2_min = 3.0,
znorm = 0.0
)
log_dNdm1 =
make_log_dNdm1_broken_powerlaw(
alphatr,
alpha,
mtr;
mlow = mlow,
mhigh = mhigh
)

log_dNdq =
    make_log_dNdq_piecewise_skewnormal(
        mtr,
        mu,
        alpha_q,
        sigma_q,
        gamma_q;
        mlow = mlow,
        mhigh = mhigh,
        m2_min = m2_min
    )

log_znorm =
    log_mdsfr(
        znorm,
        lambda,
        zp,
        kappa
    )

function log_dNdm1dqdVdt(
    m1,
    q,
    z
)
    log_m1_pop =
        log_dNdm1(m1)

    log_q_pop =
        log_dNdq(
            q,
            m1
        )

    log_VT_pop =
        log_mdsfr(
            z,
            lambda,
            zp,
            kappa
        ) -
        log_znorm

    return (
        log_m1_pop
        +
        log_q_pop
        +
        log_VT_pop
    )
end

return log_dNdm1dqdVdt

end

function pe_dataframe_to_samples_array(
df,
Nposts;
rng = Random.default_rng()
)
evts = groupby(
    df,
    :commonName,
    sort = true
)

shuffled_evts = [
    shuffle(
        rng,
        evt
    )
    for evt in evts
]

pe_samples = [
    [
        [
            evt[i, :mass_1_source],
            evt[i, :mass_ratio],
            evt[i, :redshift]
        ]
        for i in 1:np
    ]
    for (np, evt) in
    zip(
        Nposts,
        shuffled_evts
    )
]

log_pe_wts = [
    vec(
        evt[
            1:np,
            :prior_logwt_m1qz
        ]
    )
    for (np, evt) in
    zip(
        Nposts,
        shuffled_evts
    )
]

return (
    pe_samples,
    log_pe_wts
)

end

function evt_dataframe_to_kde(
df,
Nkde;
rng = Random.default_rng()
)
df = shuffle(
rng,
df
)

log_wts =
    .-
    li_nocosmo_prior_logwt_m1qz(
        df
    )

wts = exp.(
    log_wts
    .-
    logsumexp(log_wts)
)

inds = sample(
    1:size(df, 1),
    Weights(wts),
    2*Nkde
)

df_sel = df[
    inds,
    :
]

pts = Array(
    df_sel[
        1:2*Nkde,
        [
            :mass_1_source,
            :mass_ratio,
            :redshift
        ]
    ]
)'

return bw_opt_kde(
    pts[:, 1:Nkde],
    pts[:, Nkde+1:end]
)

end

function pe_dataframe_to_evt_kdes(
df,
Nkde;
rng = Random.default_rng()
)
evts = groupby(
    df,
    :commonName,
    sort = true
)

return [
    evt_dataframe_to_kde(
        evt,
        Nkde;
        rng = rng
    )
    for evt in evts
]

end

function sel_dataframe_to_samples_array(
df,
Nsamp = 1024;
rng = Random.default_rng()
)
shuffled_df = shuffle(
rng,
df
)

sel_samples = [
    [
        shuffled_df[i, :mass1_source],
        shuffled_df[i, :q],
        shuffled_df[i, :redshift]
    ]
    for i in 1:Nsamp
]

log_sel_pdraw = log.(
    shuffled_df[
        1:Nsamp,
        :sampling_pdf_q
    ]
)

return (
    sel_samples,
    log_sel_pdraw
)

end

@model function pop_model_samples(
evt_samples,
log_prior_wts,
sel_samples,
log_sel_pdraw,
Ndraw,
m_grid,
zs_interp
)
nevt =
length(
evt_samples
)

dh = dH(
    h_lvk
)

dcs_interp =
    dc_over_dh(
        zs_interp,
        Ω_M_lvk
    )

dvdz_interp =
    dvdz_over_vh(
        zs_interp,
        Ω_M_lvk,
        dcs_interp
    )

log_dV_interp =
    3 * log(dh) .+
    log.(dvdz_interp) .-
    log1p.(zs_interp)

alphatr ~ Uniform(
    -10,
    10
)

mtr ~ Uniform(
    30,
    45
)

alpha ~ Uniform(
    0,
    10
)

mu ~ Uniform(
    mtr,
    60
)

alpha_q ~ Uniform(
    0,
    3
)

sigma_q ~ Uniform(
    0.01,
    1.5
)

gamma_q ~ Uniform(
    -50,
    10
)

lambda ~ Uniform(
    0,
    10
)

zp ~ Uniform(
    0.01,
    2
)

kappa ~ Uniform(
    3,
    8
)

log_dNdm1dqdVdt =
    make_log_dNdm1dqdVdt(
        alphatr,
        alpha,
        mtr,
        mu,
        alpha_q,
        sigma_q,
        gamma_q,
        lambda,
        zp,
        kappa;
        mlow = 5.0,
        mhigh = 300.0,
        m2_min = 3.0
    )

function log_pop_density(theta)
    m1, q, z = theta

    return (
        log_dNdm1dqdVdt(
            m1,
            q,
            z
        )
        +
        interp1d(
            z,
            zs_interp,
            log_dV_interp
        )
    )
end

thetas = map(
    evt_samples
) do samples
    map(
        samples
    ) do s
        m1, q, z = s

        [
            m1,
            q,
            z
        ]
    end
end

thetas_sel = map(
    sel_samples
) do s
    m1, q, z = s

    [
        m1,
        q,
        z
    ]
end

log_likelihood_sum,
log_normalization_sum,
model_genq =
    pop_model_body(
        log_pop_density,
        thetas,
        log_prior_wts,
        thetas_sel,
        log_sel_pdraw,
        Ndraw
    )

Turing.@addlogprob!(
    log_likelihood_sum
)

Turing.@addlogprob!(
    log_normalization_sum
)

m1s = map(
    model_genq.thetas_popwt
) do tp
    tp[1]
end

qs = map(
    model_genq.thetas_popwt
) do tp
    tp[2]
end

zs = map(
    model_genq.thetas_popwt
) do tp
    tp[3]
end

m2s =
    m1s .* qs

m1_draw =
    model_genq.theta_draw[1]

q_draw =
    model_genq.theta_draw[2]

z_draw =
    model_genq.theta_draw[3]

m2_draw =
    m1_draw * q_draw

return (
    Neff_sel =
        model_genq.Neff_sel,

    R =
        model_genq.R,

    Neff_samps =
        model_genq.Neff_samps,

    m1s =
        m1s,

    m2s =
        m2s,

    qs =
        qs,

    zs =
        zs,

    m1_draw =
        m1_draw,

    m2_draw =
        m2_draw,

    q_draw =
        q_draw,

    z_draw =
        z_draw
)

end

function pe_dataframe_to_cosmo_samples_array(
df,
Nposts;
rng = Random.default_rng()
)
evts = groupby(
    df,
    :commonName,
    sort = true
)

shuffled_evts = [
    shuffle(
        rng,
        evt
    )
    for evt in evts
]

pe_samples = [
    [
        [
            evt[i, :mass_1],
            evt[i, :mass_ratio],
            evt[i, :luminosity_distance] / 1000
        ]
        for i in 1:np
    ]
    for (np, evt) in
    zip(
        Nposts,
        shuffled_evts
    )
]

log_pe_wts = [
    vec(
        evt[
            1:np,
            :prior_logwt_m1dqdl
        ]
    )
    for (np, evt) in
    zip(
        Nposts,
        shuffled_evts
    )
]

return (
    pe_samples,
    log_pe_wts
)

end

function evt_dataframe_to_cosmo_kde(
df,
Nkde;
rng = Random.default_rng()
)
df = shuffle(
rng,
df
)

log_wts =
    .-
    li_nocosmo_prior_logwt_m1dqdl(
        df
    )

wts = exp.(
    log_wts
    .-
    logsumexp(log_wts)
)

inds = sample(
    1:size(df, 1),
    Weights(wts),
    2*Nkde
)

df_sel = df[
    inds,
    :
]

pts = Array(
    df_sel[
        1:2*Nkde,
        [
            :mass_1,
            :mass_ratio,
            :luminosity_distance
        ]
    ]
)'

pts[3, :] ./= 1000

return bw_opt_kde(
    pts[:, 1:Nkde],
    pts[:, Nkde+1:end]
)

end

function pe_dataframe_to_cosmo_evt_kdes(
df,
Nkde;
rng = Random.default_rng()
)
evts = groupby(
    df,
    :commonName,
    sort = true
)

return [
    evt_dataframe_to_cosmo_kde(
        evt,
        Nkde;
        rng = rng
    )
    for evt in evts
]

end

function sel_dataframe_to_cosmo_samples_array(
df,
Nsamp = 1024;
rng = Random.default_rng()
)
shuffled_df = shuffle(
rng,
df
)

sel_samples = [
    [
        shuffled_df[i, :mass_1],
        shuffled_df[i, :q],
        shuffled_df[i, :luminosity_distance]
    ]
    for i in 1:Nsamp
]

log_sel_pdraw = log.(
    shuffled_df[
        1:Nsamp,
        :sampling_pdf_m1dqdl
    ]
)

return (
    sel_samples,
    log_sel_pdraw
)

end