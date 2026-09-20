"""
    chi_eff_marginal(chi_eff, q; amax=1)

Returns the marginal prior density on `chi_eff` conditional on `q` assuming an
isotropic, independent spin prior with flat priors on the component spins for
`a_i < amax`.

Taken from [Callister (2021)](https://arxiv.org/abs/2104.09508).
"""
function chi_eff_marginal(chi_eff, q; amax=1)
    abs_chi_eff = abs(chi_eff)
    l1 = amax*((1-q)/(1+q))
    l2 = q*amax/(1+q)
    l3 = amax/(1+q)

    if chi_eff == 0
        (1+q)/(2*amax)*(2 - log(q))
    elseif abs_chi_eff < l1 && abs_chi_eff < l2
        (1+q)/(4*q*amax^2)*( 
            q*amax*(4 + 2*log(amax) - log(q^2*amax^2 - (1+q)^2*abs_chi_eff^2)) 
            - 2*(1+q)*abs_chi_eff*atanh((1+q)*abs_chi_eff/(q*amax)) 
            + (1+q)*abs_chi_eff*(reli2(-q*amax/((1+q)*abs_chi_eff)) - reli2(q*amax/((1+q)*abs_chi_eff)))
        )
    elseif abs_chi_eff < l1 && abs_chi_eff > l2
        (1+q)/(4*q*amax^2)*(
            4*q*amax + 2*q*amax*log(amax)
            - 2*(1+q)*abs_chi_eff*atanh(q*amax/((1+q)*abs_chi_eff))
            - q*amax*log((1+q)^2*abs_chi_eff^2 - q^2*amax^2)
            + (1+q)*abs_chi_eff*(reli2(-q*amax/((1+q)*abs_chi_eff)) - reli2(q*amax/((1+q)*abs_chi_eff)))
        )
    elseif abs_chi_eff > l1 && abs_chi_eff < l2
        (1+q)/(4*q*amax^2)*(
            2*(1+q)*(amax - abs_chi_eff) - (1+q)*abs_chi_eff*log(amax)^2
            + (amax + (1+q)*abs_chi_eff*log((1+q)*abs_chi_eff))*log(q*amax/(amax - (1+q)*abs_chi_eff))
            - (1+q)*abs_chi_eff*log(amax)*(2 + log(q) - log(amax - (1+q)*abs_chi_eff))
            + q*amax*log(amax / (q*amax - (1+q)*abs_chi_eff))
            + (1+q)*abs_chi_eff*log((amax - (1+q)*abs_chi_eff)*(q*amax - (1+q)*abs_chi_eff)/q)
            + (1+q)*abs_chi_eff*(reli2(1-amax/((1+q)*abs_chi_eff)) - reli2(q*amax/((1+q)*abs_chi_eff)))
        )
    elseif abs_chi_eff > l1 && abs_chi_eff > l2 && abs_chi_eff < l3
        (1+q)/(4*q*amax^2)*(
            - abs_chi_eff*log(amax)^2 + 2*(1+q)*(amax - abs_chi_eff)
            + q*amax*log(amax/((1+q)*abs_chi_eff - q*amax)) + amax*log(q*amax/(amax - (1+q)*abs_chi_eff))
            - abs_chi_eff*log(amax)*(2*(1+q) - log((1+q)*abs_chi_eff) - q*log((1+q)*abs_chi_eff/amax))
            + (1+q)*abs_chi_eff*log((-q*amax + (1+q)*abs_chi_eff)*(amax - (1+q)*abs_chi_eff)/q)
            + (1+q)*abs_chi_eff*log(amax/((1+q)*abs_chi_eff))*log((amax - (1+q)*abs_chi_eff)/q)
            + (1+q)*abs_chi_eff*(reli2(1-amax/((1+q)*abs_chi_eff)) - reli2(q*amax/((1+q)*abs_chi_eff)))
        )
    elseif abs_chi_eff > l3 && abs_chi_eff < amax
        (1+q)/(4*q*amax^2)*(
            2*(1+q)*(amax - abs_chi_eff) - (1+q)*abs_chi_eff*log(amax)^2
            + log(amax)*(amax - 2*(1+q)*abs_chi_eff - (1+q)*abs_chi_eff*log(q/((1+q)*abs_chi_eff - amax)))
            - amax*log(((1+q)*abs_chi_eff - amax)/q)
            + (1+q)*abs_chi_eff*log(((1+q)*abs_chi_eff - amax)*((1+q)*abs_chi_eff - q*amax)/q)
            + (1+q)*abs_chi_eff*log((1+q)*abs_chi_eff)*log(q*amax / ((1+q)*abs_chi_eff - amax))
            - q*amax*log(((1+q)*abs_chi_eff - q*amax)/amax)
            + (1+q)*abs_chi_eff*(reli2(1 - amax/((1+q)*abs_chi_eff)) - reli2(q*amax/((1+q)*abs_chi_eff)))
        )
    else abs_chi_eff > amax
        zero(chi_eff)
    end
end

"""
    li_nocosmo_prior_logwt_m1qzchie(df)

Returns the LALInference "nocosmo" prior weight (i.e. un-normalized prior
density) over `m1`, `q`, `chi_eff`, `z` for each row in the given data frame.
"""

function li_nocosmo_prior_logwt_m1qzchie(df)
    z = df[!, :redshift]
    m1 = df[!, :mass_1_source]
    q = df[!, :mass_ratio]
    chi_eff = df[!, :chi_eff]

    log_opz = log1p.(z)

    dl = @. ustrip(u"Gpc", luminosity_dist((lvk_cosmology, ), z))
    dc = @. ustrip(u"Gpc", comoving_transverse_dist((lvk_cosmology, ), z))
    dh_z = @. ustrip(u"Gpc", 2.99792e8*u"m"/u"s" / Cosmology.H((lvk_cosmology, ), z))

    # According to [Callister (2021)](https://arxiv.org/abs/2104.09508), the
    # prior on m1_source, m2_source, z is (1+z)^2*dl^2*(dc + (1+z)*c/H(z))
    # the final log(m1) comes from d(m2_source)/d(q) = m1_source
    m1_m2_z_logwt = @. 2*log_opz + 2*log(dl) + log(dc + (1+z)*dh_z)
    m1_q_z_logwt = @. m1_m2_z_logwt + log(m1)

    # Also from [Callister (2021)](https://arxiv.org/abs/2104.09508)
    chi_eff_logwt = @. log(chi_eff_marginal(chi_eff, q))

    m1_q_z_logwt .+ chi_eff_logwt
end

function li_nocosmo_prior_logwt_m1qz(df)
    z = df[!, :redshift]
    m1 = df[!, :mass_1_source]
    q = df[!, :mass_ratio]

    log_opz = log1p.(z)

    dl = @. ustrip(u"Gpc", luminosity_dist((lvk_cosmology, ), z))
    dc = @. ustrip(u"Gpc", comoving_transverse_dist((lvk_cosmology, ), z))
    dh_z = @. ustrip(u"Gpc", 2.99792e8*u"m"/u"s" / Cosmology.H((lvk_cosmology, ), z))

    # According to [Callister (2021)](https://arxiv.org/abs/2104.09508), the
    # prior on m1_source, m2_source, z is (1+z)^2*dl^2*(dc + (1+z)*c/H(z))
    # the final log(m1) comes from d(m2_source)/d(q) = m1_source
    m1_m2_z_logwt = @. 2*log_opz + 2*log(dl) + log(dc + (1+z)*dh_z)
    m1_q_z_logwt = @. m1_m2_z_logwt + log(m1)

    m1_q_z_logwt
end

"""
    li_nocosmo_prior_logwt_m1dqdlchie(df)

Return the LALInference "nocosmo" prior weight (i.e. un-normalized prior
density) over `m1_detector`, `q`, `chi_eff`, and `dL`.
"""
function li_nocosmo_prior_logwt_m1dqdlchie(df)
    lw = li_nocosmo_prior_logwt_m1qzchie(df)

    z = df[!, :redshift]

    dc = @. ustrip(u"Gpc", comoving_transverse_dist((lvk_cosmology, ), z))
    dh_z = @. ustrip(u"Gpc", 2.99792e8*u"m"/u"s" / Cosmology.H((lvk_cosmology, ), z))

    log_dm1_dmd1 = @. -log1p(z)
    log_dz_ddL = @. -log(dc + (1 + z)*dh_z)

    lw + log_dm1_dmd1 + log_dz_ddL
end

function li_nocosmo_prior_logwt_m1dqdl(df)
    lw = li_nocosmo_prior_logwt_m1qz(df)

    z = df[!, :redshift]

    dc = @. ustrip(u"Gpc", comoving_transverse_dist((lvk_cosmology, ), z))
    dh_z = @. ustrip(u"Gpc", 2.99792e8*u"m"/u"s" / Cosmology.H((lvk_cosmology, ), z))

    log_dm1_dmd1 = @. -log1p(z)
    log_dz_ddL = @. -log(dc + (1 + z)*dh_z)

    lw + log_dm1_dmd1 + log_dz_ddL
end

"""
    resample_pe(log_wt_fn, samples, log_pe_prior, Nsamp)

Returns a new array of `Nsamp` PE samples drawn from a (log) prior proportional
to `log_wt_fn(m1, q, chi_eff, z)`.
"""
function resample_pe(log_wt_fn, samples, log_pe_prior, Nsamp)
    log_new_prior = [log_wt_fn(s...) for s in samples]
    log_wts = log_new_prior .- log_pe_prior
    wts = exp.(log_wts .- logsumexp(log_wts))
    inds = sample(1:length(samples), Weights(wts), Nsamp, replace=false)
    (samples[inds], log_new_prior[inds])
end