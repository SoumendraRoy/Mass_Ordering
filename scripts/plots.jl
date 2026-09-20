const m_bh_min = 10.0 # Lower bound for integration on masses.
const m_bh_max = 100.0 # Upper bound for integration on masses.
const m_plot = 35.0
const q_plot = 1.0
const z_plot = 0.5

function categorical_palette(n; l=65, c=90)
    [LCHuv(l, c, h) for h in range(0, stop=360, length=n+1)[1:n]]
end

@recipe(KDEContour, x, y) do scene
    Theme(
        levels = [0.1, 0.5],
        xexpansionfactor = 0.1,
        yexpansionfactor = 0.1,
        xgridsize=128,
        ygridsize=129
    )
end

function Makie.plot!(kdecontour::KDEContour)
    xexpansionfactor = pop!(kdecontour.attributes, :xexpansionfactor)
    yexpansionfactor = pop!(kdecontour.attributes, :yexpansionfactor)
    xgridsize = pop!(kdecontour.attributes, :xgridsize)
    ygridsize = pop!(kdecontour.attributes, :ygridsize)
    levels = pop!(kdecontour.attributes, :levels)

    x = kdecontour.x
    y = kdecontour.y

    xgrid = lift(x, xexpansionfactor, xgridsize) do x, xexpansionfactor, xgridsize
        dx = maximum(x)-minimum(x)
        fx = xexpansionfactor/2
        range(minimum(x)-fx*dx, maximum(x)+fx*dx, length=xgridsize)
    end

    ygrid = lift(y, yexpansionfactor, ygridsize) do y, yexpansionfactor, ygridsize
        dy = maximum(y)-minimum(y)
        fy = yexpansionfactor/2
        range(minimum(y)-fy*dy, maximum(y)+fy*dy, length=ygridsize)
    end

    k = lift(x, y) do x, y
        pts = vcat(x', y')
        KDE(pts)
    end

    p_levels = lift(x, y, k, levels) do x, y, k, levels
        pts = vcat(x', y')
        p_kde_pts = [pdf(k, pts[:,i]) for i in axes(pts, 2)]
        [quantile(p_kde_pts, l) for l in levels]
    end

    z = lift(x, y, k, xgrid, ygrid) do x, y, k, xgrid, ygrid
        [pdf(k, [x, y]) for x in xgrid, y in ygrid]
    end

    contour!(kdecontour, xgrid, ygrid, z; levels=p_levels, kdecontour.attributes...)
end

population_params = [:R, :alphalm1, :alphamm1, :alphahm1, :mbreakf1, :mbreaks1, :alphalm2, :alphamm2, :alphahm2, :mbreakf2, :mbreaks2, :alpharight2, :mturn_right2, :mu, :sigma, :lambda, :zp, :kappa, :Neff_sel]
cosmo_params = [:h, :Ω_M]
function traceplot(trace; params=population_params)
    f = Figure(size=(800,2400))

    # Nevt = length(dims(trace.posterior, :event))

    for (i, p) in enumerate(params)
        a_trace = Axis(f[i,1]; ylabel=string(p), xlabel="Iter")
        a_dens = Axis(f[i,2]; ylabel="Density", xlabel=string(p))

        for (ic, _) in enumerate(dims(trace.posterior, :chain))
            lines!(a_trace, trace.posterior[p][chain=ic])
            density!(a_dens, vec(trace.posterior[p][chain=ic]))
        end

        # if p == :Neff_sel
        #     hlines!(a_trace, 4*Nevt, color=:black)
        #     vlines!(a_dens, 4*Nevt, color=:black)
        # end
    end
    f
end

function log_dNdm_from_chain_sample(trace, draw, chain; mgrid=10:0.25:100)
    p = trace.posterior
    d = draw
    c = chain

    BumpCosmologyGWTC3.make_combined_log_dNdm(p.alphalm1[draw=d, chain=c], p.alphamm1[draw=d, chain=c], p.alphahm1[draw=d, chain=c], p.mbreakf1[draw=d, chain=c], p.mbreaks1[draw=d, chain=c], p.alphalm2[draw=d, chain=c], p.alphamm2[draw=d, chain=c], p.alphahm2[draw=d, chain=c], p.mbreakf2[draw=d, chain=c], p.mbreaks2[draw=d, chain=c], p.alpharight2[draw=d, chain=c], p.mturn_right2[draw=d, chain=c]; mgrid=mgrid)
end

function log_dN_from_chain_sample(trace, draw, chain)
    p = trace.posterior
    d = draw
    c = chain

    log_dN = BumpCosmologyGWTC3.make_log_dNdm1dqdVdt(p.alphalm1[draw=d, chain=c], p.alphamm1[draw=d, chain=c], p.alphahm1[draw=d, chain=c], p.mbreakf1[draw=d, chain=c], p.mbreaks1[draw=d, chain=c], p.alphalm2[draw=d, chain=c], p.alphamm2[draw=d, chain=c], p.alphahm2[draw=d, chain=c], p.mbreakf2[draw=d, chain=c], p.mbreaks2[draw=d, chain=c], p.alpharight2[draw=d, chain=c], p.mturn_right2[draw=d, chain=c], p.mu[draw=d, chain=c], p.sigma[draw=d, chain=c], p.lambda[draw=d, chain=c], p.zp[draw=d, chain=c], p.kappa[draw=d, chain=c])
    log_dN
end

function remnant_mass_dist_plot(trace; rng=Random.default_rng(), label=nothing, color=Makie.wong_colors()[1], kwargs...)
    f = Figure()
    a = Axis(f[1,1], xlabel=L"m / M_\odot", ylabel=L"m p(m)", xscale=log10, xticks=[20,30,40,50], xtickformat="{:.0f}")
    remnant_mass_dist_plot!(a, trace; rng=rng, color=color, label=label, kwargs...)
    f
end

function remnant_mass_dist_plot!(a, trace; rng=Random.default_rng(), label=nothing, color=Makie.wong_colors()[1], kwargs...)
    @progress for i in 1:100
        d = rand(rng, span(dims(trace.posterior, :draw)))
        c = rand(rng, span(dims(trace.posterior, :chain)))

        log_dN = log_dNdm_from_chain_sample(trace, d, c)

        m = exp.(log(20):0.01:log(50))
        dNdm = map(m) do m
            exp(log_dN(m))
        end
        if i == 1
            lines!(a, m, m .* dNdm ./ trapz(m, dNdm); color=(color, 0.1), label=label, kwargs...)
        else
            lines!(a, m, m .* dNdm ./ trapz(m, dNdm); color=(color, 0.1), kwargs...)
        end
    end
end

function marginal_m1_plot(trace; limits=(20.0, 50.0, 1.0, 100.0), yticks=[1, 10, 100], yminorticks=[2,3,4,5,6,7,8,9,20,30,40,50,60,70,80,90], kwargs...)
    f = Figure()
    a = Axis(f[1,1], xlabel=L"m_1/M_\odot", ylabel=L"m_1 \mathrm{d} N / \mathrm{d} m_1 \mathrm{d} V \mathrm{d} t / \mathrm{Gpc}^{-3} \, \mathrm{yr}^{-1}", xscale=log10, yscale=log10, xticks=[20, 25, 30, 35, 40, 45, 50], xtickformat="{:.0f}", yticks=yticks, yminorticks=yminorticks, yminorticksvisible=true, ytickformat="{:.0f}", limits=limits)
    marginal_m1_plot!(a, trace; kwargs...)
    f
end

function marginal_m1_plot!(a, trace; z_plot=z_plot, ms=exp.(log(m_bh_min):0.01:log(m_bh_max)), qs=0.001:0.01:1, draws=100, rng=Random.default_rng(), color=Makie.wong_colors()[1], label=nothing)
    p = trace.posterior
    @progress for i in 1:draws
        d = rand(rng, span(dims(p, :draw)))
        c = rand(rng, 1:length(dims(p, :chain)))
        log_dN = log_dN_from_chain_sample(trace, d, c)
        dNdm1 = map(ms) do m
            integrand = map(qs) do q
                m2 = q*m
                if m2 < m_bh_min
                    zero(m)
                else
                    exp(log_dN(m, q, 0.0, z_plot, 0.0; ignore_chi_eff=true))
                end
            end
            m * p.R[chain=c, draw=d] * trapz(qs, integrand)
        end
        if i == 1
            lines!(a, ms, dNdm1, color=(color, 0.1), label=label)
        else
            lines!(a, ms, dNdm1, color=(color, 0.1))
        end
    end
end

function marginal_q_plot(trace; z_plot=z_plot, ms=exp.(log(m_bh_min):0.01:log(m_bh_max)), qs=collect(0.001:0.01:1.0), draws=100, rng=Random.default_rng(), limits=(0.25, 1, 0.1, 100), yticks=[0.1, 1, 10, 100], yminorticks=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,2,3,4,5,6,7,8,9,20,30,40,50,60,70,80,90], color=Makie.wong_colors()[1])
    f = Figure()
    a = Axis(f[1,1], xlabel=L"q", ylabel=L"\mathrm{d} N / \mathrm{d} m_1 \mathrm{d} q \mathrm{d} V \mathrm{d} t / \mathrm{Gpc}^{-3} \, \mathrm{yr}^{-1}", yscale=log10, yticks=yticks, yminorticks=yminorticks, yminorticksvisible=true, ytickformat="{:.1f}", limits=limits)
    p = trace.posterior
    @progress for _ in 1:draws
        d = rand(rng, span(dims(p, :draw)))
        c = rand(rng, span(dims(p, :chain)))
        log_dN = log_dN_from_chain_sample(trace, d, c)

        dNdq = map(qs) do q
            integrand = map(ms) do m
                m2 = q*m
                if m2 < m_bh_min
                    zero(m)
                else
                    exp(log_dN(m, q, 0.0, z_plot, 0.0; ignore_chi_eff=true))
                end
            end
            p.R[chain=c, draw=d] * trapz(ms, integrand)
        end
        lines!(a, qs, dNdq, color=(color, 0.1))
    end
    f
end

function marginal_chieff_plot(trace; z_plot=z_plot, ms=exp.(log(m_bh_min):0.01:log(m_bh_max)), qs=collect(0.001:0.01:1), chi_effs=(-1:0.01:1), draws=100, rng=Random.default_rng(), limits=(-0.5, 0.5, 1, 100), yticks=[1,10,100], yminorticks=[2,3,4,5,6,7,8,9,20,30,40,50,60,70,80,90], color=Makie.wong_colors()[1])
    f = Figure()
    a = Axis(f[1,1], xlabel=L"\chi_\mathrm{eff}", ylabel=L"\mathrm{d} N / \mathrm{d} \chi_\mathrm{eff} \mathrm{d} V \mathrm{d} t / \mathrm{Gpc}^{-3} \, \mathrm{yr}^{-1}", yscale=log10, ytickformat="{:.0f}", yticks=yticks, yminorticks=yminorticks, yminorticksvisible=true, xminorticksvisible=true, limits=limits)
    p = trace.posterior
    @progress for _ in 1:draws
        d = rand(rng, span(dims(p, :draw)))
        c = rand(rng, span(dims(p, :chain)))
        log_dN = log_dN_from_chain_sample(trace, d, c)

        dNdchi = map(chi_effs) do chi_eff
            m_integrand = map(ms) do m
                q_integrand = map(qs) do q
                    m2 = q*m
                    if m2 < m_bh_min
                        zero(m)
                    else
                        exp(log_dN(m, q, chi_eff, z_plot, 0.0))
                    end
                end
                trapz(qs, q_integrand)
            end
            p.R[draw=d, chain=c] * trapz(ms, m_integrand)
        end
        lines!(a, chi_effs, dNdchi, color=(color, 0.1))
    end
    f
end

function q_chieff_corner_plot(trace;
    m_plot=m_plot,
    z_plot=z_plot,
    limits = (-0.5, 0.5, 0.5, 1),
    chi_effs = collect(-1:0.01:1),
    qs = collect(0.001:0.01:1),
    levels = range(0.1, stop=1.0, length=10),
    draws = 100,
    rng = Random.default_rng(),
    colormap = :batlow
)

    # Setup grid
    f = Figure(resolution=(800, 800))

    grid = f[1,1] = GridLayout(tellwidth=false, tellheight=false)
    top_ax = Axis(grid[1, 2], xlabel="", ylabel="", xticksvisible=false, xgridvisible=true, ygridvisible=true)
    main_ax = Axis(grid[2, 2], xlabel=L"\chi_\mathrm{eff}", ylabel=L"q", limits=limits, xgridvisible=true, ygridvisible=true)
    right_ax = Axis(grid[2, 3], xlabel=L"q", ylabel="", xticksvisible=true, xgridvisible=true, ygridvisible=true)

    colsize!(grid, 1, Relative(0.1))
    rowsize!(grid, 1, Relative(0.4))
    rowsize!(grid, 2, Relative(0.5)) 

    #############################
    # 1. 2D q-chi_eff contour plot
    #############################

    dchi_eff = chi_effs[2] - chi_effs[1]
    dq = qs[2] - qs[1]

    p = trace.posterior

    i = 0
    N = length(dims(p, :chain)) * length(dims(p, :draw))

    pqc_mean = [0.0 for ce in chi_effs, q in qs]

    @withprogress begin
        for c in span(dims(p, :chain))
            for d in span(dims(p, :draw))
                log_dN = log_dN_from_chain_sample(trace, d, c)

                pqc = [
                    if q * m_plot > m_bh_min
                        exp(log_dN(m_plot, q, ce, z_plot, 0.0))
                    else
                        zero(m_plot)
                    end
                    for ce in chi_effs, q in qs
                ]

                pqc ./= sum(pqc) * dchi_eff * dq

                pqc_mean .+= pqc

                i += 1
                @logprogress i/N
            end
        end
        pqc_mean ./= length(chi_effs) * length(qs)
    end

    vp = sort(vec(pqc_mean))
    vp_cum = cumsum(vp)
    vp_cum ./= vp_cum[end]

    vp_levels = [vp[argmin(abs.(vp_cum .- l))] for l in levels]

    blue_colormap = cgrad(:blues)  # built-in in Makie, or you can customize if needed

    # Then plot with it:
    contourf!(main_ax, chi_effs, qs, pqc_mean, levels=vp_levels, colormap=blue_colormap)

    #####################################
    # 2. Marginal over chi_eff (top plot) - updated with median + uncertainty band
    #####################################

    ms = exp.(range(log(m_bh_min), log(m_bh_max), length=100))
    dNdchi_samples = []

    @progress for i in 1:draws
        d = rand(rng, span(dims(p, :draw)))
        c = rand(rng, span(dims(p, :chain)))
        log_dN = log_dN_from_chain_sample(trace, d, c)

        dNdchi = map(chi_effs) do chi_eff
            m_integrand = map(ms) do m
                q_integrand = map(qs) do q
                    m2 = q*m
                    if m2 < m_bh_min
                        zero(m)
                    else
                        exp(log_dN(m, q, chi_eff, z_plot, 0.0))
                    end
                end
                trapz(qs, q_integrand)
            end
            p.R[draw=d, chain=c] * trapz(ms, m_integrand)
        end

        push!(dNdchi_samples, dNdchi)
    end

    # Stack samples into a matrix
    dNdchi_matrix = reduce(hcat, dNdchi_samples)

    # Compute percentiles at each chi_eff
    dNdchi_median = mapslices(x -> median(x), dNdchi_matrix; dims=2)[:]
    dNdchi_p10    = mapslices(x -> quantile(x, 0.10), dNdchi_matrix; dims=2)[:]
    dNdchi_p90    = mapslices(x -> quantile(x, 0.90), dNdchi_matrix; dims=2)[:]

    println("Standard deviation of dNdchi_p10: ", std(dNdchi_p10))
    println("Standard deviation of dNdchi_median: ", std(dNdchi_median))
    println("Standard deviation of dNdchi_p90: ", std(dNdchi_p90))

    # Plot
    band!(top_ax, chi_effs, dNdchi_p10, dNdchi_p90, color=(:blue, 0.2))  
    lines!(top_ax, chi_effs, dNdchi_median, color=:blue, linewidth=2)  

    top_ax.xlabel = L"\chi_\mathrm{eff}"
    top_ax.ylabel = L"\frac{dN}{d\chi_\mathrm{eff}} \mid_{m_1=35M_{\odot}, z=0.5}"
    top_ax.xticksvisible = true
    top_ax.yticksvisible = true

    
    ##################################
    # 3. Marginal over q (right plot)
    ##################################

    dNdq_samples = []

    @progress for _ in 1:draws
        d = rand(rng, span(dims(p, :draw)))
        c = rand(rng, span(dims(p, :chain)))
        log_dN = log_dN_from_chain_sample(trace, d, c)

        dNdq = map(qs) do q
            integrand = map(ms) do m
                m2 = q*m
                if m2 < m_bh_min
                    zero(m)
                else
                    exp(log_dN(m, q, 0.0, z_plot, 0.0; ignore_chi_eff=true))
                end
            end
            p.R[draw=d, chain=c] * trapz(ms, integrand)
        end

        push!(dNdq_samples, dNdq)
    end

    # Stack samples into a matrix
    dNdq_matrix = reduce(hcat, dNdq_samples)

    # Compute percentiles at each q
    dNdq_median = mapslices(x -> median(x), dNdq_matrix; dims=2)[:]
    dNdq_p10    = mapslices(x -> quantile(x, 0.10), dNdq_matrix; dims=2)[:]
    dNdq_p90    = mapslices(x -> quantile(x, 0.90), dNdq_matrix; dims=2)[:]

    # Plot
    band!(right_ax, qs, dNdq_p10, dNdq_p90, color=(:blue, 0.2))  
    lines!(right_ax, qs, dNdq_median, color=:blue, linewidth=2)  

    right_ax.xlabel = L"q"
    right_ax.ylabel = L"\frac{dN}{dq} \mid_{m_1=35M_{\odot}, z=0.5}"
    right_ax.xticksvisible = true
    right_ax.yticksvisible = true

    hidespines!(right_ax, :l)
    hidespines!(top_ax, :b)

    f
end

function marginal_z_plot(trace; ms=exp.(log(m_bh_min):0.01:log(m_bh_max)), qs=0.001:0.01:1, zs=expm1.(range(log(1), stop=log(1+2), length=128)), draws=100, rng=Random.default_rng(), limits=(0, 2, 1, 1000), yticks=[1, 10, 100, 1000], yminorticks=[2,3,4,5,6,7,8,9,20,30,40,50,60,70,80,90,200,300,400,500,600,700,800,900], plot_md_line=true, color=Makie.wong_colors()[1])
    f = Figure()
    a = Axis(f[1,1], xlabel=L"z", ylabel=L"\mathrm{d} N / \mathrm{d} V \mathrm{d} t / \mathrm{Gpc}^{-3} \, \mathrm{yr}^{-1}", yscale=log10, ytickformat="{:.0f}", yticks=yticks, yminorticks=yminorticks, yminorticksvisible=true, xminorticksvisible=true, limits=limits)

    p = trace.posterior

    dNs = zeros(length(zs), draws)

    @progress for j in 1:draws
        c = rand(rng, span(dims(p, :chain)))
        d = rand(rng, span(dims(p, :draw)))

        log_dN = log_dN_from_chain_sample(trace, d, c)

        z1 = zs[1]
        dNdms = map(ms) do m
            integrand = map(qs) do q
                m2 = q*m
                if m2 < m_bh_min
                    zero(m)
                else
                    exp(log_dN(m, q, 0.0, z1, 0.0; ignore_chi_eff=true))
                end
            end

            trapz(qs, integrand)
        end
        norm = trace.posterior.R[draw=d, chain=c] * trapz(ms, dNdms) / exp(log_dN(m_plot, q_plot, 0.0, z1, 0.0; ignore_chi_eff=true))

        dNdz = map(zs) do z
            norm * exp(log_dN(m_plot, q_plot, 0.0, z, 0.0; ignore_chi_eff=true))
        end
        lines!(a, zs, dNdz, color=(color, 0.1))
        dNs[:, j] = dNdz
    end

    if plot_md_line
        vs = vec(var(log.(dNs), dims=2))
        imin = argmin(vs)
        norm = median(dNs[imin, :])

        mds = @. (1 + zs)^2.7 / (1 + ((1+zs)/(1+1.9))^5.6)
        mds = mds .* norm ./ mds[imin]

        lines!(a, zs, mds, color=:black)
    end

    f
end

function m1_q_plot(trace; z_plot=z_plot, ms=exp.(log(m_bh_min):0.01:log(m_bh_max)), qs=collect(0.001:0.01:1), levels=range(0.1, stop=1.0, length=10), limits=(20, 45, 0.25, 1), colormap=:batlow)
    f = Figure()
    a = Axis(f[1,1], ylabel=L"q", xlabel=L"m_1 / M_\odot", limits=limits)
    p = trace.posterior

    dm = ms[2]-ms[1]
    dq = qs[2]-qs[1]

    i = 0
    N = length(dims(p, :chain))*length(dims(p, :draw))
    dN = [0.0 for m in ms, q in qs]
    @withprogress begin
        for c in span(dims(p, :chain))
            for d in span(dims(p, :draw))
                log_dN = log_dN_from_chain_sample(trace, d, c)

                dNdm1q = [
                    if m*q > m_bh_min
                        exp(log_dN(m, q, 0.0, z_plot, 0.0; ignore_chi_eff=true))
                    else
                        zero(q)
                    end
                    for m in ms, q in qs
                ]
                dNdm1q ./= sum(dNdm1q)*dm*dq
                dN .+= dNdm1q

                i += 1
                @logprogress i/N
            end                
        end
        dN ./= length(ms)*length(qs)
    end

    dN_vec = sort(vec(dN))

    dN_cum_vec = cumsum(dN_vec)
    dN_cum_vec ./= dN_cum_vec[end]

    dN_levels = [dN_vec[argmin(abs.(dN_cum_vec .- l))] for l in levels]

    contourf!(a, ms, qs, dN, levels=dN_levels, colormap=colormap)
    f
end

function q_chieff_plot(trace; m_plot=m_plot, z_plot=z_plot, limits=(-0.5, 0.5, 0.5, 1), chi_effs=collect(-1:0.01:1), qs=collect(0.001:0.01:1), levels=range(0.1, stop=1.0, length=10), plot_q_chieff_lines=true, rng=Random.default_rng(), colormap=:batlow)
    f = Figure()
    a = Axis(f[1,1], xlabel=L"\chi_\mathrm{eff}", ylabel=L"q", limits=limits)

    dchi_eff = chi_effs[2]-chi_effs[1]
    dq = qs[2]-qs[1]

    p = trace.posterior

    i = 0
    N = length(dims(p, :chain))*length(dims(p, :draw))

    pqc_mean = [0.0 for c in chi_effs, q in qs]
    @withprogress begin
        for c in span(dims(p, :chain))
            for d in span(dims(p, :draw))
                log_dN = log_dN_from_chain_sample(trace, d, c)

                pqc = [
                    if q*m_plot > m_bh_min
                        exp(log_dN(m_plot, q, ce, z_plot, 0.0))
                    else
                        zero(m_plot)
                    end
                    for ce in chi_effs, q in qs
                ]

                pqc ./= sum(pqc)*dchi_eff*dq

                pqc_mean .+= pqc

                i += 1
                @logprogress i/N
            end
        end
        pqc_mean ./= length(chi_effs)*length(qs)
    end
    
    vp = sort(vec(pqc_mean))
    vp_cum = cumsum(vp)
    vp_cum ./= vp_cum[end]

    vp_levels = [vp[argmin(abs.(vp_cum .- l))] for l in levels]

    contourf!(a, chi_effs, qs, pqc_mean, levels=vp_levels, colormap=colormap)

    if plot_q_chieff_lines
        for _ in 1:100
            c = rand(rng, span(dims(p, :chain)))
            d = rand(rng, span(dims(p, :draw)))

            mul = p.mul[draw=d, chain=c]
            muh = p.muh[draw=d, chain=c]

            lines!(a, chieff_interp.(qs, mul, muh), qs, color=(:black, 0.1))
        end
    end
    f
end

function beta_chi_eff_plot(trace)
    bs = vec(trace.posterior.beta_chi_eff)

    bm = median(bs)
    bl = quantile(bs, 0.16)
    bh = quantile(bs, 0.84)
    
    
    f = Figure()
    a = Axis(f[1,1], xlabel=L"\beta_{\chi_\mathrm{eff}}", title=L"\beta_{\chi_\mathrm{eff}} = %$(round(bm, digits=2))^{+%$(round(bh-bm, digits=2))}_{-%$(round(bm-bl, digits=2))}")

    density!(a, bs)

    f
end

function conditional_chieff_plot(trace; q_plot = q_plot, rng=Random.default_rng(), color=Makie.wong_colors()[1], chi_eff_min=-0.5, chi_eff_max=0.5)
    chi_effs = collect(-1:0.01:1)

    f = Figure()
    a = Axis(f[1,1], xlabel=L"\chi_\mathrm{eff}", limits=(chi_eff_min, chi_eff_max, 0, nothing), title=L"p\left( \chi_\mathrm{eff} \mid q = %$(round(q_plot, digits=2)) \right)")
    @progress for _ in 1:100
        c = rand(rng, span(dims(trace.posterior, :chain)))
        d = rand(rng, span(dims(trace.posterior, :draw)))

        mul = trace.posterior.mul[draw=d, chain=c]
        muh = trace.posterior.muh[draw=d, chain=c]
        sigmal = trace.posterior.sigmal[draw=d, chain=c]
        sigmah = trace.posterior.sigmah[draw=d, chain=c]

        mu = chieff_interp(q_plot, mul, muh)
        sigma = chieff_interp(q_plot, sigmal, sigmah)

        lines!(a, chi_effs, pdf(Normal(mu, sigma), chi_effs), color=(color, 0.1))
    end

    f
end

function frac_mass_bounds(df, m1_min, m2_min, m2_max)
    m1s = df[!, :mass_1_source]
    m2s = df[!, :mass_2_source]
    p_in = sum((m1s .> m1_min) .&& (m2s .> m2_min) .&& (m2s .< m2_max)) / length(m1s)
    p_in
end

function m1m2_kde_plot(df;
    m1_min = 20.0,
    m2_min = 3.0,
    m2_max = 50.0,
    include_legend=false,
    p_cut=[0.9, 0.5, 0.1],
    p_cut_eventlist=0.0,
    rng=Random.default_rng(),
    far_thresh=1,
    alpha_factor=0.5,
    level=0.1,
    draw_cut_lines=false
)
    if p_cut != p_cut_eventlist
        @info("Events will be plotted with p > $(p_cut_eventlist), but reduced by alpha*=$(alpha_factor) at thresholds p < $(p_cut)")
    end

    evt_groups = DataFrames.groupby(df, :gwname)

    # Only apply chirp mass and FAR filtering — no m1/m2 filter
    evt_groups = [
        evt for evt in evt_groups
        if frac_mass_bounds(evt, m1_min, m2_min, m2_max) >= p_cut_eventlist &&
           evt[1, :far] < far_thresh
    ]

    cs = categorical_palette(length(evt_groups))

    mlow = 0.0
    mhigh = 100.0

    f = Figure()
    a = Axis(f[1, 1];
        xlabel=L"\mathrm{Primary~Mass~(in~} M_\odot)",
        ylabel=L"\mathrm{Secondary~Mass~(in~} M_\odot)",
        xlabelsize=22, ylabelsize=22, titlesize=24,
        limits=(mlow,mhigh,mlow,mhigh),
        title=L"\mathrm{PE~Pos}~(m_1>20M_{\odot},~3M_{\odot}<m_2<50M_{\odot})>0.5",
        xminorgridvisible=true, xminorticksvisible=true,
        yminorgridvisible=true, yminorticksvisible=true
    )

    # Exclusion regions (shaded)
    if draw_cut_lines
        m1_min = 20.0
        m2_min = 3.0
        m2_max = 50.0

        # Left of m1=20
        poly!(a, [0.0, m1_min, m1_min], [0.0, 0.0, m1_min], color=(:grey, 0.25))
        # Below m2=3
        poly!(a, [m1_min, m1_min, 100.0, 100.0], [0.0, m2_min , m2_min, 0.0], color=(:grey, 0.25))
        # Above m2=50
        poly!(a, [m2_max, 100.0, 100.0], [m2_max, 100.0, m2_max], color=(:grey, 0.25))
    end

    # m1 < m2 region
    band!(a, [mlow, mhigh], [mlow, mhigh], [mhigh, mhigh], color=(:grey, 0.25))

    @progress for (i, evt) in enumerate(evt_groups)
        alpha = 1.0
        p = frac_mass_bounds(evt, m1_min, m2_min, m2_max)
        for pc in p_cut
            if p > pc
                break
            else
                alpha *= alpha_factor
            end
        end

        c = (cs[i], alpha)
        npe = size(evt, 1)
        ndraw = min(256, npe)
        draw_inds = randperm(rng, npe)[1:ndraw]

        evt_draw = evt[draw_inds, :]

        x = evt_draw[!, :mass_1_source]
        y = evt_draw[!, :mass_2_source]
        kde_pts = vcat(x', y')
        k = KDE(kde_pts)
        p_kde_pts = [pdf(k, [kde_pts[1,i], kde_pts[2,i]]) + pdf(k, [kde_pts[2,i], kde_pts[1,i]]) for i in axes(kde_pts, 2)]

        p = quantile(p_kde_pts, level)

        dx = maximum(x) - minimum(x)
        xgrid = range(minimum(x)-0.1*dx, maximum(x)+0.1*dx, length=128)

        dy = maximum(y) - minimum(y)
        ygrid = range(minimum(y)-0.1*dy, maximum(y)+0.1*dy, length=129)

        zgrid = [(x ≥ y ? pdf(k, [x, y]) + pdf(k, [y, x]) : zero(x)) for x in xgrid, y in ygrid]

        contour!(a, xgrid, ygrid, zgrid; levels=[p], color=c)
    end

    if include_legend
        Legend(f[2,1],
            [LineElement(color=cs[i], linestyle=nothing) for i in 1:length(evt_groups)],
            [evt[1, :gwname] for evt in evt_groups],
            nbanks=3)
    end

    return f
end





function qchieff_kde_plot(df; mc_min = chirp_mass(20.0, 20.0), mc_max=chirp_mass(50.0, 50.0), include_legend=false, p_cut=0.5, p_cut_eventlist=0.5, rng=Random.default_rng(), far_thresh=1)
    if p_cut != p_cut_eventlist
        @info("Colors will correspond to p > $(p_cut_eventlist), but only events with p > $(p_cut) will be shown")
    end

    level = 0.1

    evt_groups = DataFrames.groupby(df, :gwname)
    evt_groups = [evt for evt in evt_groups if frac_chirp_mass_bounds(evt, mc_min, mc_max) >= p_cut_eventlist && evt[1,:far] < far_thresh]

    cs = categorical_palette(length(evt_groups))

    f = Figure()
    a = Axis(f[1, 1]; xlabel=L"\chi_\mathrm{eff}", ylabel=L"q", limits=(-1,1,0,1), title=L"p = %$(p_cut)", xminorgridvisible=true, xminorticksvisible=true, yminorgridvisible=true, yminorticksvisible=true)

    @progress for (i, evt) in enumerate(evt_groups)
        if frac_chirp_mass_bounds(evt, mc_min, mc_max) < p_cut
            continue
        end

        c = cs[i]
        npe = size(evt, 1)
        ndraw = min(256, npe)
        draw_inds = randperm(rng, npe)[1:ndraw]

        evt_draw = evt[draw_inds, :]

        x = evt_draw[!, :chi_eff]
        y = evt_draw[!, :mass_ratio]
        kde_pts = vcat(x', y')
        k = KDE(vcat(x', y'))
        p_kde_pts = [pdf(k, [kde_pts[1,i], kde_pts[2,i]]) + pdf(k, [kde_pts[1,i], 2-kde_pts[2,i]]) for i in axes(kde_pts, 2)]

        p = quantile(p_kde_pts, level)

        dx = maximum(x)-minimum(x)
        xgrid = range(minimum(x)-0.1*dx, maximum(x)+0.1*dx, length=128)

        dy = maximum(y)-minimum(y)
        ygrid = range(minimum(y)-0.1*dy, maximum(y)+0.1*dy, length=129)

        zgrid = [pdf(k, [x, y]) + pdf(k, [x, 2-y]) for x in xgrid, y in ygrid]

        contour!(a, xgrid, ygrid, zgrid; levels=[p], color=c)
    end

    if include_legend
        Legend(f[2,1],
            [LineElement(color=cs[i], linestyle=nothing) for i in 1:length(evt_groups)],
            [evt[1, :gwname] for evt in evt_groups],
            nbanks=3)
    end

    f
end

function m1_m2_posterior_contours(trace; levels=[0.1], rng=Random.default_rng(), ndraw_levels=128)
    p = trace.posterior
    cs = categorical_palette(size(p.m1s, :event))

    f = Figure()
    ax = Axis(f[1,1], xlabel=L"m_1 / M_\odot", ylabel=L"m_2 / M_\odot")

    @progress for (i, c) in zip(axes(p.m1s, :event), cs)
        m1 = vec(p.m1s[event=i])
        q = vec(p.qs[event=i])
        m2 = m1 .* q

        pts = hcat([[m, n] for (m,n) in zip(m1, m2)]...)
        kde = KDE(pts)

        level_inds = rand(rng, axes(m1, 1), ndraw_levels)
        p_pts = [pdf(kde, [m, n]) + pdf(kde, [n, m]) for (m,n) in zip(m1[level_inds], m2[level_inds])]

        plevel = [quantile(p_pts, l) for l in levels]

        dm1 = maximum(m1) - minimum(m1)
        dm2 = maximum(m2) - minimum(m2)

        m1grid = range(minimum(m1)-0.1*dm1, stop=maximum(m1)+0.1*dm1, length=32)
        m2grid = range(minimum(m2)-0.1*dm2, stop=maximum(m2)+0.1*dm2, length=31)

        probs = [(m > n ? pdf(kde, [m,n]) + pdf(kde, [n,m]) : 0.0) for m in m1grid, n in m2grid]
        contour!(ax, m1grid, m2grid, probs, levels=plevel, color=c)
    end
    f
end

function mchirp_dl_result_plot(trace; rng=Random.default_rng(), nsamp=256, levels=[0.1], mc_max=125.0, dl_max=10.0, q_plot=0.8)
    p = trace.posterior
    Nevt = length(dims(p, :event))

    zs_grid = expm1.(range(log(1), log(1+4), length=1024))
    dls_grid = @. ustrip(u"Gpc", luminosity_dist((BumpCosmologyGWTC3.lvk_cosmology, ), zs_grid))

    f = Figure()
    a = Axis(f[1,1], xlabel=L"d_L / \mathrm{Gpc}", ylabel=L"\mathcal{M}_{\mathrm{det}} / M_\odot", limits=(0, dl_max, 0, mc_max))

    # Selection band
    band!(a, dls_grid, zeros(length(dls_grid)), default_mc_min .* (1 .+ zs_grid), color=(:grey, 0.1))
    band!(a, dls_grid, default_mc_max .* (1 .+ zs_grid), mc_max .* ones(length(dls_grid)), color=(:grey, 0.1))

    # @progress for (evt, clr) in zip(dims(p, :event), categorical_palette(Nevt))
    #     mcs = Float64[]
    #     dls = Float64[]
    #     for _ in 1:nsamp
    #         c = rand(rng, span(dims(p, :chain)))
    #         d = rand(rng, span(dims(p, :draw)))

    #         m1 = p.m1s[chain=At(c), draw=At(d), event=At(evt)]
    #         q = p.qs[chain=At(c), draw=At(d), event=At(evt)]
    #         m2 = q .* m1

    #         z = p.zs[chain=At(c), draw=At(d), event=At(evt)]

    #         c = cosmology(; h=p.h[chain=At(c), draw=At(d)], OmegaM=p.Ω_M[chain=At(c), draw=At(d)])

    #         dl = ustrip(u"Gpc", luminosity_dist(c, z))

    #         push!(mcs, BumpCosmologyGWTC3.chirp_mass(m1, m2)*(1+z))
    #         push!(dls, dl)
    #     end

    #     kdecontour!(a, dls, mcs, color=clr, levels=levels)
    # end
    for (evt, clr) in zip(dims(p, :event), categorical_palette(Nevt))
        m1 = p.m1s[event=At(evt)]
        q = p.qs[event=At(evt)]
        z = p.zs[event=At(evt)]

        mc = BumpCosmologyGWTC3.chirp_mass.(m1, q.*m1) .* (1 .+ z)
        dl = [ustrip(u"Gpc", luminosity_dist(cosmology(; h=h, OmegaM=Ω_M), z)) for (z, h, Ω_M) in zip(p.zs[event=At(evt)], p.h, p.Ω_M)]

        mcm = mean(mc)
        dlm = mean(dl)

        dmc = std(mc)
        ddl = std(dl)

        scatter!(a, [dlm], [mcm], color=clr)
        errorbars!(a, [dlm], [mcm], [dmc], direction=:y, color=clr)
        errorbars!(a, [dlm], [mcm], [ddl], direction=:x, color=clr)
    end

    mcs_grid = []
    for c in eachindex(dims(p, :chain))
        for d in eachindex(dims(p, :draw))
            mbm = p.mbhmax[chain=c, draw=d]

            mc = BumpCosmologyGWTC3.chirp_mass(mbm, q_plot*mbm)
    
            cosm = cosmology(; h=p.h[chain=c, draw=d], OmegaM=p.Ω_M[chain=c, draw=d])
            d = @. ustrip(u"Gpc", luminosity_dist((cosm, ), zs_grid))
            m = mc .* (1 .+ zs_grid)

            push!(mcs_grid, [BumpCosmologyGWTC3.interp1d(dg, d, m) for dg in dls_grid])
        end
    end
    mcs_grid = hcat(mcs_grid...)
    mm = vec(median(mcs_grid, dims=2))
    ml = [quantile([mcs_grid[i, j] for j in axes(mcs_grid, 2)], 0.16) for i in axes(mcs_grid, 1)]
    mh = [quantile([mcs_grid[i, j] for j in axes(mcs_grid, 2)], 0.84) for i in axes(mcs_grid, 1)]
    mll = [quantile([mcs_grid[i, j] for j in axes(mcs_grid, 2)], 0.025) for i in axes(mcs_grid, 1)]
    mhh = [quantile([mcs_grid[i, j] for j in axes(mcs_grid, 2)], 0.975) for i in axes(mcs_grid, 1)]

    lines!(a, dls_grid, mm, color=:black)
    band!(a, dls_grid, ml, mh, color=(:black, 0.25))
    band!(a, dls_grid, mll, mhh, color=(:black, 0.25))

    f
end

function h_plot(trace)

end