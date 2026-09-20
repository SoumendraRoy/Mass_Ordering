## Preliminaries

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using AdvancedHMC
using ArviZ
using BumpCosmologyGWTC3
using CairoMakie
using Colors
using Cosmology
using DataFrames
using DimensionalData
using Distributions
using GaussianKDEs
using HDF5
using InferenceObjects
using JSON
using LaTeXStrings
using MCMCChainsStorage
using NCDatasets
using PairPlots
using PolyLog
using PopModels
using Printf
using ProgressLogging
using Random
using StatsBase
using SpecialFunctions
using StatsFuns
using Trapz
using Turing
using Unitful
using UnitfulAstro
using UnitfulChainRules

include("plots.jl")

## Samples and Selection

## Set up paths
struct Paths
    gwtc_2_dir::String
    gwtc_3_dir::String
    evt_table_file::String
    o1o2o3_sensitivity_file::String
end

system = :rusty
if system == :rusty
    paths = Paths(
        "/mnt/home/ccalvk/ceph/GWTC-2.1", 
        "/mnt/home/ccalvk/ceph/GWTC-3", 
        "/mnt/home/ccalvk/ceph/gwosc-snapshots/snapshot-2023-11-04/GWTC/GWTC.json",
        "/mnt/home/ccalvk/ceph/sensitivity-estimates/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5"
    )
elseif system == :wmflaptop
    paths = Paths(
        "/Users/wfarr/Research/gwtc-2.1",
        "/Users/wfarr/Research/o3b_data/PE",
        "/Users/wfarr/Research/gwosc-snapshots/snapshot-2023-11-04/GWTC/GWTC.json",
        "/Users/wfarr/Research/o3b_data/O1O2O3-Sensitivity/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5"
    )
end

## Load PE
all_pe = load_pe(; gwtc_2_dir=paths.gwtc_2_dir, gwtc_3_dir=paths.gwtc_3_dir)
evt_table = load_event_table(paths.evt_table_file)
all_pe = join_pe_evt_tables(all_pe, evt_table)
all_pe[:, :prior_logwt_m1qzchie] = li_nocosmo_prior_logwt_m1qzchie(all_pe)
pe_table = far_cut(chirp_mass_cut(all_pe))

## Check number of events
n0p1, n0p5, n0p9 = length(groupby(far_cut(chirp_mass_cut(all_pe, thresh=0.1)), :gwname)), length(groupby(far_cut(chirp_mass_cut(all_pe, thresh=0.5)), :gwname)), length(groupby(far_cut(chirp_mass_cut(all_pe, thresh=0.9)), :gwname))
@info "N(p > 0.1) = $(n0p1); N(p > 0.5) = $(n0p5); N(p > 0.9) = $(n0p9)"

## m1-m2 KDE plot
f = m1m2_kde_plot(all_pe; p_cut=[0.5, 0.1], p_cut_eventlist=0.0, draw_cut_lines=true)
save(joinpath(@__DIR__, "..", "figures", "m1m2_kde.pdf"), f)
f

## m1-chi_eff contour plot
# Just plot the 0.5 sample:
pe = far_cut(chirp_mass_cut(all_pe, thresh=0.5))
pe_grouped = groupby(pe, :gwname)
n = length(pe_grouped)

cs = categorical_palette(n)
rng = Random.Xoshiro(0x7d53228aaa6650d2)

f = Figure()
a = Axis(f[1,1], xlabel=L"m_1 / M_\odot", ylabel=L"\chi_\mathrm{eff}", xscale=log10, xticks=[10, 100], xminorticks=[20,30,40,50,60,70,80,90], limits=(10, 100, nothing, nothing), xminorticksvisible=true)

@progress for (c, evt) in zip(cs, pe_grouped)
    inds = rand(rng, 1:size(evt, 1), 256)
    e = evt[inds, :]
    kdecontour!(a, e.mass_1_source, e.chi_eff, color=c, levels=[0.1])
end
save(joinpath(@__DIR__, "..", "figures", "m1_chi_eff_kde.pdf"), f)
f

## q-chi_eff contour plot
# Just plot the 0.5 sample:
pe = far_cut(chirp_mass_cut(all_pe, thresh=0.5))
pe_grouped = groupby(pe, :gwname)
n = length(pe_grouped)

cs = categorical_palette(n)
rng = Random.Xoshiro(0xaf91589a0f32ada4)

f = Figure()
a = Axis(f[1,1], xlabel=L"q", ylabel=L"\chi_\mathrm{eff}")

@progress for (c, evt) in zip(cs, pe_grouped)
    inds = rand(rng, 1:size(evt, 1), 256)
    e = evt[inds, :]
    kdecontour!(a, e.mass_ratio, e.chi_eff, color=c, levels=[0.1])
end
f

## Load Traces
trace_samples_0p9 = from_netcdf(joinpath(@__DIR__, "..", "data", "chains.nc"))
ne_min = dropdims(minimum(trace_samples_0p9.posterior.Neff_samps, dims=(:chain, :draw)); dims=(:chain, :draw))
ns_min = minimum(trace_samples_0p9.posterior.Neff_sel)
@info "p_cut = 0.9, minimum Neff_samps: $(round(minimum(ne_min), digits=2)), minimum Neff_sel = $(round(ns_min, digits=1)), 4*Nobs = $(4*length(dims(trace_samples_0p9.posterior, :event)))"

trace_samples_0p5 = from_netcdf(joinpath(@__DIR__, "..", "data", "chains_0.5.nc"))
ne_min = dropdims(minimum(trace_samples_0p5.posterior.Neff_samps, dims=(:chain, :draw)); dims=(:chain, :draw))
ns_min = minimum(trace_samples_0p5.posterior.Neff_sel)
@info "p_cut = 0.5, minimum Neff_samps: $(round(minimum(ne_min), digits=2)), minimum Neff_sel = $(round(ns_min, digits=1)), 4*Nobs = $(4*length(dims(trace_samples_0p5.posterior, :event)))"

trace_samples_0p1 = from_netcdf(joinpath(@__DIR__, "..", "data", "chains_0.1.nc"))
ne_min = dropdims(minimum(trace_samples_0p1.posterior.Neff_samps, dims=(:chain, :draw)); dims=(:chain, :draw))
ns_min = minimum(trace_samples_0p1.posterior.Neff_sel)
@info "p_cut = 0.1, minimum Neff_samps: $(round(minimum(ne_min), digits=2)), minimum Neff_sel = $(round(ns_min, digits=1)), 4*Nobs = $(4*length(dims(trace_samples_0p1.posterior, :event)))"

## Assign the relevant trace
trace = trace_samples_0p5

## Traceplot
Nevt = length(dims(trace.posterior, :event))
f = traceplot(trace; params=population_params) # vcat(population_params, cosmo_params))
save(joinpath(@__DIR__, "..", "figures", "traceplot.pdf"), f)
f

## Summary Table, Check the Stats
sstats = ArviZ.summarize(trace)
min_ess = minimum(Tables.getcolumn(sstats, :ess_bulk))
imin = argmin(Tables.getcolumn(sstats, :ess_bulk))

@info "Minimum ESS: $(round(min_ess, digits=2))"
Tables.subset(sstats, imin)

## Corner of Everything vs Everything
df = DataFrame([
    k => vec(trace.posterior[k])
    for k in population_params]...
)
f = pairplot(df)
save(joinpath(@__DIR__, "..", "figures", "corner.pdf"), f)
f

## Corner Compare Selection
c1 = Makie.wong_colors(0.33)[1]
c2 = Makie.wong_colors(0.33)[2]
c3 = Makie.wong_colors(0.33)[3]
f = pairplot(
    PairPlots.Series(
        DataFrame(
            [k => vec(trace_samples_0p9.posterior[k]) for k in population_params]...
        ),
        label=L"p > 0.9",
        color=c1,
        strokecolor=c1
    ),
    PairPlots.Series(
        DataFrame(
            [k => vec(trace_samples_0p5.posterior[k]) for k in population_params]...
        ),
        label=L"p > 0.5",
        color=c2,
        strokecolor=c2
    ),
    PairPlots.Series(
        DataFrame(
            [k => vec(trace_samples_0p1.posterior[k]) for k in population_params]...
        ),
        label=L"p > 0.1",
        color=c3,
        strokecolor=c3
    )
)
save(joinpath(@__DIR__, "..", "figures", "corner_compare_selection.pdf"), f)
f

## High-Mass Parameters Versus Selection
# It is useful to look at the high-mass power law across selection cuts as
# well---you can see that weakening the selection cut (i.e. including more
# events) changes the high-mass power law slope (it makes it less steep), and
# increases the high-mass power law rate.  This might be expected if you thought
# that the events on the boundary were "high-mass" and not really contributing
# to the "peak" at $35 \, M_\odot$.
high_mass_pl_params = [:log_rhm, :alphahm]

c1 = Makie.wong_colors(0.33)[1]
c2 = Makie.wong_colors(0.33)[2]
c3 = Makie.wong_colors(0.33)[3]
f = pairplot(
    PairPlots.Series(
        DataFrame(
            [k => vec(trace_samples_0p9.posterior[k]) for k in high_mass_pl_params]...
        ),
        label=L"p > 0.9",
        color=c1,
        strokecolor=c1
    ),
    PairPlots.Series(
        DataFrame(
            [k => vec(trace_samples_0p5.posterior[k]) for k in high_mass_pl_params]...
        ),
        label=L"p > 0.5",
        color=c2,
        strokecolor=c2
    ),
    PairPlots.Series(
        DataFrame(
            [k => vec(trace_samples_0p1.posterior[k]) for k in high_mass_pl_params]...
        ),
        label=L"p > 0.1",
        color=c3,
        strokecolor=c3
    )
)
save(joinpath(@__DIR__, "..", "figures", "corner_high_mass_compare_selection.pdf"), f)
f

## Peak Parameters Versus Selection
# We can see that the addition of the high-mass power law has stabilized the
# peak parameters across selection cuts:
peak_params = [:alpha, :mtr, :mbhmax, :sigma]

c1 = Makie.wong_colors(0.33)[1]
c2 = Makie.wong_colors(0.33)[2]
c3 = Makie.wong_colors(0.33)[3]
f = pairplot(
    PairPlots.Series(
        DataFrame(
            [k => vec(trace_samples_0p9.posterior[k]) for k in peak_params]...
        ),
        label=L"p > 0.9",
        color=c1,
        strokecolor=c1
    ),
    PairPlots.Series(
        DataFrame(
            [k => vec(trace_samples_0p5.posterior[k]) for k in peak_params]...
        ),
        label=L"p > 0.5",
        color=c2,
        strokecolor=c2
    ),
    PairPlots.Series(
        DataFrame(
            [k => vec(trace_samples_0p1.posterior[k]) for k in peak_params]...
        ),
        label=L"p > 0.1",
        color=c3,
        strokecolor=c3
    )
)
save(joinpath(@__DIR__, "..", "figures", "corner_peak_compare_selection.pdf"), f)
f

## Population Plots

## Beta chi_eff
f = beta_chi_eff_plot(trace)
save(joinpath(@__DIR__, "..", "figures", "beta_chi_eff.pdf"), f)
f

## Chi_eff sigmas corner
f = pairplot(PairPlots.Series(DataFrame(L"\sigma_{q=0}" => vec(trace.posterior.sigmal), L"\sigma_{q=1}" => vec(trace.posterior.sigmah))))
save(joinpath(@__DIR__, "..", "figures", "chi_eff_sigmas.pdf"), f)
f

## Chi_eff sigmas scatter
# We can see that the scatter at small q is much less well constrained, so most
# of the samples have sigma(q=1) < sigma(q=0), but it's not certain
f = Figure()
a = Axis(f[1,1], xlabel=L"\sigma_{q=0}", ylabel=L"\sigma_{q=1}")
scatter!(a, vec(trace.posterior.sigmal), vec(trace.posterior.sigmah))
lines!(a, [0, 1], [0, 1], color=:black)
save(joinpath(@__DIR__, "..", "figures", "chi_eff_sigmas_scatter.pdf"), f)
f

## alpha chi_eff
f = Figure()
a = Axis(f[1,1], xlabel=L"\alpha_{\chi_\mathrm{eff}}", ylabel="Density")
density!(a, vec(trace.posterior.alpha_chi_eff))
f

## Chi_eff mus corner
f = pairplot(PairPlots.Series(DataFrame(L"\mu_{q=0}" => vec(trace.posterior.mul), L"\mu_{q=1}" => vec(trace.posterior.muh))))
save(joinpath(@__DIR__, "..", "figures", "chi_eff_mus.pdf"), f)
f

## Marginal m1 distribution
f = marginal_m1_plot(trace)
save(joinpath(@__DIR__, "..", "figures", "m1_marginal.pdf"), f)
f

## Compare marginal m1 distribution on selection
f = marginal_m1_plot(trace_samples_0p9, label=L"p_\mathrm{cut} = 0.9")
a = f.content[1,1]

marginal_m1_plot!(a, trace_samples_0p5; color=Makie.wong_colors()[2], label=L"p_\mathrm{cut} = 0.5")
marginal_m1_plot!(a, trace_samples_0p1; color=Makie.wong_colors()[3], label=L"p_\mathrm{cut} = 0.1")
axislegend(a)
save(joinpath(@__DIR__, "..", "figures", "m1_marginal_compare_selection.pdf"), f)
f

## Remnant mass distribution
f = remnant_mass_dist_plot(trace)
save(joinpath(@__DIR__, "..", "figures", "remnant_mass_dist.pdf"), f)
f

## Remnant mass distribution on selection
f = remnant_mass_dist_plot(trace_samples_0p9, label=L"p_\mathrm{cut} = 0.9")
a = f.content[1,1]

remnant_mass_dist_plot!(a, trace_samples_0p5; color=Makie.wong_colors()[2], label=L"p_\mathrm{cut} = 0.5")
remnant_mass_dist_plot!(a, trace_samples_0p1; color=Makie.wong_colors()[3], label=L"p_\mathrm{cut} = 0.1")
axislegend(a)
save(joinpath(@__DIR__, "..", "figures", "remnant_mass_dist_compare_selection.pdf"), f)
f

## Marginal q distribution
f = marginal_q_plot(trace)
save(joinpath(@__DIR__, "..", "figures", "q_marginal.pdf"), f)
f

## Marginal chi_eff distribution
f = marginal_chieff_plot(trace)
save(joinpath(@__DIR__, "..", "figures", "chi_eff_marginal.pdf"), f)
f

## Marginal redshift
f = marginal_z_plot(trace)
save(joinpath(@__DIR__, "..", "figures", "z_marginal.pdf"), f)
f

## m1-q plot
f = m1_q_plot(trace)
save(joinpath(@__DIR__, "..", "figures", "m1_q.pdf"), f)
f

## q-chi_eff plot
f = q_chieff_plot(trace)
save(joinpath(@__DIR__, "..", "figures", "q_chieff.pdf"), f)
f

## Antonini mu-sigma versus our fit
antonini_limit = 0.7
antonini_width = 2*antonini_limit
antonini_std = antonini_width / sqrt(12)

q_antonini = 1.0 # For 2G+2G mergers

f = Figure()
a = Axis(f[1,1], xlabel=L"\mu \mid_{q=%$(q_antonini)}", ylabel=L"\sigma \mid_{q=%$(q_antonini)}")
kdecontour!(a, chieff_interp.(q_antonini, vec(trace.posterior.mul), vec(trace.posterior.muh)), chieff_interp.(q_antonini, vec(trace.posterior.sigmal), vec(trace.posterior.sigmah)), levels=0.1:0.1:0.9)
scatter!(a, 0.0, antonini_std, markersize=10, color=:black)
save(joinpath(@__DIR__, "..", "figures", "antonini_mu_sigma.pdf"), f)
f

## m1-m2 contours pop informed.
f = m1_m2_posterior_contours(trace)
save(joinpath(@__DIR__, "..", "figures", "m1_m2_posterior_contours.pdf"), f)
f

## The population-informed q distributions for each event
n = length(dims(trace.posterior, :event))
kdes = [BoundedKDE(vec(trace.posterior.qs[event=At(evt)]), upper=1.0, lower=0.0) for evt in dims(trace.posterior, :event)]
p_at_q1 = [pdf(k, 1.0) for k in kdes]
intensity = [0.9*(sum(p_at_q1 .>= p)/length(p_at_q1)) + 0.1 for p in p_at_q1]

qs = 0:0.01:1

cs = categorical_palette(length(dims(trace.posterior, :event)))
f = Figure()
a = Axis(f[1,1], xlabel=L"q", ylabel=L"p(q)", limits=(0,1,nothing,nothing))
for i in 1:length(dims(trace.posterior, :event))
    lines!(qs, pdf.((kdes[i],), qs), color=(cs[i], intensity[i]))
end
save(joinpath(@__DIR__, "..", "figures", "q_posterior_contours.pdf"), f)
f

## Bump Structural Parameters
df = DataFrame([
    k => vec(trace.posterior[k])
    for k in [:alpha, :mtr, :mbhmax, :sigma]
])
f = pairplot(df)
save(joinpath(@__DIR__, "..", "figures", "bump_structural_params.pdf"), f)
f

## mbh_max 
mbm = vec(trace.posterior.mbhmax)
m = median(mbm); l = quantile(mbm, 0.16); h = quantile(mbm, 0.84)
f = Figure()
a = Axis(f[1,1], xlabel=L"m_{\mathrm{BH},\mathrm{max}}", title=L"m_{\mathrm{BH},\mathrm{max}} = %$(round(m, digits=1))^{+%$(round(h-m, digits=1))}_{-%$(round(m-l, digits=1))} \, M_\odot", limits=(25, 40, 0, nothing))
density!(a, mbm)
save(joinpath(@__DIR__, "..", "figures", "mbh_max.pdf"), f)
f