"""
    load_pe_from_dir(dir)

Load parameter estimation samples from all the `_nocosmo.h5` files in the given
directory.

Returns a DataFrame with all the samples, with an additional column `gwname`
giving the full (GWYYMMDD_NNNNNNN) name of the event to which the samples
belong.
"""
function load_pe_from_dir(dir)
    dfs = []
    @progress name="Loading Directory" for file in readdir(dir)
        if occursin(r"GW[0-9]+.*_nocosmo.h5", file)
            gwname = match(r".*(GW[0-9]+_[0-9]+).*", file).captures[1]
            h5open(joinpath(dir, file), "r") do f
                k = keys(f)
                if "C01:Mixed" in keys(f)
                    samps = read(f["C01:Mixed/posterior_samples"])

                    d = DataFrame(samps)
                    d[!, :gwname] .= gwname

                    push!(dfs, d)
                else
                    @info "Could not read $file"
                end
            end
        end
    end
    vcat(dfs...; cols=:intersect)
end

function load_pe_from_dir_O4(; dir, run)
    dfs = DataFrame[]

    @showprogress "Loading IGWN HDF5 Files" for file in readdir(dir)
        if startswith(file, "IGWN") && endswith(file, ".hdf5")

            m = match(r"(GW[0-9]+_[0-9]+)", file)

            if m === nothing
                @warn "Could not extract GW name from $file"
                continue
            end

            gwname = m.captures[1]
            fullpath = joinpath(dir, file)

            try
                h5open(fullpath, "r") do f

                    available = keys(f)

                    # Choose PE analysis according to observing-run priority
                    if run == :O4a

                        if "C00:NRSur7dq4" in available
                            analysis = "C00:NRSur7dq4"

                        elseif "C00:Mixed" in available
                            analysis = "C00:Mixed"

                        else
                            @warn "No suitable O4a PE analysis found in $file"
                            @info "Available analyses: $(collect(available))"
                            return
                        end

                    elseif run == :O4b

                        if "C00:NRSur7dq4" in available
                            analysis = "C00:NRSur7dq4"

                        elseif "C00:IMRPhenomXPHM-SpinTaylor" in available
                            analysis = "C00:IMRPhenomXPHM-SpinTaylor"

                        else
                            @warn "No suitable O4b PE analysis found in $file"
                            @info "Available analyses: $(collect(available))"
                            return
                        end

                    else
                        error("run must be :O4a or :O4b")
                    end

                    @info "Using $analysis for $gwname"

                    samps = read(f["$analysis/posterior_samples"])
                    d = DataFrame(samps)

                    d[!, :gwname] .= gwname
                    push!(dfs, d)
                end

            catch e
                @warn "Failed to read $file: $e"
            end
        end
    end

    return isempty(dfs) ? DataFrame() : vcat(dfs...; cols=:intersect)
end

"""
    load_selection(file)

Given a path to a file containing mock injections, load it into a data frame.

Returns `(df, T_in_years, N)` where `df` is the data frame, `T_in_years` is the
analysis time in years over which the injections have been generated, and `N` is
the total number of injections generated.

The data frame will include generated columns `q` and `chi_eff` and
`sampling_pdf_qchieff` which gives the (marginal) sampling PDF for parameters
`mass1`, `q`, `chi_eff`, `z`.
"""

function load_selection(file)
    h5open(file, "r") do f
        # info
        info = f["info"]
        T_yr = info["analysis_time_yr"][]
        N_selection = info["total_injections"][]

        # injections
        df = DataFrame(read(f["injections"]))

        # ensure required columns exist, then overwrite with the ratio
        if all(in(names(df)), ["sampling_pdf_m1dqdlchieff", "weights"])
            df.sampling_pdf_m1dqdlchieff = df.sampling_pdf_m1dqdlchieff ./ df.weights
        end

        return (df, T_yr, N_selection)
    end
end

"""
function load_selection(file)
    h5open(file, "r") do f
        df = DataFrame(read(f["injections"]))
        T_yr = attrs(f)["analysis_time_s"] / (3600 * 24 * 365.25)
        N = attrs(f)["total_generated"]

        df.mass_1 = @. df.mass1_source * (1 + df.redshift)
        df.q = @. df.mass2_source ./ df.mass1_source
        df.chi_eff = @. (df.spin1z + df.q * df.spin2z) / (1 + df.q)
        df.luminosity_distance = @. ustrip(u"Gpc", luminosity_dist((lvk_cosmology, ), df.redshift))

        a1 = @. sqrt(df.spin1x^2 + df.spin1y^2 + df.spin1z^2)
        a2 = @. sqrt(df.spin2x^2 + df.spin2y^2 + df.spin2z^2)

        spin_sampling_pdf = @. 1 / (16 * π^2 * a1^2 * a2^2)

        df.sampling_pdf_qchieff = @. df.sampling_pdf / spin_sampling_pdf * df.mass1_source * chi_eff_marginal(df.chi_eff, df.q)
        df.sampling_pdf_q = @. df.sampling_pdf * df.mass1_source

        dc = @. ustrip(u"Gpc", comoving_transverse_dist((lvk_cosmology, ), df.redshift))
        dh_z = @. ustrip(u"Gpc", 2.99792e8*u"m"/u"s" / Cosmology.H((lvk_cosmology, ), df.redshift))

        df.sampling_pdf_m1dqdlchieff = @. df.sampling_pdf_qchieff / (1 + df.redshift) / (dc + (1+df.redshift)*dh_z)

        (df, T_yr, N)
    end
end
"""

"""
    load_event_table(file)

Loads the file containing a JSON representation of a GWOSC event table and
returns a DataFrame.
"""
function load_event_table(file)
    open(file, "r") do f
        j = JSON.parse(f)["events"]
        df = DataFrame()
        for k in keys(j)
            push!(df, j[k], cols=:union)
        end
        df[:, :fullname] = [k for k in keys(j)]
        dropmissing(df)
    end
end

"""
    name_or_prefix_in_nameset(n, set)

`true` if `n` or the `GWYYMMDD` prefix of `n` (before the '_' character) is
found in the `Set` of names.

This function is useful for filtering a DataFrame of event PE to correspond to
some collection of events from GWOSC (i.e. a catalog).
"""
function name_or_prefix_in_nameset(n, set)
    (n in set) || (n[1:prevind(n, findfirst('_', n))] in set)
end

"""
    name_to_common_name(n, common_names)

Given a `GWYYMMDD_NNNNNNN` name and a set of "common" names (e.g. `GW150914` for
the famous first signal), return the common name associated to `n`.

Raises an error if `n` or its "common" prefix is not found in the set of common
names.
"""
function name_to_common_name(n, common_names)
    if n in common_names
        n
    else
        pn = n[1:prevind(n, findfirst('_', n))]
        if pn in common_names
            pn
        else
            error("Could not find $(n) in common name set")
        end
    end
end

function merge_pe_evt_tables(pe_df, evt_df)
    pe_df = subset(groupby(pe_df, :gwname), :gwname => gwns -> name_or_prefix_in_nameset(gwns[1], evt_df[!, :commonName]), ungroup=true)
    pe_df.commonName = name_to_common_name.(pe_df.gwname, (Set(evt_df.commonName), ))
    innerjoin(pe_df, select(evt_df, :commonName, :far), on=:commonName)
end

function load_pe( ; gwtc_2_dir = "/Users/wfarr/Research/gwtc-2.1", gwtc_3_dir = "/Users/wfarr/Research/o3b_data/PE")
    df2 = load_pe_from_dir(gwtc_2_dir)
    df3 = load_pe_from_dir(gwtc_3_dir)
    df = vcat(df2, df3; cols=:intersect)

    df
end

function join_pe_evt_tables(pe_df, evt_df)
    pe_df = subset(groupby(pe_df, :gwname), :gwname => gwns -> name_or_prefix_in_nameset(gwns[1], evt_df[!, :commonName]), ungroup=true)
    pe_df.commonName = name_to_common_name.(pe_df.gwname, (Set(evt_df.commonName), ))
    innerjoin(pe_df, select(evt_df, :commonName, :far), on=:commonName)
end
