# From https://dcc.ligo.org/LIGO-P2000318/public/
const h_lvk = 0.679
const Ω_M_lvk = 0.3065
const lvk_cosmology = cosmology(h=h_lvk, OmegaM=Ω_M_lvk, w0=-1, wa=0)

function dH(h)
    h_lvk * 4.415205567010309 / h
end

function ez(z, Om)
    Ol = one(Om) - Om
    opz = one(z) + z
    opz2 = opz*opz
    opz3 = opz*opz2

    return sqrt(Om*opz3 + Ol)
end

function dc_over_dh(zs, Om)
    ezs = one(zs[1]) ./ ez.(zs, (Om,))
    cumtrapz(zs, ezs)
end 

function dl_over_dh(zs, dcs)
    dcs .* (one(zs[1]) .+ zs)
end

function dvdz_over_vh(zs, Om, dcs)
    (4 * pi) .* dcs .* dcs ./ ez.(zs, (Om,))
end

function ddldz_over_dh(zs, Om, dcs)
    dcs .+ (one(zs[1]) .+ zs) ./ ez.(zs, (Om,))
end