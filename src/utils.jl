function cumtrapz(xs, ys)
    dx = xs[2:end] - xs[1:end-1]
    vcat(zero(ys[1]), cumsum((ys[1:end-1] + ys[2:end]) .* dx / 2))
end

function interp1d(x, xs, ys)
    i = searchsortedfirst(xs, x)

    if i == 1
        return ys[1]
    elseif i == lastindex(xs) + 1
        return ys[end]
    else
        r = (x - xs[i-1]) / (xs[i] - xs[i-1])
        return r*ys[i] + (one(r) - r)*ys[i-1]
    end
end

function chirp_mass(m1, m2)
    mt = m1+m2
    eta = m1*m2/(mt*mt)

    mt * eta^(3/5)
end

function mass1(mc, q)
    mc * (one(q) + q)^(1/5) / q^(3/5)
end