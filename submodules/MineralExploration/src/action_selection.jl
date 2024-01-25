struct GPNextAction{I}
    l::Float64
    sf::Float64
    sn::Float64
    init_a::I
end

# function GPNextAction(l::Float64, sf::Float64, sn::Float64, init_a::I=nothing) where I
#     return GPNextAction{I}(l, sf, sn, init_a)
# end

function kernel(x::MEAction, y::MEAction, l::Float64, sf::Float64)
    if x.type == y.type
        d = (x.coords[1] - y.coords[1])^2 + (x.coords[2] - y.coords[2])^2
        return sf*exp(-d/l^2)
    else
        return Inf
    end
end

function gp_posterior(X::Vector{MEAction}, x::Vector{MEAction},
                    y::Vector{Float64}, ns, l, sf, sn)
    m = length(X)
    n = length(x)
    Kxx = zeros(Float64, n, n)
    Kxz = zeros(Float64, n, m)
    Kzz = zeros(Float64, m)
    for i = 1:n
        for j = 1:n
            Kxx[i, j] = kernel(x[i], x[j], l, sf)
            if i == j
                Kxx[i, j] += sn/ns[i]
            end
        end
        for j = 1:m
            Kxz[i, j] = kernel(x[i], X[j], l, sf)
            Kzz[j] = kernel(X[j], X[j], l, sf)
        end
    end
    α = inv(Kxx)
    σ² = Kzz - diag(transpose(Kxz)*α*Kxz, 0)
    μ = transpose(Kxz)*α*y
    return (μ, σ²)
end

function approx_posterior(X::Vector{MEAction}, x::Vector{MEAction},
                        y::Vector{Float64}, ns, l, sf, sn)
    m = length(X)
    n = length(x)
    W = zeros(Float64, m, n)
    for i = 1:m
        for j = 1:n
            W[i, j] = kernel(X[i], x[j], l, sf)/sf*ns[j]
        end
    end

    w = sum(W, dims=2)
    σ² = sf./(1.0 .+ w)
    μ = W*y./w
    return (μ, σ²)
end

function expected_improvement(μ, σ², f)
    σ = sqrt.(σ²)
    dist = Normal(0.0, 1.0)
    Δ = μ .- f
    δ = Δ./σ
    ei = Δ.*(Δ .>= 0.0)
    for (i, d) in enumerate(δ)
        ei[i] += σ[i]*pdf(dist, d) - abs(Δ[i])*cdf(dist, d)
    end
    return ei
end

function POMCPOW.next_action(o::GPNextAction, pomdp::MineralExplorationPOMDP,
                            b::MEBelief, h)
    a_idxs = h.tree.tried[h.node]
    tried_actions = h.tree.a_labels[a_idxs]::Vector{MEAction}
    action_values = h.tree.q[a_idxs]
    action_ns = h.tree.n[a_idxs]
    actions = POMDPs.actions(pomdp, b)::Vector{MEAction}

    if length(tried_actions) > 0
        # μ, σ² = gp_posterior(actions, tried_actions, action_values, action_ns, o.l, o.sf, o.sn)
        μ, σ² = approx_posterior(actions, tried_actions, action_values, action_ns, o.l, o.sf, o.sn)
        f = maximum(action_values)
        ei = expected_improvement(μ, σ², f)
        a_idx = argmax(ei)
        a = actions[a_idx]
    elseif o.init_a != nothing
        a = POMCPOW.next_action(o.init_a, pomdp, b, h)
    else
        a = rand(actions)
    end
    return a
end

function POMCPOW.next_action(o::GPNextAction, pomdp::MineralExplorationPOMDP,
                            b::POMCPOW.StateBelief, h)
    a_idxs = h.tree.tried[h.node]
    tried_actions = h.tree.a_labels[a_idxs]::Vector{MEAction}
    action_values = h.tree.q[a_idxs]
    action_ns = h.tree.n[a_idxs]
    actions = POMDPs.actions(pomdp, b)::Vector{MEAction}

    if length(tried_actions) > 0
        # μ, σ² = gp_posterior(actions, tried_actions, action_values, action_ns, o.l, o.sf, o.sn)
        μ, σ² = approx_posterior(actions, tried_actions, action_values, action_ns, o.l, o.sf, o.sn)
        f = maximum(action_values)
        ei = expected_improvement(μ, σ², f)
        a_idx = argmax(ei)
        a = actions[a_idx]
    elseif o.init_a != nothing
        a = POMCPOW.next_action(o.init_a, pomdp, b, h)
    else
        a = rand(actions)
    end
    return a
end
