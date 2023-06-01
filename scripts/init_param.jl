# Credit: https://github.com/LAMDA-POMDP/Test/blob/main/test/Test.jl

## init_param for AdaOPS
@everywhere function init_param(m, bounds::AdaOPS.IndependentBounds)
    lower = init_param(m, bounds.lower)
    upper = init_param(m, bounds.upper)
    AdaOPS.IndependentBounds(lower, upper, bounds.check_terminal, bounds.consistency_fix_thresh)
end

@everywhere function init_param(m, bound::PORollout)
    policy = typeof(bound.solver) <: Solver ? solve(bound.solver, m) : bound.solver
    if typeof(bound.updater) <: BasicParticleFilter
        PORollout(policy, BasicParticleFilter(m,
                                                bound.updater.resampler,
                                                bound.updater.n_init,
                                                bound.updater.rng))
    else
        PORollout(policy, bound.updater)
    end
end

@everywhere function init_param(m, bound::FORollout)
    policy = typeof(bound.solver) <: Solver ? solve(bound.solver, m) : bound.solver
    FORollout(policy)
end

@everywhere function init_param(m, bound::SemiPORollout)
    policy = typeof(bound.solver) <: Solver ? solve(bound.solver, m) : bound.solver
    SemiPORollout(policy)
end

@everywhere function init_param(m, bound::POValue)
    policy = typeof(bound.solver) <: Solver ? solve(bound.solver, m) : bound.solver
    POValue(policy)
end

@everywhere function init_param(m, bound::FOValue)
    policy = typeof(bound.solver) <: Solver ? solve(bound.solver, UnderlyingMDP(m)) : bound.solver
    FOValue(policy)
end


## init_param for ARDESPOT
@everywhere function init_param(m, bounds::ARDESPOT.IndependentBounds)
    lower = init_param(m, bounds.lower)
    upper_policy = init_param(m, bounds.upper)
    if typeof(upper_policy) <: Policy
        upper = (p, b)->value(upper_policy, b)
    else
        upper = upper_policy
    end
    ARDESPOT.IndependentBounds(lower, upper, bounds.check_terminal, bounds.consistency_fix_thresh)
end

@everywhere function init_param(m, bound::ARDESPOT.FullyObservableValueUB)
    # if typeof(bound.p) <: QMDPSolver || typeof(bound.p) <: RSQMDPSolver
    if typeof(bound.p) <: RSQMDPSolver
        policy = solve(bound.p, m)
    elseif typeof(bound.p) <: Solver
        policy = solve(bound.p, UnderlyingMDP(m))
    else
        policy = bound.p
    end
    ARDESPOT.FullyObservableValueUB(policy)
end

@everywhere function ParticleFilters.unnormalized_util(p::AlphaVectorPolicy, b::AbstractParticleBelief)
    util = zeros(length(alphavectors(p)))
    for (i, s) in enumerate(particles(b))
        util .+= weight(b, i) .* getindex.(p.alphas, stateindex(p.pomdp, s))
    end
    return util
end

@everywhere function lower_bounded_zeta(d, k, zeta=0.8)
    max(zeta, 1 - (0.2*k + 0.2*(1-d)))
end

init_param(m, param) = param
init_param(m, param::S) where S <: Solver = solve(param, m)