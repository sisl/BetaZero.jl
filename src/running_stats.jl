using Parameters
using Statistics

# https://stackoverflow.com/a/17637351
@with_kw mutable struct RunningStats
    n = 0
    old_μ = 0
    new_μ = 0
    old_σ = 0
    new_σ = 0
end

clear!(rs::RunningStats) = rs.n = 0

function Base.push!(rs::RunningStats, x)
    rs.n += 1

    if rs.n == 1
        rs.old_μ = rs.new_μ = x
        rs.old_σ = 0
    else
        rs.new_μ = rs.old_μ + (x - rs.old_μ) / rs.n
        rs.new_σ = rs.old_σ + (x - rs.old_μ) * (x - rs.new_μ)

        rs.old_μ = rs.new_μ
        rs.old_σ = rs.new_σ
    end
end

Statistics.mean(rs::RunningStats) = rs.n == 0 ? 0.0 : rs.new_μ
Statistics.var(rs::RunningStats) = rs.n > 1 ? rs.new_σ / (rs.n - 1) : 0.0
Statistics.std(rs::RunningStats) = sqrt(var(rs))
