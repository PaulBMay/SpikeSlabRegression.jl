using SpikeSlabRegression

using LinearAlgebra
using ProgressBars
using Random, Distributions
using Plots
using StatsFuns


n = 1_000

nω = 100
ωlb = 5
ωub = 10

ω = range(ωlb, ωub, nω)

σ = 1

zprob = 0.05

Random.seed!(92)

t = sort(rand(n))

amp = 3 .* randn(nω, 2)
phase = rand(nω)
z = rand(nω) .< zprob
sum(z)

α = 2

y = fill(α, n) + σ*randn(n)

for j in 1:nω
    y += z[j]*( amp[j,1]*cospi.(2*ω[j]*t .- phase[j]) + amp[j,2]*sinpi.(2*ω[j]*t .- phase[j]) )
end


plot(t, y)



priors = (zprob = zprob, Bprec = [1,1], τ = SpikeSlabRegression.gammashaperate(1, 0.25))
nsamps = 10_000

samples = BSSR(y, t, ω, priors, nsamps; progress = false)

plot(samples.τ)
plot(sqrt.(1 ./ samples.τ))

plot(ω, mean(samples.z, dims = 1)')
scatter!(ω, z)
z


plot(ω, samples.amp[(end-100):end,:]')

ampmu = mean(samples.amp, dims = 1)[1,:]

amptrue = norm.(eachrow(amp)) .* z

ampmu

scatter(ampmu, amptrue)


