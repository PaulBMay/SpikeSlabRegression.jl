using SpikeSlabRegression

using LinearAlgebra
using ProgressBars
using Random, Distributions
using Plots
using StatsFuns


n = 5_000

L = 80

nω = 100
ωlb = 0.01
ωub = 1

ω = range(ωlb, ωub, nω)

σ = 1

zprob = 0.05

Random.seed!(123)

t = sort(L*rand(n))

amp = 3 .* randn(nω, 2)
amptrue = vec(sqrt.(sum(amp.^2, dims = 2)))
phase = rand(nω)
z = rand(nω) .< zprob
sum(z)

scatter(ω, z)

amptrue = vec(sqrt.(sum(amp.^2, dims = 2))) .* z


α = 2

y = fill(α, n) + σ*randn(n)

for j in 1:nω
    y += z[j]*( amp[j,1]*cospi.(2*ω[j]*t .- phase[j]) + amp[j,2]*sinpi.(2*ω[j]*t .- phase[j]) )
end


plot(t, y)



priors = (zprob = zprob, Bprec = [1,1], τ = SpikeSlabRegression.gammashaperate(1, 0.25))
nsamps = 10_000

samples = BSSR(y, t, ω, priors, nsamps; progress = false)

ampmu = mean(samples.amp, dims = 1)[1,:]
amplb = quantile.(eachcol(samples.amp), 0.025)
ampub = quantile.(eachcol(samples.amp), 0.975)

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

######################

Y = 1.0*randn(nsamps, n) .+ y'


samples2 = BSSR(Y, t, ω, priors, nsamps; progress = true)

ampmu2 = mean(samples2.amp, dims = 1)[1,:]
ampmed2 = median.(eachcol(samples2.amp))
amplb2 = quantile.(eachcol(samples2.amp), 0.025)
ampub2 = quantile.(eachcol(samples2.amp), 0.975)




plot(ω, ampmu2)
plot!(ω, amptrue)


priors3 = (zshape = [0.25, 5], Bprec = [1,1], τ = SpikeSlabRegression.gammashaperate(1, 0.25))

samples3 = SpikeSlabRegression.BSSR2(Y, t, ω, priors3, nsamps; progress = true)


ampmu3 = mean(samples3.amp, dims = 1)[1,:]
ampmed3 = median.(eachcol(samples3.amp))
amplb3 = quantile.(eachcol(samples3.amp), 0.025)
ampub3 = quantile.(eachcol(samples3.amp), 0.975)

plot(ω, ampmu3)
plot!(ω, amptrue)