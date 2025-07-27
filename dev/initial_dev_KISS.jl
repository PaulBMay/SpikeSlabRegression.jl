using LinearAlgebra
using ProgressBars
using Random, Distributions
using Plots

n = 1_000

nω = 10
ωlb = 1
ωub = 5

ω = range(ωlb, ωub, nω)

σ = 0.1

zprob = 0.2

Random.seed!(96)

t = sort(rand(n))

amp = randn(nω, 2)
phase = rand(nω)
z = rand(nω) .< zprob

α = 2

y = fill(α, n) + σ*randn(n)

for j in 1:nω
    y += z[j]*( amp[j,1]*cospi.(2*ω[j]*t .- phase[j]) + amp[j,2]*sinpi.(2*ω[j]*t .- phase[j]) )
end


plot(t, y)

#############################
# Fit
###################

#= X = zeros(n, 2*nω + 1)

X[:,1] .= 1.0

for j in 1:nω
    col = 2 + 2*(j-1)
    X[:,col] .= cospi.(2*ω[j]*t)
    X[:,col+1] .= sinpi.(2*ω[j]*t)
end

B = (X'*X) \ (X'*y)

fit = X*B

scatter(t, y)
plot!(t, fit) # Looks great!

amp_hat = reshape(B[2:end], 2, nω)'

norm.(eachrow(amp_hat))
z.* norm.(eachrow(amp)) # Looks terrible!


##################
# If I know z?
###############

X2 = X[:, [true; repeat(z, inner = 2) ] ]

B2 = (X2'*X2) \ (X2'*y)

fit2 = X2*B2

scatter(t, y)
plot!(t, fit2) # Still looks great!

amp_hat2 = reshape(B2[2:end], 2, sum(z))'

norm.(eachrow(amp_hat2))
norm.(eachrow(amp))[z] # Phew, that's better! =#

function BSSR_KISS(y::AbstractVector, t::AbstractVector, ω::AbstractVector, priors::NamedTuple, nsamps::Integer)

    # dims
    n = length(y)
    nω = length(ω)
    p = 1 + 2*nω

    # Form design matrix
    X = zeros(n, p)
    X[:,1] .= 1.0
    for j in 1:nω
        col = 2*j
        X[:,col] .= cospi.(2*ω[j]*t)
        X[:,col+1] .= sinpi.(2*ω[j]*t)
    end

    # initial values and allocations

    B = (X'*X) \ (X'*y)
    fit = X*B
    
    z = fill(true, nω)
    zsamps = fill(false, nsamps, nω)
    zfull = view(z, repeat(1:nω, inner = 2))
    active = [true; zfull]
    
    τ = n / norm(y - fit)^2
    τsamps = zeros(nsamps)
    τshape = priors.τ[1] + n/2

    ampsamps = zeros(nsamps, nω)

    Qprior = Diagonal(1.0*[priors.Bprec[1]; fill(priors.Bprec[2], 2*nω)])
    Qpost = (X'*X) + Qprior
    Bmu = similar(B)

    Xy = X'*y
    XX = X'*X


    for m in ProgressBar(1:nsamps)


        # Sample z_j; j = 1,…,nω

        for j in 1:nω

            # Inactive?
            z[j] = false
            active[2:end] .= zfull
            @views ll_negative = log(1 - priors.zprob) + τ*( dot(B[active], Xy[active]) )



        end

        # Sample B

        Qpost .= Symmetric(XzXz + (1/τ)*Qprior)
        cholesky!(Qpost)
        QpU = UpperTriangular(Qpost)

        ldiv!(Bmu, QpU', Xzy)
        ldiv!(QpU, Bmu)

        B .= Bmu + sqrt(1/τ)*(QpU \ randn(p))

        fit .= Xz*B
        fit_temp .= fit

        # Sample τ

        sse = sum((y - fit).^2)
        τscale = 1 / (priors.τ[2] + sse/2)
        τ = rand(Gamma(τshape, τscale))
        # Write samples

        ampsamps[m,:] .= z .* norm.(eachrow(reshape(B[2:end], 2, nω)'))
        τsamps[m] = τ
        zsamps[m,:] .= z


    end


    return (amp = ampsamps, τ = τsamps, z = zsamps)




end

function gammashaperate(mu, std)

    shape = @. (mu/std)^2
    rate = @. mu / (std ^2)

    return shape, rate

end


priors = (zprob = 0.2, Bprec = [1,1], τ = gammashaperate(100, 10))
nsamps = 10_000

samples = BSSR(y, t, ω, priors, nsamps)

plot(samples.τ)
plot(sqrt.(1 ./ samples.τ))

mean(samples.z, dims = 1)


plot(samples.amp[(end-100):end,:]')

ampmu = mean(samples.amp, dims = 1)[1,:]

amptrue = norm.(eachrow(amp)) .* z

ampmu


