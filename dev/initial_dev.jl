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

function BSSR(y::AbstractVector, t::AbstractVector, ω::AbstractVector, priors::NamedTuple, nsamps::Integer)

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

    Xz = copy(X)

    # initial values and allocations

    B = (X'*X) \ (X'*y)
    fit = X*B
    fit_temp = copy(fit)
    
    z = fill(true, nω)
    
    τ = n / norm(y - fit)^2
    τsamps = zeros(nsamps)
    τshape = priors.τ[1] + n/2

    ampsamps = zeros(nsamps, nω)

    Qprior = Diagonal(1.0*[priors.Bprec[1]; fill(priors.Bprec[2], 2*nω)])
    Qpost = (X'*X) + Qprior
    Bmu = similar(B)

    Xzy = Xz'*y
    Xy = X'*y
    XzXz = Xz'*Xz
    XX = X'*X


    for m in ProgressBar(1:nsamps)

        println(m)

        # Sample z_j; j = 1,…,nω

        for j in 1:nω

            zj = z[j]
            inds = (2*j):(2*j + 1)
            Xblock = view(X, :, inds)

            # z_j = 0?
            if zj
                fit_temp .-= Xblock*view(B, inds)
            end
            negative = log(1 - priors.zprob) - 0.5*τ*sum((y - fit_temp).^2)

            # z_j = 1?
            if !zj
                fit_temp .+= Xblock*view(B, inds)
            end
            affirmative = log(priors.zprob) - 0.5*τ*sum((y - fit_temp).^2)

            logsum = max(affirmative, negative) + log1p(exp(-abs(affirmative - negative)))
            affirmative_prob = exp(affirmative - logsum)

            z[j] = rand() < affirmative_prob

            # Need to update Xz and related mats?
            if zj & !z[j] 
                Xz[:,inds] .= 0
                Xzy[inds] .= 0
                XzXz[inds,:] .= 0
                XzXz[:,inds] .= 0
                fit .= fit_temp
            elseif !zj & z[j]
                Xz[:,inds] .= Xblock
                Xzy[inds] .= view(Xy, inds)
                XzXz[inds,:] .= view(XX, inds, :)
                XzXz[:,inds] .= view(XX, :, inds)
                fit .= fit_temp
            else
                fit_temp .= fit
            end
        end

        # Sample B

        Qpost .= Symmetric(XzXz + (1/τ)*Qprior)
        println(Qpost)
        Qpostchol = cholesky(Qpost)
        QpU = UpperTriangular(Qpostchol.U)

        ldiv!(Bmu, QpU', Xzy)
        ldiv!(QpU, Bmu)

        B .= Bmu + sqrt(1/τ)*(QpU \ randn(p))

        fit .= Xz*B
        fit_temp .= fit

        # Sample τ

        sse = sum((y - fit).^2)
        τscale = 1 / (priors.τ[2] + sse/2)
        τ = rand(Gamma(τshape, τscale))
        println(τ)
        println("###############")
        # Write samples

        ampsamps[m,:] .= z .* norm.(eachrow(reshape(B[2:end], 2, nω)'))
        τsamps[m] = τ


    end


    return (ampsamps = ampsamps, τsamps = τsamps)




end


priors = (zprob = zprob, Bprec = [1,1], τ = [1,1])

samples = BSSR(y, t, ω, priors, 100)

plot(samples.ampsamps[(end-100):end,:]')

ampmu = mean(samples.ampsamps, dims = 1)[1,:]


