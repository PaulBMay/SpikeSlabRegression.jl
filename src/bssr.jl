function sindesign(t::AbstractVector, ω::AbstractVector)

    n = length(t)
    nω = length(ω)
    p = 2*nω + 1

    X = zeros(n, p)
    X[:,1] .= 1.0
    for j in 1:nω
        col = 2*j
        X[:,col] .= cospi.(2*ω[j]*t)
        X[:,col+1] .= sinpi.(2*ω[j]*t)
    end

    return X

end

function BSSR(y::AbstractVector, t::AbstractVector, ω::AbstractVector, priors::NamedTuple, nsamps::Integer)

    # dims
    n = length(y)
    nω = length(ω)
    p = 1 + 2*nω

    # Form design matrix
    X = sindesign(t, ω)

    # initial values and allocations

    B = (X'*X) \ (X'*y)
    Bsamps = zeros(nsamps, p)
    fit = X*B
    
    z = fill(true, nω + 1) # An extra leading true to represent the intercept
    active = view(z, [1; repeat(2:(nω+1), inner = 2)])

    zsamps = fill(false, nsamps, nω)

    
    τ = n / norm(y - fit)^2
    τsamps = zeros(nsamps)
    τshape = priors.τ[1] + n/2

    ampsamps = zeros(nsamps, nω)

    Qprior = Diagonal(1.0*[priors.Bprec[1]; fill(priors.Bprec[2], 2*nω)])
    Qpost = (X'*X) + Qprior
    Bmu = similar(B)

    Xy = X'*y
    Xzy = copy(Xy)
    XX = X'*X
    XzXz = copy(XX)


    for m in ProgressBar(1:nsamps)


        # Sample z_j; j = 1,…,nω

        for j in 1:nω

            zj_prev = z[j+1]

            z[j+1] = true
            lp_affirmative = @views log(priors.zprob) + τ*( dot(Xy[active], B[active]) - 0.5*dot(B[active], XX[active, active], B[active]) )

            z[j+1] = false
            lp_negative = @views log(1 - priors.zprob) + τ*( dot(Xy[active], B[active]) - 0.5*dot(B[active], XX[active, active], B[active]) )

            affirmative_prob = exp(lp_affirmative - logsumexp([lp_affirmative, lp_negative]))

            z[j+1] = rand() < affirmative_prob

        end

        # Sample B

        XzXz .= @. active * XX * active'
        Xzy .= Xy .* active

        Qpost .= Symmetric(XzXz  + (1/τ)*Qprior)
        cholesky!(Qpost)
        QpU = UpperTriangular(Qpost)

        ldiv!(Bmu, QpU', Xzy)
        ldiv!(QpU, Bmu)

        B .= Bmu + sqrt(1/τ)*(QpU \ randn(p))

        # Sample τ

        fit .= @views X[:,active]*B[active]


        sse = sum((y - fit).^2)
        τscale = 1 / (priors.τ[2] + sse/2)
        τ = rand(Gamma(τshape, τscale))
        # Write samples

        ampsamps[m,:] .= @views z[2:(nω+1)] .* norm.(eachrow(reshape(B[2:p], 2, nω)'))
        τsamps[m] = τ
        zsamps[m,:] .= view(z, 2:(nω+1))
        Bsamps[m,:] .= B


    end


    return (amp = ampsamps, τ = τsamps, z = zsamps, X = X, B = Bsamps)




end

function gammashaperate(mu, std)

    shape = @. (mu/std)^2
    rate = @. mu / (std ^2)

    return shape, rate

end