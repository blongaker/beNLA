include("sketch.jl")
include("../problems/matrixlib.jl")

using LinearAlgebra
using StatsBase
using Plots

function compute_sparse_sign_embedding_projection_error(k::Integer, ζ::Integer)
    R = 2.0; N = 1024;
    K = far_field_interaction_matrix(R, N)
    Y = sparse_sign_embedding(K, k, ζ)
    F = qr(Y); Q = Matrix(F.Q);
    QQᵀK = Q * Q' * K
    error = norm(K - QQᵀK)
    # print("||(I-QQᵀ)K||₂ = $error")
    return error
end

function sparse_sign_embedding_vary_k()
    k_size = 50
    ζ = 8
    errors = zeros(k_size-ζ+1)
    for k=ζ:k_size
        errors[k-ζ+1] = compute_sparse_sign_embedding_projection_error(k, ζ)
    end
    plot(ζ:k_size, errors, ls=:dot, lw=3, yscale=:log10)
end


function compute_SRHT_projection_error(k::Integer)
    R = 2.0; N = 1024;
    K = far_field_interaction_matrix(R, N)
    Y = SRFT(K, k)
    F = qr(Y); Q = Matrix(F.Q);
    QQᵀK = Q * Q' * K
    error = norm(K - QQᵀK)
    # print("||(I-QQᵀ)K||₂ = $error")
    return error
end


function SRHT_vary_k()
    k_size = 50
    errors = zeros(k_size)
    for k=1:k_size
        errors[k] = compute_SRHT_projection_error(k)
    end
    plot(1:k_size, errors, ls=:dot, lw=3, yscale=:log10)
end



function measure_distortion(sketch::String)
    """
    Goal is to measure ϵ in the embedding definition: (1-ϵ)|x| ≤ |Sx| ≤ (1+ϵ)|x|.
    Compute a bunch of unit vectors in a subspace, and sketch them.
    Then compare the norms of the original vectors to the sketched vectors.
    It is equivalent to measure
        1-ϵ ≤ |Sx|/|x| ≤ 1+ϵ.
    So compute the ratios, then measure the largest/smallest value for the quantity.
    Each sample should be a row of the matrix to sketch (we make these smaller).
    WLOG we can take x to be unit vectors.
    """
    N = 16834   # number of samples
    d = 1000    # original dimension
    k = 100     # sketching dimension

    # Get random samples in all directions
    A = randn(N, d)
    
    # Sketch
    if sketch === "SRFT"
        Y = SRFT(A, k)
    elseif sketch === "SRHT"
        Y = SRHT(A, k)
    elseif sketch === "gaussian"
        Y = gaussian_sketch(A, k)
    elseif sketch === "sparse_sign"
        Y = sparse_sign_embedding(A, k, 10)
    else
        print("Sketch not recognized")
        return
    end

    # Measure epsilon
    ratios = norm.(eachrow(Y)) ./ norm.(eachrow(A))
    ϵ = max(maximum(ratios) - 1, 1 - minimum(ratios))
    print("Dimension n: $d, Sketch dim k: $k ⇒ ϵ=$ϵ")

    # Plot histogram
    histogram(ratios)

end


measure_distortion("SRHT")