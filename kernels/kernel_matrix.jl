using LinearAlgebra


@doc raw"""
    KernelMatrix(kernel_function::Function, pts::Matrix)

Represents a kernel matrix. We do not evaluate the entries of the matrix because this is very expensive.

Rather, the entries can be computed on the fly using `build_kernel_matrix` and related functions.
"""
struct KernelMatrix
    kernel_function::Function
    pts::Matrix
end


@doc raw"""
    build_kernel_matrix(kernelmat::KernelMatrix)

Evaluates a kernel matrix, where $K_{ij} = k(x_{i}, x_{j})$.

TODO: should the kernel function be vectorized or is this fine?
"""
function build_kernel_matrix(kernelmat::KernelMatrix)
    d, N = size(kernelmat.pts)
    K = zeros(N, N)
    for i=1:N
        for j=1:N
            K[i,j] = kernelmat.kernel_function(
                kernelmat.pts[:,i],
                kernelmat.pts[:,j]
            )
        end
    end
    return K
end


@doc raw"""
    build_kernel_submatrix(kernelmat::KernelMatrix, row_inds::Vector{<:Integer}, col_inds::Vector{<:Integer})

Evaluates a kernel submatrix, based on the indices specified.

It will be the case that $K_{ij} = k(x_{s_i}, x_{t_j})$, where $s,t$ are the list of subindices.
"""
function build_kernel_submatrix(
    kernelmat::KernelMatrix,
    row_inds::AbstractVector{<:Integer},
    col_inds::AbstractVector{<:Integer}
)
    M = length(row_inds); N = length(col_inds)
    K_sub = zeros(M, N)
    for (j, id2) in enumerate(col_inds)
        for (i, id1) in enumerate(row_inds)
            K_sub[i,j] = kernelmat.kernel_function(
                kernelmat.pts[:,id1],
                kernelmat.pts[:,id2]
            )
        end
    end
    return K_sub
end
