include("../kernels/kernel_matrix.jl")
include("../kernels/kernels.jl")
include("../data/load_data.jl")
using LinearAlgebra
using Clustering
using AbstractTrees
using BlockDiagonals


# Structs: for cluster tree (organizing the points) and HODLR tree
struct ClusterTreeNode
    range::UnitRange{Int}
    left::Union{ClusterTreeNode, Nothing}
    right::Union{ClusterTreeNode, Nothing}
end

struct ClusterTree
    root::ClusterTreeNode
    points::Matrix
end

struct HODLRNode
    range::UnitRange{Int}
    U::Matrix        # column basis for this cluster (0×0 at the root)
    V::Matrix        # row basis for this cluster (0×0 at the root)
    α::Union{HODLRNode, Nothing}
    β::Union{HODLRNode, Nothing}
    Bᵅᵝ::Matrix      # coupling: A[α, β] ≈ α.U * Bᵅᵝ * β.V'
    Bᵝᵅ::Matrix      # coupling: A[β, α] ≈ β.U * Bᵝᵅ * α.V'
    D::Matrix        # dense diagonal block (leaves only; 0×0 otherwise)
end

struct HSSNode
    range::UnitRange{Int}
    U::Matrix           # column basis
    V::Matrix           # row basis
    U_interp::Matrix    # column hierarchical relation
    V_interp::Matrix    # row hierarchical relation
    α::Union{HSSNode, Nothing}
    β::Union{HSSNode, Nothing}
    Bᵅᵝ::Matrix         # coupling: A[α, β] ≈ Uᵅ * Bᵅᵝ * Vᵝ'
    Bᵝᵅ::Matrix         # coupling: A[β, α] ≈ Uᵝ * Bᵝᵅ * Vᵅ'
    D::Matrix           # dense diagonal block (leaves only; 0×0 otherwise)
end


# Helper functions that help with tree traversal
function isleaf(node::ClusterTreeNode)
    return node.left === nothing
end

function isleaf(node::HODLRNode)
    return node.α === nothing
end

function isleaf(node::HSSNode)
    return node.α === nothing
end

AbstractTrees.children(node::ClusterTreeNode) = isleaf(node) ? () : (node.left, node.right)
AbstractTrees.children(node::HODLRNode) = isleaf(node) ? () : (node.α, node.β)
AbstractTrees.children(node::HSSNode) = isleaf(node) ? () : (node.α, node.β)


# Build cluster tree to order points for hierarchical construction
function build_kd_cluster_tree(points::Matrix, k::Integer)
    d, N = points.size
    root, order = build_kd_cluster_tree_helper(points, 1:N, k, 1)
    return ClusterTree(root, points[:,order])
end

function build_kd_cluster_tree_helper(
        points::Matrix,
        active_indices::Union{UnitRange{Int}, Vector{Int}},
        k::Integer,
        count::Integer
    )
    """
    points is a d*N matrix, where d is the dimension and N is the number of points
    Returns the root of a tree and a permutation of the points,
        that corresponds to the correct ordering.
    """
    active_points = points[:,active_indices]
    d, N = active_points.size

    # Base case: N < k. Build a leaf.
    if N <= k
        return ClusterTreeNode(count:(count+N-1), nothing, nothing), active_indices
    end

    # Otherwise: split points in half...
    R = Clustering.kmeans(active_points, 2)
    a = assignments(R)
    left_count = sum(a.==1)
    
    # Permute the points and recurse
    ord = sortperm(a)
    left_inds = active_indices[ord][1:left_count]
    right_inds = active_indices[ord][left_count+1:end]
    left_node, left_ord = build_kd_cluster_tree_helper(points, left_inds, k, count)
    right_node, right_ord = build_kd_cluster_tree_helper(points, right_inds, k, count+left_count)
    ord = [left_ord; right_ord]
    return ClusterTreeNode(count:(count+N-1), left_node, right_node), ord

end



### HODLR Matrix algorithms

# Traverse the tree recursively (equivalent to depth first search)
# This is pre-order: compress blocks before moving on to children.
function build_HODLR_matrix(
    node::ClusterTreeNode,
    kernel_matrix::KernelMatrix,
    k::Integer,
    U=zeros(0,0),
    V=zeros(0,0)
    )

    # Leaf: compute diagonal block
    if isleaf(node)
        zero = zeros(0,0)
        D = build_kernel_submatrix(kernel_matrix, node.range, node.range)
        return HODLRNode(node.range, U, V, nothing, nothing, zero, zero, D)
    end

    # Otherwise, compute low-rank approximation
    α, β = node.left, node.right
    Aᵅᵝ = build_kernel_submatrix(kernel_matrix, α.range, β.range)
    Aᵝᵅ = build_kernel_submatrix(kernel_matrix, β.range, α.range)
    Uᵅ, Bᵅᵝ, Vᵝ = compress_block_fixed_k(Aᵅᵝ, k)
    Uᵝ, Bᵝᵅ, Vᵅ = compress_block_fixed_k(Aᵝᵅ, k)
    left = build_HODLR_matrix(α, kernel_matrix, k, Uᵅ, Vᵅ)
    right = build_HODLR_matrix(β, kernel_matrix, k, Uᵝ, Vᵝ)
    return HODLRNode(node.range, U, V, left, right, Bᵅᵝ, Bᵝᵅ, zeros(0,0))
end

function compress_block_fixed_k(A, k)
    F = svd(A)
    if length(F.S) < k
        U = F.U
        B = Diagonal(F.S)
        V = F.V
    else
        U = F.U[:,1:k]
        B = Diagonal(F.S[1:k])
        V = F.V[:,1:k]
    end
    return U, B, V
end

# HODLR matvec
function HODLR_matvec(A::HODLRNode, x::Vector)

    # Loop over all nodes of the tree.
    N_A = length(A.range)
    N_x = length(x)
    if N_A != N_x
        return nothing
    end

    b = zeros(N_x)
    for node in PreOrderDFS(A)
        # If it is a leaf: add the diagonal block
        if isleaf(node)
            b[node.range] += (node.D * x[node.range])
        # Otherwise; add low-rank approximation
        else
            α = node.α; β = node.β
            b[α.range] += (α.U * (node.Bᵅᵝ * (β.V' * x[β.range])))
            b[β.range] += (β.U * (node.Bᵝᵅ * (α.V' * x[α.range])))
        end
    end
    return b
end



### HSS Matrix algorithms

# Construct HSS Matrix.
# Use the ID to compute the low-rank approximations. This simplifies coupling.
# Unlike HODLR, this is post-order. We construct the bases at the leaves before recursively
# moving up to their parents, to construct the next blocks.
function build_HSS_matrix(node::ClusterTreeNode, kernel_matrix::KernelMatrix, k::Integer)
    d, N = kernel_matrix.pts.size
    range = node.range
    rangeᶜ = range_complement(node.range, N)

    # Leaf: compute diagonal block AND the row/column bases.
    if isleaf(node)
        zero = zeros(0,0)
        D = build_kernel_submatrix(kernel_matrix, range, range)
        HSS_block_row = build_kernel_submatrix(kernel_matrix, range, rangeᶜ)
        HSS_block_column = build_kernel_submatrix(kernel_matrix, rangeᶜ, range)
        U, _, _ = compress_block_fixed_k(HSS_block_row, k)
        V, _, _ = compress_block_fixed_k(HSS_block_column', k)
        return HSSNode(node.range, U, V, zero, zero, nothing, nothing, zero, zero, D)
    end

    # Node: post-order traversal. Build low-rank approx at leaves first
    α = build_HSS_matrix(node.left, kernel_matrix, k)
    β = build_HSS_matrix(node.right, kernel_matrix, k)

    # Build interpolation matrices that connects leaves (same as HODLR). Bᵅᵝ = Uᵅ' * Aᵅᵝ * Vᵝ.
    Aᵅᵝ = build_kernel_submatrix(kernel_matrix, α.range, β.range)
    Aᵝᵅ = build_kernel_submatrix(kernel_matrix, β.range, α.range)
    Bᵅᵝ = α.U' * Aᵅᵝ * β.V; Bᵝᵅ = β.U' * Aᵝᵅ * α.V

    # Build the small 2k*k matrix that defines recursive bases.
    # We will do the stupid way for simplicity, by rebuilding HSS block row and column.
    # Compute the SVD of the block row as before, then project bases from previous level.
    HSS_block_row = build_kernel_submatrix(kernel_matrix, range, rangeᶜ)
    HSS_block_column = build_kernel_submatrix(kernel_matrix, rangeᶜ, range)
    U, _, _ = compress_block_fixed_k(HSS_block_row, k) # n*k
    V, _, _ = compress_block_fixed_k(HSS_block_column', k) # n*k
    U_interp = BlockDiagonal([α.U', β.U']) * U # 2k*k
    V_interp = BlockDiagonal([α.V', β.V']) * V # 2k*k

    return HSSNode(node.range, U, V, U_interp, V_interp, α, β, Bᵅᵝ, Bᵝᵅ, zeros(0,0))
    # We could post-process to remove U/U_interp and V/V_interp, but for now just leave both.
end

function range_complement(range, N)
    return setdiff(1:N, range)
end

# HSS matvec
# Works like a FMM: go bottom-up first, followed by top-down.
function HSS_matvec(A::HSSNode, x::Vector)
    # Upwards pass: compute q_tilde at ALL levels and save
    q_tilde = IdDict{HSSNode, Vector}()
    for node in PostOrderDFS(A)
        if isleaf(node)
            q_tilde[node] = node.V' * x[node.range]
        else
            q_tilde[node] = node.V_interp' * vcat(q_tilde[node.α], q_tilde[node.β])
        end
    end
    # Downwards pass: compute b_tilde
    b_tilde = IdDict{HSSNode, Vector}()
    b_tilde[A] = zeros(0)
    b = zeros(length(x))
    for node in PreOrderDFS(A)
        if !isleaf(node)
            kᵅ, _ = node.Bᵅᵝ.size
            b_tilde_current = node.U_interp * b_tilde[node] + vcat(node.Bᵅᵝ * q_tilde[node.β], node.Bᵝᵅ * q_tilde[node.α])
            b_tilde[node.α] = b_tilde_current[1:kᵅ]; b_tilde[node.β] = b_tilde_current[(kᵅ+1):end]
        else
            b[node.range] = node.U * b_tilde[node] + node.D * x[node.range]
        end
    end
    return b
end



function main()

    # X, _, _, _ = load_pendigits_dataset()
    # d, N = X.size
    # println("Pendigits dataset: d=$(d), N=$(N)")
    # k = 32

    d = 4; N = 3000
    X = rand(d, N)
    k = 20

    # Build tree
    @info "Building cluster tree..."
    tree = build_kd_cluster_tree(X, 2*k)

    # Build HODLR matrix
    @info "Building full kernel matrix..."
    kernel_matrix = KernelMatrix(rbf_kernel, tree.points)
    full_kernel_matrix = build_kernel_matrix(kernel_matrix)

    # @info "Building HODLR matrix..."
    # compressed_kernel_matrix = build_HODLR_matrix(tree.root, kernel_matrix, k)
    @info "Building HSS matrix..."
    compressed_kernel_matrix = build_HSS_matrix(tree.root, kernel_matrix, k)

    # Compute error with power interation
    @info "Beginning power iteration..."
    x_next = rand(N)
    x_next ./ norm(x_next)
    it = 0
    while true
        it += 1
        x = x_next
        if typeof(compressed_kernel_matrix) == HSSNode
            x_next = HSS_matvec(compressed_kernel_matrix, x)
        elseif typeof(compressed_kernel_matrix) == HODLRNode
            x_next = HODLR_matvec(compressed_kernel_matrix, x)
        end
        x_next -= full_kernel_matrix * x
        x_next ./ norm(x_next)
        norm(x_next - x) <= 1e-10 && break
    end
    error = x_next' * (HSS_matvec(compressed_kernel_matrix, x_next) - full_kernel_matrix * x_next) / norm(full_kernel_matrix)
    println("||A - A_compressed|| / ||A|| = $(error); # power iterations = $it")

end

main()