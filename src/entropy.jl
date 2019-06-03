#
# Functions to compute Schmidt decomposition and entanglement entropy.
#

"""
    do_schmidt_decomposition(L::Int, N::Int, i::Int,
                             state::Vector{<:Number})

Do Schmidt decomposition of `state` at site `i`.
"""
function do_schmidt_decomposition(L::Int, N::Int, i::Int,
                                  state::Vector{<:Number})
    # Get basis of states.
    basis = get_LN_states(L, N)

    # Matrix to store the decomposed state in the A and B subspaces.
    dim_A = 1<<(L-i)
    dim_B = 1<<i
    M = zeros(eltype(state), (dim_A, dim_B))

    mask = 1<<i - 1
    # s, sA, and sB refer to the indices of states in the whole space, A, and B.
    for s=1:length(basis)
        sA = basis[s]>>i
        sB = mask&basis[s]
        M[sA+1, sB+1] = state[s]
    end

    svals = svdvals(M)
    return svals
end

"""
    get_entanglement_entropy(L::Int, N::Int, i::Int,
                             state::Vector{<:Number})

Compute the entanglement entropy of `state` at site `i`.
"""
function get_entanglement_entropy(L::Int, N::Int, i::Int,
                                  state::Vector{<:Number})
    svals = do_schmidt_decomposition(L, N, i, state)
    svals2 = svals.^2
    S = -sum(@. svals2*log(svals2))
    return S
end
