"""
Functions to build many-body operators.
"""

include("./states.jl")

function build_many_body_op(L::Int, N::Int,
                            J::Array{T, 2} where T=Union{Float64, ComplexF64},
                            C::Union{Float64, ComplexF64}=0.)
    """
    Build many-body operator from the hopping matrix J.

    # Arguments:
    - `L::Int`: number of sites.
    - `N::Int`: number of particles.
    - `J::Array{T, 2}`: hopping matrix:
        ``J = \\sum_{ij} J_{i,j} b^\\dagger_i b_j``
    - `N::Union{Float64, ComplexF64}=0.`: constant term added to the
        diagonal of the many-body operator.
    """
    # Basis of states and dimension of the Hilbert space.
    states = _get_LN_states(L, N)
    dH = length(states)

    # Make the many-body operator of the same type as J.
    Op = zeros(eltype(J), (dH, dH))

    for s=1:dH
        state = states[s]
        # Constant term.
        Op[s, s] += C

        # Diagonal terms.
        for i=0:L-1
            if (state>>i)&1 == 1
                Op[s, s] += J[i+1, i+1]
            end
        end

        # Off-diagonal terms.
        for i=0:L-1
            for j=0:L-1
                if (!iszero(J[i+1, j+1])
                    && (state>>i)&1 == 0 && (state>>j)&1 == 1)
                    Jstate = state + 1<<i - 1<<j
                    js = searchsortedfirst(states, Jstate)
                    Op[js, s] = J[i+1, j+1]
                end
            end
        end
    end
    return Op
end
