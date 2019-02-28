"""
Functions to build many-body operators.
"""

include("./states.jl")

function build_many_body_op(L, N, J)
    """
    Build many-body operator from the hopping matrix J.

    The basis states have L sites and N particles. J is the hopping
    matrix:
        ``J = \\sum_{ij} b^\\dagger_i b_j``
    """
    # Basis of states and dimension of the Hilbert space.
    states = _get_states(L, N)
    dH = length(states)

    # Make the many-body operator of the same type as J.
    Op = zeros(eltype(J), (dH, dH))

    for s=1:dH
        state = states[s]

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
