"""
Functions to compute Fock basis states.
"""

"""
Compute states with L sites and N particles.
"""
function _get_LN_states(L::Int, N::Int)
    states = zeros(Int, binomial(L, N))
    tmp = 1
    for i=0:(1<<L)-1
        bits = 0
        for j=0:L-1
            bits += (i>>j)&1
        end
        if bits == N
            states[tmp] = i
            tmp += 1
        end
    end
    return states
end
