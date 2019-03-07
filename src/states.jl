"""
Functions to compute Fock basis states.
"""

"""
    _get_LN_states(L::Int, N::Int)

Compute fock basis states with L sites and N particles.
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

"""
    _get_spin1_LSz_states(L::Int, Sz::Int)

Compute states with L spins s=1 and Sz magnetization.

There can be 3 states per individual site/spin. In the usual notation
these are: |1, 0>, |1, 1>, |1, -1>. Because we can't represent these
states with only one bit of information we map each site to the three
possible states:
* |1, 0> = (00)
* |1, 1> = (01)
* |1,-1> = (10)
For example, a state with three spins, with magnetization -1, 1, 0 in
each site, respectively, is represented as: 100100.
"""
function _get_spin1_LSz_states(L::Int, Sz::Int)
    # Number of states.
    num_states = 0
    l = iseven(L+Sz) ? L : L-1
    # Number of spins with m=1 and m=-1 in a chain of l sites.
    Nu = (l+Sz)>>1
    Nd = (l-Sz)>>1
    while l >= 0 && Nu >= 0 && Nd >= 0
        num_states += binomial(l, Nu)*binomial(L, l)
        l -= 2
        Nu -= 1
        Nd -= 1
    end

    states = zeros(Int, num_states)
    cont = 1
    for i=0:(1<<(2L))-1
        state_Sz = 0
        is_valid = true
        for j=0:L-1
            state_Sz += (i>>(2j))&1
            state_Sz -= (i>>(2j+1))&1
            if (i>>(2j))&1 == 1 && (i>>(2j+1))&1 == 1
                is_valid = false
            end
        end
        if state_Sz == Sz && is_valid
            states[cont] = i
            cont += 1
        end
    end
    return states
end
