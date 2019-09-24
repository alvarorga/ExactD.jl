export expected_n,
       expected_pq

"""
    expected_n(basis::Vector{Int},
               state::Vector{<:Number},
               p::Vector{Int})

Compute the expectation value <state|n_p[1]*n_p[2]*...*n_p[n]|state>.
"""
function expected_n(basis::Vector{Int},
                    state::Vector{<:Number},
                    p::Vector{Int})
    o = zero(Float64)
    for i in eachindex(basis)
        bs = basis[i]
        has_all_n = true
        for pi in p
            (bs>>(pi-1))&1 == 0 && (has_all_n = false)
        end
        has_all_n && (o += abs2(state[i]))
    end
    return o
end

expected_n(basis, state, p::Int) = expected_n(basis, state, Int[p])

"""
    expected_pq(basis::Vector{Int},
                state::Vector{<:Number},
                p::Vector{Int},
                q::Vector{Int})

Compute <state|b^+_p[1]*...*b^+_p[n]*b_q[1]*...*b_q[m]|state>.
"""
function expected_pq(basis::Vector{Int},
                     state::Vector{<:Number},
                     p::Vector{Int},
                     q::Vector{Int})

    p = deepcopy(p)
    q = deepcopy(q)
    allunique(p) || return zero(eltype(state))
    allunique(q) || return zero(eltype(state))
    # Number operators.
    n_i = p ∩ q
    # Remove those indices where there are number ops from p and q.
    setdiff!(p, n_i)
    setdiff!(q, n_i)

    o = zero(eltype(state))
    for i in eachindex(basis)
        bs = basis[i]
        has_all_pq = true
        # See if b^+_p*b_q acting on bs is not zero.
        for pi in p
            (bs>>(pi-1))&1 != 0 && (has_all_pq = false)
        end
        for qi in q
            (bs>>(qi-1))&1 != 1 && (has_all_pq = false)
        end
        for ni in n_i
            (bs>>(ni-1))&1 != 1 && (has_all_pq = false)
        end
        # If b^+_q*b_q|bs> not zero, compute the resulting state.
        if has_all_pq
            rs = bs
            for pi in p
                rs += 1<<(pi-1)
            end
            for qi in q
                rs -= 1<<(qi-1)
            end
            j = searchsorted(basis, rs)
            # If rs ∈ basis then length(j) = 1, else 0.
            if length(j) == 1
                # Do state[j][] because j is a range.
                o += state[i]*conj(state[j][])
            end
        end
    end
    return o
end

expected_pq(basis, state, p::Int, q::Int) = expected_pq(basis, state, [p], [q])
