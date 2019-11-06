export expected_n,
       expected_pq

"""
    expected_n(basis::Vector{Int},
               phi::Vector{T},
               psi::Vector{T},
               p::Vector{Int}) where {T<:Number}

Compute the expectation value <phi|n_p[1]*n_p[2]*...*n_p[n]|psi>.
"""
function expected_n(basis::Vector{Int},
                    phi::Vector{T},
                    psi::Vector{T},
                    p::Vector{Int}) where {T<:Number}
    o = zero(Float64)
    for i in eachindex(basis)
        bs = basis[i]
        has_all_n = true
        for pi in p
            (bs>>(pi-1))&1 == 0 && (has_all_n = false)
        end
        if has_all_n
            o += conj(phi[i])*psi[i]
        end
    end
    return o
end

"""
    expected_n(basis::Vector{Int},
               psi::Vector{T},
               p::Vector{Int}) where {T<:Number}

Compute the expectation value <psi|n_p[1]*n_p[2]*...*n_p[n]|psi>.
"""
function expected_n(basis::Vector{Int},
                    psi::Vector{T},
                    p::Vector{Int}) where {T<:Number}
    return expected_n(basis, psi, psi, p)
end

expected_n(basis, psi, p::Int) = expected_n(basis, psi, Int[p])
expected_n(basis, phi, psi, p::Int) = expected_n(basis, phi, psi, Int[p])

"""
    expected_pq(basis::Vector{Int},
                phi::Vector{T},
                psi::Vector{T},
                p::Vector{Int},
                q::Vector{Int}) where {T<:Number}

Compute <phi|b^+_p[1]*...*b^+_p[n]*b_q[1]*...*b_q[m]|psi>.
"""
function expected_pq(basis::Vector{Int},
                     phi::Vector{T},
                     psi::Vector{T},
                     p::Vector{Int},
                     q::Vector{Int}) where {T<:Number}

    p = deepcopy(p)
    q = deepcopy(q)
    (allunique(p) && allunique(q)) || return zero(T)
    # Number operators.
    n_i = p ∩ q
    # Remove those indices where there are number ops from p and q.
    setdiff!(p, n_i)
    setdiff!(q, n_i)

    o = zero(T)
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
            j = searchsortedfirst(basis, rs)
            o += psi[i]*conj(phi[j])
        end
    end
    return o
end

"""
    expected_pq(basis::Vector{Int},
                psi::Vector{<:Number},
                p::Vector{Int},
                q::Vector{Int})

Compute <psi|b^+_p[1]*...*b^+_p[n]*b_q[1]*...*b_q[m]|psi>.
"""
function expected_pq(basis::Vector{Int},
                     psi::Vector{<:Number},
                     p::Vector{Int},
                     q::Vector{Int})
    return expected_pq(basis, psi, psi, p, q)
end

expected_pq(basis, psi, p::Int, q::Int) = expected_pq(basis, psi, [p], [q])
expected_pq(basis, phi, psi, p::Int, q::Int) = expected_pq(basis, phi, psi, [p], [q])
