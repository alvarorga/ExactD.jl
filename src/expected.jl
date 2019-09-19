export expected_n

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
