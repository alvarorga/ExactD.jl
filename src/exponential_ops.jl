export exp_J2!,
       exp_J1!

"""
    exp_J2!(state::Vector{T},
            basis::Vector{Int},
            α::AbstractMatrix{T}) where {T<:Number}

Apply the exponential operator e^{∑_pq α_pq n_p n_q} to `state`.
"""
function exp_J2!(state::Vector{T},
                 basis::Vector{Int},
                 α::AbstractMatrix{T}) where {T<:Number}
    # Deduce the number of sites from the highest basis state.
    L = 0
    max_bs = basis[end]
    for i=0:31 # Arbitrary very high number.
        if (max_bs>>i)&1 == 1
            L = i+1
        end
    end

    for (ibs, bs) in enumerate(basis)
        c = zero(T)
        for i=0:L-1
            (bs>>i)&1 != 1 && continue
            c += α[i+1, i+1]
            for j=i+1:L-1
                if (bs>>j)&1 == 1
                    c += α[i+1, j+1] + α[j+1, i+1]
                end
            end
        end
        state[ibs] *= exp(c)
    end
    return state
end

"""
    exp_J1!(state::Vector{T},
            basis::Vector{Int},
            α::AbstractMatrix{T}) where {T<:Number}

Apply the exponential operator e^{∑_p α_p n_p} to `state`.
"""
function exp_J1!(state::Vector{T},
                 basis::Vector{Int},
                 α::Vector{T}) where {T<:Number}
    # Deduce the number of sites from the highest basis state.
    L = 0
    max_bs = basis[end]
    for i=0:31 # Arbitrary very high number.
        if (max_bs>>i)&1 == 1
            L = i+1
        end
    end

    for (ibs, bs) in enumerate(basis)
        c = zero(T)
        for i=0:L-1
            if (bs>>i)&1 == 1
                c += α[i+1]
            end
        end
        state[ibs] *= exp(c)
    end
    return state
end
