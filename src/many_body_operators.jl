"""
Functions to build many-body operators.
"""

using SparseArrays

include("./states.jl")

"""
    build_many_body_op(L::Int, N::Int,
                       J::Array{T, 2} where T=Union{Float64, ComplexF64},
                       C::Union{Float64, ComplexF64}=0.)

Build many-body operator from the hopping matrix J.

# Arguments:
- `L::Int`: number of sites.
- `N::Int`: number of particles.
- `J::Array{T, 2}`: hopping matrix:
    ``J = ∑_{ij} J_{i,j} b^†_i b_j``
- `C::Union{Float64, ComplexF64}=0.`: constant term added to the
    diagonal of the many-body operator.
"""
function build_many_body_op(L::Int, N::Int,
                            J::Array{T, 2} where T=Union{Float64, ComplexF64},
                            C::Union{Float64, ComplexF64}=0.)
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

"""
    build_sparse_many_body_op(L::Int, N::Int,
                              J::Array{T, 2} where T=Union{Float64, ComplexF64},
                              C::Union{Float64, ComplexF64}=0.)

Build sparse many-body operator from the matrix J.

# Arguments:
- `L::Int`: number of sites.
- `N::Int`: number of particles.
- `J::Array{T, 2}`: hopping matrix:
    ``J = ∑_{ij} J_{i,j} b^†_i b_j``
- `C::Union{Float64, ComplexF64}=0.`: constant term added to the
    diagonal of the many-body operator.
"""
function build_sparse_many_body_op(L::Int, N::Int,
                                   J::Array{T, 2} where T=Union{Float64,
                                                                ComplexF64},
                                   C::Union{Float64, ComplexF64}=0.)
    # Basis of states and dimension of the Hilbert space.
    states = _get_LN_states(L, N)
    dH = length(states)

    # Number of non-zero off-diagonal elts in J.
    nnz_J = 0
    for i=1:L, j=1:L
        if i != j && !iszero(J[i, j])
            nnz_J += 1
        end
    end
    # Number of non-zero values in Op due to one correlator b^d_i b_j.
    elts_correlator = binomial(L-2, N-1)
    # Preallocate rows, cols and indices vectors.
    # Number of nnz elts in Op due to off-diag terms and diag terms.
    nnz_in_Op = nnz_J*elts_correlator + dH
    rows = zeros(Int, nnz_in_Op)
    cols = zeros(Int, nnz_in_Op)
    vals = zeros(eltype(J), nnz_in_Op)

    cont = 1
    for s=1:dH
        state = states[s]
        # Constant term.
        vals[cont] += C

        # Diagonal term.
        for i=0:L-1
            if (state>>i)&1 == 1
                vals[cont] += J[i+1, i+1]
            end
        end
        rows[cont] = s
        cols[cont] = s
        cont += 1

        # Off-diagonal terms.
        for i=0:L-1
            for j=0:L-1
                if (!iszero(J[i+1, j+1])
                        && (state>>i)&1 == 0 && (state>>j)&1 == 1)
                    Jstate = state + 1<<i - 1<<j
                    js = searchsortedfirst(states, Jstate)
                    rows[cont] = js
                    cols[cont] = s
                    vals[cont] = J[i+1, j+1]
                    cont += 1
                end
            end
        end
    end
    Op = sparse(rows, cols, vals)
    return Op
end

"""
    build_spin1_many_body_op(L::Int, Sz::Int,
                             J::Array{T1, 2} where T1=Union{Float64, ComplexF64},
                             JmJp::Vector{T2} where T2=Union{Float64, ComplexF64},
                             Jz::Vector{T3} where T3=Union{Float64, ComplexF64},
                             C::Union{Float64, ComplexF64}=0.)

Build a spin 1 many-body operator from the hopping matrix J.

# Arguments:
- `L::Int`: number of sites.
- `Sz::Int`: magentization of the states.
- `J::Array{T, 2}`: hopping matrix:
    ``J = ∑_{ij} J_{i,j} S^+_i S^-_j``
- `JmJp::Array{T, 1}`: local terms:
    ``JmpJp = ∑_i JmJp_i S^-_i S^+_i``
- `Jz::Array{T, 1}`: local Sz terms:
    ``Jz = ∑_i Jz_i S^z_i``
- `C::Union{Float64, ComplexF64}=0.`: constant term added to the
    diagonal of the many-body operator.
"""
function build_spin1_many_body_op(L::Int, Sz::Int,
                                  J::Array{T1, 2} where T1=Union{Float64, ComplexF64},
                                  JmJp::Vector{T2} where T2=Union{Float64, ComplexF64},
                                  Jz::Vector{T3} where T3=Union{Float64, ComplexF64},
                                  C::Union{Float64, ComplexF64}=0.)
    # Basis of states and dimension of the Hilbert space.
    states = _get_spin1_LSz_states(L, Sz)
    dH = length(states)

    # Make the many-body operator of the same type as J.
    Op = zeros(eltype(J), (dH, dH))

    for s=1:dH
        state = states[s]
        # Constant term.
        Op[s, s] += C

        # Sz terms.
        for i=0:L-1
            Op[s, s] += Jz[i+1]*((state>>(2i))&1 - (state>>(2i+1))&1)
        end

        # S^+_i S^-_i and S^+_i S^-_i terms.
        for i=0:L-1
            Op[s, s] += 2*J[i+1, i+1]*(1 - (state>>(2i+1))&1)
            Op[s, s] += 2*JmJp[i+1]*(1 - (state>>2i)&1)
        end

        # S^+_i S^-_j terms.
        for i=0:L-1
            for j=0:L-1
                i == j && continue
                iszero(J[i+1, j+1]) && continue
                # Four possibilities to find the state. In positions i and j,
                # respectively: (0, +), (0, 0), (-, +), (-, 0).
                if (state>>2i)&3 == 0 && (state>>2j)&3 == 1
                    Jstate = state + 1<<2i - 1<<2j
                    js = searchsortedfirst(states, Jstate)
                    Op[js, s] = 2*J[i+1, j+1]
                elseif (state>>2i)&3 == 0 && (state>>2j)&3 == 0
                    Jstate = state + 1<<2i + 2<<2j
                    js = searchsortedfirst(states, Jstate)
                    Op[js, s] = 2*J[i+1, j+1]
                elseif (state>>2i)&3 == 2 && (state>>2j)&3 == 1
                    Jstate = state - 2<<2i - 1<<2j
                    js = searchsortedfirst(states, Jstate)
                    Op[js, s] = 2*J[i+1, j+1]
                elseif (state>>2i)&3 == 2 && (state>>2j)&3 == 0
                    Jstate = state - 2<<2i + 2<<2j
                    js = searchsortedfirst(states, Jstate)
                    Op[js, s] = 2*J[i+1, j+1]
                end
            end
        end
    end
    return Op
end
