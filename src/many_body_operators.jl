#
# Functions to build many-body operators.
#

"""
    function build_many_body_op(L::Int, N::Int, J::Array{T, 2}, C::T=zero(T)) where T<:Number

Build many-body operator from the hopping matrix J.

# Arguments:
- `L::Int`: number of sites.
- `N::Int`: number of particles.
- `J::Array{T, 2}`: hopping matrix:
    ``J = ∑_{ij} J_{i,j} b^†_i b_j``
- `C::T=zero(T)`: constant term added to the diagonal of the many-body
    operator.
"""
function build_many_body_op(L::Int, N::Int, J::Array{T, 2}, C::T=zero(T)) where T<:Number
    # Basis of states and dimension of the Hilbert space.
    states = get_LN_states(L, N)
    dH = length(states)

    # Make the many-body operator of the same type as J.
    Op = zeros(T, (dH, dH))

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
    function build_sparse_many_body_op(L::Int, N::Int, J::Array{T, 2}, C::T=zero(T)) where T<:Number

Build sparse many-body operator from the matrix J.

# Arguments:
- `L::Int`: number of sites.
- `N::Int`: number of particles.
- `J::Array{T, 2}`: hopping matrix:
    ``J = ∑_{ij} J_{i,j} b^†_i b_j``
- `C::T=zero(T)`: constant term added to the diagonal of the many-body
    operator.
"""
function build_sparse_many_body_op(L::Int, N::Int, J::Array{T, 2}, C::T=zero(T)) where T<:Number
    # Basis of states and dimension of the Hilbert space.
    states = get_LN_states(L, N)
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
    vals_type = Float64
    vals = zeros(T, nnz_in_Op)

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
    function build_spin1_many_body_op(L::Int, Sz::Int, J::Array{T, 2},
                                      Jz::Vector{T}, Jz2::Vector{T}, C::T=zero(T))
                                      where T<:Number

Build a spin 1 many-body operator from the hopping matrix J.

# Arguments:
- `L::Int`: number of sites.
- `Sz::Int`: magnetization of the states.
- `J::Array{T, 2}`: hopping matrix, ignore diagonal values:
    ``J = ∑_{ij} J_{i,j} S^+_i S^-_j``
- `W::Array{T, 2}`: interaction matrix.
    ``W = ∑_{i ≤ j} W_{i,j} S^z_i S^z_j``
- `Jz::Vector{T}`: local Sz terms:
    ``Jz = ∑_i Jz_i S^z_i``
- `C::T=zero(T)`: constant term added to the diagonal of the many-body
    operator.
"""
function build_spin1_many_body_op(L::Int, Sz::Int, J::Array{T, 2},
                                  W::Array{T, 2}, Jz::Vector{T}, C::T=zero(T)) where T<:Number
    # Basis of states and dimension of the Hilbert space.
    states = get_spin1_LSz_states(L, Sz)
    dH = length(states)

    # Make the many-body operator of the same type as J.
    Op = zeros(T, (dH, dH))

    for s=1:dH
        state = states[s]
        # Constant term.
        Op[s, s] += C

        # Sz and Sz^2 terms.
        for i=0:L-1
            Op[s, s] += Jz[i+1]*((state>>2i)&1 - (state>>(2i+1))&1)
            Op[s, s] += W[i+1, i+1]*((state>>2i)&1 + (state>>(2i+1))&1)
        end

        # S^z_i S^z_j terms, i < j.
        for j=1:L-1
            for i=0:j-1
                iszero(W[i+1, j+1]) && continue
                # Four possibilities to find the state. In positions i and j,
                # respectively: (+, +), (-, -), (-, +), (+, -).
                if (((state>>2i)&3 == 1 && (state>>2j)&3 == 1) ||
                    ((state>>2i)&3 == 2 && (state>>2j)&3 == 2))
                    Op[s, s] += W[i+1, j+1]
                elseif (((state>>2i)&3 == 2 && (state>>2j)&3 == 1) ||
                        ((state>>2i)&3 == 1 && (state>>2j)&3 == 2))
                    Op[s, s] -= W[i+1, j+1]
                end
            end
        end

        # S^+_i S^-_j, with i != j, terms.
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
