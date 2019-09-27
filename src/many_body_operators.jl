export build_many_body_op,
       build_sparse_many_body_op,
       build_spin1_many_body_op

"""
    function build_many_body_op(L::Int, N::Int,
                                J::AbstractMatrix{T},
                                V::AbstractMatrix{T},
                                C::T=zero(T)) where T<:Number

Build many-body operator from the hopping matrix J with N particles.
"""
function build_many_body_op(L::Int, N::Int,
                            J::AbstractMatrix{T},
                            V::AbstractMatrix{T},
                            C::T=zero(T)) where T<:Number
    basis = make_LN_basis(L, N)
    return build_many_body_op(L, basis, J, V, C)
end

"""
    function build_many_body_op(L::Int,
                                basis::Vector{Int},
                                J::AbstractMatrix{T},
                                V::AbstractMatrix{T},
                                C::T=zero(T)) where T<:Number

Build many-body operator from the hopping matrix J with a given basis of states.

# Arguments:
- L::Int: number of sites.
- basis::Vector{Int}: basis of Fock states for the mb operator.
- J::AbstractMatrix{T}: hopping matrix b^dagger_i*b_j.
- V::AbstractMatrix{T}: interaction matrix n_i*n_j.
- C::T=zero(T): constant term.
"""
function build_many_body_op(L::Int,
                            basis::Vector{Int},
                            J::AbstractMatrix{T},
                            V::AbstractMatrix{T},
                            C::T=zero(T)) where T<:Number
    # Dimension of the Hilbert space.
    dH = length(basis)

    # Use only upper triangular part of `V`.
    V = deepcopy(V)
    for i=1:L-1, j=i+1:L
        V[i, j] += V[j, i]
    end

    # Make the many-body operator of the same type as J.
    Op = zeros(T, (dH, dH))

    for s=1:dH
        state = basis[s]
        # Constant term.
        Op[s, s] += C

        # Diagonal terms.
        for i=0:L-1
            if (state>>i)&1 == 1
                Op[s, s] += J[i+1, i+1]
            end
            for j=i+1:L-1
                if !iszero(V[i+1, j+1]) && (state>>i)&1==1 && (state>>j)&1==1
                    Op[s, s] += V[i+1, j+1]
                end
            end
        end

        # Off-diagonal terms.
        for i=0:L-1
            for j=0:L-1
                if (!iszero(J[i+1, j+1])
                        && (state>>i)&1 == 0 && (state>>j)&1 == 1)
                    Jstate = state + 1<<i - 1<<j
                    js = searchsortedfirst(basis, Jstate)
                    Op[js, s] = J[i+1, j+1]
                end
            end
        end
    end
    return Op
end

"""
    function build_sparse_many_body_op(L::Int,
                                       basis::Vector{Int},
                                       J::AbstractMatrix{T},
                                       V::AbstractMatrix{T},
                                       C::T=zero(T)) where {T<:Number}

Build sparse many-body operator from the matrix J with a N particles.

# Arguments:
- L::Int: number of sites.
- basis::Vector{Int}: basis of Fock states for the mb operator.
- J::AbstractMatrix{T}: hopping matrix b^dagger_i*b_j.
- V::AbstractMatrix{T}: interaction matrix n_i*n_j.
- C::T=zero(T): constant term.
"""
function build_sparse_many_body_op(L::Int,
                                   basis::Vector{Int},
                                   J::AbstractMatrix{T},
                                   V::AbstractMatrix{T},
                                   C::T=zero(T)) where {T<:Number}
    # Dimension of the Hilbert space.
    dH = length(basis)

    # Use only upper triangular part of `V`.
    V = deepcopy(V)
    for i=1:L-1, j=i+1:L
        V[i, j] += V[j, i]
    end

    # Number of non-zero off-diagonal elts in J.
    nnz_J = count(abs.(J) .> 1e-8) - count(abs.(diag(J)) .> 1e-8)
    # Number of nnz elts in Op due to off-diag terms and diag terms.
    nnz_in_Op = dH*(nnz_J + 1)
    rows = zeros(Int, nnz_in_Op)
    cols = zeros(Int, nnz_in_Op)
    vals = zeros(T, nnz_in_Op)

    cont = 1
    for s=1:dH
        state = basis[s]
        # Constant term.
        vals[cont] += C

        # Diagonal term.
        for i=0:L-1
            if (state>>i)&1 == 1
                vals[cont] += J[i+1, i+1]
            end
            for j=i+1:L-1
                if !iszero(V[i+1, j+1]) && (state>>i)&1==1 && (state>>j)&1==1
                    vals[cont] += V[i+1, j+1]
                end
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
                    js = searchsortedfirst(basis, Jstate)
                    rows[cont] = js
                    cols[cont] = s
                    vals[cont] = J[i+1, j+1]
                    cont += 1
                end
            end
        end
    end
    # Remove zeros from rows, cols, vals.
    rows = rows[1:cont-1]
    cols = cols[1:cont-1]
    vals = vals[1:cont-1]
    Op = sparse(rows, cols, vals, dH, dH)
    return Op
end

"""
    function build_spin1_many_body_op(L::Int,
                                      Sz::Int,
                                      J::AbstractMatrix{T},
                                      W::AbstractMatrix{T},
                                      Jz::Vector{T},
                                      C::T=zero(T)) where T<:Number

Build a spin 1 many-body operator from the hopping matrix J.
"""
function build_spin1_many_body_op(L::Int,
                                  Sz::Int,
                                  J::AbstractMatrix{T},
                                  W::AbstractMatrix{T},
                                  Jz::Vector{T},
                                  C::T=zero(T)) where T<:Number
    basis = make_spin1_LSz_basis(L, Sz)
    return build_spin1_many_body_op(L, basis, J, W, Jz, C)
end

"""
    function build_spin1_many_body_op(L::Int,
                                      basis::Vector{Int},
                                      J::AbstractMatrix{T},
                                      W::AbstractMatrix{T},
                                      Jz::Vector{T},
                                      C::T=zero(T)) where T<:Number

Build a spin 1 many-body operator from the hopping matrix J.

# Arguments:
- L::Int: number of sites.
- Sz::Int: magnetization of the basis.
- J::AbstractMatrix{T}: hopping matrix, ignore diagonal values S^+_i*S^-_j.
- W::AbstractMatrix{T}: interaction matrix S^z_i*S^z_j.
- Jz::Vector{T}: local Sz terms S^z_i.
- C::T=zero(T): constant term.
"""
function build_spin1_many_body_op(L::Int,
                                  basis::Vector{Int},
                                  J::AbstractMatrix{T},
                                  W::AbstractMatrix{T},
                                  Jz::Vector{T},
                                  C::T=zero(T)) where T<:Number
    # Dimension of the Hilbert space.
    dH = length(basis)

    # Make the many-body operator of the same type as J.
    Op = zeros(T, (dH, dH))

    for s=1:dH
        state = basis[s]
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
                    js = searchsortedfirst(basis, Jstate)
                    Op[js, s] = 2*J[i+1, j+1]
                elseif (state>>2i)&3 == 0 && (state>>2j)&3 == 0
                    Jstate = state + 1<<2i + 2<<2j
                    js = searchsortedfirst(basis, Jstate)
                    Op[js, s] = 2*J[i+1, j+1]
                elseif (state>>2i)&3 == 2 && (state>>2j)&3 == 1
                    Jstate = state - 2<<2i - 1<<2j
                    js = searchsortedfirst(basis, Jstate)
                    Op[js, s] = 2*J[i+1, j+1]
                elseif (state>>2i)&3 == 2 && (state>>2j)&3 == 0
                    Jstate = state - 2<<2i + 2<<2j
                    js = searchsortedfirst(basis, Jstate)
                    Op[js, s] = 2*J[i+1, j+1]
                end
            end
        end
    end
    return Op
end
