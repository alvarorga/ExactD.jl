using ExactD, Random, LinearAlgebra, Test

@testset "parity operator" begin
    @test ExactD.parity(10, 1, 2) == 1.
    @test ExactD.parity(10, 2, 0) == -1.
    @test ExactD.parity(10, 1, 3) == 1.
    @test ExactD.parity(14, 1, 3) == -1.
end

@testset "basis of Fock states" begin
@testset "states with N particles and L sites" begin
    L = 4
    N = 2
    states = make_LN_basis(L, N)

    @test states[1] == 3
    @test states[3] == 6
    @test states[5] == 10
    @test states[6] == 12
end

@testset "states with N1, N2 particles and L sites" begin
    L = 4
    N1 = 2
    N2 = 3
    states = make_LN1N2_basis(L, N1, N2)

    @test states[1] == 16+32+64 + 1+2
    @test states[5] == 16+32+64 + 2+8
    @test states[9] == 16+32+128 + 2+4
    @test states[11] == 16+32+128 + 2+8
    @test states[18] == 16+64+128 + 4+8
    @test states[23] == 32+64+128 + 2+8
end

@testset "spin 1 states with L spins and Sz magnetization" begin
    L = 4
    Sz = 2
    states = make_spin1_LSz_basis(L, Sz)

    @test length(states) == 10
    @test states[1] == 5
    @test states[3] == 20
    @test states[5] == 68
    @test states[7] == 86
    @test states[10] == 149
    @test get_num_spin1_states(L, Sz) == 10

    L = 4
    Sz = 1
    states = make_spin1_LSz_basis(L, Sz)

    @test length(states) == 16
    @test states[1] == 1
    @test states[3] == 16
    @test states[5] == 25
    @test states[8] == 70
    @test states[10] == 82
    @test states[16] == 148
    @test get_num_spin1_states(L, Sz) == 16
end
end  # testset "basis of Fock states"

@testset "hard-core boson many body operators" begin
@testset "dense many-body operator" begin
    L = 4
    N = 2
    J::Array{Float64, 2} = reshape(collect(1:L^2), (L, L))
    V::Array{Float64, 2} = reshape(collect(1:L^2), (L, L))
    Op = build_many_body_op(L, N, J, V)

    # Diagonal terms.
    @test Op[1, 1] ≈ 7. + 7.
    @test Op[4, 4] ≈ 17. + 17.
    @test Op[6, 6] ≈ 27. + 27.

    # Off-diagonal terms.
    @test Op[1, 2] ≈ 10.
    @test Op[2, 5] ≈ 0.
    @test Op[4, 2] ≈ 12.
    @test Op[6, 1] ≈ 0.
    @test Op[3, 5] ≈ 15.

    C = 1.3
    Op2 = build_many_body_op(L, N, J, V, C)

    # Diagonal terms.
    @test Op2[1, 1] ≈ 8.3 + 7.
    @test Op2[4, 4] ≈ 18.3 + 17.
    @test Op2[6, 6] ≈ 28.3 + 27.
end

@testset "sparse many-body operator" begin
    L = 6
    N = 3
    basis = make_LN_basis(L, N)
    J::Array{Float64, 2} = reshape(sin.(1:L^2), (L, L))
    V::Array{Float64, 2} = reshape(tan.(1:L^2), (L, L))
    C = 1.3
    sp_Op = build_sparse_many_body_op(L, basis, J, V, C)
    de_Op = build_many_body_op(L, basis, J, V, C)
    @test norm(Array(sp_Op) - de_Op) ≈ 0.0 atol=1e-12
end
end  # testset "hard-core boson many body operators"

@testset "dense many-body operator with L spins s=1" begin
    L = 4
    Sz = 2
    J = reshape(collect(1.:L^2), (L, L))
    W = reshape(collect(1.:L^2), (L, L))
    Jz = collect(1.:L)/2
    C = 1.33
    Op = build_spin1_many_body_op(L, Sz, J, W, Jz, C)

    # Diagonal terms: S^z + (S^z_i)^2 + S^z_i*S^z_j.
    @test Op[1, 1] ≈ 1.5 + 7 + 5 + C
    @test Op[4, 4] ≈ 2.5 + 17 + 13 + C
    @test Op[7, 7] ≈ 4. + 34 + 12 + C
    @test Op[10, 10] ≈ 1. + 34 - 18 + C

    # Off-diagonal terms.
    @test Op[1, 2] ≈ 20.
    @test Op[1, 4] ≈ 28.
    @test Op[1, 6] ≈ 0.
    @test Op[7, 6] ≈ 4.
    @test Op[6, 7] ≈ 10.
    @test Op[1, 9] ≈ 30.
    @test Op[9, 1] ≈ 24.
    @test Op[1, 10] ≈ 24.

    L = 3
    Sz = 0
    J = reshape(collect(1.:L^2), (L, L))
    W = reshape(collect(1.:L^2), (L, L))
    Jz = collect(1.:L)/4
    Op = build_spin1_many_body_op(L, Sz, J, W, Jz)
    # Off-diagonal terms.
    @test Op[4, 2] ≈ 12.
    @test Op[4, 5] ≈ 4.
end

@testset "expectation values" begin
@testset "<n_p1*n_p2*...*n_pn>" begin
    L = 5
    N = 3
    basis = make_LN_basis(L, N)
    s = rand(length(basis))
    s ./= norm(s)

    p = 1
    @test expected_n(basis, s, p) ≈ s[1]^2+s[2]^2+s[3]^2+s[5]^2+s[6]^2+s[8]^2
    p = 4
    @test expected_n(basis, s, p) ≈ s[2]^2+s[3]^2+s[4]^2+s[8]^2+s[9]^2+s[10]^2

    p = [1, 3]
    @test expected_n(basis, s, p) ≈ s[1]^2+s[3]^2+s[6]^2
    p = [4, 2]
    @test expected_n(basis, s, p) ≈ s[2]^2+s[4]^2+s[9]^2

    p = [1, 3, 5]
    @test expected_n(basis, s, p) ≈ s[6]^2
    p = [4, 2, 3]
    @test expected_n(basis, s, p) ≈ s[4]^2

    p = [1, 2, 3, 5]
    @test expected_n(basis, s, p) ≈ 0. atol=1e-10
end

@testset "<b^+_p[1]*...*b^+_p[n]*b_q[1]*...*b_q[m]>" begin
    L = 5
    N = 3
    basis = make_LN_basis(L, N)
    s = rand(length(basis))
    s ./= norm(s)

    p = 1
    q = 1
    @test expected_pq(basis, s, p, q) ≈ s[1]^2+s[2]^2+s[3]^2+s[5]^2+s[6]^2+s[8]^2
    p = 4
    q = 4
    @test expected_pq(basis, s, p, q) ≈ s[2]^2+s[3]^2+s[4]^2+s[8]^2+s[9]^2+s[10]^2

    p = [1, 2]
    q = [1, 3]
    @test expected_pq(basis, s, p, q) ≈ s[2]*s[3] + s[5]*s[6]
    p = [4, 2]
    q = [1, 3]
    @test expected_pq(basis, s, p, q) ≈ s[6]*s[9]
end
end
