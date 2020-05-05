using ExactD, Test, LinearAlgebra

@testset "parity operator" begin
    @test ExactD.parity(10, 1, 2) == 1.
    @test ExactD.parity(10, 2, 0) == -1.
    @test ExactD.parity(10, 1, 3) == 1.
    @test ExactD.parity(14, 1, 3) == -1.
end

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
    Jx = zeros(L)
    C = 1.33
    Op = build_spin1_many_body_op(L, Sz, J, W, Jz, Jx, C)

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
    Jx = zeros(L)
    Op = build_spin1_many_body_op(L, Sz, J, W, Jz, Jx)
    # Off-diagonal terms.
    @test Op[4, 2] ≈ 12.
    @test Op[4, 5] ≈ 4.

    # Test only Sx terms.
    L = 3
    basis = make_spin1_L_basis(L)
    J = zeros(L, L)
    W = zeros(L, L)
    Jz = zeros(L)
    Jx = collect(1.:L)
    Op = build_spin1_many_body_op(L, basis, J, W, Jz, Jx)

    @test all(Op .≈ transpose(Op))
    @test Op[1, 3] ≈ 1/sqrt(2)
    @test Op[1, 2] ≈ 1/sqrt(2)
    @test Op[12, 15] ≈ 2/sqrt(2)
    @test Op[12, 18] ≈ 2/sqrt(2)
    @test Op[27, 9] ≈ 3/sqrt(2)
end
