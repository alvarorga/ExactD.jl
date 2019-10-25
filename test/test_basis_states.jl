using ExactD, LinearAlgebra, Test
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
