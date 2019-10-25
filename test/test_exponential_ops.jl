using LinearAlgebra, Random, ExactD, Test

@testset "exponential operators" begin
Random.seed!(0)
L = 6
N = 3
basis = make_LN_basis(L, N)
state = rand(length(basis))

@testset "exponential J2" begin
    α = rand(L, L)
    cstate = deepcopy(state)
    exp_J2!(cstate, basis, α)
    @test cstate[1] ≈ state[1]*exp(α[1,2]+α[2,1]+α[1,3]+α[3,1]+α[2,3]+α[3,2]+α[1,1]+α[2,2]+α[3,3])
    @test cstate[4] ≈ state[4]*exp(α[4,2]+α[2,4]+α[4,3]+α[3,4]+α[2,3]+α[3,2]+α[4,4]+α[2,2]+α[3,3])
    @test cstate[8] ≈ state[8]*exp(α[1,4]+α[4,1]+α[1,5]+α[5,1]+α[4,5]+α[5,4]+α[1,1]+α[4,4]+α[5,5])
    @test cstate[16] ≈ state[16]*exp(α[4,6]+α[6,4]+α[4,3]+α[3,4]+α[6,3]+α[3,6]+α[4,4]+α[6,6]+α[3,3])
    @test cstate[20] ≈ state[20]*exp(α[6,5]+α[5,6]+α[6,4]+α[4,6]+α[5,4]+α[4,5]+α[6,6]+α[5,5]+α[4,4])
end

@testset "exponential J1" begin
    α = rand(L)
    cstate = deepcopy(state)
    exp_J1!(cstate, basis, α)
    @test cstate[1] ≈ state[1]*exp(α[1]+α[2]+α[3])
    @test cstate[4] ≈ state[4]*exp(α[4]+α[2]+α[3])
    @test cstate[8] ≈ state[8]*exp(α[1]+α[4]+α[5])
    @test cstate[16] ≈ state[16]*exp(α[4]+α[6]+α[3])
    @test cstate[20] ≈ state[20]*exp(α[6]+α[5]+α[4])
end

end
