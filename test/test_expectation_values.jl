using ExactD, Test, LinearAlgebra

@testset "expectation values" begin
L = 5
N = 3
basis = make_LN_basis(L, N)
s = normalize(rand(length(basis)))
r = normalize(rand(length(basis)))

@testset "<s|n_p1*n_p2*...*n_pn|s>" begin
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

@testset "<r|n_p1*n_p2*...*n_pn|s>" begin
    p = 1
    @test expected_n(basis, r, s, p) ≈ s[1]*r[1]+s[2]*r[2]+s[3]*r[3]+s[5]*r[5]+s[6]*r[6]+s[8]*r[8]
    p = 4
    @test expected_n(basis, r, s, p) ≈ s[2]*r[2]+s[3]*r[3]+s[4]*r[4]+s[8]*r[8]+s[9]*r[9]+s[10]*r[10]

    p = [1, 3]
    @test expected_n(basis, r, s, p) ≈ s[1]*r[1]+s[3]*r[3]+s[6]*r[6]
    p = [4, 2]
    @test expected_n(basis, r, s, p) ≈ s[2]*r[2]+s[4]*r[4]+s[9]*r[9]

    p = [1, 3, 5]
    @test expected_n(basis, r, s, p) ≈ s[6]*r[6]
    p = [4, 2, 3]
    @test expected_n(basis, r, s, p) ≈ s[4]*r[4]

    p = [1, 2, 3, 5]
    @test expected_n(basis, r, s, p) ≈ 0. atol=1e-10
end

@testset "<s|b^+_p[1]*...*b^+_p[n]*b_q[1]*...*b_q[m]|s>" begin
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

@testset "<r|b^+_p[1]*...*b^+_p[n]*b_q[1]*...*b_q[m]|s>" begin
    p = 1
    q = 1
    @test expected_pq(basis, r, s, p, q) ≈ s[1]*r[1]+s[2]*r[2]+s[3]*r[3]+s[5]*r[5]+s[6]*r[6]+s[8]*r[8]
    p = 4
    q = 4
    @test expected_pq(basis, r, s, p, q) ≈ s[2]*r[2]+s[3]*r[3]+s[4]*r[4]+s[8]*r[8]+s[9]*r[9]+s[10]*r[10]

    p = [1, 2]
    q = [1, 3]
    @test expected_pq(basis, r, s, p, q) ≈ r[2]*s[3] + r[5]*s[6]
    p = [4, 2]
    q = [1, 3]
    @test expected_pq(basis, r, s, p, q) ≈ s[6]*r[9]
end
end
