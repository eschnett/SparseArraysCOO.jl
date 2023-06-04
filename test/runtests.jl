using SparseArrays
using SparseArraysCOO
using Test

const BigRat = Rational{BigInt}

@testset "Sparse vectors" begin
    for iter in 1:100
        n = rand(1:10)
        x = SparseVectorCOO{BigRat}(n)

        moment0 = BigRat(0)
        moment1 = BigRat(0)
        moment2 = BigRat(0)
        moment3 = BigRat(0)

        nk = rand(1:n)
        for k in 1:nk
            i = rand(1:n)
            v = BigRat(rand(Int8))//rand(1:10)
            x[i] = v

            moment0 += i^0 * v
            moment1 += i^1 * v
            moment2 += i^2 * v
            moment3 += i^3 * v
        end

        x = sparse(x)

        @test eltype(x) ≡ BigRat
        @test size(x) == (n,)

        # have_overlap = nnz(x) < n

        moment0′ = BigRat(0)
        moment1′ = BigRat(0)
        moment2′ = BigRat(0)
        moment3′ = BigRat(0)

        for (i, v) in zip(findnz(x)...)
            moment0′ += i^0 * v
            moment1′ += i^1 * v
            moment2′ += i^2 * v
            moment3′ += i^3 * v
        end

        @test moment0′ == moment0
        @test moment1′ == moment1
        @test moment2′ == moment2
        @test moment3′ == moment3
    end
end

@testset "Sparse matrices" begin
    for iter in 1:100
        m = rand(1:10)
        n = rand(1:10)
        A = SparseMatrixCOO{BigRat}(m, n)

        moment00 = BigRat(0)
        moment01 = BigRat(0)
        moment02 = BigRat(0)
        moment10 = BigRat(0)
        moment11 = BigRat(0)
        moment12 = BigRat(0)
        moment20 = BigRat(0)
        moment21 = BigRat(0)
        moment22 = BigRat(0)

        nk = rand(1:(m * n))
        for k in 1:nk
            i = rand(1:m)
            j = rand(1:n)
            v = BigRat(rand(Int8))//rand(1:10)
            A[i, j] = v

            moment00 += i^0 * j^0 * v
            moment01 += i^0 * j^1 * v
            moment02 += i^0 * j^2 * v
            moment10 += i^1 * j^0 * v
            moment11 += i^1 * j^1 * v
            moment12 += i^1 * j^2 * v
            moment20 += i^2 * j^0 * v
            moment21 += i^2 * j^1 * v
            moment22 += i^2 * j^2 * v
        end

        A = sparse(A)

        @test eltype(A) ≡ BigRat
        @test size(A) == (m, n)

        # have_overlap = nnz(A) < m * n

        moment00′ = BigRat(0)
        moment01′ = BigRat(0)
        moment02′ = BigRat(0)
        moment10′ = BigRat(0)
        moment11′ = BigRat(0)
        moment12′ = BigRat(0)
        moment20′ = BigRat(0)
        moment21′ = BigRat(0)
        moment22′ = BigRat(0)

        for (i, j, v) in zip(findnz(A)...)
            moment00′ += i^0 * j^0 * v
            moment01′ += i^0 * j^1 * v
            moment02′ += i^0 * j^2 * v
            moment10′ += i^1 * j^0 * v
            moment11′ += i^1 * j^1 * v
            moment12′ += i^1 * j^2 * v
            moment20′ += i^2 * j^0 * v
            moment21′ += i^2 * j^1 * v
            moment22′ += i^2 * j^2 * v
        end

        @test moment00′ == moment00
        @test moment01′ == moment01
        @test moment02′ == moment02
        @test moment10′ == moment10
        @test moment11′ == moment11
        @test moment12′ == moment12
        @test moment20′ == moment20
        @test moment21′ == moment21
        @test moment22′ == moment22
    end
end
