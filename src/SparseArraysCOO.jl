module SparseArraysCOO

using SparseArrays

################################################################################

export SparseVectorCOO
"""
    SparseVectorCOO{T}

A sparse vector in "coordinate" format. This stores the index `i` for every value in an unsorted array.

The constructor takes a single integer argument specifying the size of the sparse vector:
    x = SparseVectorCOO{Float64}(4)
Use standard Julia assignment to insert elements:
    x[2] = 3

Use `SparseArrays.sparse` to convert to the efficient (sorted)
`SparseVector` format.
"""
struct SparseVectorCOO{T}
    n::Int
    I::Vector{Int}
    V::Vector{T}

    function SparseVectorCOO{T}(n::Int, I::Vector{Int}, V::Vector{T}) where {T}
        n ≥ 0 || throw(ArgumentError("invalid Array size"))
        I ≢ V || throw(ArgumentError("arrays must be distinct"))
        length(I) == length(V) || throw(ArgumentError("invalid Array dimensions"))
        all(1 ≤ i ≤ n for i in I) || throw(BoundsError())
        return new{T}(n, I, V)
    end
end

SparseVectorCOO(n::Int, I::Vector{Int}, V::Vector{T}) where {T} = SparseVectorCOO{T}(n, I, V)

SparseVectorCOO{T}(n::Int) where {T} = SparseVectorCOO{T}(n, Int[], T[])

SparseVectorCOO{T}(x::SparseVectorCOO{T}) where {T} = SparseVectorCOO{T}(x.n, copy(x.I), copy(x.V))
SparseVectorCOO(x::SparseVectorCOO{T}) where {T} = SparseVectorCOO{T}(x)

function SparseVectorCOO{T}(x::AbstractSparseVector) where {T}
    I, V = findnz(x)
    return SparseVectorCOO{T}(size(x, 1), I, V)
end
SparseVectorCOO(x::AbstractSparseVector{T}) where {T} = SparseVectorCOO{T}(x)

function Base.setindex!(x::SparseVectorCOO, v, i::Integer)
    1 ≤ i ≤ x.n || throw(BoundsError())
    push!(x.I, i)
    push!(x.V, v)
    return x
end

"""
    SparseArrays.sparse(x::SparseVectorCOO)::SparseVector

Convert to the standard sparse vector representation.
"""
SparseArrays.sparse(x::SparseVectorCOO) = sparsevec(x.I, x.V, x.n)

# Expensive operations are intentionally not provided. Instead, call
# `sparse` to switch to an efficient representation, and then apply
# these operations. Expensive operations include e.g. comparisons and
# `getindex`.

Base.eltype(::SparseVectorCOO{T}) where {T} = T
Base.size(x::SparseVectorCOO) = (x.n,)

Base.zero(x::SparseVectorCOO{T}) where {T} = SparseVectorCOO{T}(x.n)

Base.map(f, x::SparseVectorCOO) = SparseVectorCOO(x.n, copy(x.I), map(f, x.V))

Base.:+(x::SparseVectorCOO) = map(+, x)
Base.:-(x::SparseVectorCOO) = map(-, x)

Base.:*(a::Number, x::SparseVectorCOO) = map(b -> a * b, x)
Base.:*(x::SparseVectorCOO, a::Number) = map(b -> b * a, x)
Base.:\(a::Number, x::SparseVectorCOO) = map(b -> a \ b, x)
Base.:/(x::SparseVectorCOO, a::Number) = map(b -> b / a, x)

function Base.:+(x::SparseVectorCOO, y::SparseVectorCOO)
    size(x) == size(y) || throw(DimensionMismatch())
    return SparseVectorCOO(x.n, vcat(x.I, y.I), vcat(x.V, y.V))
end
Base.:-(x::SparseVectorCOO, y::SparseVectorCOO) = x + (-y)

################################################################################

export SparseMatrixCOO
"""
    SparseMatrixCOO{T}

A sparse matrix in "coordinate" format. This stores both indices `i` and `j` for every value in an unsorted array.

The constructor takes a two integer arguments specifying the size of the sparse matrix:
    A = SparseMatrixCOO{Float64}(4, 4)
Use standard Julia assignment to insert elements:
    A[2, 3] = 4

Use `SparseArrays.sparse` to convert to the efficient (compressed, sorted)
`SparseMatrixCSC` format.
"""
struct SparseMatrixCOO{T}
    m::Int
    n::Int
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{T}

    function SparseMatrixCOO{T}(m::Int, n::Int, I::Vector{Int}, J::Vector{Int}, V::Vector{T}) where {T}
        m ≥ 0 || throw(ArgumentError("invalid Array size"))
        n ≥ 0 || throw(ArgumentError("invalid Array size"))
        (I ≢ J && I ≢ V && J ≢ V) || throw(ArgumentError("arrays must be distinct"))
        length(I) == length(J) == length(V) || throw(ArgumentError("invalid Array dimensions"))
        all(1 ≤ i ≤ m for i in I) || throw(BoundsError())
        all(1 ≤ j ≤ n for j in J) || throw(BoundsError())
        return new{T}(m, n, I, J, V)
    end
end

SparseMatrixCOO(m::Int, n::Int, I::Vector{Int}, J::Vector{Int}, V::Vector{T}) where {T} = SparseMatrixCOO{T}(m, n, I, J, V)

SparseMatrixCOO{T}(m::Int, n::Int) where {T} = SparseMatrixCOO{T}(m, n, Int[], Int[], T[])

SparseMatrixCOO{T}(A::SparseMatrixCOO{T}) where {T} = SparseMatrixCOO{T}(A.m, A.n, copy(A.I), copy(A.j), copy(A.V))
SparseMatrixCOO(A::SparseMatrixCOO{T}) where {T} = SparseMatrixCOO{T}(A)

function SparseMatrixCOO{T}(A::AbstractSparseMatrix) where {T}
    I, J, V = findnz(A)
    return SparseMatrixCOO{T}(size(A, 1), size(A, 2), I, J, V)
end
SparseMatrixCOO(x::AbstractSparseMatrix{T}) where {T} = SparseMatrixCOO{T}(x)

function Base.setindex!(A::SparseMatrixCOO, v, i::Integer, j::Integer)
    1 ≤ i ≤ A.m || throw(BoundsError())
    1 ≤ j ≤ A.n || throw(BoundsError())
    push!(A.I, i)
    push!(A.J, j)
    push!(A.V, v)
    return A
end
"""
    SparseArrays.sparse(A::SparseMatrixCOO)::SparseMatrixCSC

Convert to the standard sparse matrix representation.
"""
SparseArrays.sparse(A::SparseMatrixCOO) = sparse(A.I, A.J, A.V, A.m, A.n)

# Expensive operations are intentionally not provided. Instead, call
# `sparse` to switch to an efficient representation, and then apply
# these operations. Expensive operations include e.g. comparisons and
# `getindex`.

Base.eltype(::SparseMatrixCOO{T}) where {T} = T
Base.size(A::SparseMatrixCOO) = (A.m, A.n)

Base.zero(A::SparseMatrixCOO{T}) where {T} = SparseMatrixCOO{T}(A.m, A.n)
function Base.one(A::SparseMatrixCOO{T}) where {T}
    A.m == A.n || throw(DimensionMismatch("multiplicative identity defined only for square matrices"))
    return SparseMatrixCOO{T}(A.m, A.m, collect(1:(A.m)), collect(1:(A.m)), ones(T, A.m))
end

Base.map(f, A::SparseMatrixCOO) = SparseMatrixCOO(A.m, A.n, copy(A.I), copy(A.J), map(f, A.V))

Base.:+(A::SparseMatrixCOO) = map(+, A)
Base.:-(A::SparseMatrixCOO) = map(-, A)

Base.:*(a::Number, A::SparseMatrixCOO) = map(b -> a * b, A)
Base.:*(A::SparseMatrixCOO, a::Number) = map(b -> b * a, A)
Base.:\(a::Number, A::SparseMatrixCOO) = map(b -> a \ b, A)
Base.:/(A::SparseMatrixCOO, a::Number) = map(b -> b / a, A)

function Base.:+(A::SparseMatrixCOO, B::SparseMatrixCOO)
    size(A) == size(B) || throw(DimensionMismatch())
    return SparseMatrixCOO(A.m, A.n, vcat(A.I, B.I), vcat(A.J, B.J), vcat(A.V, B.V))
end
Base.:-(A::SparseMatrixCOO, B::SparseMatrixCOO) = A + (-B)

end
