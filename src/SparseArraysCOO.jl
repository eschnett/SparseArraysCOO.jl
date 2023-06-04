module SparseArraysCOO

using SparseArrays

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
    function SparseVectorCOO{T}(n::Int) where {T}
        @assert n ≥ 0
        return new(n, Int[], T[])
    end
end

function Base.setindex!(S::SparseVectorCOO, v, i::Integer)
    @assert 1 ≤ i ≤ S.n
    push!(S.I, i)
    push!(S.V, v)
    return S
end
"""
    SparseArrays.sparse(S::SparseVectorCOO)::SparseVector

Convert to the standard sparse vector representation.
"""
SparseArrays.sparse(S::SparseVectorCOO) = sparsevec(S.I, S.V, S.n)

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
    function SparseMatrixCOO{T}(m::Int, n::Int) where {T}
        @assert m ≥ 0
        @assert n ≥ 0
        return new(m, n, Int[], Int[], T[])
    end
end

function Base.setindex!(S::SparseMatrixCOO, v, i::Integer, j::Integer)
    @assert 1 ≤ i ≤ S.m
    @assert 1 ≤ j ≤ S.n
    push!(S.I, i)
    push!(S.J, j)
    push!(S.V, v)
    return S
end
"""
    SparseArrays.sparse(S::SparseMatrixCOO)::SparseMatrixCSC

Convert to the standard sparse matrix representation.
"""
SparseArrays.sparse(S::SparseMatrixCOO) = sparse(S.I, S.J, S.V, S.m, S.n)

end
