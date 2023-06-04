# SparseArraysCOO.jl

Create sparse vectors and matrices conveniently and efficiently

* [![Documenter](https://img.shields.io/badge/docs-dev-blue.svg)](https://eschnett.github.io/SparseArraysCOO.jl/dev)
* [![GitHub
  CI](https://github.com/eschnett/SparseArraysCOO.jl/workflows/CI/badge.svg)](https://github.com/eschnett/SparseArraysCOO.jl/actions)
* [![Codecov](https://codecov.io/gh/eschnett/SparseArraysCOO.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/eschnett/SparseArraysCOO.jl)

This package `SparseArraysCOO.jl` provides simple convenience
functions for creating sparse vectors and sparse matrices. After
creating a `SparseMatrixCOO` matrix or a `SparseVectorCOO` vector, it
records all assignments made to it. There are no other operations
defined on these types, except converting them to the efficient
`SparseMatrixCSC` and `SparseVector` types. This is no more or less
efficient than the way described in the documentation of
`SparseArrays`, but it is more convenient.

The suffix `COO` stands for "coordinate format". This means that the
internal representation stores both indices `i` and `j` for every
nonzero value.

# Examples

Creating a sparse matrix:
```Julia
julia> using SparseArrays, SparseArraysCOO
julia> A = SparseMatrixCOO{Float64}(4, 4)
SparseMatrixCOO{Float64}(4, 4, Int64[], Int64[], Float64[])

julia> A[1, 1] = 1
julia> A[2, 3] = 4
julia> A[4, 1] = 17
julia> A = sparse(A)
4×4 SparseMatrixCSC{Float64, Int64} with 3 stored entries:
  1.0   ⋅    ⋅    ⋅
   ⋅    ⋅   4.0   ⋅
   ⋅    ⋅    ⋅    ⋅
 17.0   ⋅    ⋅    ⋅

julia> typeof(A)
SparseMatrixCSC{Float64, Int64}
```

Creating a sparse vector:
```Julia
julia> using SparseArrays, SparseArraysCOO
julia> x = SparseVectorCOO{Float64}(4)
SparseVectorCOO{Float64}(4, Int64[], Float64[])

julia> x[1] = 1
julia> x[2] = 4
julia> x[4] = 17
julia> x = sparse(x)
4-element SparseVector{Float64, Int64} with 3 stored entries:
  [1]  =  1.0
  [2]  =  4.0
  [4]  =  17.0

julia> typeof(x)
SparseVector{Float64, Int64}
```
