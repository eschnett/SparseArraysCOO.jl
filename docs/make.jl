# Generate documentation with this command:
# (cd docs && julia make.jl)

push!(LOAD_PATH, "..")

using Documenter
using SparseArraysCOO

makedocs(; sitename="SparseArraysCOO", format=Documenter.HTML(), modules=[SparseArraysCOO])

deploydocs(; repo="github.com/eschnett/SparseArraysCOO.jl.git", devbranch="main", push_preview=true)
