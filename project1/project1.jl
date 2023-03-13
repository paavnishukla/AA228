using Graphs
using Printf
using CSV
using DataFrames
using SpecialFunctions
using LinearAlgebra
using StatsBase
"""
    write_gph(dag::DiGraph, idx2names, filename)

Takes a DiGraph, a Dict of index to names and a output filename to write the graph in `gph` format.
"""
function write_gph(dag::DiGraph, idx2names, filename)
    open(filename, "w") do io
        for edge in edges(dag)
            @printf(io, "%s,%s\n", idx2names[src(edge)], idx2names[dst(edge)])
        end
    end
end

function bayesian_score_component(M, alpha)
    p = sum(loggamma.(alpha + M))
    p -= sum(loggamma.(alpha))
    p += sum(loggamma.(sum(alpha,dims=2)))
    p -= sum(loggamma.(sum(alpha,dims=2) + sum(M,dims=2)))
    return p
end

function bayesian_score(vars, G, D)
    n = length(vars)
    print("\nlength is",n)
    M = statistics(vars, G, D)
    print("\nM is",M)
    alpha = prior(vars, G)
    return sum(bayesian_score_component(M[i], alpha[i]) for i in 1:n)
end

function sub2ind(siz, x)
    k = vcat(1, cumprod(siz[1:end-1]))
    return dot(k, x .- 1) + 1
end

function statistics(vars, G, D)
    n = size(D, 1)
    r = [vars[i].r for i in 1:n]
    q = [prod([r[j] for j in inneighbors(G,i)]) for i in 1:n]
    #m = size(D,1)
    M = [zeros(q[i], r[i]) for i in 1:n]
    for o in eachcol(D)
        print("\n",o)
        for i in 1:n
            k = o[i]
            parents = inneighbors(G,i)
            j = 1
            if !isempty(parents)
                j = sub2ind(r[parents], o[parents])
            end
            M[i][j,k] += 1.0
        end
    end
    return M
end

function prior(vars, G)
    print("length is")

    n = length(vars)

    r = [vars[i].r for i in 1:n]
    q = [prod([r[j] for j in inneighbors(G,i)]) for i in 1:n]
    return [ones(q[i], r[i]) for i in 1:n]
end

struct Variable
    name::Symbol
    r::Int # number of possible values
end
const Assignment = Dict{Symbol,Int}
const FactorTable = Dict{Assignment,Float64}
struct Factor
    vars::Vector{Variable}
    table::FactorTable
end

variablenames(φ::Factor) = [var.name for var in φ.vars]
select(a::Assignment, varnames::Vector{Symbol}) = Assignment(n=>a[n] for n in varnames)

function assignments(vars::AbstractVector{Variable})
    names = [var.name for var in vars]
    return vec([Assignment(n=>v for (n,v) in zip(names, values))
    for values in product((1:v.r for v in vars)...)])
end

function normalize!(φ::Factor)
    z = sum(p for (a,p) in φ.table)
    for (a,p) in φ.table
        φ.table[a] = p/z
    end
    return φ
end
# function compute(infile, outfile)

#     # WRITE YOUR CODE HERE
#     # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
#     # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING

# end

# if length(ARGS) != 2
#     error("usage: julia project1.jl <infile>.csv <outfile>.gph")
# end

# inputfilename = ARGS[1]
# outputfilename = ARGS[2]

# compute(inputfilename, outputfilename)

df = CSV.read("./example/example.csv", DataFrame)
print(df)
trandf = permutedims(df)
print(trandf)
names2idx = Dict{String, Int64}()
varnames = []
for i in 1:size(df,2)
    names2idx[names(df)[i]] = i
    #print(names(df)[i])    
    num = countmap(df[:,names(df)[i]])
    print(length(num))

    push!(varnames,Variable(:i,3))
end

my_edges = []
open("./example/example.gph","r") do datafile
    while ! eof(datafile)
        line = readline(datafile)
        push!(my_edges, line)
        #print(my_edges)
    end
end

g = SimpleDiGraph(length(names2idx))

for edge in my_edges
    parent, child = split(edge, ",")
    add_edge!(g, names2idx[parent], names2idx[child])
end
#convert(Matrix{Int64}, df)

value = bayesian_score(varnames,g,trandf)
print("\nBayesian score is",value)
