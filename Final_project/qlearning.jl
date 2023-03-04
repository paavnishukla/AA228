using CSV
using DataFrames
using DelimitedFiles
using Plots
using LinearAlgebra

### Q learning implementation
mutable struct QLearning
S # state space (assumes 1:nstates)
A #action space (assumes 1:nactions)
Y #discount
Q #action value function
alpha #learning rate
end

function update!(model::QLearning, s, a, r, sbar)
    Y, Q, alpha = model.Y, model.Q, model.alpha
    #print(keys(Q))
    Q[(s,a)] += alpha*(r + Y*maximum([Q[(sbar,"b")],Q[(sbar,"w")]]) - Q[(s,a)])
    return model
end

function allargmax(a)
    m = maximum(a)
    filter(i -> a[i] == m, eachindex(a))
end
function create_csv(inputfile)
    outputfile = replace(inputfile,".csv" => "_mdp.csv")
    outputfile = open(outputfile,"w")
    write(outputfile, "s,a,r,sp", "\n")     
    for row in CSV.Rows(inputfile)
        flight = row.flight
        days_left = parse(Int64,row.days_left)
        price = parse(Int64,row.price)
        #print("\n",flight,days_left,price) 
        s = string(flight,'_',days_left) 
        s = replace(s,"-"=>"")
        if(days_left > 1)
            sp = string(flight,'_',days_left-1)
            sp = replace(sp,"-"=>"")
            a = "w"
            r = 0
            mdptuple = s,a,r,sp
            write(outputfile, join(mdptuple,","), "\n")

        elseif(days_left == 1)
            sp = "notravel"
            a = "w"
            r = -1000000
            mdptuple = s,a,r,sp
            write(outputfile, join(mdptuple,","), "\n")     
        end

        sp = "travel"
        a = "b"
        r = -price
        mdptuple = s,a,r,sp 
        write(outputfile, join(mdptuple,","), "\n")

    end
    mdptuple = "travel","b",0,"travel" 
    write(outputfile, join(mdptuple,","),"\n")
    mdptuple = "travel","w",0,"travel"
    write(outputfile, join(mdptuple,","),"\n")
    mdptuple = "notravel","b",0,"notravel"
    write(outputfile, join(mdptuple,","),"\n")
    mdptuple = "notravel","b",0,"notravel"
    write(outputfile, join(mdptuple,","),"\n")

    close(outputfile)
end
function parse_file2func(inputfile)
    Y = 1
    alpha = 0.1
    m = 1561*49 + 1
    n = 2
    S = collect(1:m)
    A = collect(1:n)
    Q = Dict{Tuple{String,String},Float64}()
    states = Set()
    for row in CSV.Rows(inputfile)
        s = row.s
        a = row.a
        sp = row.sp
        Q[(s,"b")] = 0
        Q[(sp,"b")] = 0
        Q[(s,"w")] = 0
        Q[(sp,"w")] = 0
        push!(states,s)
        push!(states,sp)


    end
    Q[("notravel","b")] = 0
    Q[("notravel","w")] = 0
    Q[("travel","b")] = 0
    Q[("travel","w")] = 0

    model = QLearning(S,A,Y,Q,alpha)
    for j in 1:1
        for row in CSV.Rows(inputfile)
            s = row.s
            a = row.a
            r = parse(Int64,row.r)
            sp = row.sp
            update!(model, s,a,r,sp)
        end
    end

    #print("Q is\n",Q,"\n")
    outputfile = replace(inputfile,"csv" => "policy")
    outputfile = open(outputfile,"w")

    for state in sort!(collect(states))
        maxaction = (Q[(state,"w")] <  Q[(state,"b")]) ? "b" : "w"
        write(outputfile, state,",", maxaction, "\n")     
    end
    close(outputfile)

end
inputfile = ARGS[1]
create_csv(inputfile)
inputfile = replace(inputfile,".csv" => "_mdp.csv")
parse_file2func(inputfile)