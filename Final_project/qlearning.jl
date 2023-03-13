using CSV
using DataFrames
using DelimitedFiles
using Plots
using LinearAlgebra
using Dates, DayCounts
### Q learning implementation
mutable struct QLearning
Y #discount
Q #action value function
alpha #learning rate
end

function update!(model::QLearning, s, a, r, sbar)
    Y, Q, alpha = model.Y, model.Q, model.alpha
    Q[(s,a)] += alpha*(r + Y*maximum([Q[(sbar,"b")],Q[(sbar,"w")]]) - Q[(s,a)])
    return model
end

function allargmax(a)
    m = maximum(a)
    filter(i -> a[i] == m, eachindex(a))
end

function create_csv_octoparse(inputfile,refdate)
    outputfile = replace(inputfile,".csv" => "_mdp.csv")
    o_file = open(outputfile,"w")
    write(o_file, "s,a,r,sp", "\n") 
    for row in CSV.Rows(inputfile)
        flightCode = row.flightCode
        url = row.Page_URL
        url2 = split(url,"India")
        #print("\n",url2)
        url3 = split(url2[3],"&")
        url4 = split(url3[1],"C")
        date = Date(url4[2],dateformat"d/m/y")
        days_left = Dates.value(date-refdate)
        #print(days_left)
        num = row.NumFlight
        flight = string(flightCode,num)
        #days_left = parse(Int64,row.days_left)
        price = parse(Int64,row.Price)
        #print("\n",flight,days_left,price) 
        s = string(flight,'_',days_left) 
        if(days_left > 0)
            sp = string(flight,'_',days_left-1)
            #sp = replace(sp,"-"=>"")
            a = "w"
            r = 0.0
            mdptuple = s,a,r,sp
            write(o_file, join(mdptuple,","), "\n")
        # elseif(days_left == 0)
        #     sp = "notravel"
        #     a = "w"
        #     r = -57090.0
        #     mdptuple = s,a,r,sp
        #     write(o_file, join(mdptuple,","), "\n")     
        end
        sp = "travel"
        a = "b"
        r = Float64(100000/price)
        #r = Float64(-price)
        mdptuple = s,a,r,sp 
        write(o_file, join(mdptuple,","), "\n")

    end
    mdptuple = "travel","b",0,"travel" 
    write(o_file, join(mdptuple,","),"\n")
    mdptuple = "travel","w",0,"travel"
    write(o_file, join(mdptuple,","),"\n")
    mdptuple = "notravel","b",0,"notravel"
    write(o_file, join(mdptuple,","),"\n")
    mdptuple = "notravel","w",0,"notravel"
    write(o_file, join(mdptuple,","),"\n")
    close(o_file)
    #print("hi1")

    #outputfile = open(outputfile,"r")
    Reward = Dict{Tuple{String,String},Tuple{String,Float64,Int64}}()
    #println(outputfile)
    i_file = open(outputfile, "r")
    for row in CSV.Rows(i_file)
        #println(row)
        s = row.s
        a = row.a
        r = row.r
        sp = row.sp
        if haskey(Reward, Tuple((s, a)))
            r = parse(Float64,row.r) + Reward[(s,a)][2]
            c = 1+ Reward[(s,a)][3]
        else
            r = parse(Float64,row.r)
            c = 1
        end 
        Reward[(s,a)] =  Tuple((sp,r,c))

        while(!(haskey(Reward, Tuple((sp, "b"))) || haskey(Reward, Tuple((sp, "w"))) ) && sp != "travel")

            temp_sp = split(sp,"_")
            s = sp
            days = parse(Int64,temp_sp[2])-1
            if days >= 0
                sp = string(temp_sp[1],"_",days)
                Reward[(s,"w")] =  Tuple((sp,0.0,0))
                Reward[(s,"b")] =  Tuple((sp,0.0,0))

            elseif days == -1
                Reward[(s,"w")] =  Tuple(("notravel",0.0,0))
                Reward[(s,"b")] =  Tuple(("travel",0.0,0)) 
            else            
                break
            end
        end
    end
    #print("hi2",Reward)
    o_filestr = replace(outputfile,"_mdp" => "_mdp_modified")
    o_file2 = open(o_filestr,"w")
    write(o_file2, "s,a,r,sp", "\n") 
    for (key,value) in sort(collect(Reward))
        s = key[1]
        a = key[2]
        sp = value[1]
        if value[3] == 0
            r = 0.0
        else
            r = value[2]/Float64(value[3])
        end
        mdptuple = s,a,r,sp
        #print(mdptuple)
        write(o_file2, join(mdptuple,","), "\n")
    end
    close(o_file2)
    # outputfile2 = replace(outputfile,".csv" => "_modified.csv")
    # outputfile2 = open(outputfile,"w")
    # write(outputfile2, "s,a,r,sp", "\n") 
    # for row in CSV.Rows(outputfile) 
    # end
    #print(Reward)
    return Reward
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
    mdptuple = "notravel","w",0,"notravel"
    write(outputfile, join(mdptuple,","),"\n")
    close(outputfile)
end

function parse_file2func(inputfile)
    Y = 1
    alpha = 1
    #m = 1561*49 + 1
    #n = 2
    #S = collect(1:m)
    #A = collect(1:n)
    Q = Dict{Tuple{String,String},Float64}()
    states = Set()
    for row in CSV.Rows(inputfile)
        s = row.s
        #a = row.a
        sp = row.sp
        Q[(s,"b")] = 0.0
        Q[(sp,"b")] = 0.0
        Q[(s,"w")] = 0.0
        Q[(sp,"w")] = 0.0
        push!(states,s)
        push!(states,sp)
    end
    Q[("notravel","b")] = 0
    Q[("notravel","w")] = 0
    Q[("travel","b")] = 0
    Q[("travel","w")] = 0
    model = QLearning(Y,Q,alpha)
    count = 0
    for j in 1:100
        for row in CSV.Rows(inputfile)
            s = row.s
            a = row.a
            r = parse(Float64,row.r)
            sp = row.sp
            #print("\nline=",count,",s=",s,",a=",a,",r=",r,",sp=",sp,",Q=",Q[(s,a)],",w=",Q[(sp,"w")],",b=",Q[(sp,"b")])
            update!(model, s,a,r,sp)
            #print("Qupdate =",Q[(s,a)])
        end
        count += 1
    end
    #print("Q is\n",Q,"\n")
    outputfile = replace(inputfile,"csv" => "policy")
    out_1 = open(outputfile,"w")
    write(out_1,"state,action\n")
    for state in sort!(collect(states))
        maxaction = (Q[(state,"w")] <  Q[(state,"b")]) ? "b" : "w"
        write(out_1, state,",", maxaction, "\n")     
    end
    close(out_1)
end

function runtest(traininput,testinput,refdate)
    #   df = CSV.read(inputfile, DataFrame)
    #  for i in 1:size(df,2)
    trainpolicy = Dict{String,String}()
    for row in CSV.Rows(traininput)
        state = row.state
        action = row.action
        trainpolicy[state] = action
    end
    Reward = Dict{Tuple{String,String},Tuple{String,Float64,Int64}}()
    Reward = create_csv_octoparse(testinput,refdate)
    testmdp = replace(testinput,".csv" => "_mdp_modified.csv")
    parse_file2func(testmdp)
    testoutput = replace(testmdp,".csv" => ".policy")
    testmetric = replace(testoutput,".policy"=>"_testmetrics.csv","_mdp_modified"=>"")
    testmetric_out = open(testmetric,"w")
    write(testmetric_out,"State,Random_action,Predicted_action,Optimal_action,Performance,Optimal Performance,Normalized Performance\n")
    totalperf = 0.0
    totaloptimalperf = 0.0
    totalnormperf = 0.0
    count = 0
    for row in CSV.Rows(testoutput)
        count += 1
        s = row.state
        optimal_action = row.action
        predicted_action = haskey(trainpolicy,s) ? trainpolicy[s] : rand(["w","b"])
        #print("\n",s)
        if((s == "travel") || (s == "notravel"))
            random_action = "w"
        else
            if parse(Int64,(split(s,"_"))[2]) != 0
                random_action = rand(["w","b"])
            else
                random_action = optimal_action
            end
        end
        randprice = (random_action == "w") ? 0.0 : ((Reward[(s,random_action)][3] == 0) ? 0.0 : Reward[(s,random_action)][2]/Float64(Reward[(s,random_action)][3]))
        predictprice = (predicted_action == "w") ? 0.0 : ((Reward[(s,random_action)][3] == 0) ? 0.0 : Reward[(s,predicted_action)][2]/Float64(Reward[(s,predicted_action)][3]))
        optimalprice = (optimal_action == "w") ? 0.0 : ((Reward[(s,random_action)][3] == 0) ? 0.0 : Reward[(s,optimal_action)][2]/Float64(Reward[(s,optimal_action)][3]))
        reward_wait = 0.0
        reward_buy = (Reward[(s,"b")][3] == 0) ? 0.0 : Reward[(s,"b")][2]/Float64(Reward[(s,"b")][3])
        #print("\nreward=",s,",",Reward[(s,"b")][3],",",Reward[(s,"b")][3] == 0,",",Reward[(s,"b")][3],",",reward_buy)
        perf = (randprice == 0.0) ? 0.0 : ((randprice-predictprice))/randprice
        optimalperf = (randprice == 0.0) ? 0.0 : (randprice-optimalprice)/randprice
        normperf = (((randprice-optimalprice) == 0.0) && ((randprice-predictprice) == 0.0)) ? 1.0 : (((randprice-optimalprice) == 0.0) ? 0.0 : (randprice-predictprice)/(randprice-optimalprice))
        totalperf += perf
        totaloptimalperf += optimalperf
        totalnormperf += normperf
        testtuple = s,random_action,predicted_action,optimal_action,perf,optimalperf,normperf
        write(testmetric_out, join(testtuple,","), "\n")
    end

    if count == 0
        testtuple = "","","","",0,0,0
    else
        testtuple = "","","","",totalperf/Float64(count),totaloptimalperf/Float64(count),totalnormperf/Float64(count)
    end
    write(testmetric_out, join(testtuple,","), "\n")
    close(testmetric_out)
end

if length(ARGS) == 1
    print("Only 1 file available, treating it as train")
    trainfile = ARGS[1]
    refdate = Date("13/03/2023",dateformat"d/m/y")
    create_csv_octoparse(trainfile,refdate)
    trainfile2func = replace(trainfile,".csv" => "_mdp_modified.csv")
    parse_file2func(trainfile2func)
    trainpolicy = replace(trainfile2func,".csv"=>".policy")
    print("The optimal policy is in ",trainpolicy," Please check")
elseif length(ARGS) == 2
    trainfile = ARGS[1]
    testfile = ARGS[2]
    refdate = Date("13/03/2023",dateformat"d/m/y")
    create_csv_octoparse(trainfile,refdate)
    trainfile2func = replace(trainfile,".csv" => "_mdp_modified.csv")
    parse_file2func(trainfile2func)
    trainpolicy = replace(trainfile2func,".csv"=>".policy")
    refdate = Date("14/03/2023",dateformat"d/m/y")
    runtest(trainpolicy,testfile,refdate)
    testmetric = replace(trainfile,".csv" => "_testmetrics.csv")
    print("Testmetrics is available. Please check ",testmetric)
else
    error("usage: julia qlearning.jl trainfile.csv testfile.csv")
end
