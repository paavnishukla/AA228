using CSV
using DataFrames
using DelimitedFiles
using Plots
using LinearAlgebra
using Dates, DayCounts
using Random
using StatsBase

DATASET_DIR = "dataset/"

"""
    Q learning data structure
"""
mutable struct QLearning
Y           #discount
Q           #action value function
alpha       #learning rate
end

"""
    The update function for Q learning
"""
function update!(model::QLearning, s, a, r, sbar)
    Y, Q, alpha = model.Y, model.Q, model.alpha
    Q[(s,a)] += alpha*(r + Y*maximum([Q[(sbar,"b")],Q[(sbar,"w")]]) - Q[(s,a)])
    return model
end

function allargmax(a)
    m = maximum(a)
    filter(i -> a[i] == m, eachindex(a))
end

"""
    Function to visualize the price trend of 10 flight numbers 
    for all the routes (source-destination pairs) over 7 days
    for a departure date of 1st May 2023.
    
    The flight numbers are chosen at random, with the constraint
    that data was available on each of the 6 days data was collected
"""
function visualize_price_trend()
    sources = ["Delhi", "Bangalore", "Kolkata"]
    # sources = ["Bangalore"]
    files = readdir(string(DATASET_DIR, "train/"))
    # Loop over all the files
    for source in sources
        unique_flights = Vector{String}()
        for file in files
            i_file = string(DATASET_DIR, "train/", file)
            # Read the CSV file into a DataFrame
            df = CSV.read(i_file, DataFrame)    
            # Extract rows with the particular source
            fil_df = filter(row -> row[5] == source, df)
            # concatenate FlightCode, NumFlight into a single column
            fil_df.Flight_Id = string.(fil_df.FlightCode, fil_df.NumFlight)
            if isempty(unique_flights)
                unique_flights = unique(fil_df[!, size(fil_df, 2)])
                # println(source, unique_flights)
            else
                unique_flights = intersect(unique_flights, unique(fil_df[!, size(fil_df, 2)]))
                # println(source, unique_flights)
            end
        end

        # Choose 10 random flight IDs from the unique list for each route
        unique_flights = sample(unique_flights, 10, replace=false)
        # println(source, unique_flights)

        # Get the price of these 10 flights over the 7 days 
        # Plotting data contains 7 price points for each of the 10 flights
        plot_data = Dict{String, Vector{Float64}}()
        x = ["13 Mar", "14 Mar", "15 Mar", "16 Mar", "17 Mar", "18 Mar"]
        p = plot(title = "Flight Price vs Time", lw = 2)
        for flight in unique_flights
            price_data = Vector{Float64}()
            for file in files
                i_file = string(DATASET_DIR, "train/", file)
                # Read the CSV file into a DataFrame
                df = CSV.read(i_file, DataFrame)    
                # concatenate FlightCode, NumFlight into a single column
                df.Flight_Id = string.(df.FlightCode, df.NumFlight)
                # Extract rows with the particular source
                df = filter(row -> row[11] == flight, df)
                price = df[1, 9] 
                push!(price_data, price)
            end
            # println(price_data)
            plot_data[flight] = price_data
            
            # Plot the data
            plot!(p, x, price_data, xlabel = "Dates", ylabel = "Price in INR", label = flight)
        end
        savefig(string(source, "_price_trend.png"))
    end
end

"""
    Builds a reward dictionary that maps every unique (s, a) to (sp, r, c)
"""
function compute_reward(input_mdp)
    Reward = Dict{Tuple{String,String},Tuple{String,Float64,Int64}}()
    for row in eachrow(input_mdp)
        s = row.s
        a = row.a
        r = row.r
        sp = row.sp
        if haskey(Reward, Tuple((s, a)))
            r = row.r + Reward[(s,a)][2]
            c = 1+ Reward[(s,a)][3]
        else
            r = row.r
            c = 1
        end 
        Reward[(s,a)] =  Tuple((sp,r,c))

        # If the next state (the next day) has no entries in the CSV and the next state is
        # still not a terminal state => set a 0 reward for both buying and waiting
        # This is while there is still time left. Otherwise, set the next state to
        # a terminal state
        if !(s == "travel" || s == "notravel")
            while(!(haskey(Reward, Tuple((sp, "b"))) || haskey(Reward, Tuple((sp, "w")))) && sp != "travel" && sp != "notravel")
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
    end
    
    return Reward
end

"""
    Transform the input rows into the reqd. mdp tuples
"""
function get_mdp(df, refdate, o_file)
    nodenames = ["s", "a", "r", "sp"]
    mdp_df = DataFrame([name => [] for name in nodenames])
    for row in eachrow(df)
        FlightCode = row.FlightCode
        # Compute the days left
        url = row.Page_URL
        url2 = split(url,"India")
        url3 = split(url2[3],"&")
        url4 = split(url3[1],"C")
        date = Date(url4[2],dateformat"d/m/y")
        days_left = Dates.value(date-refdate)
        num = row.NumFlight
        flight = string(FlightCode,num)
        price = row.Price
        s = string(flight,'_',days_left) 

        # Add the transition from this date to the next date
        # This happens when the user decides to wait
        # Waiting has a reward of 0 associated to it
        if(days_left > 0)
            sp = string(flight,'_',days_left-1)
            a = "w"
            r = 0.0
            mdptuple = s,a,r,sp
            push!(mdp_df, mdptuple)
        end

        # Add the transition when the user decides to purchase
        # the ticket on this date - action is buy
        # Buying today has a reward that is proportional to inverse of today's price
        sp = "travel"
        a = "b"
        r = Float64(100000/price)
        #r = Float64(-price)
        mdptuple = s,a,r,sp 
        push!(mdp_df, mdptuple)
    end
    
    # Add some more state transitions associated with terminal states
    mdptuple = "travel","b",0,"travel" 
    push!(mdp_df, mdptuple)
    mdptuple = "travel","w",0,"travel"
    push!(mdp_df, mdptuple)
    mdptuple = "notravel","b",0,"notravel"
    push!(mdp_df, mdptuple)
    mdptuple = "notravel","w",0,"notravel"
    push!(mdp_df, mdptuple)
    
    # Some state tuples appear multiple times in the CSV
    # For eg: Flight 2 at 5th day before departure with multiple
    # price points - average out the rewards associated
    nodenames = ["s", "a", "r", "sp"]
    mdp_df_mod = DataFrame([name => [] for name in nodenames])
    Reward = compute_reward(mdp_df)
    
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
        push!(mdp_df_mod, mdptuple)
    end

    # Concatenate to the output file
    CSV.write(o_file, mdp_df_mod, header=false, append=true)
    # println(mdp_df_mod)
end

"""
    Function that processes input files which contains flight price details
    (scraped using Octoparse) and creates a new file which contains rows
    representing the MDP tuple (s,a,r,sp,price)
"""
function create_csv_mdp(is_test, ref_date_str)
    # Get a list of files in the current directory
    files = Vector{String}()
    if !is_test
        println()
        files = readdir(string(DATASET_DIR, "train/"))
    else
        push!(files, "test_Delhi.csv")
        push!(files, "test_Bangalore.csv")
        push!(files,  "test_Kolkata.csv")
    end

    sources = ["Delhi", "Bangalore", "Kolkata"]

    # Remove any output MDP files if they exist
    for source in sources
        rm_file = string()
        if !is_test
            rm_file = string(DATASET_DIR, source, "_mdp.csv")
        else 
            rm_file = string(DATASET_DIR, source, "_test_mdp.csv")
        end

        if isfile(rm_file)
            rm(rm_file)
            println("Removed the MDP file for ", source)
        end
        
        # Write the header for each file
        out_file = open(rm_file,"w")
        write(out_file, "s,a,r,sp", "\n")
        close(out_file)
    end

    # Loop over all the files
    for file in files
        ref_date_s = string("")
        i_file = ""
        if !is_test
            i_file = string(DATASET_DIR, "train/", file)
            ref_date_s = replace(file, "_" => "/")
            ref_date_s = replace(ref_date_s, ".csv" => "")
            ref_date_s = string(ref_date_s, "/2023")
        else
            i_file = string(DATASET_DIR, file)
            ref_date_s = ref_date_str
        end
        
        println(ref_date_s) 
        ref_date = Date(ref_date_s, dateformat"d/m/y")
        for source in sources
            o_file = string()
            if !is_test
                o_file = string(DATASET_DIR, source, "_mdp.csv")
            else 
                o_file = string(DATASET_DIR, source, "_test_mdp.csv")
            end
        
            # Read the CSV file into a DataFrame
            println("Reading CSV file ", i_file)
            df = CSV.read(i_file, DataFrame)
            
            # Extract rows with the particular source
            filtered_df = filter(row -> row[5] == source, df)
            # println("Size: ", size(filtered_df, 1))
            
            # Concatenate to the output file 
            out_file = open(o_file,"a")
            get_mdp(filtered_df, ref_date, out_file)
            close(out_file)
        end
    end
    return
end

"""
    Function that generates a policy file for each route 
    from the given MDP file for the route
"""
function generate_policy(inputfile)
    Y = 1
    alpha = 1
    Q = Dict{Tuple{String,String},Float64}()
    states = Set()
    # Initializing the Q-value matrix
    for row in CSV.Rows(inputfile)
        s = row.s
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
    # count = 0
    
    # Fixing an iteration count of 100 for updates
    for j in 1:100
        for row in CSV.Rows(inputfile)
            s = row.s
            a = row.a
            r = parse(Float64, row.r)
            sp = row.sp
            #print("\nline=",count,",s=",s,",a=",a,",r=",r,",sp=",sp,",Q=",Q[(s,a)],",w=",Q[(sp,"w")],",b=",Q[(sp,"b")])
            update!(model, s,a,r,sp)
            #print("Qupdate =",Q[(s,a)])
        end
        # count += 1
    end
    
    # Remove any policy files if they exist
    outputfile = replace(inputfile, "csv" => "policy")
    if isfile(outputfile)
        rm(outputfile)
        println("Removed the policy file: ", outputfile)
    end

    out_f = open(outputfile,"w")
    write(out_f,"state,action\n")
    for state in sort!(collect(states))
        maxaction = (Q[(state,"w")] <  Q[(state,"b")]) ? "b" : "w"
        write(out_f, state,",", maxaction, "\n")     
    end
    close(out_f)
end

"""
    Function that tests the policy generated during the training phase
    on a test file that is presented. The reference date for computing
    days left must be provided for the test file.
"""
function runtest(traininput,testinput,refdate)
    trainpolicy = Dict{String,String}()

    # Initialize the policy matrix - optimal action for 
    # every state as dictated by the training phase
    for row in CSV.Rows(traininput)
        state = row.state
        action = row.action
        trainpolicy[state] = action
    end

    Reward = Dict{Tuple{String,String},Tuple{String,Float64,Int64}}()
    
    # Create the MDP for the test file 
    create_csv_mdp(true, refdate)

    # Compute the reward for this MDP file
    test_mdp_file = replace(traininput, "_mdp.policy" => "_test_mdp.csv")
    println(test_mdp_file)
    test_mdp = CSV.read(test_mdp_file, DataFrame)   
    Reward = compute_reward(test_mdp)
    
    # Generate the best policy for this test file
    generate_policy(test_mdp_file)
    testoutput = replace(test_mdp_file, ".csv" => ".policy")
    testmetric = replace(testinput, ".csv" => "metrics.csv")
    # testmetric = replace(testoutput,".policy"=>"_testmetrics.csv","_mdp_modified"=>"")
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

"""
    Main function - Point of entry to the program 
    The program needs to be fed the train and test files
"""
function main()
    sources = ["Delhi", "Bangalore", "Kolkata"]
    if (length(ARGS)!=0) && (length(ARGS) != 1)
        error("usage: julia qlearning.jl \n or \n julia qlearning.jl <test_file>")
    elseif length(ARGS) == 0
        println("Generating the MDP file from the scraped web data!")
        create_csv_mdp(false, "")
        println("Generating an optimal policy for each route...")
        for source in sources
            mdp_file = string(DATASET_DIR, source, "_mdp.csv")
            if !isfile(mdp_file)
                error("MDP file not found for ", source)
            end
            generate_policy(mdp_file)
            policy_file = replace(mdp_file,".csv"=>".policy")
            println("The optimal policy for ", source, " generated is in ",policy_file)
        end
    else
        testfile = ARGS[1]
        # Split the given test file into chunks containing flight details
        # for every route
        for source in sources   
            t_file = string(DATASET_DIR, "test_", source, ".csv")
            if isfile(t_file)
                rm(t_file)
                println("Removed the split test file for ", source)
            end
            
            # Read the CSV file into a DataFrame
            df = CSV.read(testfile, DataFrame)    
            # Extract rows with the particular source
            filtered_df = filter(row -> row[5] == source, df)
            # Concatenate to the output file 
            t_file_o = open(t_file,"w")
            CSV.write(t_file_o, filtered_df)
            println("Generated split test data file: ", t_file)
            close(t_file_o)  
        end
        
        # Generating evaluation metrics for each route
        for source in sources
            # Policy file is already generated
            p_file = string(DATASET_DIR, source, "_mdp.policy")
            if !isfile(p_file)
                error("Policy file not found for ", source)
            end
            t_file = string(DATASET_DIR, "test_", source, ".csv")
            if !isfile(t_file)
                error("Test file not found for ", source)
            end
            println("Testing policy on the test data: ", t_file)
            ref_test_d = "19/03/2023"
            # Run the test on the given file
            runtest(p_file, t_file, ref_test_d)
            metric_file = replace(t_file,".csv" => "metrics.csv")
            println("Testmetrics for ", source, "  available in ", metric_file)
        end
    end
end

# To train/test uncomment this call
main()
# To visualize price trend graphs uncomment this call
# visualize_price_trend()