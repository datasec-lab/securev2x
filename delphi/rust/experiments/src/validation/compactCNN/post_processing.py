"""
Created 10/15/23

Author: @Yoshi234
"""
import matplotlib.pyplot as plt
import pandas as pd

def plot_latencies(srv_latencies, cli_latencies, run_num):
    latencies = pd.DataFrame({"server latency":srv_latencies, "client latency":cli_latencies})
    ax = latencies.plot(kind="box")
    ax.set_title("Latency Measures")
    ax.set_ylabel("Latency time")
    plt.savefig("figures/latencies{}.png".format(run_num))

def plot_preproc(srv_preproc, cli_preproc, run_num):
    preproc = pd.DataFrame({"server preprocessing":srv_preproc, "client preprocessing": cli_preproc})
    ax = preproc.plot(kind = "box")
    ax.set_title("Preprocessing Measures")
    ax.set_ylabel("Preprocessing time")
    plt.savefig("figures/preprocessing{}.png".format(run_num))

def plot_through_time(srv_totals, cli_totals, run_num):
    total = pd.DataFrame({"server totals":srv_totals, "client totals": cli_totals})
    ax = total.plot(kind = "box")
    ax.set_title("Throughput Measures")
    ax.set_ylabel("Throughput time")
    plt.savefig("figures/throughput_time{}.png".format(run_num))

def append_time(times_list, tracker_val, min_param, max_param, line):
        val = ""
        unit = ""
        for i in range(84, len(line)):
            if line[i].isdigit() or line[i]==".": val += line[i]
        for j in range(84, len(line)):
            if line[j].isalpha(): unit += line[j]
        val = float(val)
        if unit == "ms":
            val /= 1000
            tracker_val += val
            if val > max_param: max_param = val
            if val < min_param: min_param = val
            times_list.append(val)
        if unit == "s":
            tracker_val += val
            if val > max_param: max_param = val
            if val < min_param: min_param = val
            times_list.append(val)
        return (tracker_val, max_param, min_param, times_list)

def main():
    run_number = "7"
    processing_output_file = "/home/jjl20011/snap/snapd-desktop-integration/current/Lab/Projects/Project1-V2X-Secure2PC/v2x-delphi-2pc/delphi/rust/experiments/src/validation/compactCNN/validation_runs/validation_run{}.txt".format(run_number)
    num_comp = 316

    srv_tot = 0
    srv_totals = []
    cli_tot = 0
    cli_totals = []

    max_srv_pre = 0
    min_srv_pre = 1000
    srv_preproc = 0
    srv_preprocessing = []
    
    max_cli_pre = 0
    min_cli_pre = 1000
    cli_preproc = 0
    cli_preprocessing = []

    max_srv_lat = 0
    min_srv_lat = 1000
    server_latency = 0
    srv_latencies = []

    max_cli_lat = 0
    min_cli_lat = 1000
    client_latency = 0
    cli_latencies = []

    progress = 0

    with open(processing_output_file, "r") as f:
        cont = True
        cli_subtotal = 0
        srv_subtotal = 0
        lines = f.readlines()
        for line in lines:
            if "End:     Server online phase" in line:
                server_latency, max_srv_lat, min_srv_lat, srv_latencies = append_time(srv_latencies, 
                                                                                      server_latency, 
                                                                                      min_srv_lat, 
                                                                                      max_srv_lat, 
                                                                                      line)
                srv_subtotal += srv_latencies[-1]
            elif "End:     Client online phase" in line:
                # this is the last step to appear in the log
                client_latency, max_cli_lat, min_cli_lat, cli_latencies = append_time(cli_latencies, 
                                                                                      client_latency, 
                                                                                      min_cli_lat, 
                                                                                      max_cli_lat, 
                                                                                      line)
                cli_subtotal += cli_latencies[-1]
                progress += 1
                cli_totals.append(cli_subtotal)
                srv_totals.append(srv_subtotal)
                cli_tot += cli_subtotal
                srv_tot += srv_subtotal
                cli_subtotal = 0
                srv_subtotal = 0
                print(progress/num_comp*100,"%")
            elif "End:     Client offline phase" in line:
                cli_preproc, max_cli_pre, min_cli_pre, cli_preprocessing = append_time(cli_preprocessing, 
                                                                                       cli_preproc, 
                                                                                       min_cli_pre, 
                                                                                       max_cli_pre, 
                                                                                       line)
                cli_subtotal += cli_preprocessing[-1]
            elif "End:     Server offline phase" in line:
                srv_preproc, max_srv_pre, min_srv_pre, srv_preprocessing = append_time(srv_preprocessing, 
                                                                                       srv_preproc, 
                                                                                       min_srv_pre, 
                                                                                       max_srv_pre, 
                                                                                       line)
                srv_subtotal += srv_preprocessing[-1]
                

            # if progress == num_comp: cont = False
    
    # get the number of observations read
    num_comp = len(cli_totals)

    avg_cl_lat = client_latency / num_comp
    avg_srv_lat = server_latency / num_comp
    avg_cli_preproc = cli_preproc / num_comp
    avg_srv_preproc = srv_preproc / num_comp
    srv_throughput = num_comp/srv_tot
    cli_throughput = num_comp/cli_tot

    with open("latency.txt{}".format(run_number), "w") as f:
        f.write("Avg Client Latency = {} s\n".format(avg_cl_lat))
        f.write("\tmax latency = {} s\n".format(max_cli_lat))
        f.write("\tmin latency = {} s\n".format(min_cli_lat))
        f.write("Avg Server Latency = {} s\n".format(avg_srv_lat))
        f.write("\tmax latency = {} s\n".format(max_srv_lat))
        f.write("\tmin latency = {} s\n".format(min_srv_lat))
        f.write("Avg Client Preprocessing = {} s\n".format(avg_cli_preproc))
        f.write("\tmax preproc = {} s\n".format(max_cli_pre))
        f.write("\tmin preproc = {} s\n".format(min_cli_pre))
        f.write("Avg Server Preprocessing = {} s\n".format(avg_srv_preproc))
        f.write("\tmax preproc = {} s\n".format(max_srv_pre))
        f.write("\tmin preproc = {} s\n".format(min_srv_pre))
        f.write("Server Throughput - {} s per process\n".format(srv_throughput))
        f.write("Client Throughput - {} s per process\n".format(cli_throughput))
    
    plot_preproc(srv_preprocessing, cli_preprocessing, run_number)
    plot_through_time(srv_totals, cli_totals, run_number)
    plot_latencies(srv_latencies, cli_latencies, run_number)

if __name__ == "__main__":
    main()