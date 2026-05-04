import os
import argparse
import pandas as pd

from mem3dmapper.cli.benchmark_eval_2 import EvaluationCost, DeviceParams, MultiplierConst, AdderConst, UltraCost, SimplerCost, LogicCost, AutoCost, AdderNorConst
from mem3dmapper.cli.benchmark_eval_2 import evaluate_MVM_cost_2D, evaluate_MVM_cost_3D, plot_multi_frameworks
# overhead factors per 2D framework — justified by their gate counts

def get_dims():
    parser = argparse.ArgumentParser(description="Extract N and M columns from CSV files")
    parser.add_argument("folder_path", help="Path to folder containing CSV files")
    args = parser.parse_args()

    folder_path = args.folder_path

    if not os.path.isdir(folder_path):
        print("Invalid folder path!")
        exit(1)

    data_dict = {}  # key = filename, value = list of (N, M)

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)

            if 'N' in df.columns and 'M' in df.columns:
                pairs = list(zip(df['N'], df['M']))
                data_dict[file.split(".")[0]] = pairs
            else:
                print(f"Skipping {file}: Columns N or M not found")

    # Now you can use it
    # for filename, pairs in data_dict.items():
    #     print(f"File: {filename}")
    #     for N, M in pairs:
    #         print(f"  N: {N}, M: {M}")

    return data_dict

if __name__ == "__main__":
    model_dims = get_dims()


    multiplier_cost = EvaluationCost(K=4, device_params=DeviceParams(), gateCount=MultiplierConst())
    adder_cost = EvaluationCost(K=4, device_params=DeviceParams(), gateCount=AdderConst())


    ultra_cost = EvaluationCost(K=4, device_params=DeviceParams(), gateCount=UltraCost())
    simpler_cost = EvaluationCost(K=4, device_params=DeviceParams(), gateCount=SimplerCost())
    logic_cost = EvaluationCost(K=4, device_params=DeviceParams(), gateCount=LogicCost())
    auto_cost = EvaluationCost(K=4, device_params=DeviceParams(), gateCount=AutoCost())
    
    # multiplier_nor_cost = EvaluationCost(K=4, device_params=DeviceParams(), gateCount=MultiplierNorConst())
    adder_nor_cost = EvaluationCost(K=4, device_params=DeviceParams(), gateCount=AdderNorConst())

    multiplier_cost.calculate()
    adder_cost.calculate()
    
    # multiplier_nor_cost.calculate()
    ultra_cost.calculate()
    simpler_cost.calculate()
    logic_cost.calculate()
    auto_cost.calculate()
    adder_nor_cost.calculate()

    print('multiplier_cost', multiplier_cost.get_results())
    print('adder_cost', adder_cost.get_results())

    # print(multiplier_nor_cost.get_results())
    print('ultra_cost', ultra_cost.get_results())
    print('simpler_cost', simpler_cost.get_results())
    print('logic_cost', logic_cost.get_results())
    print('auto_cost', auto_cost.get_results())
    print('adder_nor_cost', adder_nor_cost.get_results())

    energies_3d = []
    latencies_3d = []
    areas_3d = []

    ENERGY_CONV = 1e-15
    LATENCY_CONV = 1e-9
    AREA_CONV = 1e-9

    for model_name, dims in model_dims.items():
        e = 0
        l = 0
        a = 0
        for N, M in dims:
            e3, l3, a3 = evaluate_MVM_cost_3D(N, 0, multiplier_cost, adder_cost)
            e += (e3 * ENERGY_CONV) * M
            l += (l3 * LATENCY_CONV) * M
            a += (a3 * AREA_CONV) * M

        energies_3d.append(e)
        latencies_3d.append(l)
        areas_3d.append(a)

    print("3D Energy Consumption:", energies_3d)
    # print("3D Latency:", latencies_3d)
    # print("3D Area:", area_3d)

    frameworks = ["Ultra", "Simpler", "Logic", "Auto"]
    framework_costs = [ultra_cost, simpler_cost, logic_cost, auto_cost]
    
    # evaluate each framework across all benchmarks
    energies_2d_all = []
    latencies_2d_all = []
    areas_2d_all = []
    for fw_cost, framework_name in zip(framework_costs, frameworks):
        e_list = []
        l_list = []
        a_list = []
        for model_name, dims in model_dims.items():
            e = 0
            l = 0
            a = 0
            for N, M in dims:
                e2, l2, a2 = evaluate_MVM_cost_2D(N, 0, fw_cost, adder_nor_cost, framework_name)
                e += (e2 * ENERGY_CONV) * M
                l += (l2 * LATENCY_CONV) * M
                a += (a2 * AREA_CONV) * M

            e_list.append(e)
            l_list.append(l)
            a_list.append(a)
        energies_2d_all.append(e_list)
        latencies_2d_all.append(l_list)
        areas_2d_all.append(a_list)

    # print("2D Energy Consumption:", energies_2d_all)

    import numpy as np
    energy_matrix = np.vstack(energies_2d_all + [energies_3d])    # shape (m + 1, n) with 3D last
    latency_matrix = np.vstack(latencies_2d_all + [latencies_3d])
    area_matrix = np.vstack(areas_2d_all + [areas_3d])

    # per-benchmark normalization across all frameworks (including 3D)
    eps = 1e-30
    max_energy_per_bench = np.maximum(energy_matrix.max(axis=0), eps)
    max_latency_per_bench = np.maximum(latency_matrix.max(axis=0), eps)
    max_area_per_bench = np.maximum(area_matrix.max(axis=0), eps)

    norm_energy_matrix = energy_matrix / max_energy_per_bench
    norm_latency_matrix = latency_matrix / max_latency_per_bench
    norm_area_matrix = area_matrix / max_area_per_bench

    # labels: frameworks first, then 3D last
    labels = frameworks + ["PRISM"]
    # print("2D Energy:", energies_2d_all)
    print("2D Latency:", latencies_2d_all)
    # print("2D Area:", areas_2d_all)

    print(norm_energy_matrix)

    benchmarks = [f for f in model_dims.keys()]
    plot_multi_frameworks(benchmarks, labels, norm_energy_matrix, norm_latency_matrix, norm_area_matrix)
    # plot_actual_log_scale(benchmarks, labels, norm_energy_matrix, norm_latency_matrix, norm_area_matrix)
    # plot_results_log_scale(benchmarks, labels, norm_energy_matrix, norm_latency_matrix, norm_area_matrix)

