import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import pyreadr

# ==========================
# Data Loading and Preprocessing
# ==========================
data = pyreadr.read_r('/Users/xingkun/Desktop/FYP/code/main1/Cruise/dataContact.RData')
pair_df = list(data.values())[0]
pair_df.rename(columns={"ID.x": "ID 1", "ID.y": "ID 2", "DURATION": "Weight", "SAIL": 'Sail', "DAY_INTERACT": "Day"}, inplace=True)

data = pyreadr.read_r('/Users/xingkun/Desktop/FYP/code/main1/Cruise/dataNodes.RData')
individual_df = list(data.values())[0]
individual_df.rename(columns={"ID": "ID", "DEGREE": "Degree", "STRENGTH": "Strenth", "EIGEN": 'Eigen', "BETWEEN": 'Between', "CC": 'CC', "SAIL": 'Sail', "DAY_INTERACT": 'Day'}, inplace=True)
individual_df = individual_df.drop(columns=["CABIN_NO", "AGE", "GENDER"])
individual_df = individual_df.dropna()

pair1_df = pair_df[pair_df["Sail"] == 1]
global cumulative_pcr_transmission, cumulative_pcr_quarantine
cumulative_pcr_transmission = 0
cumulative_pcr_quarantine = 0
# ==========================
# Resolution Filter Function
# ==========================
def filter_by_resolution(resolution):
    if resolution == "close":
        df = pair1_df[pair1_df["CONTACT_TYPE"] == "close"]
    elif resolution == "close and casual":
        df = pair1_df[(pair1_df["CONTACT_TYPE"] == "close") | (pair1_df["CONTACT_TYPE"] == "casual")]
    elif resolution == "all contact":
        df = pair1_df[(pair1_df["CONTACT_TYPE"] == "close") | (pair1_df["CONTACT_TYPE"] == "casual")| (pair1_df["CONTACT_TYPE"] == "transient")]
    else:
        df = pair1_df.copy()
    # Only use day 2 (full day)
    df = df[df["Day"] == 2.0]
    return df

# ==========================
# Graph Construction
# ==========================
def build_graph(filtered_df):
    aggregated_weights = (
        filtered_df.groupby(['ID 1', 'ID 2'])['Weight']
        .sum()
        .reset_index()
    )
    G = nx.Graph()
    for _, row in aggregated_weights.iterrows():
        G.add_edge(row["ID 1"], row["ID 2"], weight=row["Weight"])
    if len(G.nodes) == 0:
        return G
    largest_component = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_component).copy()

# ==========================
# PCR Test Function (does NOT modify state)
# ==========================

def pcr_test(individual_states, sensitivity=0.95, specificity=0.98):
    updated_states = individual_states.copy()
    PCR_state = {}

    for node, state in individual_states.items():
        rand_val = random.random()
        if state == "I":
            # Infected: true positive with probability sensitivity.
            PCR_state[node] = 1 if rand_val < sensitivity else 0
        elif state in ["S", "E"]:
            # Susceptible/Exposed: false positive with probability (1 - specificity).
            PCR_state[node] = 1 if rand_val < (1 - specificity) else 0
        else:
            # For any other state (if any), default to negative.
            PCR_state[node] = 0

    return updated_states, PCR_state

# ==========================
# Quarantine Function (as in your code)
# ==========================
def quarantine(
    individual_states,
    quarantine_state,
    time_exposed,
    time_infectious,
    consecutive_PCR,
    had_positive_PCR,
    pcr_result=None,
    sensitivity=0.95,
    specificity=0.98,
    exposure_threshold=3,
    quarantine_threshold=7, #max quarantine days
    exposure_durations=None,
    infectious_durations=None,
    use_new_testing_method=False,
    quarantine_testing_frequency=1
):
    """
    update quarantine state (once in quarantine, not in transmission loop)

    quarantine condition: 
    1. weight > exposure duration
    2. PCR positive 

    release condition: 
    1. hit consecutive quarantine_threshold (all -ve throughout)

    release as S is ALL -ve during quarantine
    release as R is at least ONE +ve during quarantine

    sequence: 
    1. check weight for ALL regardless of quarantine -> may go quarantine
    2. test PCR everyday in quarantine
    3. +ve -> go quarantine 
    4. -ve -> check release condition
    """
    updated_quarantine_state = quarantine_state.copy()
    updated_consecutive_PCR = consecutive_PCR.copy()
    updated_had_positive_PCR = had_positive_PCR.copy()
    updated_states = individual_states.copy()
    global cumulative_pcr_quarantine
    
    # Initialize counters for logging
    quarantine_entries_pcr = 0
    quarantine_entries_exposure = 0
    quarantine_releases = 0

    for node in individual_states.keys():
        if quarantine_state[node] == 0:  # Not in quarantine
            # Check if individual should enter quarantine
            if pcr_result is not None and pcr_result.get(node, 0) == 1:
                updated_quarantine_state[node] = 1
                updated_consecutive_PCR[node] = 0
                updated_had_positive_PCR[node] = True
                quarantine_entries_pcr += 1  # Count PCR-based entries
            elif time_exposed[node] > exposure_threshold:
                updated_quarantine_state[node] = 1  # Enter quarantine
                updated_consecutive_PCR[node] = 0  # Reset PCR counter
                updated_had_positive_PCR[node] = False  # Reset positive PCR flag
                quarantine_entries_exposure += 1  # Count exposure-based entries

        elif quarantine_state[node] == 1:  # In quarantine
            # Progress through states while in quarantine
            if individual_states[node] == "E":  # Exposed
                time_exposed[node] += 1
                if time_exposed[node] >= exposure_durations[node]:
                    updated_states[node] = "I"  # Become Infectious
                    time_infectious[node] = 0
            elif individual_states[node] == "I":  # Infectious
                time_infectious[node] += 1
                if time_infectious[node] >= infectious_durations[node]:
                    updated_states[node] = "R"  # Recover
            current_state = updated_states[node]
            
            # Perform PCR testing for quarantined individual
            if use_new_testing_method:
                # New testing method implementation
                updated_consecutive_PCR[node] += 1  # Increment days in quarantine
                
                if updated_consecutive_PCR[node] % quarantine_testing_frequency == 0:  # Test on 7th day
                    _, quarantine_PCR_state = pcr_test(
                        {node: current_state},
                        sensitivity=sensitivity,
                        specificity=specificity
                    )
                    cumulative_pcr_quarantine += 1 
                    
                    if quarantine_PCR_state[node] == 1:  # Positive test
                        updated_had_positive_PCR[node] = True
                        updated_consecutive_PCR[node] = 0  # Reset counter
                        #updated_states[node] = "I"
                    else:  # Negative test on 7th day
                        #updated_consecutive_PCR[node] += 1
                        if updated_consecutive_PCR[node] >= quarantine_threshold:
                            updated_quarantine_state[node] = 0  # Exit quarantine
                            quarantine_releases += 1
                            if updated_states[node] == "R":
                                pass
                            elif updated_had_positive_PCR[node]:
                                updated_states[node] = "R"  
                            else:
                                updated_states[node] = "S"
            else:
                # Original testing method: test daily
                _, quarantine_PCR_state = pcr_test(
                    {node: current_state},
                    sensitivity=sensitivity,
                    specificity=specificity
                )
                cumulative_pcr_quarantine += 1 
                
                if quarantine_PCR_state[node] == 1:  # Positive test
                    updated_consecutive_PCR[node] = 0  # Reset counter
                    updated_had_positive_PCR[node] = True
                    #updated_states[node] = "I"
                else:  # Negative test
                    updated_consecutive_PCR[node] += 1
                    if updated_consecutive_PCR[node] >= quarantine_threshold:
                        updated_quarantine_state[node] = 0  # Exit quarantine
                        quarantine_releases += 1
                        if updated_states[node] == "R":
                            pass
                        elif updated_had_positive_PCR[node]:
                            updated_states[node] = "R"  
                        else:
                            updated_states[node] = "S"

    # Print logging information
    #print(f"Quarantine entries (PCR): {quarantine_entries_pcr}, Entries (Exposure): {quarantine_entries_exposure}, Releases: {quarantine_releases}")

    return (
        updated_quarantine_state,
        updated_consecutive_PCR,
        updated_had_positive_PCR,
        updated_states,
    )

# ==========================
# Vaccination Function (affects susceptibility and r_values, not state)
# ==========================


def mass_vaccinate(
    individual_states,
    vaccination_status,
    vaccination_start_time,
    susceptibility,
    r_values,
    time_step,
    rollout_rate=0.02,                # 2% of the total population per day
    coverage_threshold=0.8,           # Stop if 80% of the population is vaccinated
    vaccination_efficacy=0.8,         # 80% reduction in susceptibility
    r_reduction=0.5                   # 50% reduction in infectiousness if breakthrough occurs
):
    """
    Apply mass vaccination: vaccinate individuals in S and E compartments.
    - The rollout_rate determines that 2% of the total population is vaccinated per day.
    - Vaccination stops when overall vaccination coverage reaches the coverage_threshold.
    """
    total_population = len(individual_states)
    current_coverage = sum(vaccination_status.values()) / total_population
    if current_coverage >= coverage_threshold:
        # Vaccination stops when the overall coverage threshold is reached.
        return vaccination_status, susceptibility, r_values

    # Determine the target number to vaccinate for today.
    target_num = int(rollout_rate * total_population)
    
    # Identify eligible individuals (in S or E compartments and not already vaccinated).
    eligible_nodes = [node for node, state in individual_states.items()
                      if (not vaccination_status[node]) and (state in ["S", "E"])]
    
    # Vaccinate a random subset up to the daily target.
    num_to_vaccinate = min(target_num, len(eligible_nodes))
    nodes_to_vaccinate = np.random.choice(eligible_nodes, size=num_to_vaccinate, replace=False)
    
    for node in nodes_to_vaccinate:
        vaccination_status[node] = True
        vaccination_start_time[node] = time_step
        # Update protection: reduce susceptibility and r_value.
        susceptibility[node] *= (1 - vaccination_efficacy)
        r_values[node] *= r_reduction

    return vaccination_status, susceptibility, r_values


def ring_vaccinate(
    individual_states,
    vaccination_status,
    vaccination_start_time,
    susceptibility,
    r_values,
    time_step,
    rollout_rate=0.02,                # Use rollout rate to vaccinate 2% of total population per day (applied to eligible contacts)
    coverage_threshold=0.8,           # Overall vaccination threshold for stopping vaccination
    vaccination_efficacy=0.8,         # 80% reduction in susceptibility
    r_reduction=0.5                   # 50% reduction in infectiousness if breakthrough occurs
):
    """
    Apply ring vaccination: vaccinate individuals only in the Exposed (E) compartment.
    This function vaccinates a daily target number (rollout_rate * total population)
    of exposed individuals (E) if they haven't been vaccinated yet.
    """
    total_population = len(individual_states)
    current_coverage = sum(vaccination_status.values()) / total_population
    if current_coverage >= coverage_threshold:
        # Stop if overall vaccination coverage is reached.
        return vaccination_status, susceptibility, r_values

    # Determine today's target number.
    target_num = int(rollout_rate * total_population)
    
    # Identify eligible nodes: those in the "E" compartment who haven't been vaccinated.
    eligible_nodes = [node for node, state in individual_states.items()
                      if (not vaccination_status[node]) and (state == "E")]
    
    # Vaccinate a random subset of eligible individuals up to the target number.
    num_to_vaccinate = min(target_num, len(eligible_nodes))
    nodes_to_vaccinate = np.random.choice(eligible_nodes, size=num_to_vaccinate, replace=False)
    
    for node in nodes_to_vaccinate:
        vaccination_status[node] = True
        vaccination_start_time[node] = time_step
        susceptibility[node] *= (1 - vaccination_efficacy)
        r_values[node] *= r_reduction

    return vaccination_status, susceptibility, r_values


def vaccinate(
    individual_states,
    vaccination_status,
    vaccination_start_time,
    susceptibility,
    r_values,
    time_step,
    rollout_rate=0.02,                # 2% rollout per day
    coverage_threshold=0.8,           # Stop when 80% coverage is reached
    vaccination_efficacy=0.8,
    r_reduction=0.5,
    strategy="mass"                 # "mass" for mass vaccination, "ring" for ring vaccination
):
    """
    Wrapper for vaccination that applies either mass or ring vaccination strategy.
    """
    if strategy == "mass":
        return mass_vaccinate(
            individual_states=individual_states,
            vaccination_status=vaccination_status,
            vaccination_start_time=vaccination_start_time,
            susceptibility=susceptibility,
            r_values=r_values,
            time_step=time_step,
            rollout_rate=rollout_rate,
            coverage_threshold=coverage_threshold,
            vaccination_efficacy=vaccination_efficacy,
            r_reduction=r_reduction
        )
    elif strategy == "ring":
        return ring_vaccinate(
            individual_states=individual_states,
            vaccination_status=vaccination_status,
            vaccination_start_time=vaccination_start_time,
            susceptibility=susceptibility,
            r_values=r_values,
            time_step=time_step,
            rollout_rate=rollout_rate,
            coverage_threshold=coverage_threshold,
            vaccination_efficacy=vaccination_efficacy,
            r_reduction=r_reduction
        )
    else:
        raise ValueError("Invalid vaccination strategy. Choose 'mass' or 'ring'.")

# ==========================
# Simulation Function
# ==========================
def simulate_epidemic(resolution,
                      pcr_active,
                      quarantine_active,
                      vaccination_active,
                      sensitivity=0.95,
                      specificity=0.98,
                      vaccination_coverage=0.5,
                      vaccination_efficacy=0.8,
                      r_reduction=0.5,
                      time_steps=600,
                      strategy="mass",
                      pcr_testing_frequency=1,
                      pcr_testing_percentage=0.5,
                      quarantine_testing_method=False,
                      quarantine_testing_frequency=1):
    # Initialize counters at the start of each simulation
    global cumulative_pcr_transmission, cumulative_pcr_quarantine
    cumulative_pcr_transmission = 0
    cumulative_pcr_quarantine = 0
    total_vaccinations = 0
    #global quarantine_pcr_tests
    #quarantine_pcr_tests = 0
    
    # Build network based on resolution
    df_filtered = filter_by_resolution(resolution)
    graph = build_graph(df_filtered)
    if len(graph.nodes) == 0:
        return None
    # Initialize states and parameters
    individual_states = {node: "S" for node in graph.nodes}
    initial_infected = np.random.choice(list(graph.nodes), size=3, replace=False)
    for node in initial_infected:
        individual_states[node] = "I"
    exposure_durations = {node: np.random.gamma(3, 4/3) for node in graph.nodes}
    infectious_durations = {node: np.random.gamma(6, 5/6) for node in graph.nodes}
    time_exposed = {node: 0 for node in graph.nodes}
    time_infectious = {node: 0 for node in graph.nodes}
    cumulative_exposure = {node: 0 for node in graph.nodes}
    quarantine_state = {node: 0 for node in graph.nodes}
    consecutive_PCR = {node: 0 for node in graph.nodes}
    had_positive_PCR = {node: False for node in graph.nodes}
    PCR_state = {node: 0 for node in graph.nodes}
    vaccination_status = {node: False for node in graph.nodes}
    vaccination_start_time = {node: None for node in graph.nodes}
    susceptibility = {node: 1.0 for node in graph.nodes}
    r_values = {node: 1.0 for node in graph.nodes}
    total_infected = set()
    state_transition_log = []
    beta = 0.0029
    last_pcr_test = {node: -1 for node in graph.nodes}

    for t in range(time_steps):
        if quarantine_active:
            # Track PCR tests done in quarantine

            #pre_quarantine_pcr = cumulative_pcr
            quarantine_state, consecutive_PCR, had_positive_PCR, individual_states = quarantine(
                individual_states=individual_states,
                quarantine_state=quarantine_state,
                time_exposed=time_exposed,
                time_infectious=time_infectious,
                consecutive_PCR=consecutive_PCR,
                had_positive_PCR=had_positive_PCR,
                pcr_result=PCR_state,
                sensitivity=sensitivity,
                specificity=specificity,
                exposure_threshold=3,
                quarantine_threshold=7,
                exposure_durations=exposure_durations,
                infectious_durations=infectious_durations,
                use_new_testing_method=quarantine_testing_method,
                quarantine_testing_frequency=quarantine_testing_frequency
            )
            # Calculate PCR tests done in quarantine
            #quarantine_pcr_tests = cumulative_pcr - pre_quarantine_pcr

        # Interventions
        if vaccination_active:
            prev_vaccinated = sum(1 for v in vaccination_status.values() if v)
            vaccination_status, susceptibility, r_values = vaccinate(
                                                            individual_states,
                                                            vaccination_status,
                                                            vaccination_start_time,
                                                            susceptibility,
                                                            r_values,
                                                            t,
                                                            rollout_rate=0.02,                # 2% rollout per day
                                                            coverage_threshold=vaccination_coverage,           # Stop when 80% coverage is reached
                                                            vaccination_efficacy=vaccination_efficacy,
                                                            r_reduction=r_reduction,
                                                            strategy=strategy                 # "mass" for mass vaccination, "ring" for ring vaccination
                                                        )
            new_vaccinations = sum(1 for v in vaccination_status.values() if v) - prev_vaccinated
            if new_vaccinations > 0:
                total_vaccinations += new_vaccinations
        # Transmission dynamics
        new_states = individual_states.copy()
        for node in graph.nodes:
            if quarantine_state[node] > 0:  # If quarantine_state > 0, the individual is in quarantine
                continue
            elif individual_states[node] == "S":
                for neighbor in graph.neighbors(node):
                    if quarantine_state[neighbor] > 0:
                        continue
                    if individual_states[neighbor] == "I":
                        weight = graph[node][neighbor]['weight']
                        cumulative_exposure[node] += weight
                        infection_prob = 1 - np.exp(-susceptibility[node] * beta * r_values[neighbor] * weight)
                        if np.random.rand() < infection_prob:
                            new_states[node] = "E"
                            time_exposed[node] = 0
                            break
            elif individual_states[node] == "E":
                time_exposed[node] += 1
                if time_exposed[node] >= exposure_durations[node]:
                    new_states[node] = "I"
                    time_infectious[node] = 0
            elif individual_states[node] == "I":
                total_infected.add(node)
                time_infectious[node] += 1
                if time_infectious[node] >= infectious_durations[node]:
                    new_states[node] = "R"
        individual_states = new_states.copy()
        state_transition_log.append({
            "individual_states": individual_states.copy(),
            "quarantine_state": quarantine_state.copy()
        })
        if pcr_active:
            should_test_today = (pcr_testing_frequency == 1) or (t % pcr_testing_frequency == 0)
            if should_test_today:
                eligible_nodes = [node for node in graph.nodes 
                                if quarantine_state[node] == 0]
                sample_size = int(len(eligible_nodes) * pcr_testing_percentage)
                
                # Convert eligible_nodes to a list if it isn't already
                eligible_nodes = list(eligible_nodes)
                
                # Only proceed with random choice if we have eligible nodes
                if eligible_nodes and sample_size > 0:
                    nodes_to_test = np.random.choice(eligible_nodes, 
                                                   size=min(sample_size, len(eligible_nodes)), 
                                                   replace=False)
                    
                    # Convert nodes_to_test to a list for safer iteration
                    nodes_to_test = nodes_to_test.tolist()
                    
                    if nodes_to_test:  # Check if we have any nodes to test
                        test_states = {node: individual_states[node] for node in nodes_to_test}
                        _, PCR_state_subset = pcr_test(test_states, sensitivity, specificity)
                        
                        PCR_state = {node: 0 for node in graph.nodes}
                        for node in nodes_to_test:
                            PCR_state[node] = PCR_state_subset[node]
                            last_pcr_test[node] = t
                        
                        # Add regular PCR tests to cumulative count
                        #global cumulative_pcr
                        cumulative_pcr_transmission += len(nodes_to_test)
        #cumulative_pcr += quarantine_pcr_tests

        if all(state in {"S", "R"} for state in individual_states.values()):
            break

    s_counts = [sum(1 for s in log["individual_states"].values() if s == "S") for log in state_transition_log]
    e_counts = [sum(1 for s in log["individual_states"].values() if s == "E") for log in state_transition_log]
    i_counts = [sum(1 for s in log["individual_states"].values() if s == "I") for log in state_transition_log]
    r_counts = [sum(1 for s in log["individual_states"].values() if s == "R") for log in state_transition_log]
    total_population = len(graph.nodes)
    final_outbreak_size = total_population - s_counts[-1]
    peak_infected_count = max(i_counts)
    peak_infecting_time = i_counts.index(peak_infected_count)
    recovery_time = None
    for t, (e, i) in enumerate(zip(e_counts, i_counts)):
        if e == 0 and i == 0:
            recovery_time = t
            break
    if recovery_time is None:
        recovery_time = len(state_transition_log)
    metrics = {
        "final_outbreak_size": final_outbreak_size,
        "peak_infecting_time": peak_infecting_time,
        "peak_infected_count": peak_infected_count,
        "recovery_time": recovery_time,
        "total_pcr_transmission": cumulative_pcr_transmission,
        "total_pcr_quarantine": cumulative_pcr_quarantine,
        "total_vaccinations": total_vaccinations,
    }
    return metrics

# ==========================
# Experiment Blocks
# ==========================

# -----------------------------
# Run each simulation 10 times and aggregate results
# -----------------------------
num_runs = 50

# Create lists to collect results for each block
block1_results = []  # For default parameters per resolution & intervention combination
block2_results = []  # For varying PCR parameters (PCR only)
block3_results = []  # For varying Vaccination parameters (PCR+Vaccination)
block4_results = []

block1_all_runs = [] 
block2_all_runs = [] 
block3_all_runs = [] 
block4_all_runs = [] 

# -----------------------------
# Block 1: Default parameters for each resolution and intervention combination
# -----------------------------
print("=== Block 1: Default Parameters ===")
resolutions = ["close", "close and casual", "all contact"]
interventions = {
    "no intervention": {"pcr_active": False, "quarantine_active": False, "vaccination_active": False,"strategy":"mass"},
    "PCR+ring Vaccination": {"pcr_active": True, "quarantine_active": False, "vaccination_active": False,"strategy":"ring"},
    "PCR+Quarantine": {"pcr_active": True, "quarantine_active": True, "vaccination_active": False,"strategy":"mass"},
    "PCR+mass Vaccination": {"pcr_active": True, "quarantine_active": False, "vaccination_active": True,"strategy":"mass"},
}
for res in resolutions:
    for intv_name, flags in interventions.items():
        run_metrics = []
        for run in range(num_runs):
            metrics = simulate_epidemic(
                resolution=res,
                pcr_active=flags["pcr_active"],
                quarantine_active=flags["quarantine_active"],
                vaccination_active=flags["vaccination_active"],
                sensitivity=0.95,
                specificity=0.98,
                vaccination_coverage=0.6,
                vaccination_efficacy=0.9,
                r_reduction=0.5,
                time_steps=600,
                strategy=flags["strategy"],
                pcr_testing_frequency=1,
                pcr_testing_percentage = 0.8,
                quarantine_testing_method=False,
                quarantine_testing_frequency=1
            )
            if metrics is not None:
                run_metrics.append(metrics)
                                # Store the individual run data for box plot
                block1_all_runs.append({
                    "Resolution": res,
                    "Intervention": intv_name,
                    "Final Outbreak Size": metrics["final_outbreak_size"],
                    "Peak Infecting Time": metrics["peak_infecting_time"],
                    "Peak Infected Count": metrics["peak_infected_count"],
                    "Recovery Time": metrics["recovery_time"]
                })
        if run_metrics:
            # Calculate mean for each metric over the runs
            avg_final_outbreak = np.mean([m["final_outbreak_size"] for m in run_metrics])
            avg_peak_time = np.mean([m["peak_infecting_time"] for m in run_metrics])
            avg_peak_count = np.mean([m["peak_infected_count"] for m in run_metrics])
            avg_recovery_time = np.mean([m["recovery_time"] for m in run_metrics])
            avg_vaccinations = np.mean([m["total_vaccinations"] for m in run_metrics])
            block1_results.append({
                "Resolution": res,
                "Intervention": intv_name,
                "Mean Final Outbreak Size": avg_final_outbreak,
                "Mean Peak Infecting Time": avg_peak_time,
                "Mean Peak Infected Count": avg_peak_count,
                "Mean Recovery Time": avg_recovery_time,
                "Mean Total Vaccinations": avg_vaccinations
            })
            print(f"Resolution: {res}, Intervention: {intv_name}")
            print({
                "final_outbreak_size": avg_final_outbreak,
                "peak_infecting_time": avg_peak_time,
                "peak_infected_count": avg_peak_count,
                "recovery_time": avg_recovery_time,
                "total_vaccinations": avg_vaccinations
            })
block1_df_all_runs = pd.DataFrame(block1_all_runs)
import matplotlib.pyplot as plt
import seaborn as sns

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
palette = "Set2"

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Final Outbreak Size
sns.boxplot(x="Resolution", y="Final Outbreak Size", hue="Intervention",
            data=block1_df_all_runs, palette=palette, dodge=True, ax=axes[0,0])

axes[0,0].set_title("Final Outbreak Size")
axes[0,0].legend_.remove()  # Remove duplicate legend

# 2. Peak Infecting Time
sns.boxplot(x="Resolution", y="Peak Infecting Time", hue="Intervention",
            data=block1_df_all_runs, palette=palette, dodge=True, ax=axes[0,1])

axes[0,1].set_title("Peak Infecting Time")
axes[0,1].legend_.remove()

# 3. Peak Infected Count
sns.boxplot(x="Resolution", y="Peak Infected Count", hue="Intervention",
            data=block1_df_all_runs, palette=palette, dodge=True, ax=axes[1,0])

axes[1,0].set_title("Peak Infected Count")
axes[1,0].legend_.remove()

# 4. Recovery Time (Epidemic Period)
sns.boxplot(x="Resolution", y="Recovery Time", hue="Intervention",
            data=block1_df_all_runs, palette=palette, dodge=True, ax=axes[1,1])

axes[1,1].set_title("Epidemic Period")
axes[1,1].legend_.remove()

# Add a single legend for all subplots (using the handles from one of them)
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(labels))
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.show()
# Save the figure as an image file
plt.savefig("block1_boxplot.png", dpi=300)
plt.close(fig)
# -----------------------------
# Block 2: Vary PCR Parameters (PCR + Quarantine)
# -----------------------------

block1_df = pd.DataFrame(block1_results)









# Write the DataFrames to an Excel file with separate sheets
with pd.ExcelWriter("/Users/xingkun/Desktop/FYP/simulation_resultsTT(Baseline).xlsx", engine="xlsxwriter") as writer:
    block1_df.to_excel(writer, sheet_name="Default_Parameters", index=False)
    workbook  = writer.book

    # Insert Block 1 plot
    worksheet1 = writer.sheets["Default_Parameters"]
    worksheet1.insert_image('J2', "block1_boxplot.png")

    # Insert Block 2 plots (all 8 images) into "PCR_Parameters" sheet.
  

print("All graphs and data have been saved to simulation_resultsTT.xlsx")
