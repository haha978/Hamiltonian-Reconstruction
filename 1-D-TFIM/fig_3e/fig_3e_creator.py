import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

run_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "heatmaps_11q_TFIM"))

saved_dir = os.path.join(run_dir, 'hr_p1p2_plot_n_qbts_11_J_0.5_depolarization_only')
n_qbts = 11
points_per_axis = 150

## P2 Values
y_axis_start = 10**-4
y_axis_end = .2

## P1 Values
x_axis_start = 10**-6
x_axis_end = 10**-1

def round_to_n(x, n):
    return round(x, -int(np.floor(np.sign(x) * np.log10(abs(x)))) + n)

### Axes
y_axis = np.zeros(points_per_axis)
y_axis[0] = y_axis_start
y_axis[-1] = y_axis_end
x_axis = np.zeros(points_per_axis)
x_axis[0] = x_axis_start
x_axis[-1] = x_axis_end

### Making Log Log Grid
for i in range(1, points_per_axis-1):

    x_axis[i] = 10**(np.log10(x_axis[i-1]) + (np.log10(x_axis_end) - np.log10(x_axis_start))/(points_per_axis-1))
    y_axis[i] = 10**(np.log10(y_axis[i-1]) + (np.log10(y_axis_end) - np.log10(y_axis_start))/(points_per_axis-1))

### Order of Data Points for collection and plotting
p1_vals = [round_to_n(i,8) for i in x_axis for j in y_axis]
p2_vals = [round_to_n(j,8) for i in x_axis for j in y_axis]
# p1_vals = [i for i in x_axis for j in y_axis]
# p2_vals = [j for i in x_axis for j in y_axis]

os.chdir(saved_dir)

HR_distance_p1_p2 = np.zeros(points_per_axis)
energy_p1_p2 = np.zeros(points_per_axis)
fidelity_p1_p2 = np.zeros(points_per_axis)

HR_distance_p1_p2 = pickle.load(open("HR_distance_p1_p2",'rb'))

energy_p1_p2 = pickle.load(open("energy_p1_p2",'rb'))

fidelity_p1_p2 = pickle.load(open("fidelity_p1_p2",'rb'))
    
print(len(energy_p1_p2),round_to_n(0.010637648543162674,8))
graph_energy = []
graph_p2 = []
graph_HR = []
graph_fidelity = []
for i in range(len(p1_vals)):
    if round_to_n(0.010637648543162674,8) == p1_vals[i]:
        graph_energy.append(energy_p1_p2[i])
        graph_p2.append(p2_vals[i])
        graph_HR.append(HR_distance_p1_p2[i])
        graph_fidelity.append(100*((fidelity_p1_p2[i]/100)**2))
        #graph_fidelity.append(fidelity_p1_p2[i])

os.chdir(run_dir + "/plots_p1p2_folder")

new_rc_params = {
"font.family": 'serif',
"font.size": 12,
"font.serif": ['Times New Roman'] + plt.rcParams['font.serif'],
"svg.fonttype": 'none',
'text.usetex': False,
"svg.fonttype": 'none'}
plt.rcParams.update(new_rc_params)

quantum_names = ["IonQ Harmony"]
quantum_values = [1-96.02]

fig = plt.figure(figsize=(5,5))
ax = plt.axes()

plt.axvspan(.01, .1, facecolor='lightgreen', alpha=0.5, zorder=-100, label = "NISQ Range")
ax.scatter(graph_p2, graph_energy, c = 'cornflowerblue', alpha = 0.8, marker = ".", label = "Energy", zorder = 1)
ax.set_xlabel('Two Qubit Gate Error Rate',fontsize = 12)
ax.set_ylabel("Energy",fontsize = 12)
ax.set_xscale("log")
#ax.set_ylim([0, 100])

ax.invert_xaxis()
title = "11 Qubit Energy and HR Distance versus "+ '\n' + "Two Qubit Error Rate for 99% P1 Value"
plt.title(title,fontsize = 12)

ax1_2 = ax.twinx()
ax1_2.scatter(graph_p2, graph_HR, c = 'salmon', marker=".", label = "HR distance",zorder = 1)
ax1_2.set_ylabel("HR distance",fontsize = 12)
ax1_2.set_ylim([0, 1])
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
plt.savefig("HR_Energy_vs_P2_" + str(n_qbts)+ "qubits.svg",dpi = 300, bbox_inches='tight')

fig2 = plt.figure(figsize=(5,5))
ax2 = plt.axes()

plt.axvspan(.01, .1, facecolor='lightgreen', alpha=0.5, zorder=-100, label = "NISQ Range")
ax2.scatter(graph_p2, graph_fidelity, c = 'cornflowerblue', alpha = 0.8, marker = ".", label = "Fidelity")
ax2.set_xlabel('Two Qubit Gate Error Rate',fontsize = 12)
ax2.set_ylabel("Fidelity",fontsize = 12)
ax2.set_xscale("log")
ax2.set_yticks([0,25,50,75, 100])
ax2.set_yticklabels(["0%","25%", "50%","75%", "100%"])
ax2.invert_xaxis()
title = "11 Qubit Fidelity and HR Distance versus "+ '\n' + "Two Qubit Error Rate for 99% P1 Value"
plt.title(title,fontsize = 12)
ax2_2 = ax2.twinx()
ax2_2.scatter(graph_p2, graph_HR, c = 'salmon', marker=".", label = "HR distance")
ax2_2.set_ylabel("HR distance",fontsize = 12)
ax2_2.set_ylim([0, 1])
fig2.legend(loc='center right', bbox_to_anchor=(1,.7), bbox_transform=ax2.transAxes)
plt.savefig("HR_Fidelity_vs_P2_" + str(n_qbts)+ "qubits.svg",dpi = 300, bbox_inches='tight')
