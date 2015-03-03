from operator import itemgetter
import os.path as op
import numpy as np
import brian as br
import time
import pudb
import snn 
import train
import initial as init

"""
    This script simulates a basic set of feedforward spiking neurons (Izhikevich model).

    So far the ReSuMe algorithm has been implemented (it seems).

    TO DO:
        X Clean it up so that it is easier to tweak parameters
        X Test the firing time range of a neuron
        X Add multiple layers
        X Tweak parameters to make it solve the XOR problem efficiently
        X Work this into a script which takes arguments that denote whether it should train the weights and save them to a text file, or read the weights from a text file and just run it.
        Add liquid state machine layer
        Make SetNumSpikes(args...) more efficient, and have the weights made to no be on the border of a change in number of spikes
        Make SetNumSpikes(args...) better at fine tuning the weights, esp. when there are very large numbers of hidden layer neurons
"""

weight_file = "weights.txt"

objects = []
N = 1

vt = -15 * br.mV
vr = -74 * br.mV

A=0.02
B=0.2
C=-65
D=6
tau=5
bench='xor'
levels=4

N_in = 2
N_liquid = [14, 3, 3] # Total, liquid in, liquid out
#CP_liquid = 0.7
N_hidden = [5]
N_out = 1

#file_array = ["Si", "Sl", "Sa", "Sb"]
#synapes_array = []
Pc = 0.05

'''     0 - Calculates number of filters    '''
#if bench == 'mnist':
#    levels = levels
#    n = 32  # Dimension of largest image in pyramid
#    img_dims = (n, n)
#    N = 0  # Number of filters
#    dir = 'minst/'
#    data_dir = 'on_pyr/'
#
#    for h in range(levels):
#        N += ((n / 2) - 2)**2
#        n /= 2
#
#elif bench == 'LI':
#    levels = 1
#    n = 3   # Dimension of single image (no pyramid)
#    img_dims = (n, n)
#    N = 1
#    dir = 'li-data/'
#    data_dir = 'noise/'
#
if bench == 'xor':
    levels = 1
    img_dims = (1, 2)
    N = 1
    #dir = 'li-data/'
    #data_dir = 'noise/'

#if bench == 'xor':
#    dA = 10*br.ms
#    dB = 16*br.ms
#    desired_times = {0:dB, 1:dA}


simtime = 1 #duration of the simulation in s
number = 1 #number of hidden_neurons
a = A/br.ms
b = B/br.ms
c = C*br.mvolt
d = D*br.mV/br.ms
tau = tau*br.ms
bench = bench

parameters = [a, b, c, d, tau]

eqs_hidden_neurons = '''
    dv/dt = (0.04/ms/mV)*v**2 + (5/ms) * v + 140*mV/ms - u + ge/ms + I/ms : mvolt
    du/dt = a*((b*v) - u) : mvolt/msecond
    dge/dt = -ge/tau : mvolt
    I: mvolt
'''

"""     USING MATHEMATICA

    u = 25. (-5 a b + A B)
    v = 25. (-5. + a b)
"""

reset = '''
    v = c
    u += d
'''

#B = b*br.ms

#pudb.set_trace()
u0 = (25*(-5*A*B + A**2 * B**2)) * br.mV
v0 = (25*(-5 + A**2 * B**2)) * br.mV
I0 = 0*br.mV
ge0 = 0*br.mV

img = np.empty(img_dims)

count = 0
g = 2

T = 40
N_h = 1
N_o = 1
# DEFINE OBJECTS
# pudb.set_trace()
neuron_groups = init.SetNeuronGroups(N_in, N_liquid, N_hidden, N_out, vt, \
        parameters, eqs_hidden_neurons, reset)
synapse_groups = init.SetSynapses(neuron_groups)
output_monitor = init.StateMonitors(neuron_groups, 'out')
spike_monitors = init.AllSpikeMonitors(neuron_groups)
net = init.AddNetwork(neuron_groups, synapse_groups, output_monitor, spike_monitors)

snn.Run(T, net, v0, u0, I0, ge0, bench, 0,\
        neuron_groups, synapse_groups, output_monitor, spike_monitors)

snn.SetNumSpikes(T, N_h, N_o, v0, u0, I0, ge0, bench, number, \
        neuron_groups, synapse_groups, output_monitor, spike_monitors)

# LIQUID STATE MACHINE

#pudb.set_trace()
#for i in range(len(hidden_neurons)):
#    S_hidden.append(br.SpikeMonitor(hidden_neurons[i], record=True))
#
#S_out = br.SpikeMonitor(output_neurons, record=True)

#objects.append(input_neurons)
#objects.append(output_neurons)
#for i in range(len(hidden_neurons)):
#    objects.append(hidden_neurons[i])
#
#objects.append(S_in)
#objects.append(S_out)
#for i in range(len(hidden_neurons)):
#    objects.append(S_hidden[i])
#
#for i in range(len(N_hidden)):
#    objects.append(Sa[i])
#
#objects.append(Sb)
#
#objects.append(M)
#objects.append(Mv)
#objects.append(Mu)

#net = br.Network(objects)
#pudb.set_trace()

'''         TRAINING        '''
#Net = br.Network(objects)
#OUT = open('weights.txt', 'a')

#number = 3
#N_o = 1
#N_h = 1
#for i in range(10):
#    for number in range(3, -1, -1):
#        snn.SetNumSpikes(T, N_h, N_o, v0, u0, bench, number, \
#            input_neurons, liquid_neurons, hidden_neurons, output_neurons, \
#            Si, Sl, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=False, letter=None)
#        print "\tDone! for number = ", number


"""
print "======================================================================"
print "\t\t\tSetting number of spikes"
print "======================================================================"

pudb.set_trace()
if op.isfile(weight_file):
    #pudb.set_trace()
    Si, Sl, Sa, Sb = snn.ReadWeights(Si, Sl, Sa, Sb, weight_file)

else:
    snn.SaveWeights(Si, Sl, Sa, Sb, "weights.txt")

#pudb.set_trace()
#Sa[0].w[:] = '0*br.mV'
snn.Run(T, v0, u0, bench, 0, input_neurons, hidden_neurons, output_neurons, Si, Sl, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out)
#pudb.set_trace()
#snn.Plot(N, Nu, Nv, 1)
#snn.Plot(M, Mu, Mv, 1)

print "======================================================================"
print "\t\t\tTraining with ReSuMe"
print "======================================================================"

if bench == 'xor':
    if op.isfile("times.txt"):
        desired_times = train.ReadTimes("times.txt")
    else:

        desired_times = [-1, -1]
        extreme_spikes = train.TestNodeRange(T, N, v0, u0, bench, number, input_neurons, hidden_neurons, output_neurons, Sa, Sl, Sb, M, Mv, Mu, S_in, S_hidden, S_out)
        diff = extreme_spikes[1] + extreme_spikes[0]
        diff_r = diff / 10

        extreme_spikes[0] = extreme_spikes[0] + diff_r
        extreme_spikes[1] = extreme_spikes[1] + diff_r

        desired_times[0] = extreme_spikes[0]*br.second
        desired_times[1] = extreme_spikes[1]*br.second

        f = open("times.txt", 'w')
        f.write(str(float(desired_times[0])))
        f.write("\n")
        f.write(str(float(desired_times[1])))
        f.write("\n")

else:
    pudb.set_trace()

for number in range(4):
    print "\tTRAINING: number = ", number
    train.ReSuMe(desired_times, Pc, T, N, v0, u0, bench, number, input_neurons, hidden_neurons, output_neurons, Sa, Sl, Sb, M, Mv, Mu, S_in, S_hidden, S_out)

print "======================================================================"
print "\t\t\tTesting"
print "======================================================================"

#pudb.set_trace()
for number in range(4):
    snn.Run(T, v0, u0, bench, number, \
            input_neurons, hidden_neurons, output_neurons, \
            Sa, Sl, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=True, letter=None)

    if number < 2:
        desired = desired_times[0]
    else:
        desired = desired_times[1]

    print "Number, Desired, Actual = ", number, ", ", desired, ", ", S_out.spiketimes[0]
"""
