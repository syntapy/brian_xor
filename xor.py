from operator import itemgetter
import numpy as np
import brian2 as br
import time
import pudb
#import winpdb
import snn 
import train
import sys

"""
    This script simulates a basic set of feedforward spiking neurons (Izhikevich model).

    So far the ReSuMe algorithm has been implemented (it seems).

    TO DO:
        X Clean it up so that it is easier to tweak parameters
        X Test the firing time range of a neuron
        Add multiple layers
        Tweak parameters to make it solve the XOR problem efficiently
        Work this into a script which takes arguments that denote whether it should train the 
            weights and save them to a text file, or read the weights from a text file and 
            just run it.
"""

#objects = []
N = 1

vt = -15 * br.mV
vr = -74 * br.mV

A=0.02 / 1000
B=0.2 / 1000
C=-65.0 / 1000
D=6.0 / 1000
TAU=15.0 / 1000
bench='xor'
levels=4

N_in = 2
N_liquid = [3, 3, 14]
N_hidden = 4
N_out = 1

Pc = 0.005


'''     0 - Calculates number of filters    '''
if bench == 'mnist':
    levels = levels
    n = 32  # Dimension of largest image in pyramid
    img_dims = (n, n)
    N = 0  # Number of filters
    dir = 'minst/'
    data_dir = 'on_pyr/'

    for h in range(levels):
        N += ((n / 2) - 2)**2
        n /= 2

elif bench == 'LI':
    levels = 1
    n = 3   # Dimension of single image (no pyramid)
    img_dims = (n, n)
    N = 1
    dir = 'li-data/'
    data_dir = 'noise/'

elif bench == 'xor':
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
a = A/br.second
b = B
c = C*br.volt
d = D*br.volt
tau = TAU*br.second
bench = bench

eqs_hidden_neurons = '''
    dv/dt = (0.04/ms/mV)*v**2 + (5/ms) * v + 140*mV/ms - u + ge/ms + I/ms : volt
    du/dt = A*((B*v) - u) : volt/second
    dge/dt = -ge/tau : volt
    I: volt
'''

v0 = (br.volt*br.sqrt((((5 - B) / 0.08)**2-140) / 0.04) - br.volt*((5 - B) / 0.08)) / 1000.0
u0 = B*v0

reset_t = '''
    v = C
    u += D
    ge = ge / 2.0
'''

thresh='v>vt'

img = np.empty(img_dims)

count = 0
g = 2

input_neurons = br.SpikeGeneratorGroup(N=N_in+1, indices=np.array([]), times=np.array([])*br.ms)
liquid_neurons = br.NeuronGroup(N=N_liquid[-1], \
        model=eqs_hidden_neurons, \
        threshold=thresh, \
        refractory=2*br.ms, \
        reset=reset_t)
pudb.set_trace()
liquid_in = liquid_neurons.subgroup(N_liquid[0])
liquid_hidden = liquid_neurons.subgroup(N_liquid[-1] - N_liquid[0] - N_liquid[1])
liquid_out = liquid_neurons.subgroup(N_liquid[1])

hidden_neurons = br.NeuronGroup(N_hidden, model=eqs_hidden_neurons, threshold=vt, \
        refractory=2*br.ms, reset=reset_t)
output_neurons = br.NeuronGroup(N_out, model=eqs_hidden_neurons, threshold=vt, \
        refractory=2*br.ms, reset=reset_t)

#objects.append(hidden_neurons)
Si = br.Synapses(input_neurons, liquid_in, model='w:1', pre='ge+=w')#, max_delay=9*br.ms)
Sl = br.Synapses(liquid_neurons, liquid_neurons, model='w:1', pre='ge+=w')#, max_delay=9*br.ms)

Sa = br.Synapses(liquid_out, hidden_neurons, model='w:1', pre='ge+=w')#, max_delay=9*br.ms)
Sb = br.Synapses(hidden_neurons, output_neurons, model='w:1', pre='ge+=w*(2)')

v0, u0 = -70*br.mV, -14*br.mV/br.msecond

liquid_neurons.v = v0
liquid_neurons.u = u0
liquid_neurons.I = 0

hidden_neurons.v = v0
hidden_neurons.u = u0
hidden_neurons.I = 0

output_neurons.v = v0
output_neurons.u = u0
output_neurons.I = 0


print "v0 = ", v0
print "u0 = ", u0

Si[:,:]=True
Sl[:,:]=True
Sa[:,:]=True
Sb[:,:]=True

Si.w[:]='6.04*(0.4+0.4*rand())*br.mV'
Sl.w[:]='6.04*(0.3+0.2*rand())*br.mV'
Sa.w[:]='2.04*(0.5+0.2*rand())*br.mV'
Sb.w[:]='0.25*(0.0+0.2*rand())*br.mV'

Si.delay='(3*rand())*ms'
Sl.delay='(3*rand())*ms'
Sa.delay='(1)*ms'
Sb.delay='(1)*ms'
print "n.v0, n.u0 = ", hidden_neurons.v, ", ", hidden_neurons.u

M =br.StateMonitor(output_neurons,'ge',record=0)
Mv=br.StateMonitor(output_neurons,'v',record=0)
Mu=br.StateMonitor(output_neurons,'u',record=0)

N =br.StateMonitor(hidden_neurons,'ge',record=0)
Nv=br.StateMonitor(hidden_neurons,'v',record=0)
Nu=br.StateMonitor(hidden_neurons,'u',record=0)

S_in = br.SpikeMonitor(input_neurons, record=True)
S_hidden = br.SpikeMonitor(hidden_neurons, record=True)
S_out = br.SpikeMonitor(output_neurons, record=True)

#objects.append(M)
#objects.append(Mv)
#objects.append(Mu)

'''         TRAINING        '''
#Net = br.Network(objects)
#OUT = open('weights.txt', 'a')

number = 3
T = 20
N_o = 1
N_h = 1

print "======================================================================"
print "\t\t\tSetting number of spikes"
print "======================================================================"

"""
while True:
    snn.Run(2*T, v0, u0, bench, number, input_neurons, liquid_in, \
            liquid_hidden, liquid_out, liquid_neurons, \
            hidden_neurons, output_neurons, \
            Si, Sl, Sa, Sb, M, Mv, Mu, \
            S_in, S_hidden, S_out, train=True, letter=None)

    snn.Plot(Nv, 0)
"""
    

snn.SetNumSpikes(0, T, N_h, N_o, v0, u0, bench, number, input_neurons, \
        liquid_in, liquid_hidden, liquid_out, liquid_neurons, \
        hidden_neurons, output_neurons, \
        Si, Sl, Sa, Sb, M, Mv, Mu, \
        S_in, S_hidden, S_out, train=False, letter=None)

snn.PrintSpikes(T, N_h, N_o, v0, u0, bench, number, input_neurons, \
        liquid_in, liquid_hidden, liquid_out, liquid_neurons, \
        hidden_neurons, output_neurons, \
        Si, Sl, Sa, Sb, M, Mv, Mu, \
        S_in, S_hidden, S_out, train=False, letter=None)

if bench == 'xor':
    desired_times = [-1, -1]
    mid_times = [-1, -1]
    extreme_spikes = train.TestNodeRange(T, N, v0, u0, bench, number, input_neurons, \
            liquid_in, liquid_hidden, liquid_out, liquid_neurons, \
            hidden_neurons, output_neurons, \
            Si, Sl, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out)
    diff = extreme_spikes[1] + extreme_spikes[0]
    diff_r = diff / 8.0

    #pudb.set_trace()
    extreme_spikes[0] = extreme_spikes[0] + diff_r
    extreme_spikes[1] = extreme_spikes[1] - diff_r

    mid_times[0] = (extreme_spikes[0] + extreme_spikes[1]) / 2.0
    mid_times[1] = (extreme_spikes[0] + extreme_spikes[1]) / 2.0

    mid_times[0] = mid_times[0]*br.second
    mid_times[1] = mid_times[1]*br.second

    desired_times[0] = extreme_spikes[0]*br.second
    desired_times[1] = extreme_spikes[1]*br.second

else:
    pudb.set_trace()

snn.SetNumSpikes(1, T, N_h, N_o, v0, u0, bench, number, input_neurons, \
        liquid_in, liquid_hidden, liquid_out, liquid_neurons, \
        hidden_neurons, output_neurons, \
        Si, Sl, Sa, Sb, M, Mv, Mu, \
        S_in, S_hidden, S_out, train=False, letter=None)


for number in range(4):
    print "\tTRAINING: number = ", number
    #pudb.set_trace()
    train.ReSuMe(mid_times, Pc, T, N, v0, u0, bench, number, input_neurons, \
            liquid_in, liquid_hidden, liquid_out, liquid_neurons, \
            hidden_neurons, output_neurons, \
            Si, Sl, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out)

print "======================================================================"
print "\t\t\tTraining with ReSuMe"
print "======================================================================"

for number in range(4):
    print "\tTRAINING: number = ", number
    #pudb.set_trace()
    train.ReSuMe(desired_times, Pc, T, N, v0, u0, bench, number, input_neurons, \
            liquid_in, liquid_hidden, liquid_out, liquid_neurons, \
            hidden_neurons, output_neurons, \
            Si, Sl, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out)
    #pudb.set_trace()

print "======================================================================"
print "\t\t\tTesting"
print "======================================================================"

#pudb.set_trace()
for number in range(4):
    snn.Run(T, v0, u0, bench, number, \
            input_neurons, liquid_in, liquid_hidden, liquid_out, liquid_neurons, \
            hidden_neurons, output_neurons, \
            Si, Sl, Sa, Sb, M, Mv, Mu, \
            S_in, S_hidden, S_out, train=True, letter=None)

    if number < 2:
        desired = desired_times[0]
    else:
        desired = desired_times[1]

    print "Number, Desired, Actual = ", number, ", ", desired, ", ", S_out.spiketimes[0]
