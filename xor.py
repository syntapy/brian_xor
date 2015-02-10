from operator import itemgetter
import numpy as np
import brian as br
import time
import pudb
#import winpdb
import snn 
import train

"""
    This script simulates a basic set of feedforward spiking neurons (Izhikevich model).

    So far the ReSuMe algorithm has been implemented (it seems).

    TO DO:
        Clean it up so that it is easier to tweak parameters
        Test the firing time range of a neuron
        Tweak parameters to make it solve the XOR problem efficiently
        Work this into a script which takesarguments that denote whether it should train the weights and save them to a text file, or read the weights from a text file and just run it.
"""

#objects = []
N = 1

vt = -15 * br.mV
vr = -74 * br.mV

a=0.02
b=0.2
c=-65
d=6
tau=5
bench='xor'
levels=4

N_in = 2
N_hidden = 4
N_out = 1

Pc = 0.05


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

if bench == 'xor':
    dA = 10*br.ms
    dB = 16*br.ms
    in_out = {0:dB, 1:dA}


simtime = 1 #duration of the simulation in s
number = 1 #number of hidden_neurons
a = a/br.ms
b = b/br.ms
c = c * br.mvolt
d = d*br.mV/br.ms
tau = tau*br.ms
bench = bench

eqs_hidden_neurons = '''
    dv/dt = (0.04/ms/mV)*v**2 + (5/ms) * v + 140*mV/ms - u + ge/ms + I/ms : mvolt
    du/dt = a*((b*v) - u) : mvolt/msecond
    dge/dt = -ge/tau : mvolt
    I: mvolt
'''

B = b*br.ms

v0 = br.mV*br.sqrt((((5 - B) / 0.08)**2-140) / 0.04) - br.mV*((5 - B) / 0.08)
u0 = b*v0

reset = '''
    v = c
    u += d
'''

img = np.empty(img_dims)

count = 0
g = 2

spikes = []
input_neurons = br.SpikeGeneratorGroup(N_in+1, spikes)
hidden_neurons = br.NeuronGroup(N_hidden, model=eqs_hidden_neurons, threshold=vt, refractory=20*br.ms, reset=reset)
output_neurons = br.NeuronGroup(N_out, model=eqs_hidden_neurons, threshold=vt, refractory=20*br.ms, reset=reset)

#objects.append(hidden_neurons)
Sa = br.Synapses(input_neurons, hidden_neurons, model='w:1', pre='ge+=w')#, max_delay=9*br.ms)
Sb = br.Synapses(hidden_neurons, output_neurons, model='w:1', pre='ge+=w*(2)')

v0, u0 = -70*br.mV, -14*br.mV/br.msecond

hidden_neurons.v = v0
hidden_neurons.u = u0
hidden_neurons.I = 0

output_neurons.v = v0
output_neurons.u = u0
output_neurons.I = 0


print "v0 = ", v0
print "u0 = ", u0

Sa[:,:]=True
Sb[:,:]=True

Sa.w[:]='8.04*(0.7+0.2*rand())*br.mV'
Sb.w[:]='9.0*(0.1+0.2*rand())*br.mV'

Sa.delay='(4)*ms'
Sb.delay='(4)*ms'
print "n.v0, n.u0 = ", hidden_neurons.v, ", ", hidden_neurons.u

M =br.StateMonitor(output_neurons,'ge',record=0)
Mv=br.StateMonitor(output_neurons,'v',record=0)
Mu=br.StateMonitor(output_neurons,'u',record=0)

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
T = 30
N_o = 1
N_h = 1

print "======================================================================"
print "\t\t\tSetting number of spikes"
print "======================================================================"

for i in range(10):
    for number in range(3, -1, -1):
        snn.SetNumSpikes(T, N_h, N_o, v0, u0, bench, number, input_neurons, hidden_neurons, output_neurons, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=False, letter=None)
        print "\tDone! for number = ", number
    #winpdb.set_trace()

print "======================================================================"
print "\t\t\tTraining with ReSuMe"
print "======================================================================"

for number in range(4):
    print "\tTRAINING: number = ", number
    #if number == 3:
    #    pudb.set_trace()
    train.ReSuMe(in_out, Pc, T, N, v0, u0, bench, number, input_neurons, hidden_neurons, output_neurons, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out)

print "======================================================================"
print "\t\t\tTesting"
print "======================================================================"

#pudb.set_trace()
for number in range(4):
    snn.Run(T, v0, u0, bench, number, \
            input_neurons, hidden_neurons, output_neurons, \
            Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=True, letter=None)

    if number < 2:
        desired = 0.010
    else:
        desired = 0.016

    print "Number, Desired, Actual = ", number, ", ", desired, ", ", S_out.spiketimes
#
#    snn.Plot(Mv, number)
#
#br.show()
#for number in range(4):

#print "spike times: ", S_out.spikes

#br.plot(210)
"""
dw = 0.07
spikes_store = []
for i in range(40):
    snn.Run(40, v0, u0, bench, number, \
            input_neurons, hidden_neurons, output_neurons, \
            Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=True, letter=None)

    Sb.w[0] += dw

    spikes_store.append([])
    for j in range(len(S_out.spiketimes)):
        if len(S_out.spiketimes[j]) > 0:
            spikes_store[i].append(([i, j], S_out.spiketimes[j][0]))

for i in range(len(spikes_store)):
    for j in range(len(spikes_store[i])):
        print spikes_store[i][j], ", ",
    print "\n"
"""
