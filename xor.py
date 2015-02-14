from operator import itemgetter
import os.path as op
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
        X Clean it up so that it is easier to tweak parameters
        X Test the firing time range of a neuron
        X Add multiple layers
        Tweak parameters to make it solve the XOR problem efficiently
        X Work this into a script which takes arguments that denote whether it should train the weights and save them to a text file, or read the weights from a text file and just run it.
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

#if bench == 'xor':
#    dA = 10*br.ms
#    dB = 16*br.ms
#    desired_times = {0:dB, 1:dA}


simtime = 1 #duration of the simulation in s
number = 1 #number of hidden_neurons
a = A/br.ms
b = B/br.ms
c = C * br.mvolt
d = D*br.mV/br.ms
tau = tau*br.ms
bench = bench

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

B = b*br.ms

#pudb.set_trace()
#v0 = -72.28*br.mV#olt#br.mV*br.sqrt((((5 - B) / 0.08)**2-140) / 0.04) - br.mV*((5 - B) / 0.08)
#u0 = -13960*br.mV#olt#b*v0
u0 = (25*(-5*A*B + A**2 * B**2)) * br.mV
v0 = (25*(-5 + A**2 * B**2)) * br.mV

reset = '''
    v = c
    u += d
'''

img = np.empty(img_dims)

count = 0
g = 2

spikes = []
N_hidden = [3, 3]
hidden_neurons = []# * len(N_hidden)
input_neurons = br.SpikeGeneratorGroup(N_in+1, spikes)

for i in range(len(N_hidden)):
    hidden_neurons.append(br.NeuronGroup(N_hidden[i], model=eqs_hidden_neurons, threshold=vt, refractory=2*br.ms, reset=reset))

output_neurons = br.NeuronGroup(N_out, model=eqs_hidden_neurons, threshold=vt, refractory=2*br.ms, reset=reset)

#pudb.set_trace()
Sa = []
Sa.append(br.Synapses(input_neurons, hidden_neurons[0], model='w:1', pre='ge+=w'))
for i in range(len(N_hidden) - 1):
    Sa.append(br.Synapses(hidden_neurons[i], hidden_neurons[i+1], model='w:1', pre='ge+=w'))

Sb = br.Synapses(hidden_neurons[-1], output_neurons, model='w:1', pre='ge+=w')

v0, u0 = -70*br.mV, -14*br.mV/br.msecond
for i in range(len(hidden_neurons)):
    hidden_neurons[i].v = v0
    hidden_neurons[i].u = u0
    hidden_neurons[i].I = 0
    hidden_neurons[i].ge = 0

output_neurons.v = v0
output_neurons.u = u0
output_neurons.I = 0
output_neurons.ge = 0

print "v0 = ", v0
print "u0 = ", u0

for i in range(len(Sa)):
    Sa[i][:,:]=True
    Sa[i].w[:]='8.04*(0.7+0.2*rand())*br.mV'
    Sa[i].delay='(4)*ms'

Sb[:,:]=True
Sb.w[:]='9.0*(0.1+0.2*rand())*br.mV'
Sb.delay='(4)*ms'
#print "n.v0, n.u0 = ", hidden_neurons.v, ", ", hidden_neurons.u

M =br.StateMonitor(output_neurons,'ge',record=0)
Mv=br.StateMonitor(output_neurons,'v',record=0)
Mu=br.StateMonitor(output_neurons,'u',record=0)

N =br.StateMonitor(hidden_neurons[1],'ge',record=0)
Nv=br.StateMonitor(hidden_neurons[1],'v',record=0)
Nu=br.StateMonitor(hidden_neurons[1],'u',record=0)

S_in = br.SpikeMonitor(input_neurons, record=True)
S_hidden = []

#pudb.set_trace()
for i in range(len(hidden_neurons)):
    S_hidden.append(br.SpikeMonitor(hidden_neurons[i], record=True))

S_out = br.SpikeMonitor(output_neurons, record=True)

objects.append(input_neurons)
objects.append(output_neurons)
for i in range(len(hidden_neurons)):
    objects.append(hidden_neurons[i])

objects.append(S_in)
objects.append(S_out)
for i in range(len(hidden_neurons)):
    objects.append(S_hidden[i])

for i in range(len(N_hidden)):
    objects.append(Sa[i])

objects.append(Sb)

objects.append(M)
objects.append(Mv)
objects.append(Mu)

#pudb.set_trace()
net = br.Network(objects)

'''         TRAINING        '''
#Net = br.Network(objects)
#OUT = open('weights.txt', 'a')

number = 3
T = 40
N_o = 1
N_h = 1

print "======================================================================"
print "\t\t\tSetting number of spikes"
print "======================================================================"


if op.isfile(weight_file):
    #pudb.set_trace()
    Sa, Sb = snn.ReadWeights(Sa, Sb, weight_file)

else:
    for i in range(10):
        for number in range(3, -1, -1):
            #if i == 9 and number == 0:
            #    pudb.set_trace()
            snn.SetNumSpikes(T, N_h, N_o, v0, u0, bench, number, input_neurons, hidden_neurons, output_neurons, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=False, letter=None)
            print "\tDone! for number = ", number

    snn.SaveWeights(Sa, Sb, "weights.txt")

#pudb.set_trace()
#Sa[0].w[:] = '0*br.mV'
snn.Run(T, v0, u0, bench, 0, input_neurons, hidden_neurons, output_neurons, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out)
#pudb.set_trace()
#snn.Plot(N, Nu, Nv, 1)
#snn.Plot(M, Mu, Mv, 1)

print "======================================================================"
print "\t\t\tTraining with ReSuMe"
print "======================================================================"

if bench == 'xor':
    desired_times = [-1, -1]
    extreme_spikes = train.TestNodeRange(T, N, v0, u0, bench, number, input_neurons, hidden_neurons, output_neurons, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out)
    diff = extreme_spikes[1] + extreme_spikes[0]
    diff_r = diff / 10

    extreme_spikes[0] = extreme_spikes[0] + diff_r
    extreme_spikes[1] = extreme_spikes[1] + diff_r

    desired_times[0] = extreme_spikes[0]*br.second
    desired_times[1] = extreme_spikes[1]*br.second

else:
    pudb.set_trace()

for number in range(4):
    print "\tTRAINING: number = ", number
    train.ReSuMe(desired_times, Pc, T, N, v0, u0, bench, number, input_neurons, hidden_neurons, output_neurons, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out)

print "======================================================================"
print "\t\t\tTesting"
print "======================================================================"

#pudb.set_trace()
for number in range(4):
    snn.Run(T, v0, u0, bench, number, \
            input_neurons, hidden_neurons, output_neurons, \
            Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=True, letter=None)

    if number < 2:
        desired = desired_times[0]
    else:
        desired = desired_times[1]

    print "Number, Desired, Actual = ", number, ", ", desired, ", ", S_out.spiketimes[0]
