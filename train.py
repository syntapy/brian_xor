import brian as br
import numpy as np
import math as ma
import initial as init
import pudb
import snn

def DesiredOut(label, bench):
    return_val = None

    if bench == 'xor':
        if label == 1:
            return_val = 1*br.ms
        else:
            return_val = 7*br.ms

    return return_val

def WeightChange(s):
    A = 10**-1
    tau = 2.0*br.ms
    return A*ma.exp(-s / tau)

def L(t):
    tau = 5.0*br.ms
    if t > 0:
        return ma.exp(float(-t / tau))
    else:
        return 0

def P_Index(S_l, S_d):
    return_val = 0

    return_val += abs(L(S_d) - L(S_l[0][0]*br.second))

    return return_val

def ReadTimes(filename):
    f = open(filename, 'r')

    #pudb.set_trace()
    lines = f.readlines()
    f.close()

    desired_times = [-1] * len(lines)

    for i in range(len(lines)):
        tmp = float(lines[i][:-1])
        desired_times[i] = tmp * br.second

    return desired_times


def ReSuMe(desired, number, net, N_liquid, N_hidden, T, N_h, N_o, v0, u0, I0, ge0, \
        neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters):

    img, label = snn.ReadImg(number=number)
    N_hidden_last = len(net[neuron_names[-2][-1]])
    N_out = len(net[neuron_names[-1]])

    N_h = 1
    N_o = 1

    trained = False

    while trained == False:
        trained = True
        for i in range(N_hidden_last):

            #pudb.set_trace()
            #print "\t\ti = ", i
            net = snn.Run(net, T, v0, u0, I0, ge0, \
                        neuron_names, synapse_names, state_monitor_names, \
                        spike_monitor_names, parameters, number)

            #print "Hidden Times: ", 
            #for j in range(len(S_hidden)):
            #    print S_hidden[j].spiketimes, " ", 

            #print "\nOutput Times: ", S_out.spiketimes
            right_spike_numbers_hidden = init.check_number_spikes(net, 0, \
                        T, N_h, N_o, v0, u0, I0, ge0, \
                        neuron_names, spike_monitor_names)

            if right_spike_numbers_hidden == False:
                print "ERROR!! WRONG NUMBER OF SPIKES!! Resetting No. Spikes!!!"
                init.set_number_spikes(net, 0, T, N_h, N_o, v0, u0, I0, ge0, \
                        neuron_names, synapse_names, state_monitor_names, spike_monitor_names, \
                        parameters)

            right_spike_numbers_out = init.check_number_spikes(net, 1, \
                        T, N_h, N_o, v0, u0, I0, ge0, \
                        neuron_names, spike_monitor_names)

            if right_spike_numbers_out == False:
                print "ERROR!! WRONG NUMBER OF SPIKES!! Resetting No. Spikes!!!"
                init.set_number_spikes(net, 1, T, N_h, N_o, v0, u0, I0, ge0, \
                        neuron_names, synapse_names, state_monitor_names, spike_monitor_names, \
                        parameters)

            #pudb.set_trace()
            indices_l, spikes_l = net[spike_monitor_names[-1]]
            indices_i, spikes_i = net[spike_monitor_names[-2][-1]]
            S_l = init.collect_spikes(indices_l, spikes_l, 1)
            S_i = init.collect_spikes(indices_i, spikes_i, N_hidden[-1])
            #S_l = S_out.spiketimes
            S_d = desired_time

            P = P_Index(S_l, S_d)
            print "\t\t\tP = ", P
            if P < Pc:
                trained = True
                break

            print "i = ", i
            #if i == 2:
            #    pudb.set_trace()
            sd = max(0, float(S_d) - S_i[i][0])
            sl = max(0, S_l[0][0] - S_i[i][0])
            Wd = WeightChange(sd)
            Wl = -WeightChange(sl)
            net.restore()
            net[synapse_names[3]].w[i] = net[synapse_names[3]].w[i] + Wd + Wl
            net.store()

def SpikeSlopes(Mv, S_out, d_i=3):
    
    """
        Returns a list of values that indicate the difference in voltage 
        between each spike's starting threshold voltage and the voltage 
        d_i time steps before it

        NOTE: This assumes that the brian equation solver uses a constant time step
        throught computation.
    """

    N = len(S_out.spikes)
    dt = Mv.times[1] - Mv.times[0]
    v_diffs = []
    i_diffs = []

    for i in range(N):
        time = S_out.spikes[i]
        index_a = time / dt
        index_b = index_a - d_i 

        v_diffs.append(Mv.values[index_a] - Mv.values[index_b])
        i_diffs.append(index_a - index_b)

    return v_diffs, dt

def PickWeightIndexA(Sa, S_hidden, S_out):
    pass

def PickWeightIndicesB(Mv, Sb, S_hidden, S_out, d_i=3):

    """
        Depending on the delays of the synapses, and the spike times
        in the hidden layer and output layer, modification of only certain of the weights
        in the hidden to output synapses will have an effect on each output spike

    """

    v_diffs, i_diffs = SpikeSlopes(Mv, S_out, d_i)


"""
def TestNodeRange(T, v0, u0, I0, ge0, neuron_names, synapse_names, state_monitor_names, spike_monitor_names, \
        parameters, number, net)

    n_hidden_last = len(net[neuron_names[2][-1]])
    old_weights = np.empty(n_hidden_last)

    return_val = [-1, -1]

    for i in range(n_hidden_last):
        old_weights[i] = Sb.w[i]
        Sb.w[i] = 0

    j = 0
    Sb.w[0] = 0
    while True:

        snn.Run(T, v0, u0, I0, ge0, neuron_names, synapse_names, state_monitor_names, spike_monitor_names, \
                parameters, number, net)
        #pudb.set_trace()
        spikes_out = S_out.spiketimes[0]
        #spikes_hidden = S_hidden.spiketimes[0]
        n_outspikes = len(spikes_out)
        print "n_outspikes, Sb.w[0] = ", n_outspikes, ", ", Sb.w[0]

        if n_outspikes == 1:
            if return_val[0] == -1:
                #pudb.set_trace()
                return_val[0] = spikes_out[0]# - spikes_hidden[0]
            return_val[1] = spikes_out[0]
        elif n_outspikes > 1:
            #pudb.set_trace()
            break

        Sb.w[0] = Sb.w[0] + 0.001

        #if j % 1 == 0:
        #    snn.Plot(Mv, 0)
        #    
        #j += 1


    for i in range(n_hidden_last):
        Sb.w[i] = old_weights[i]

    return return_val
"""
