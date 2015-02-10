import brian as br
import math as ma
import pudb
import snn

def DesiredOut(label, bench):
    return_val = None

    if bench == 'xor':
        if label == 1:
            return_val = 0*br.ms
        else:
            return_val = 6*br.ms

    return return_val

def WeightChange(s):
    A = 10**1
    tau = 1.0*br.ms
    return A*ma.exp(-s / tau)

def L(t):
    tau = 5.0*br.ms
    if t > 0:
        return ma.exp(float(-t / tau))
    else:
        return 0

def P_Index(S_l, S_d):
    return_val = 0

    #if len(S_l) == len(S_d):
    #for i in range(len(S_l)):
        #if len(S_l[i]) != len(S_d[i]):
        #    return None
        #if len(S_l[i]) != 0:
        #    pudb.set_trace()
    return_val += abs(L(S_d) - L(S_l[0][0]*br.second))

    #pudb.set_trace()
    return return_val
    #else:
    #    return None

def ReSuMe(in_out, Pc, T, N, v0, u0, bench, number, input_neurons, hidden_neurons, output_neurons, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out):

    img, label = snn.ReadImg(number=number, bench=bench)
    N_hidden = len(hidden_neurons)
    N_out = len(output_neurons)

    N_h = 1
    N_o = 1

    trained = False

    while trained == False:
        for i in range(N_hidden):

            #print "\t\ti = ", i
            label = snn.Run(T, v0, u0, bench, number, input_neurons, hidden_neurons, output_neurons, \
                Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=True, letter=None)

            print "Spike Times: ", S_hidden.spiketimes, " ", S_out.spiketimes
            done = snn.CheckNumSpikes(T, N_h, N_o, v0, u0, bench, number, input_neurons, hidden_neurons, output_neurons, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=False, letter=None)

            if done == False:
                print "ERROR!! WRONG NUMBER OF SPIKES!! Resetting No. Spikes!!!"
                #pudb.set_trace()
                snn.SetNumSpikes(T, N_h, N_o, v0, u0, bench, number, input_neurons, hidden_neurons, output_neurons, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=False, letter=None)

            #pudb.set_trace()
            S_l = S_out.spiketimes
            S_i = S_hidden.spiketimes
            S_d = in_out[label]

            P = P_Index(S_l, S_d)
            print "\t\t\tP = ", P
            if P < Pc:
                trained = True
                break

            #pudb.set_trace()
            sd = max(0, float(S_d) - S_i[i][0])
            sl = max(0, S_l[0][0] - S_i[i][0])
            Wd = WeightChange(sd)
            Wl = -WeightChange(sl)
            Sb.w[i] = Sb.w[i] + Wd + Wl

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

    

def dWa(Sa, v_diffs, i_diffs):
    pass
