import initial as init
import numpy as np
import brian2 as br
import cProfile
import pudb

def make2dList(rows, cols):
    a=[]
    for row in xrange(rows): a += [[None]*cols]

    return a

def ReadImg(number=1, letter=None, bench='LI', levels=None):

    if bench == 'xor':
        if levels == None:
            levels = 1
        n = 3   # Dimension of single image (no pyramid)
        img_dims = (1, 2)
        img = np.empty(img_dims)

        '''
            00:0 11:0 10:1 01:1
        '''

        if number == 0:
            img[0][0] = 0
            img[0][1] = 0
            label = 0

        elif number == 1:
            img[0][0] = 1
            img[0][1] = 1
            label = 0

        elif number == 2:
            img[0][0] = 1
            img[0][1] = 0
            label = 1

        elif number == 3:
            img[0][0] = 0
            img[0][1] = 1
            label = 1
        else:
            img = None
            label = -1

    elif bench == 'mnist':
        if levels == None:
            levels = 4
        n = 32  # Dimension of largest image in pyramid
        img_dims = (n, n)
        N = 0  # Number of filters
        dir = 'minst/'
        data_dir = 'on_pyr/'

        for h in range(levels):
            N += ((n / 2) - 2)**2
            n /= 2

    elif bench == 'LI':
        if levels == None:
            levels = 1
        n = 3   # Dimension of single image (no pyramid)
        img_dims = (n, n)
        img = np.empty(img_dims)
        N = 1
        dir = 'li-data/'
        data_dir = 'noise/'
        if letter == None:
            letter = 'L'

        F = open(dir + data_dir + letter + str(number) + '.txt')

        for i in range(img_dims[0]):
            line = F.readline()

            for j in range(img_dims[1]):
                img[i][j] = int(line[j])

        if letter == 'L':
            label = 0
        else:
            label = 1

        F.close()

    return img, label

def GetInSpikes(img, bench='LI'):

    if img != None:
        img_dims = np.shape(img)
        if bench == 'LI' or bench == 'xor':

            spikes = [-1, -1, -1]
            h = 0
            x_range = range(img_dims[0])
            y_range = range(img_dims[1])

            for i in x_range:
                for j in y_range:
                    pixel = img[i][j]
                    if pixel == 1:
                        spikes[h] = 7
                    else:
                        spikes[h] = 1
                    h += 1

            spikes[h] = 7

            return spikes

def d_w(S_d, S_l, S_in):
    sd = S_d - S_in
    sl = S_l - S_in

    if len(sd) == len(sl):
        return_val = A*np.sum(np.exp(sd) - np.exp(sl))

    else:
        return_val = None

    return return_val

def P_Index(S_d, S_l):
    return_val = 0

    if len(S_d) == len(S_l):
        for i in range(len(S_d)):
            return_val += ma.exp(abs(S_d[i] - S_l[i]))
    else:
        return_val = None

    return return_val

def ModifyWeights(S, dv):
    n = len(S.w[:])
    for i in range(n):
        weet = S.w[i]
        weet = weet*br.volt + dv*br.mV
        S.w[i] = weet

    return S

def CollectSpikes(indices, spikes, N_neurons):
    """
    This takes the indices and spike times and converts them into a list of lists
    """
    spikes_hidden = []
    spikes_out = []

    spikes_list = []

    j = 0
    arg_sort = br.argsort(indices)
    sorted_indices = br.sort(indices)
    for i in range(N_neurons):
        spikes_list.append([])

    for i in range(len(sorted_indices)):
        index = arg_sort[i]
        spikes_list[sorted_indices[i]].append(spikes[arg_sort[i]])

    return spikes_list

def CheckNumSpikes(layer, T, N_h, N_o, v0, u0, I0, ge0, neuron_names, spike_monitor_names, net):

    """
    Returns True if each hidden neuron is emmitting N_h spikes
    and if each output neuron is emmitting N_o spikes
    """

    N_out_spikes = []
    N_hidden_spikes = []

    N_hidden = len(neuron_names[2])
    N_out = 1

    if layer == 0:
        for i in range(N_hidden):
            hidden_layer = net[neuron_names[2][i]]
            spike_monitor = net[spike_monitor_names[2][i]]
            indices, spikes = spike_monitor.it
            if len(indices) == 0:
                if N_h > 0:
                    return False
            else:
                indices = br.unique(indices)
                if len(indices) != N_h*len(hidden_layer):
                    return False

    else:
        output_layer = net[neuron_names[3]]
        spike_monitor = net[spike_monitor_names[3]]
        indices, spikes = spike_monitor.it
        if len(indices) == 0:
            if N_o > 0:
                return False
        else:
            indicies = br.unique(indices)
            if len(indices) != N_o*len(output_layer):
                return False

    return True

def ModifyNeuronWeights(net, neuron_str, synapse_str, neuron_index, dv, N_neurons):
    N_neurons = len(net[neuron_str])
    N_synapses = len(net[synapse_str])
    for i in range(neuron_index, N_synapses, N_synapses / N_neurons):
        net[synapse_str].w[i] += dv*br.random()

    return net

def ModifyLayerWeights(net, spikes, neuron_str, synapse_str, number, dw_abs, D_spikes):

    modified = False
    N_neurons = len(net[neuron_str])
    if N_neurons == 1:
        index = 0
    else:
        index = number
    #pudb.set_trace()
    for i in range(index, N_neurons, 4):
        for j in range(N_neurons):
            if len(spikes[j]) > D_spikes:
                modified = True
                net = ModifyNeuronWeights(net, neuron_str, synapse_str, j, -dw_abs, N_neurons)
            elif len(spikes[j]) < D_spikes:
                modified = True
                net = ModifyNeuronWeights(net, neuron_str, synapse_str, j, dw_abs, N_neurons)

    return modified, net

def BasicTraining(net, neuron_str, synapse_str, spike_monitor_str, number, dw_abs, D_spikes):
    """
    Modifies the weights leading to each neuron in either the hidden layer or the output layer,
    in order to take it a step closer to having the desired number of spikes
    """
    layer_neurons = net[neuron_str]
    layer_synapses = net[synapse_str]
    spike_monitor = net[spike_monitor_str]
    N_neurons = len(layer_neurons)

    indices, spikes = spike_monitor.it
    #pudb.set_trace()
    spikes = CollectSpikes(indices, spikes, N_neurons)
    net.restore(str(number))
    modified, net = ModifyLayerWeights(net, spikes, neuron_str, synapse_str, number, dw_abs, D_spikes)
    net.store(str(number))

    return modified, net

def SetNumSpikes(layer, T, N_h, N_o, v0, u0, I0, ge0, net, \
        neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters, number):
    """
    This sets the number of spikes in the last hidden layer, and in the output layer, to
    N_h and N_o, respectively

    The network should start off with the last hidden layer having small but random weights 
    feeding into it such that the last hidden layer produces no spikes. Then,
    for each neuron in the last hidden layer, the weights feeding into it are gradually increased
    randomly through addiction of small numbers. If the number of spikes is too much, small random
    values are subtracted from the weights leading to it, until the desired number of spikes is
    emitted for every single input value produced in the input neurons.

    One issue is that it may take a long time to do this for more than one input sequence to the
    network as a whole, because the operations done for one input would be somewhat reversing
    the operations done for the other input, hence the likely usefullness of modifcation through
    random values.

    For each input combination that is fed into the network as a whole, it might help to have 
    different vectors which corresond to modification of weights. For instance, for network input
    0, you could modify every 4th neuron, for network input 1 you could modify every forth neuron
    but with an offset of 1, for network input 2 you modify every 4th neuron with an offset of 2,
    and so forth. That might be usefull.
    """

    dw_abs = 0.02
    min_dw_abs = 0.001
    i = 0
    #last = 0 # -1, 0, 1: left, neither, right

    print "layer = ", layer
    if layer == 0:
        dw_abs = 0.5
        #right_dw_abs = True
    else:
        dw_abs = 0.0005
        #div = 0
    modified = True
    j = 0
    while modified == True:
        modified = False
        print "\tj = ", j
        j += 1
        k = 0
        for number in range(4):
            desired_spikes = False
            print "\t\tNumber = ", number, "\t"
            while desired_spikes == False:
                #pudb.set_trace()
                Run(T, v0, u0, I0, ge0, neuron_names, synapse_names, state_monitor_names, \
                        spike_monitor_names, parameters, number, net)

                print "\t\t\tk = ", k
                #pudb.set_trace()
                desired_spikes = CheckNumSpikes(layer, T, N_h, N_o, v0, u0, I0, ge0, \
                        neuron_names, spike_monitor_names, net)

                if layer == 0:
                    modified, net = BasicTraining(net, neuron_names[2][-1], synapse_names[2][-1], spike_monitor_names[2][-1], number, dw_abs, N_h)
                else:
                    modified, net = BasicTraining(net, neuron_names[3], synapse_names[3], spike_monitor_names[3], number, dw_abs, N_o)
                k += 1
    return net

def Run(T, v0, u0, I0, ge0, neuron_names, synapse_names, \
    state_monitor_names, spike_monitor_names, parameters, number, net):

    #print "STARTING RUN FUNCTION"
    #pudb.set_trace()
    a = parameters[0]
    b = parameters[1]
    c = parameters[2]
    d = parameters[3]
    tau = parameters[4]
    vt = parameters[5]
    vr = parameters[6]

    #print "RESTORING NETWORK"
    net.restore(str(number))

    #print "STARTING COMPUTATIONS"
    net.run(T*br.msecond,report=None)
    #print "DONE"

    #return label

def Plot(monitor, number):
    #pudb.set_trace()
    br.plot(211)
    if type(monitor) == list or type(monitor) == tuple:
        br.plot(monitor[0].t/br.ms,monitor[0].v[0]/br.mV, label='v')
        #br.plot(monitor[1].t/br.ms,monitor[1].u[0]/br.mV, label='u')
        #br.plot(monitor[2].t/br.ms,monitor[0].ge[0]/br.mV, label='v')
    else:
        br.plot(monitor.t/br.ms,monitor.v[0]/br.mV, label='v')
    #br.plot(monitor[0].t/br.ms,monitor[0].u[0]/br.mV, label='u')
    #br.plot((monitor[2].t)/br.ms,(monitor[2][0]/br.mV), label='ge')
    br.legend()
    br.show()

"""
def SaveWeights(Si, Sl, Sa, Sb, filename):
    N_hidden = len(Sa)

    f = open(filename, 'w')
    for i in range(len(Sa)):
        for j in range(len(Sa[i].w[:]) - 1):
            f.write(str(Sa[i].w[j]))
            f.write(", ")
        f.write(str(Sa[i].w[-1]))
        f.write('\n')

    #pudb.set_trace()
    for i in range(len(Sb.w[:]) - 1):
        f.write(str(Sb.w[i]))
        f.write(", ")
    f.write(str(Sb.w[-1]))
    f.write('\n')
    f.close()

def IsNumber(character):
    if character == ' ' or character == ',': 
        return False

    if character == ';' or character == '\n' or character == '':
        return False

    return True

def ReadNumber(line, j):
    k = j
    while line[k] != ',' and line[k] != '\n':
        k += 1

    number = float(line[j:k])
    if line[k] != '\n':
        while IsNumber(line[k]) == False and k < len(line):
            k += 1

    return number, k

def ReadWeights(Si, Sl, Sa, Sb, filename):
    f = open(filename, 'r')

    lines = f.readlines()

    if len(Si) + len(Sl.w[:]) + len(Sa) + len(Sb) != len(lines):
        print "ERROR!"
        pudb.set_trace()

    #for i in range(len(

    line = lines[-1]
    j = 0
    index = 0
    while j < len(line) - 1:
        number, j = ReadNumber(line, j)
        Sb.w[index] = number
        index += 1

    for i in range(len(Sa)):
        line = lines[i]
        j = 0
        index = 0
        while j < len(line) - 1:
            number, j = ReadNumber(line, j)
            Sa[i].w[index] = number
            index += 1

    line = lines[-1]
    j = 0
    index = 0
    while j < len(line) - 1:
        number, j = ReadNumber(line, j)
        Sb.w[index] = number
        index += 1

    return Si, Sl, Sa, Sb
"""





"""
    dv = 0.2
    k = 0
    done = False

    while done == False: 

        pudb.set_trace()
        Run(T, v0, u0, I0, ge0, bench, number, \
            neuron_groups, synapse_groups, output_monitor, spike_monitors)

        N_hidden = len(neuron_groups[2])
        done = CheckNumSpikes(T, N_h, N_o, v0, u0, I0, ge0, bench, number, \
            neuron_groups, synapse_groups, output_monitor, spike_monitors)

        spikes_hidden, spikes_out = CollectSpikes(spike_monitors)
        N_out = len(spikes_out)

        print "SETTING NO. SPIKES "
        print "hidden: ", 
        for i in range(len(spike_monitors[2])):
            for j in range(len(spike_monitors[2][i])):
                print spike_monitors[2][i][j].spiketimes, " ",

        print "\nout: ", spike_monitors[3].spiketimes

        hidden_are_set = True
        #pudb.set_trace()
        net.restore(str(number))
        for i in range(len(neuron_groups[2])):
            for j in range(len(neuron_groups[2][i])):
                if len(spike_monitors[2][i][j].spiketimes[0]) < N_h:
                    ModifyWeights(synapse_groups[2][i][j], dv)
                    hidden_are_set = False
                elif len(spike_monitors[2][i][j]) > N_h:
                    ModifyWeights(synapse_groups[2][i][j.spiketimes[0]], -dv)
                    hidden_are_set = False
        net.store(str(number))

        if hidden_are_set == True:
            if N_out < N_o:
                ModifyWeights(synapse_groups[3], dv)
                net.store(str(number))
            elif N_out > N_o:
                ModifyWeights(synapse_groups[3], -dv)
                net.store(str(number))
            elif N_out == N_o:# or i == 100:
                break

        #if k % 100 == 0:
        #    Plot(Mu, Mv, 0)

        k += 1

"""














"""
print "\t\t\t\t."
for i in range(len(hidden_neurons)):
    if len(S_hidden[i]) < N_h:
        ModifyWeights(Sa[i], dv, 0)
        modified = True
    elif len(S_hidden[i]) > N_h:
        ModifyWeights(Sa[i], -dv, 0)
        modified = True
"""
"""
print "\t\t\t\tdiv = ", div
print S_hidden.spiketimes
if N_out < N_o:
    if last == 1:
        if dv > min_dv:
            dv = dv / 2
            div += 1
    elif last == 0:
        last = -1
    ModifyWeights(Sb, 0*dv, 0)
    modified = True
elif N_out > N_o:
    ModifyWeights(Sb, -0*dv, 0)
    modified = True
    #last = 1
elif N_out == N_o:
    done = True
    if dv > min_dv:
        ModifyWeights(Sb, -dv, 1)
        modified = True
        last = 1
    else:
    """
