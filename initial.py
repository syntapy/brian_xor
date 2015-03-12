import brian2 as br
import numpy as np
import pudb
import snn

def NeuronIndices(N_hidden):
    """
        Nin, Nli, Nlh, Nlo, N_liq, Nhid, Nout
    """
    return 0, 1, 2, 3, 4, 5, 5+N_hidden

def SynapseIndices(N_hidden):
    """Si, Sl, Sa, Sb"""
    return 0, 1, 2, 3+N_hidden

def SetNeuronGroups(N_in, N_liquid, N_hidden, N_out, parameters, \
        eqs_hidden_neurons, reset, neuron_names):

    #pudb.set_trace()
    #x = np.array([0, 1, 2])
    #y = np.array([0, 2, 1])*br.msecond
    #input_neurons = br.SpikeGeneratorGroup(N=N_in+1, indices=x, times=y)
    input_neurons = br.NeuronGroup(3, '''dv/dt = (vt - vr)/period : volt (unless refractory)
                                        period: second
                                        fire_once: boolean ''', \
                                    threshold='v>vt', reset='v=vr',
                                    refractory='fire_once', \
                                    name=neuron_names[0])

    a = parameters[0]
    b = parameters[1]
    c = parameters[2]
    d = parameters[3]
    tau = parameters[4]
    vt = parameters[5]
    vr = parameters[6]

    liquid_neurons = br.NeuronGroup(N_liquid[-1], model=eqs_hidden_neurons, \
            threshold='v>vt', refractory=2*br.ms, reset=reset, \
            method='euler', name=neuron_names[1][3])

    liquid_in = br.Subgroup(liquid_neurons, 0, N_liquid[0], \
            name=neuron_names[1][0])
    liquid_hidden = br.Subgroup(liquid_neurons, N_liquid[0], N_liquid[-1] - N_liquid[1], \
            name=neuron_names[1][1])
    liquid_out = br.Subgroup(liquid_neurons, N_liquid[-1] - N_liquid[1], N_liquid[-1], \
            name=neuron_names[1][2])

    #liquid_in.indices = np.arange(0, N_liquid[0])
    #liquid_hidden.indices = np.arange(N_liquid[0], N_liquid[-1] - N_liquid[1])
    #liquid_out.indices = np.arange(N_liquid[-1] - N_liquid[1], N_liquid[-1])

    liquid_in.v = 0*br.mV
    liquid_hidden.v = 0*br.mV
    liquid_out.v = 0*br.mV

    hidden_neurons = []
    for i in range(len(N_hidden)):
        hidden_neurons.append(br.NeuronGroup(N_hidden[i], \
            model=eqs_hidden_neurons, threshold='v>vt', refractory=2*br.ms, reset=reset, \
            method='rk4', name=neuron_names[2][i]))

    output_neurons = br.NeuronGroup(N_out, model=eqs_hidden_neurons,\
        threshold='v>vt', refractory=2*br.ms, reset=reset, method='rk4', name=neuron_names[3])

    neuron_groups = [input_neurons, \
        [liquid_in, \
        liquid_hidden, \
        liquid_out, \
        liquid_neurons], \
        hidden_neurons, \
        output_neurons]

    return neuron_groups

def InitConditions(net, string, v0, u0, I0, ge0):
    net[string].v = v0
    net[string].u = u0
    net[string].I = I0
    net[string].ge = ge0

def NeuronInitConditions(net, neuron_names, v0, u0, I0, ge0):
    N_groups = len(neuron_names)

    for i in range(N_groups):
        if type(neuron_names[i]) == list:
            N = len(neuron_names[i])
            for j in range(N):
                InitConditions(net, neuron_names[i][j], v0, u0, I0, ge0)
        else:
            InitConditions(net, neuron_names[i], v0, u0, I0, ge0)

    return net

def SetSynapseInitialWeights(net, synapse_names):
    net[synapse_names[0]].connect(True)
    net[synapse_names[0]].w[:]='42.2*(0.3+0.8*rand())'
    net[synapse_names[0]].delay='(1)*ms'

    net[synapse_names[1]].connect(True)
    net[synapse_names[1]].w[:,:]='6.2'
    net[synapse_names[1]].delay='3*rand()*ms'

    for i in range(len(synapse_names[2])):
        net[synapse_names[2][i]].connect(True)
        net[synapse_names[2][i]].w[:, :]='15.1*(0.5+0.5*rand())'
        net[synapse_names[2][i]].delay='(1)*ms'

    """
    non-zero index     non-zero neuron
    0:                 0
    1:                 1
    2:                 2
    3:                 
    4:                 
    5:                 0
    6:                 1
    7:                 2
    8:                 
    """
    net[synapse_names[2][-1]].w[:, :]='15.1*(0.0+0.5*rand())'
    #net[synapse_names[2][-1]].w[9] = 10
    net[synapse_names[-1]].connect(True)
    net[synapse_names[-1]].w[:, :]='0.9*(0.1+0.2*rand())'
    net[synapse_names[-1]].delay='(1)*ms'

    return net

def SetSynapses(neuron_groups, synapse_names):

    #synapse_names = ['Si', 'Sl', [], 'Sb']
    s = 1.0
    N_hidden_layers = len(neuron_groups[2])
    #pudb.set_trace()
    Si = br.Synapses(neuron_groups[0], neuron_groups[1][0], model='w:1', pre='ge+=w*mV', \
            name=synapse_names[0])
    Sl = br.Synapses(neuron_groups[1][-1], neuron_groups[1][-1], model='w:1', pre='ge+=w*mV', \
            name=synapse_names[1])

    Sa = []
    Sa.append(br.Synapses(neuron_groups[1][2], neuron_groups[2][0], model='w:1', pre='ge+=w*mV', \
            name=synapse_names[2][0]))
    for i in range(N_hidden_layers - 1):
        Sa.append(br.Synapses(neuron_groups[2][i], neuron_groups[2][i+1], model='w:1', \
                pre='ge+=w*mV'), name=synapse_names[2][i+1])
    Sb = br.Synapses(neuron_groups[2][-1], neuron_groups[3], model='w:1', pre='ge+=w*mV', \
            name=synapse_names[3])

    synapse_groups = [Si, Sl, Sa, Sb]

    return synapse_groups

def NeuronGroupIndex(index_str):

    if index_str == 'input':
        index_a = 0
        index_b = None
    elif index_str == 'liquid_in':
        index_a = 1
        index_b = 0
    elif index_str == 'liquid_hidden':
        index_a = 1
        index_b = 1
    elif index_str == 'liquid_out':
        index_a = 1
        index_b = 2
    elif index_str == 'liquid_all':
        index_a = 1
        index_b = 3
    elif index_str[:-1] == 'hidden_':
        index_a = 2
        index_b = int(index_str[-1])
    elif index_str == 'out':
        index_a = 3
        index_b = None

    return index_a, index_b

def StateMonitors(neuron_groups, index_str, index_record=0):

    index_a, index_b = NeuronGroupIndex(index_str)
    if index_str == 'input':
        M = br.StateMonitor(neuron_groups[index_a], 'v', record=index_record, \
                    name=(index_str + '_v'))

        return M

    else:
        if index_b == None:
            Mge = br.StateMonitor(neuron_groups[index_a], 'ge', record=index_record, \
                        name=(index_str + '_ge' + str(index_record)))
            Mv = br.StateMonitor(neuron_groups[index_a], 'v', record=index_record, \
                        name=(index_str + '_v' + str(index_record)))
            Mu = br.StateMonitor(neuron_groups[index_a], 'u', record=index_record, \
                        name=(index_str + '_u' + str(index_record)))
        else:
            Mge = br.StateMonitor(neuron_groups[index_a][index_b], 'ge', record=index_record, \
                        name=(index_str + '_ge' + str(index_record)))
            Mv = br.StateMonitor(neuron_groups[index_a][index_b], 'v', record=index_record, \
                        name=(index_str + '_v' + str(index_record)))
            Mu = br.StateMonitor(neuron_groups[index_a][index_b], 'u', record=index_record, \
                        name=(index_str + '_u' + str(index_record)))

        return Mv, Mu, Mge

def SpikeMonitor(neuron_groups, index_str):
    index_a, index_b, index_aux = NeuronGroupIndex(index_str)

    if index_b == None and index_aux == None:
        S = br.SpikeMonitor(neuron_groups[index_a], record=0)
    elif index_a == 2:
        S = br.SpikeMonitor(neuron_groups[index_a][index_aux], record=0)
    elif index_b != None:
        S = br.SpikeMonitor(neuron_groups[index_a][index_b], record=0)

    return S

def AllSpikeMonitors(neuron_groups, spike_monitor_names):
    N = len(neuron_groups)

    spike_monitors = []

    spike_monitors.append(br.SpikeMonitor(neuron_groups[0], record=0, name=spike_monitor_names[0]))
    spike_monitors.append(br.SpikeMonitor(neuron_groups[1][-1], record=0, \
            name=spike_monitor_names[1]))
    spike_monitors.append([])
    for i in range(len(neuron_groups[2])):
        spike_monitors[2].append(br.SpikeMonitor(neuron_groups[2][i], \
                record=0, name=spike_monitor_names[2][i]))
    spike_monitors.append(br.SpikeMonitor(neuron_groups[3], record=0, name=spike_monitor_names[3]))

    return spike_monitors

def _network(net, group):
    N_groups = len(group)

    if N_groups > 0:
        for i in range(N_groups):
            if type(group[i]) == list:
                N = len(group[i])
                for j in range(N):
                    net.add(group[i][j])
            else:
                net.add(group[i])
    else:
        net.add(group)

    return net

def AddNetwork(neuron_groups, synapse_groups, state_monitors, spike_monitors, parameters):
    a = parameters[0]
    b = parameters[1]
    c = parameters[2]
    d = parameters[3]
    tau = parameters[4]
    vt = parameters[5]
    vr = parameters[6]

    net = br.Network()

    net = _network(net, neuron_groups)
    net = _network(net, synapse_groups)
    net = _network(net, spike_monitors)
    if type(state_monitors) == list or type(state_monitors) == tuple:
        for i in range(len(state_monitors)):
            net = _network(net, state_monitors[i])
    else:
        net = _network(net, state_monitors)

    return net

def SetInitStates(net, vr, v0, u0, I0, ge0, neuron_names, bench='xor'):

    net.store()
    for number in range(4):
        net = NeuronInitConditions(net, neuron_names[1:], v0, u0, I0, ge0)
        letter = None
        label = 0
        img, label = snn.ReadImg(number=number, bench=bench, letter=letter)
        spikes = snn.GetInSpikes(img, bench=bench)
        net[neuron_names[0]].period = spikes * br.ms
        net[neuron_names[0]].fire_once = [True, True, True]
        net[neuron_names[0]].v = vr
        net.store(str(number))

    return net

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
                    k_modified, net = BasicTraining(net, neuron_names[2][-1], synapse_names[2][-1], spike_monitor_names[2][-1], number, dw_abs, N_h)
                else:
                    k_modified, net = BasicTraining(net, neuron_names[3], synapse_names[3], spike_monitor_names[3], number, dw_abs, N_o)

                if k_modified == True:
                    modified = True
                k += 1
    return net

def SaveWeights(synapses, file_name):

    F = open(file_name, 'w')
    n = len(synapses.w[:])
    for i in range(n):
        F.write(synapses.w[i])
        F.write('\n')
    F.close()

def StringToWeights(string):
    """
    string is a set of floating point numbers or integers separated by newline characters
    """

    n = len(string)
    weights = np.empty(n, dtype=float)
    for i in xrange(n):
        weights[i] = float(string[i][:-1])

    return weights

def ReadWeights(file_names):
    weight_list = []
    for i in range(len(synapse_names)):
        if type(synapse_names[i]) == list:
            string.append([])
            for j in range(len(synapse_names[i])):
                file_name = 'weights/' + synapse_names[i][j] + '.txt'
                F = open(file_name, 'r')
                weight_array = StringToWeights(F.readlines())
                weight_list[i].append(weight_array)
                F.close()
        else:
            file_name = 'weights/' + synapse_names[i] + '.txt'
            F = open(file_name, 'r')
            weight_array = StringToWeights(F.readlines())
            weight_list.append(weight_array)
            F.close()

    return weight_list

def SaveNetworkWeights(net, synapse_names):
    for i in range(len(synapse_names)):
        if type(synapse_names[i]) == list:
            for j in range(len(synapse_names[i])):
                SaveWeights(net[synapse_names[i][j]], 'weights/' + synapse_names[i][j] + '.txt')
        else:
            SaveWeights(net[synapse_names[i]], 'weights/' + synapse_names[i] + '.txt')

def ReadNetworkWeights(net, synapse_names):
    for i in xrange(len(synapse_names)):
        if type(synapse_names[i]) == list:
            for j in xrange(len(synapse_names[i])):
                net[synapse_names[i][j]].w[:] = ReadWeights('weights/' + synapse_names[i][j] + '.txt')
        else:
            net[synapse_names[i]].w[:] = ReadWeights('weights/' + synapse_names[i] + '.txt')

    return net

def NumberLines(synapse_name_single):
    with open(synapse_name_single) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def CompatibleDimensions(net, synapse_name_single):
    n_f = NumberLines(synapse_name_single)
    n_net = len(net[synapse_name_single].w[:])

    return n_f == n_net

def CorrectWeightsExist(net, synapse_names, N_liquid, N_hidden):

    for i in range(len(synapse_names)):
        if type(synapse_names[i]) == list:
            for j in range(len(synapse_names[i])):
                if op.isfile(synapse_names[i][j]) == False:
                    return False
                elif CompatibleDimensions(net, synapse_names[i][j]) == False:
                    return False
        else:
            if op.isfile(synapse_names[i]) == False:
                return False
            elif CompatibleDimensions(net, synapse_names[i][j]) == False:
                return False

    return True

def SetWeights(net, synapse_names):

    if CorrectWeightsExist(synapse_names, N_liquid, N_hidden):
        weights = snn.ReadWeights(synapse_Names)
        for i in range(len(synapse_names)):
            if type(synapse_names[i]) == list:
                for j in range(len(synapse_names)):
                    net[synapse_names[i][j]].w[:] = weights[i][j]
            else:
                net[synapse_names[i]].w[:] = weights[i]
    else:
        net = snn.SetNumSpikes(0, T, N_h, N_o, v0, u0, I0, ge0, net, \
                neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters, number)

        net = snn.SetNumSpikes(1, T, N_h, N_o, v0, u0, I0, ge0, net, \
                neuron_names, synapse_names, state_monitor_names, spike_monitor_names, parameters, number)

        snn.SaveWeights(net, synapse_names)

    return net
