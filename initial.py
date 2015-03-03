import brian2 as br
import numpy as np
import pudb

def NeuronIndices():
    """Nin, Nli, Nlh, Nlo, Nliq, Nhid, Nout"""
    return 0, 1, 2, 3, 4, 5, 6

def SynapseIndices():
    """Si, Sl, Sa, Sb"""
    return 7, 8, 9, 10

def OutputMonitorIndex():
    return 11

def SpikeMonitorIndices():
    return 12

def SetNeuronGroups(N_in, N_liquid, N_hidden, N_out, vt, parameters, eqs_hidden_neurons, reset):

    a = np.array([0])
    b = np.array([0])*br.msecond
    input_neurons = br.SpikeGeneratorGroup(N=N_in+1, indices=a, times=b)

    a = parameters[0]
    b = parameters[1]
    c = parameters[2]
    d = parameters[3]
    tau = parameters[4]

    pudb.set_trace()
    liquid_neurons = br.NeuronGroup(N_liquid[0], model=eqs_hidden_neurons, \
        threshold=vt, \
        refractory=2*br.ms, \
        reset=reset)

    liquid_in = br.Subgroup(liquid_neurons, 0, N_liquid[0])
    liquid_hidden = br.Subgroup(liquid_neurons, N_liquid[0], N_liquid[-1] - N_liquid[1])
    liquid_out = br.Subgroup(liquid_neurons, N_liquid[-1] - N_liquid[1], N_liquid[-1])

    hidden_neurons = []
    for i in range(len(N_hidden)):
        hidden_neurons.append(br.NeuronGroup(N_hidden[i], \
            model=eqs_hidden_neurons, threshold=vt, refractory=2*br.ms, reset=reset))

    output_neurons = br.NeuronGroup(N_out, model=eqs_hidden_neurons,\
        threshold=vt, refractory=2*br.ms, reset=reset)

    net = br.Network()
    net.add(input_neurons, \
        liquid_neurons_in, \
        iquid_neurons_hidden, \
        liquid_neurons_out, \
        liquid_neurons, \
        hidden_neurons, \
        output_neurons)

    return net

def InitConditions(single_neuron_group, v0, u0, I0, ge0):
    single_neuron_group.v = v0
    single_neuron_group.u = u0
    single_neuron_group.I = I0
    single_neuron_group.ge = ge0

def NeuronInitConditions(neuron_groups, v0, u0, I0, ge0):
    N_groups = len(neuron_groups)

    for i in range(N_groups):
        if type(neuron_groups[i]) == list:
            N = len(neuron_groups[i])
            for j in range(N):
                InitConditions(neuron_groups[i][j], v0, u0, I0, ge0)
        else:
            InitConditions(neuron_groups[i], v0, u0, I0, ge0)

def SetSynapseInitialWeights(synapse_groups):

    synapse_groups[0][:,:]=True
    synapse_groups[0].w[:]='8.04*(0.3+0.8*rand())*br.mV'
    synapse_groups[0].delay='(4)*ms'

    synapse_groups[1][:,:]=True
    synapse_groups[1].w[:,:]='8.0*rand()*br.mV'
    synapse_groups[1].delay='3*rand()*ms'

    for i in range(len(synapse_groups[2])):
        synapse_groups[2][i][:,:]=True
        synapse_groups[2][i].w[:]='8.04*(0.3+0.8*rand())*br.mV'
        synapse_groups[2][i].delay='(4)*ms'

    synapse_groups[-1][:,:]=True
    synapse_groups[-1].w[:]='9.0*(0.1+0.2*rand())*br.mV'
    synapse_groups[-1].delay='(4)*ms'

    """
    all_files_present = True
    for i in range(len(synapse_groups)):
        file_name = "weights/" + str(i) + ".txt"
        if not op.isfile(file_name):
            all_files_present = False
            break

    if all_files_present:
        file_name = "weights/" + array_file[0] + ".txt"
        Si.load_connectivity(file_name)
        file_name
    """

def SetSynapses(neuron_groups):

    N_hidden_layers = len(neuron_groups[2])
    Nin, Nli, Nlh, Nlo, Nliq, Nhid, Nout = NeuronIndices()
    Si = br.Synapses(net._added_groups[Nin], net._added_groups[Nli], model='w:1', pre='ge+=w')
    Sl = br.Synapses(net._added_groups[Nliq], net._added_groups[Nliq], model='w:1', pre='ge+=w')

    Sa = []
    Sa.append(br.Synapses(net._added_groups[Nlo], net._added_groups[Nhid], model='w:1', pre='ge+=w'))
    for i in range(N_hidden_layers - 1):
        Sa.append(br.Synapses(neuron_groups[2][i], neuron_groups[2][i+1], model='w:1', pre='ge+=w'))
    Sb = br.Synapses(neuron_groups[2][-1], neuron_groups[-1], model='w:1', pre='ge+=w')

    synapse_groups = [Si, Sl, Sa, Sb]
    SetSynapseInitialWeights(synapse_groups)

    return synapse_groups

def NeuronGroupIndex(index_str, index_aux=None):

    if index_str == 'input':
        index_a = 0
        index_b = None
        index_aux = None
    elif index_str == 'liquid':
        index_a = 1
        index_b = 1
        index_aux = None
    elif index_str == 'liquid_in':
        index_a = 1
        index_b = 0
        index_aux = None
    elif index_str == 'liquid_out':
        index_a = 1
        index_b = 2
        index_aux = None
    elif index_str == 'hidden':
        index_a = 2
        index_b = None
    elif index_str == 'out':
        index_a = 3
        index_b = None
        index_aux = None

    return index_a, index_b, index_aux

def StateMonitors(neuron_groups, index_str, index_record=0, index_aux=None):

    """
        index_str: tells which group / subgroup in which to pick a neuron
            'input', 'liquid_in', 'liquid_hidden', 'liquid_out', 'hidden', 'out'

        index_record: index of neuron in group / subgroup
        index_aux: if hidden neuron is picked, this is the index of the layer
            otherwise it is None
    """

    index_a, index_b, index_aux = NeuronGroupIndex(index_str, index_aux)
    if index_b == None and index_aux == None:
        Mge = br.StateMonitor(neuron_groups[index_a], 'ge', record=index_record)
        Mv = br.StateMonitor(neuron_groups[index_a], 'v', record=index_record)
        Mu = br.StateMonitor(neuron_groups[index_a], 'u', record=index_record)
    elif index_a == 2:
        Mge = br.StateMonitor(neuron_groups[index_a][index_aux], 'ge', record=index_record)
        Mv = br.StateMonitor(neuron_groups[index_a][index_aux], 'v', record=index_record)
        Mu = br.StateMonitor(neuron_groups[index_a][index_aux], 'u', record=index_record)
    elif index_b != None:
        Mge = br.StateMonitor(neuron_groups[index_a][index_b], 'ge', record=index_record)
        Mv = br.StateMonitor(neuron_groups[index_a][index_b], 'v', record=index_record)
        Mu = br.StateMonitor(neuron_groups[index_a][index_b], 'u', record=index_record)

    return [Mv, Mu, Mge]

def SpikeMonitor(neuron_groups, index_str):
    index_a, index_b, index_aux = NeuronGroupIndex(index_str)

    if index_b == None and index_aux == None:
        S = br.SpikeMonitor(neuron_groups[index_a], record=True)
    elif index_a == 2:
        S = br.SpikeMonitor(neuron_groups[index_a][index_aux], record=True)
    elif index_b != None:
        S = br.SpikeMonitor(neuron_groups[index_a][index_b], record=True)

    return S

def AllSpikeMonitors(neuron_groups):
    #pudb.set_trace()
    N = len(neuron_groups)
    spike_monitors = []
    spike_monitors.append(SpikeMonitor(neuron_groups, 'input'))
    spike_monitors.append(SpikeMonitor(neuron_groups, 'liquid'))
    spike_monitors.append([])
    for i in range(len(neuron_groups[2])):
        spike_monitors[2].append([])
        for j in range(len(neuron_groups[2][i])):
            spike_monitors[2][i].append(SpikeMonitor(neuron_groups[2][i], 'hidden'))
    spike_monitors.append(SpikeMonitor(neuron_groups[3], 'out'))

    return spike_monitors

def AddNetwork(neuron_groups, synapse_groups, output_monitor, spike_monitors):

    pudb.set_trace()
    net = br.Network()

    N_groups = len(neuron_groups)

    for i in range(N_groups):
        if type(neuron_groups[i]) == list:
            N = len(neuron_groups[i])
            for j in range(N):
                net.add(neuron_groups[i][j])
        else:
            net.add(neuron_groups[i])


    N_groups = len(synapse_groups)

    for i in range(N_groups):
        if type(synapse_groups[i]) == list:
            N = len(synapse_groups[i])
            for j in range(N):
                net.add(synapse_groups[i][j])
        else:
            net.add(synapse_groups[i])


    N_groups = len(output_monitor)

    for i in range(N_groups):
        if type(output_monitor[i]) == list:
            N = len(output_monitor[i])
            for j in range(N):
                net.add(output_monitor[i][j])
        else:
            net.add(output_monitor[i])

    N_groups = len(spike_monitors)

    for i in range(N_groups):
        if type(spike_monitors[i]) == list:
            N = len(spike_monitors[i])
            for j in range(N):
                net.add(spike_monitors[i][j])
        else:
            net.add(spike_monitors[i])

    return net
