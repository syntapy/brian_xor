import brian2 as br
import numpy as np
import pudb

def NeuronIndices(N_hidden):
    """
        Nin, Nli, Nlh, Nlo, N_liq, Nhid, Nout
    """
    return 0, 1, 2, 3, 4, 5, 5+N_hidden

def SynapseIndices(N_hidden):
    """Si, Sl, Sa, Sb"""
    return 0, 1, 2, 3+N_hidden

def SetNeuronGroups(N_in, N_liquid, N_hidden, N_out, parameters, eqs_hidden_neurons, reset):

    #pudb.set_trace()
    #x = np.array([0, 1, 2])
    #y = np.array([0, 2, 1])*br.msecond
    #input_neurons = br.SpikeGeneratorGroup(N=N_in+1, indices=x, times=y)
    input_neurons = br.NeuronGroup(3, '''dv/dt = vt/period : volt (unless refractory)
                                        period: second
                                        fire_once: boolean ''', \
                                    threshold='v>vt', reset='v=0*volt',
                                    refractory='fire_once')

    a = parameters[0]
    b = parameters[1]
    c = parameters[2]
    d = parameters[3]
    tau = parameters[4]
    vt = parameters[5]

    liquid_neurons = br.NeuronGroup(N_liquid[-1], model=eqs_hidden_neurons, \
        threshold='v>vt', \
        refractory=2*br.ms, \
        reset=reset)

    liquid_in = br.Subgroup(liquid_neurons, 0, N_liquid[0])
    liquid_hidden = br.Subgroup(liquid_neurons, N_liquid[0], N_liquid[-1] - N_liquid[1])
    liquid_out = br.Subgroup(liquid_neurons, N_liquid[-1] - N_liquid[1], N_liquid[-1])

    #liquid_in.indices = np.arange(0, N_liquid[0])
    #liquid_hidden.indices = np.arange(N_liquid[0], N_liquid[-1] - N_liquid[1])
    #liquid_out.indices = np.arange(N_liquid[-1] - N_liquid[1], N_liquid[-1])

    liquid_in.v = 0*br.mV
    liquid_hidden.v = 0*br.mV
    liquid_out.v = 0*br.mV

    hidden_neurons = []
    for i in range(len(N_hidden)):
        hidden_neurons.append(br.NeuronGroup(N_hidden[i], \
            model=eqs_hidden_neurons, threshold='v>vt', refractory=2*br.ms, reset=reset))

    output_neurons = br.NeuronGroup(N_out, model=eqs_hidden_neurons,\
        threshold='v>vt', refractory=2*br.ms, reset=reset)

    neuron_groups = [input_neurons, \
        [liquid_in, \
        liquid_hidden, \
        liquid_out, \
        liquid_neurons], \
        hidden_neurons, \
        output_neurons]

    return neuron_groups

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

    #pudb.set_trace()
    synapse_groups[0].connect(True)
    synapse_groups[0].w[:]='8.04*(0.3+0.8*rand())'
    synapse_groups[0].delay='(1)*ms'

    synapse_groups[1].connect(True)
    synapse_groups[1].w[:,:]='8.0*rand()'
    synapse_groups[1].delay='3*rand()*ms'

    for i in range(len(synapse_groups[2])):
        synapse_groups[2][i].connect(True)
        synapse_groups[2][i].w[:, :]='8.04*(0.3+0.8*rand())'
        synapse_groups[2][i].delay='(1)*ms'

    synapse_groups[-1].connect(True)
    synapse_groups[-1].w[:, :]='9.0*(0.1+0.2*rand())'
    synapse_groups[-1].delay='(1)*ms'

def SetSynapses(neuron_groups):

    N_hidden_layers = len(neuron_groups[2])
    #pudb.set_trace()
    Si = br.Synapses(neuron_groups[0], neuron_groups[1][0], model='w:1', pre='ge+=w*mV')
    Sl = br.Synapses(neuron_groups[1][-1], neuron_groups[1][-1], model='w:1', pre='ge+=w*mV')

    Sa = []
    Sa.append(br.Synapses(neuron_groups[1][2], neuron_groups[2][0], model='w:1', pre='ge+=w*mV'))
    for i in range(N_hidden_layers - 1):
        Sa.append(br.Synapses(neuron_groups[2][i], neuron_groups[2][i+1], model='w:1', pre='ge+=w*mV'))
    Sb = br.Synapses(neuron_groups[2][-1], neuron_groups[3], model='w:1', pre='ge+=w*mV')

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

def AllSpikeMonitors(neuron_groups):
    N = len(neuron_groups)

    spike_monitors = []

    spike_monitors.append(br.SpikeMonitor(neuron_groups[0], record=0))
    spike_monitors.append(br.SpikeMonitor(neuron_groups[1][-1], record=0))
    spike_monitors.append([])
    for i in range(len(neuron_groups[2])):
        spike_monitors[2].append(br.SpikeMonitor(neuron_groups[2][i], record=0))
    spike_monitors.append(br.SpikeMonitor(neuron_groups[3], record=0))

    return spike_monitors

def _network(net, group):
    N_groups = len(group)

    for i in range(N_groups):
        if type(group[i]) == list:
            N = len(group[i])
            for j in range(N):
                net.add(group[i][j])
        else:
            net.add(group[i])

    return net

def AddNetwork(neuron_groups, synapse_groups, output_monitor, spike_monitors, parameters):
    a = parameters[0]
    b = parameters[1]
    c = parameters[2]
    d = parameters[3]
    tau = parameters[4]
    vt = parameters[5]

    net = br.Network()

    net = _network(net, neuron_groups)
    net = _network(net, synapse_groups)
    net = _network(net, output_monitor)
    net = _network(net, spike_monitors)

    return net
