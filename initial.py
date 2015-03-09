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
            name=neuron_names[1][3])

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
            name=neuron_names[2][i]))

    output_neurons = br.NeuronGroup(N_out, model=eqs_hidden_neurons,\
        threshold='v>vt', refractory=2*br.ms, reset=reset, name=neuron_names[3])

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

def SetSynapseInitialWeights(net, synapse_names):

    #pudb.set_trace()
    net[synapse_names[0]].connect(True)
    net[synapse_names[0]].w[:]='42.2*(0.3+0.8*rand())'
    net[synapse_names[0]].delay='(1)*ms'

    net[synapse_names[1]].connect(True)
    net[synapse_names[1]].w[:,:]='0.02*rand()'
    net[synapse_names[1]].delay='3*rand()*ms'

    for i in range(len(synapse_names[2])):
        net[synapse_names[2][i]].connect(True)
        net[synapse_names[2][i]].w[:, :]='0.1*(0.3+0.8*rand())'
        net[synapse_names[2][i]].delay='(1)*ms'

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

def NeuronGroupIndex(index_str, index_aux=None):

    if index_str == 'input':
        index_a = 0
        index_b = None
        index_aux = None
    elif index_str == 'liquid_in':
        index_a = 1
        index_b = 0
        index_aux = None
    elif index_str == 'liquid_hidden':
        index_a = 1
        index_b = 1
        index_aux = None
    elif index_str == 'liquid_out':
        index_a = 1
        index_b = 2
        index_aux = None
    elif index_str == 'liquid_all':
        index_a = 1
        index_b = 3
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
    if index_str == 'input':
        M = br.StateMonitor(neuron_groups[index_a], 'v', record=index_record, \
                    name=(index_str + '_v'))

        return M

    else:
        if index_b == None and index_aux == None:
            Mge = br.StateMonitor(neuron_groups[index_a], 'ge', record=index_record, \
                        name=(index_str + '_ge'))
            Mv = br.StateMonitor(neuron_groups[index_a], 'v', record=index_record, \
                        name=(index_str + '_v'))
            Mu = br.StateMonitor(neuron_groups[index_a], 'u', record=index_record, \
                        name=(index_str + '_u'))
        elif index_a == 2:
            Mge = br.StateMonitor(neuron_groups[index_a][index_aux], 'ge', record=index_record, \
                        name=(index_str + '_ge'))
            Mv = br.StateMonitor(neuron_groups[index_a][index_aux], 'v', record=index_record, \
                        name=(index_str + '_v'))
            Mu = br.StateMonitor(neuron_groups[index_a][index_aux], 'u', record=index_record, \
                        name=(index_str + '_u'))
        elif index_b != None:
            Mge = br.StateMonitor(neuron_groups[index_a][index_b], 'ge', record=index_record, \
                        name=index_str + '_ge')
            Mv = br.StateMonitor(neuron_groups[index_a][index_b], 'v', record=index_record, \
                        name=index_str + '_v')
            Mu = br.StateMonitor(neuron_groups[index_a][index_b], 'u', record=index_record, \
                        name=index_str + '_u')

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

def AddNetwork(neuron_groups, synapse_groups, output_monitor, spike_monitors, parameters):
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
    net = _network(net, output_monitor)
    net = _network(net, spike_monitors)

    return net
