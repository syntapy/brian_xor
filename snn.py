import initial as init
import numpy as np
import brian2 as br
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
                    if pixel == 0:
                        spikes[h] = 6
                    else:
                        spikes[h] = 0
                    h += 1

            spikes[h] = 0

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

def CollectSpikes(spike_monitors):
    spikes_hidden = []
    spikes_out = []

    # hidden spikes
    #pudb.set_trace()
    for i in range(len(spike_monitors[2])):
        spikes_hidden.append([])
        for j in range(len(spike_monitors[2][0][i].spiketimes)):
            spikes_hidden[i].append(list(spike_monitors[2][0][i].spiketimes[j]))

    for i in range(len(spike_monitors[3].spiketimes)):
        spikes_out.append(spike_monitors[3].spiketimes[i])

    return spikes_hidden, spikes_out

def CheckNumSpikes(T, N_h, N_o, v0, u0, I0, ge0, bench, number, \
        neuron_groups, synapse_groups, output_monitor, spike_monitors):

    N_out_spikes = []
    N_hidden_spikes = []

    N_hidden = len(neuron_groups[2])
    N_out = len(neuron_groups[3])
    for i in range(len(neuron_groups[2])):
        N_hidden_spikes.append([])
        for j in range(len(neuron_groups[2][i])):
            N_hidden_spikes[i].append(len(neuron_groups[2][i]))

    spikes_hidden, spikes_out = CollectSpikes(spike_monitors)
    for i in range(len(spikes_hidden)):
        for j in range(len(spikes_hidden[i])):
            if len(spikes_hidden[i][j]) != N_h:
                return False

    for i in range(len(spikes_out)):
        if len(spikes_out[i]) != N_o:
            return False

    return True

def SetNumSpikes(T, N_h, N_o, v0, u0, I0, ge0, bench, number, \
    neuron_groups, synapse_groups, output_monitor, spike_monitors, train=False, letter=None):

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
        for i in range(len(neuron_groups[2])):
            for j in range(len(neuron_groups[2][i])):
                if len(spike_monitors[2][i][j].spiketimes[0]) < N_h:
                    ModifyWeights(synapse_groups[2][i][j], dv)
                    hidden_are_set = False
                elif len(spike_monitors[2][i][j]) > N_h:
                    ModifyWeights(synapse_groups[2][i][j.spiketimes[0]], -dv)
                    hidden_are_set = False

        if hidden_are_set == True:
            if N_out < N_o:
                ModifyWeights(synapse_groups[3], dv)
            elif N_out > N_o:
                ModifyWeights(synapse_groups[3], -dv)
            elif N_out == N_o:# or i == 100:
                break

        #if k % 100 == 0:
        #    Plot(Mu, Mv, 0)

        k += 1

def Run(T, net, v0, u0, I0, ge0, bench, number, \
    neuron_groups, synapse_groups, output_monitor, spike_monitors, \
    parameters, train=False, letter=None):

    a = parameters[0]
    b = parameters[1]
    c = parameters[2]
    d = parameters[3]
    tau = parameters[4]
    vt = parameters[5]

    net.restore()
    img, label = ReadImg(number=number, bench=bench, letter=letter)
    spikes = GetInSpikes(img, bench=bench)
    neuron_groups[0].period = spikes * br.ms
    neuron_groups[0].fire_once = [True, True, True]
    neuron_groups[0].v = 0
    #if number >= 0 and number < 4:
    #    neuron_groups[0].set_spiketimes(spikes)
    #else:
    #    neuron_groups[0].set_spiketimes([])

    init.NeuronInitConditions(neuron_groups[1:], v0, u0, I0, ge0)
    #pudb.set_trace()
    net.run(T*br.msecond,report='text')

    return label

def Plot(monitor, number):
    #pudb.set_trace()
    br.plot(211)
    br.plot(monitor[0].t/br.ms,monitor[0].v[0]/br.mV, label='v')
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
