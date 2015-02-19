import numpy as np
import brian as br
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

            spikes = []
            h = 0
            x_range = range(img_dims[0])
            y_range = range(img_dims[1])

            for i in x_range:
                for j in y_range:
                    pixel = img[i][j]
                    if pixel == 0:
                        spikes.append((h, 6*br.ms))
                    else:
                        spikes.append((h, 0*br.ms))
                    h += 1
            spikes.append((h, 0*br.ms))

            return spikes

def d_w(S_d, S_l, S_in):
    sd = S_d - S_in
    sl = S_l - S_in

    if len(sd) == len(sl):
        return_val = A*np.sum(np.exp(sd) - np.exp(sl))

    else:
        return_val = None

    return return_val

def LowPass(S):
    pass

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

def CollectSpikes(N_hidden, S_hidden, S_out):
    spikes_hidden = []
    spikes_out = []

    #pudb.set_trace()
    for i in range(N_hidden):
        spikes_hidden.append([])
        for j in range(len(S_hidden[i].spiketimes)):
            spikes_hidden[i].append(list(S_hidden[i].spiketimes[j]))

    for i in range(len(S_out.spiketimes)):
        spikes_out.append(list(S_out.spiketimes[i]))

    return spikes_hidden, spikes_out

def CheckNumSpikes(T, N_h, N_o, v0, u0, bench, number, input_neurons, liquid_neurons, hidden_neurons, output_neurons, Sin, Sliq, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=False, letter=None):
    #Run(T, v0, u0, bench, number, input_neurons, hidden_neurons, output_neurons, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=True, letter=None)
    N_out_spikes = []
    N_hidden_spikes = []

    N_hidden = len(hidden_neurons)
    N_out = len(output_neurons)
    for i in range(N_hidden):
        N_hidden_spikes.append([])
        for j in range(len(hidden_neurons[i])):
            #pudb.set_trace()
            N_hidden_spikes[i].append(len(hidden_neurons[i]))

    #pudb.set_trace()
    spikes_hidden, spikes_out = CollectSpikes(len(N_hidden_spikes), S_hidden, S_out)

    #pudb.set_trace()
    for i in range(len(spikes_hidden)):
        for j in range(len(spikes_hidden[i])):
            if len(spikes_hidden[i][j]) != N_h:
                return False

    for i in range(len(spikes_out)):
        if len(spikes_out[i]) != N_o:
            return False

    return True

def SetNumSpikes(T, N_h, N_o, v0, u0, bench, number, input_neurons, liquid_neurons, hidden_neurons, output_neurons, Sin, Sliq, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=False, letter=None):

    dv = 0.2
    k = 0
    done = False

    while done == False: 

        #pudb.set_trace()
        Run(T, v0, u0, bench, number, input_neurons, liquid_neurons, hidden_neurons, output_neurons, Sin, Sliq, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=True, letter=None)
        N_hidden = len(hidden_neurons)
        done = CheckNumSpikes(T, N_h, N_o, v0, u0, bench, number, input_neurons, liquid_neurons, hidden_neurons, output_neurons, Sin, Sliq, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=False, letter=None)

        spikes_hidden, spikes_out = CollectSpikes(N_hidden, S_hidden, S_out)
        N_out = len(spikes_out[0])

        print "SETTING NO. SPIKES "
        print "hidden: ", 
        for i in range(len(S_hidden)):
            print S_hidden[i].spiketimes, " ",

        print "\nout: ", S_out.spiketimes

        #pudb.set_trace()
        hidden_are_set = True
        for i in range(len(hidden_neurons)):
            for j in range(len(hidden_neurons[i])):
                if len(S_hidden[i][j]) < N_h:
                    ModifyWeights(Sa[i][j], dv)
                    hidden_are_set = False
                elif len(S_hidden[i][j]) > N_h:
                    ModifyWeights(Sa[i][j], -dv)
                    hidden_are_set = False

        if hidden_are_set == True:
            if N_out < N_o:
                ModifyWeights(Sb, dv)
            elif N_out > N_o:
                ModifyWeights(Sb, -dv)
            elif N_out == N_o:# or i == 100:
                break

        #if k % 100 == 0:
        #    Plot(Mu, Mv, 0)

        k += 1

def SaveWeights(Sa, Sb, filename):
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

def ReadWeights(Sa, Sb, filename):
    f = open(filename, 'r')

    lines = f.readlines()

    if len(Sa) + 1 != len(lines):
        print "ERROR!"
        pudb.set_trace()

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

    return Sa, Sb

def Run(T, v0, u0, bench, number, input_neurons, liquid_neurons, hidden_neurons, output_neurons, Sin, Sliq, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=False, letter=None):

    br.forget(Sin, Sliq)
    for i in range(len(Sa)):
        br.forget(Sa[i])
    br.forget(Sb)
    br.reinit(states=False)
    br.recall(Sin, Sliq)
    for i in range(len(Sa)):
        br.recall(Sa[i])
    br.recall(Sb)

    img, label = ReadImg(number=number, bench=bench, letter=letter)
    spikes = GetInSpikes(img, bench=bench)
    if number >= 0 and number < 4:
        input_neurons.set_spiketimes(spikes)
    else:
        input_neurons.set_spiketimes([])

    for i in range(len(liquid_neurons)):
        liquid_neurons[i].v = v0
        liquid_neurons[i].u = u0
        liquid_neurons[i].I = 0
        liquid_neurons[i].ge = 0

    for i in range(len(hidden_neurons)):
        hidden_neurons[i].v = v0
        hidden_neurons[i].u = u0
        hidden_neurons[i].I = 0
        hidden_neurons[i].ge = 0

    #pudb.set_trace()
    output_neurons.v = v0
    output_neurons.u = u0
    output_neurons.I = 0
    output_neurons.ge = 0
    br.run(T*br.msecond,report='text')

    return label

def Plot(M, Mu, Mv, number):
    br.plot(211)
    #pudb.set_trace()
    br.plot(Mu.times/br.ms,Mu[0]/br.mvolt, label='u')
    br.plot((Mv.times)/br.ms,2000*(Mv[0]/br.mvolt) - 58000, label='v')
    br.plot((M.times)/br.ms,2000*(M[0]/br.mvolt) - 58000, label='ge')
    #br.plot(Mv.times/br.ms,Mv[
    br.legend()
    br.show()
