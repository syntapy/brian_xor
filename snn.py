import numpy as np
import brian as br
import pudb

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
    for i in range(N_hidden):
        spikes_hidden.append([])

    spikes_out = S_out.spikes

    return spikes_hidden, spikes_out

def CheckNumSpikes(T, N_h, N_o, v0, u0, bench, number, input_neurons, hidden_neurons, output_neurons, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=False, letter=None):
    #Run(T, v0, u0, bench, number, input_neurons, hidden_neurons, output_neurons, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=True, letter=None)
    N_out = len(output_neurons)
    N_hidden = len(hidden_neurons)
    spikes_hidden, spikes_out = CollectSpikes(N_hidden, S_hidden, S_out)

    for i in range(N_out):
        if len(S_out.spiketimes[i]) != N_o:
            return False

    #pudb.set_trace()
    for i in range(N_hidden):
        if len(S_hidden.spiketimes[i]) != N_h:
            return False

    return True

def SetNumSpikes(T, N_h, N_o, v0, u0, bench, number, input_neurons, hidden_neurons, output_neurons, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=False, letter=None):

    dv = 0.2
    i = 0
    done = False

    while done == False: 

        Run(T, v0, u0, bench, number, input_neurons, hidden_neurons, output_neurons, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=True, letter=None)
        done = CheckNumSpikes(T, N_h, N_o, v0, u0, bench, number, input_neurons, hidden_neurons, output_neurons, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=False, letter=None)

        N_hidden = len(hidden_neurons)
        spikes_hidden, spikes_out = CollectSpikes(N_hidden, S_hidden, S_out)
        N_out = len(spikes_out)

        print "SETTING NO. SPIKES "
        print "Spike Times: ", S_hidden.spiketimes, " ", S_out.spiketimes

        hidden_are_set = True
        for i in range(len(hidden_neurons)):
            if len(S_hidden[i]) < N_h:
                ModifyWeights(Sa[i], dv)
                hidden_are_set = False
            elif len(S_hidden[i]) > N_h:
                ModifyWeights(Sa[i], -dv)
                hidden_are_set = False

        if hidden_are_set == True:
            if N_out < N_o:
                ModifyWeights(Sb, dv)
            elif N_out > N_o:
                ModifyWeights(Sb, -dv)
            elif N_out == N_o or i == 100:
                break

        i += 1

def Run(T, v0, u0, bench, number, input_neurons, hidden_neurons, output_neurons, Sa, Sb, M, Mv, Mu, S_in, S_hidden, S_out, train=False, letter=None):

    hidden_neurons.v = v0
    hidden_neurons.u = u0
    hidden_neurons.I = 0
    output_neurons.v = v0
    output_neurons.u = u0
    output_neurons.I = 0

    br.forget(Sa, Sb)
    br.reinit(states=False)
    br.recall(Sa, Sb)

    img, label = ReadImg(number=number, bench=bench, letter=letter)
    spikes = GetInSpikes(img, bench=bench)
    input_neurons.set_spiketimes(spikes)
    hidden_neurons.v = v0
    output_neurons.v = v0
    hidden_neurons.u = u0
    output_neurons.u = u0
    hidden_neurons.I = 0
    output_neurons.I = 0
    br.run(T*br.msecond,report='text')

    return label

def Plot(Mv, number):
    #if number < 2:
    br.plot(211)
    #pudb.set_trace()
    br.plot(Mv.times/br.ms,Mv[0]/br.mvolt, label=str(number))
    br.legend()
    #elif number < 4:
    #    br.subplot(212)
    #    br.plot(Mv.times/br.ms,Mv[0], label='1')
    #    br.legend()
    #print "SAVED!"
    #br.savefig('plot.png')
    br.show()

#def dT_dW(T, v0, u0, bench, number, input_neurons, hidden_neurons, output_neurons, Sa, Sb, M, Mv, Mu, S_in, S_out, train=False, letter=None):
#
#    Run(T, v0, u0, bench, number, input_neurons, hidden_neurons, output_neurons, Sa, Sb, M, Mv, Mu, S_in, S_out, train=False, letter=None)    



"""
def Train(bench, number, input_neurons, hidden_neurons, output_neurons, Sa, Sb, M, Mv, Mu, S_in, S_out, train=False, letter=None):
    br.reinit(states=False)
    br.run(30*br.msecond,report='text')

            if len_d > 0 and n == 0:

                    Sa.w[i_a] = weet
                else:
                    n_c = len(Sb.w[:])
                    i_c = br.randint(n_c)
                    weet = Sb.w[i_c]
                    weet = weet*br.volt + dv*br.mV

                    Sb.w[i_c] = weet

            elif n < 5:
                index = in_spikes[i]
                weet = Sb.w[index]
                if len_l < len_d:
                    weet = weet*br.volt + dv*br.mV
                else:
                    weet = weet*br.volt - dv*br.mV
                Sb.w[index] = weet
                i += 1
            elif n > 4:
                dv = dv / 2
"""
        

