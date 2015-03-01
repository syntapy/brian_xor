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

def ModifyWeights(S, dv, rand=0):
    n = len(S.w[:])
    if rand == 0:
        for i in range(n):
            weet = S.w[i]
            weet = weet*br.volt + dv*br.mV
            S.w[i] = weet
    else:
        for i in range(n):
            weet = S.w[i]
            weet = weet*br.volt + dv*br.rand()*br.mV
            S.w[i] = weet

def CollectSpikes(N_hidden, S_hidden, S_out):
    spikes_hidden = []
    for i in range(N_hidden):
        spikes_hidden.append([])

    spikes_out = S_out.spikes

    return spikes_hidden, spikes_out

def CheckNumSpikes(layer, T, N_h, N_o, v0, u0, bench, number, \
        input_neurons, liquid_in, liquid_hidden, liquid_out, liquid_neurons, \
        hidden_neurons, output_neurons, \
        Si, Sl, Sa, Sb, M, Mv, Mu, \
        S_in, S_hidden, S_out, train=False, letter=None):

    N_out = len(output_neurons)
    N_hidden = len(hidden_neurons)
    spikes_hidden, spikes_out = CollectSpikes(N_hidden, S_hidden, S_out)

    if layer == 0 or layer == -1:
        for i in range(N_hidden):
            if len(S_hidden.spiketimes[i]) != N_h:
                return False

    elif layer == 1 or layer == -1:
        for i in range(N_out):
            if len(S_out.spiketimes[i]) != N_o:
                return False

    return True

def SetNumSpikes(layer, T, N_h, N_o, v0, u0, bench, number, input_neurons, \
        liquid_in, liquid_hidden, liquid_out, liquid_neurons, \
        hidden_neurons, output_neurons, \
        Si, Sl, Sa, Sb, M, Mv, Mu, \
        S_in, S_hidden, S_out, train=False, letter=None):

    dv = 0.02
    min_dv = 0.001
    i = 0
    last = 0 # -1, 0, 1: left, neither, right

    print "layer = ", layer
    if layer == 0:
        dv = 0.02
        right_dv = True
    else:
        dv = 0.0005
        div = 0
    modified = True
    j = 0
    while modified == True:
        modified = False
        print "\tj = ", j
        j += 1
        for number in range(4):
            done = False
            print "\t\tNumber = ", number, "\t"
            while done == False:
                #pudb.set_trace()
                Run(T, v0, u0, bench, number, input_neurons, liquid_in, \
                        liquid_hidden, liquid_out, liquid_neurons, \
                        hidden_neurons, output_neurons, \
                        Si, Sl, Sa, Sb, M, Mv, Mu, \
                        S_in, S_hidden, S_out, train=True, letter=None)

                done = CheckNumSpikes(layer, T, N_h, N_o, v0, u0, bench, number, input_neurons, \
                                liquid_in, liquid_hidden, liquid_out, liquid_neurons, \
                                hidden_neurons, output_neurons, \
                                Si, Sl, Sa, Sb, M, Mv, Mu, \
                                S_in, S_hidden, S_out, train=False, letter=None)

                N_hidden = len(hidden_neurons)
                spikes_hidden, spikes_out = CollectSpikes(N_hidden, S_hidden, S_out)
                N_out = len(spikes_out)

                hidden_are_set = True
                if layer == 0:
                    print "\t\t\t\t."
                    for i in range(len(hidden_neurons)):
                        if len(S_hidden[i]) < N_h:
                            ModifyWeights(Sa[i], dv, 0)
                            modified = True
                        elif len(S_hidden[i]) > N_h:
                            ModifyWeights(Sa[i], -dv, 0)
                            modified = True
                else:
                    print "\t\t\t\tdiv = ", div
                    print S_hidden.spiketimes
                    if N_out < N_o:
                        """
                        if last == 1:
                            if dv > min_dv:
                                dv = dv / 2
                                div += 1
                        elif last == 0:
                            last = -1
                        """
                        ModifyWeights(Sb, 0*dv, 0)
                        modified = True
                    elif N_out > N_o:
                        ModifyWeights(Sb, -0*dv, 0)
                        modified = True
                        #last = 1
                    elif N_out == N_o:
                        done = True
                        """
                        if dv > min_dv:
                            ModifyWeights(Sb, -dv, 1)
                            modified = True
                            last = 1
                        else:
                        """

def Run(T, v0, u0, bench, number, input_neurons, \
        liquid_in, liquid_hidden, liquid_out, liquid_neurons, \
        hidden_neurons, output_neurons, \
        Si, Sl, Sa, Sb, M, Mv, Mu, \
        S_in, S_hidden, S_out, train=False, letter=None):

    pudb.set_trace()
    br.restore()
    #br.forget(Si, Sl, Sa, Sb)
    #br.reinit(states=False)
    #br.recall(Si, Sl, Sa, Sb)

    print Sa.w[:]

    img, label = ReadImg(number=number, bench=bench, letter=letter)
    spikes = GetInSpikes(img, bench=bench)
    input_neurons.set_spiketimes(spikes)

    liquid_out.v = v0
    liquid_out.u = u0
    liquid_out.I = 0
    liquid_out.ge = 0

    liquid_hidden.v = v0
    liquid_hidden.u = u0
    liquid_hidden.I = 0
    liquid_hidden.ge = 0

    liquid_in.v = v0
    liquid_in.u = u0
    liquid_in.I = 0
    liquid_in.ge = 0

    liquid_neurons.v = v0
    liquid_neurons.u = u0
    liquid_neurons.I = 0
    liquid_neurons.ge = 0

    hidden_neurons.v = v0
    hidden_neurons.u = u0
    hidden_neurons.I = 0
    hidden_neurons.ge = 0

    output_neurons.v = v0
    output_neurons.u = u0
    output_neurons.I = 0
    output_neurons.ge = 0

    br.run(T*br.msecond)

    return label

def PrintSpikes(T, N_h, N_o, v0, u0, bench, number, input_neurons, \
        liquid_in, liquid_hidden, liquid_out, liquid_neurons, \
        hidden_neurons, output_neurons, \
        Si, Sl, Sa, Sb, M, Mv, Mu, \
        S_in, S_hidden, S_out, train=False, letter=None):
    
        for number in range(4):
            Run(T, v0, u0, bench, number, input_neurons, liquid_in, \
                    liquid_hidden, liquid_out, liquid_neurons, \
                    hidden_neurons, output_neurons, \
                    Si, Sl, Sa, Sb, M, Mv, Mu, \
                    S_in, S_hidden, S_out, train=True, letter=None)

            print "Number = ", number
            print "\t", S_hidden.spiketimes
            print "\t", S_out.spiketimes
            print "---------------------------------------------------------------------------------"
            print "---------------------------------------------------------------------------------"

def Plot(Mv, number):
    br.plot(211)
    br.plot(Mv.times/br.ms,Mv[0]/br.mvolt, label=str(number))
    br.legend()
    br.show()

def SaveConnectivity(Si, Sl, Sa, Sb, number):
    Si.save_connectivity("weights/Si-" + str(number) + ".txt")
    Sl.save_connectivity("weights/Sl-" + str(number) + ".txt")
    Sa.save_connectivity("weights/Sa-" + str(number) + ".txt")
    Sb.save_connectivity("weights/Sb-" + str(number) + ".txt")

def LoadConnectivity(Si, Sl, Sa, Sb, number):
    Si.load_connectivity("weights/Si-" + str(number) + ".txt")
    Sl.load_connectivity("weights/Sl-" + str(number) + ".txt")
    Sa.load_connectivity("weights/Sa-" + str(number) + ".txt")
    Sb.load_connectivity("weights/Sb-" + str(number) + ".txt")

"""
def SetHiddenNumSpikes(T, N_h, N_o, v0, u0, bench, number, input_neurons, \
        liquid_in, liquid_hidden, liquid_out, liquid_neurons, \
        hidden_neurons, output_neurons, \
        Si, Sl, Sa, Sb, M, Mv, Mu, \
        S_in, S_hidden, S_out, train=False, letter=None):

    dv = 0.02
    min_dv = 0.0001
    i = 0
    last = 0 # -1, 0, 1: left, neither, right

    for layer in range(2):
        print "layer = ", layer
        if layer == 0:
            dv = 0.02
            right_dv = True
        else:
            dv = 0.02
            div = 0
        modified = True
        j = 0
        while modified == True:
            modified = False
            print "\tj = ", j
            j += 1
            for number in range(4):
                done = False
                print "\t\tNumber = ", number, "\t"
                while done == False:
                    #pudb.set_trace()
                    Run(T, v0, u0, bench, number, input_neurons, liquid_in, \
                            liquid_hidden, liquid_out, liquid_neurons, \
                            hidden_neurons, output_neurons, \
                            Si, Sl, Sa, Sb, M, Mv, Mu, \
                            S_in, S_hidden, S_out, train=True, letter=None)

                    done = CheckNumSpikes(layer, T, N_h, N_o, v0, u0, bench, number, input_neurons, \
                                    liquid_in, liquid_hidden, liquid_out, liquid_neurons, \
                                    hidden_neurons, output_neurons, \
                                    Si, Sl, Sa, Sb, M, Mv, Mu, \
                                    S_in, S_hidden, S_out, train=False, letter=None)

                    N_hidden = len(hidden_neurons)
                    spikes_hidden, spikes_out = CollectSpikes(N_hidden, S_hidden, S_out)
                    N_out = len(spikes_out)

                    hidden_are_set = True
                    if layer == 0:
                        print "\t\t\t\t."
                        for i in range(len(hidden_neurons)):
                            if len(S_hidden[i]) < N_h:
                                ModifyWeights(Sa[i], dv)
                                modified = True
                            elif len(S_hidden[i]) > N_h:
                                ModifyWeights(Sa[i], -dv)
                                modified = True
                    else:
                        print "\t\t\t\tdiv = ", div
                        if N_out < N_o:
                            if last == 1:
                                if dv > min_dv:
                                    dv = dv / 2
                                    div += 1
                            elif last == 0:
                                last = -1
                            ModifyWeights(Sb, dv)
                            modified = True
                        elif N_out > N_o:
                            ModifyWeights(Sb, -dv)
                            modified = True
                            last = 1
                        elif N_out == N_o:
                            if dv > min_dv:
                                ModifyWeights(Sb, -dv)
                                modified = True
                                last = 1
                            else:
                                done = True
"""

