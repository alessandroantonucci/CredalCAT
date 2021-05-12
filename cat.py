import os
import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.patches as mpatches
from tikzplotlib import save as tikz_save

#directory = os.listdir(".")
#for filename in directory:
#	print(filename)
n_students = 1024
n_questions = 18

profiles_csv = 'Minimalistic1x2x9.profiles.csv'
profiles = pd.read_csv(profiles_csv,names=['pr'])

head_c = ['s','skill','q','t','sc','a','l0','l1','u0','u1']
head_b = ['s','skill','q','t','sc','a','p0','p1']

c_mode_csv = 'Minimalistic1x2x9.progress.credal-adaptive-mode.eps01.csv'
c_ent_csv = 'Minimalistic1x2x9.progress.credal-adaptive-entropy.eps01.csv'
c_mode2_csv = 'Minimalistic1x2x9.progress.credal-adaptive-mode.csv'
c_ent2_csv = 'Minimalistic1x2x9.progress.credal-adaptive-entropy.csv'
b_ran_csv = 'Minimalistic1x2x9.progress.bayesian-non-adaptive.csv'
b_mode_csv = 'Minimalistic1x2x9.progress.bayesian-adaptive-mode.csv'
b_ent_csv = 'Minimalistic1x2x9.progress.bayesian-adaptive-entropy.csv'
b_pr_csv = 'Minimalistic1x2x9.progress.bayesian-adaptive-pright.csv'
c_pr_csv = 'Minimalistic1x2x9.progress.credal-adaptive-pright.csv'

c_mode = pd.read_csv(c_mode_csv,names=head_c)
c_ent = pd.read_csv(c_ent_csv,names=head_c)
c_mode2 = pd.read_csv(c_mode2_csv,names=head_c)
c_ent2 = pd.read_csv(c_ent2_csv,names=head_c)
c_pr = pd.read_csv(c_pr_csv,names=head_c)
b_ran = pd.read_csv(b_ran_csv,names=head_b)
b_mode = pd.read_csv(b_mode_csv,names=head_b)
b_ent = pd.read_csv(b_ent_csv,names=head_b)
b_pr = pd.read_csv(b_pr_csv,names=head_b)

nn1 = np.zeros(n_questions+1)
nn2 = np.zeros(n_questions+1)
nn3 = np.zeros(n_questions+1)
nn4 = np.zeros(n_questions+1)
nn5 = np.zeros(n_questions+1)
nn6 = np.zeros(n_questions+1)
nn7 = np.zeros(n_questions+1)
nn8 = np.zeros(n_questions+1)
nn9 = np.zeros(n_questions+1)

br1 = np.zeros(n_questions+1)
br2 = np.zeros(n_questions+1)
br3 = np.zeros(n_questions+1)
br4 = np.zeros(n_questions+1)
br5 = np.zeros(n_questions+1)
br6 = np.zeros(n_questions+1)
br7 = np.zeros(n_questions+1)
br8 = np.zeros(n_questions+1)
br9 = np.zeros(n_questions+1)

x = range(n_questions+1)

for s in range(n_students):
    p = profiles['pr'][s]
    d1 = c_ent[c_ent['s']==s]
    d2 = c_ent2[c_ent2['s']==s]
    d3 = c_mode[c_mode['s']==s]
    d4 = c_mode2[c_mode2['s']==s]
    d5 = c_pr[c_pr['s']==s]
    d6 = b_ran[b_ran['s']==s]
    d7 = b_ent[b_ent['s']==s]
    d8 = b_mode[b_mode['s']==s]
    d9 = b_pr[b_pr['s']==s]
    #x1 = range(len(d1['l0']))
    #x2 = range(len(d2['l0']))
    #x3 = range(len(d3['l0']))
    #x4 = range(len(d4['l0']))
    #x5 = range(len(d5['l0']))
    #x6 = range(len(d6['p0']))
    #x7 = range(len(d7['p0']))
    #x8 = range(len(d8['p0']))
    #x9 = range(len(d9['p0']))
    l1 = np.array(d1['l0'] if p else d1['l1'])
    u1 = np.array(d1['u0'] if p else d1['u1'])
    l2 = np.array(d2['l0'] if p else d2['l1'])
    u2 = np.array(d2['u0'] if p else d2['u1'])
    l3 = np.array(d3['l0'] if p else d3['l1'])
    u3 = np.array(d3['u0'] if p else d3['u1'])
    l4 = np.array(d4['l0'] if p else d4['l1'])
    u4 = np.array(d4['u0'] if p else d4['u1'])
    l5 = np.array(d5['l0'] if p else d5['l1'])
    u5 = np.array(d5['u0'] if p else d5['u1'])
    p6 = np.array(d6['p0'] if p else d6['p1'])
    p7 = np.array(d7['p0'] if p else d7['p1'])
    p8 = np.array(d8['p0'] if p else d8['p1'])
    p9 = np.array(d9['p0'] if p else d9['p1'])
    u1 = np.append(u1,[u1[-1] for _ in range(n_questions+1-len(u1))])
    u2 = np.append(u2,[u2[-1] for _ in range(n_questions+1-len(u2))])
    u3 = np.append(u3,[u3[-1] for _ in range(n_questions+1-len(u3))])
    u4 = np.append(u4,[u4[-1] for _ in range(n_questions+1-len(u4))])
    l1 = np.append(l1,[l1[-1] for _ in range(n_questions+1-len(l1))])
    l2 = np.append(l2,[l2[-1] for _ in range(n_questions+1-len(l2))])
    l3 = np.append(l3,[l3[-1] for _ in range(n_questions+1-len(l3))])
    l4 = np.append(l4,[l4[-1] for _ in range(n_questions+1-len(l4))])
    l5 = np.append(l5,[l5[-1] for _ in range(n_questions+1-len(l5))])
    u5 = np.append(u5,[u5[-1] for _ in range(n_questions+1-len(u5))])
    p6 = np.append(p6,[p6[-1] for _ in range(n_questions+1-len(p6))])
    p7 = np.append(p7,[p7[-1] for _ in range(n_questions+1-len(p7))])
    p8 = np.append(p8,[p8[-1] for _ in range(n_questions+1-len(p8))])
    p9 = np.append(p9,[p9[-1] for _ in range(n_questions+1-len(p9))])

    #nn1 += (l1<0.5)
    #nn3 += (l3<0.5)
    nn1 += ((l1+u1)<1.)
    nn3 += ((l3+u3)<1.)
    nn6 += (p6<0.5)
    nn7 += (p7<0.5)
    nn8 += (p8<0.5)
    nn9 += (p9<0.5)
    br1 += l1#+u1)
    br3 += l3#+u3)
    br6 += p6
    br7 += p7
    br8 += p8
    br9 += p9

    if False:
        print(np.array(d1['t']))
        print(np.array(d2['t']))
        print(np.array(d3['t']))
        print(np.array(d4['t']))
        print(np.array(d5['t']))
        print(np.array(d6['t']))
        print(np.array(d7['t']))
        print(np.array(d8['t']))
        print(np.array(d9['t']))

    if False:
        plt.fill_between(x, l1, u1, alpha=0.1, color='red',label='H')
        plt.fill_between(x, l3, u3, alpha=0.1, color='blue',label='mode')
        plt.fill_between(x,l5,u5,alpha=0.2,color='yellow')
        plt.xlabel("Questions")
        plt.ylabel("P(S)")
        plt.xticks(np.arange(n_questions+1))
        plt.plot(x,p6,'--k',label='rand')
        plt.plot(x,p7,'-r',label='H')
        plt.plot(x,p8,'-b',label='mode')
        plt.legend()
        plt.show()

nn1 /= n_students
nn3 /= n_students
nn6 /= n_students
nn7 /= n_students
nn8 /= n_students
nn9 /= n_students
br1 /= n_students
br3 /= n_students
br6 /= n_students
br7 /= n_students
br8 /= n_students
br9 /= n_students

#plt.yscale("log")


#plt.plot(range(len(totals_precise)), totals_precise, color='black', label='ciao')
#plt.fill_between(range(len(totals_precise)), totals_lower, totals_upper, color='black', alpha=.2)
#plt.xlabel("$t$")
#plt.ylabel("$cost$")
# plt.yticks(np.arange(0, 1.01, step=0.25))
fig = plt.figure()
plt.plot(np.arange(n_questions+1), nn1,'r', label='c-h')
plt.plot(np.arange(n_questions+1), nn3,'b', label='c-m')
plt.plot(np.arange(n_questions+1), nn6,'k--', label='rand')
plt.plot(np.arange(n_questions+1), nn7,'r--', label='h')
plt.plot(np.arange(n_questions+1), nn8,'b--', label='m')
plt.plot(np.arange(n_questions+1), nn9,'y--', label='r')
plt.legend()
#plt.show()
plt.savefig("acc.png", dpi=250, bbox_inches="tight",pad_inches=0.02)
tikz_save('acc.tex')

plt.plot(np.arange(n_questions+1), br1,'r', label='c-h')
plt.plot(np.arange(n_questions+1), br3,'b', label='c-m')
plt.plot(np.arange(n_questions+1), br6,'k--', label='rand')
plt.plot(np.arange(n_questions+1), br7,'r--', label='h')
plt.plot(np.arange(n_questions+1), br8,'b--', label='m')
plt.plot(np.arange(n_questions+1), br9,'y--', label='r')
plt.legend()
#plt.show()
tikz_save('brier.tex')
