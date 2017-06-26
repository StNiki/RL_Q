import numpy as np
import pickle

import matplotlib.pylab as plt
plt.style.use('ggplot')
log = pickle.load(open("thetaGR.p", "rb"))
print(log.shape)

l=log.T

print(l[:,499])

i=0
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_title("Weights Learning Curve")
ax.legend(loc=2,prop={'size':8})
ax.set_xlabel("Episode")
ax.set_ylabel("Weights")
#ax.set_yscale('log')
for w in l:
	i+=1
	st="{0}".format(i)
	ax.plot(w)
#ax.plot(xx,e[0,:], 'r',xx,e[1,:],'g',xx,e[2,:], 'b',xx,e[3,:],'m',xx,e[4,:],'c',xx,e[5,:], 'y',xx,e[6,:], 'k',xx,e[7,:],'g--',xx,e[8,:],'m--' )
print(l.shape)
fig2 = plt.figure(figsize=(24, 12))
ax2 = fig2.add_subplot(1, 1, 1)
ax2.set_title("Final Weights")
ax2.legend(loc=2,prop={'size':8})
ax2.set_xlabel("Features")
ax2.set_ylabel("Weights")

ax2.bar(np.linspace(1,20,19),l[:,500])

fig.tight_layout()
fig.savefig('w_GR.pdf',dpi=200)
fig2.tight_layout()
fig2.savefig('b_GR.pdf',dpi=200)


