import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(5,5))
ax = plt.gca()

# removing the default axis on all sides:
for side in ['bottom','right','top','left']:
    ax.spines[side].set_visible(False)

# removing the axis ticks
plt.xticks([]) # labels
plt.yticks([])
ax.xaxis.set_ticks_position('none') # tick markers
ax.yaxis.set_ticks_position('none')

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
plt.style.use('seaborn-white')
plt.ylabel('y', fontsize=20); plt.xlabel('x', fontsize=20)

# get width and height of axes object to compute
# matching arrowhead length and width
dps = fig.dpi_scale_trans.inverted()
bbox = ax.get_window_extent().transformed(dps)
width, height = bbox.width, bbox.height

# manual arrowhead width and length
hw = 1./20.*(ymax-ymin)
hl = 1./20.*(xmax-xmin)
lw = 1. # axis line width
ohg = 0.3 # arrow overhang

# compute matching arrowhead length and width
yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

# draw x and y axis
ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw,
         head_width=hw, head_length=hl, overhang = ohg,
         length_includes_head= True, clip_on = False)

ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw,
         head_width=yhw, head_length=yhl, overhang = ohg,
         length_includes_head= True, clip_on = False)
	
np.random.seed(2015)
N = 50
val1 = np.random.normal(loc=0.2, scale=0.05,size=N)
val2 = np.random.normal(loc=0.8, scale=0.04, size=N)

num_samples = 10**4
idx = np.random.choice(np.arange(len(val1)), num_samples)

plt.plot(val1[idx], val2[idx], 'k.')


np.random.seed(2015)
N = 25
val1 = np.random.normal(loc=0.8, scale=0.015,size=N)
val2 = np.random.normal(loc=0.5, scale=0.02, size=N)

num_samples = 10**4
idx = np.random.choice(np.arange(len(val1)), num_samples)

plt.plot(val1[idx], val2[idx], 'k.')


np.random.seed(2015)
N = 50
val1 = np.random.normal(loc=0.55, scale=0.05,size=N)
val2 = np.random.normal(loc=0.2, scale=0.06, size=N)

num_samples = 10**4
idx = np.random.choice(np.arange(len(val1)), num_samples)

plt.plot(val1[idx], val2[idx], 'k.')


plt.plot([0.5, 0.7, 0.2],[0.9, 0.6, 0.2], 'k.')

plt.xlim([0,1]); plt.ylim([0,1])

plt.text(0.2,0.22,'$o_1$', fontsize=15)
plt.text(0.7,0.615,'$o_2$', fontsize=15)
plt.text(0.5,0.915,'$o_3$', fontsize=15)
plt.text(0.8,0.545,'$O_4$', fontsize=15)
plt.text(0.54,0.345,'$N_1$', fontsize=15)
plt.text(0.22,0.91,'$N_2$', fontsize=15)

plt.savefig('figures/anomalies.png', dpi=300, transparent=True, bbox_inches='tight')
plt.show()