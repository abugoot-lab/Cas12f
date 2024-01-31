import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import palettable.cubehelix as ch


sns.set(context='talk', font_scale=0.5, color_codes=True, palette='deep', style='ticks',
        rc={'mathtext.fontset': 'cm', 'xtick.direction': 'out','ytick.direction': 'out',
            'axes.linewidth': 1.5, 'figure.dpi':100, 'text.usetex':False})



df=pd.read_csv('round5.csv')
label=df['name']
function=df['average']
group=df['Group']
x=pd.read_csv('psa_round5_representation.csv',header=None)
df['negave']=-1*df['average']

pca = PCA(n_components=2)
new=pca.fit_transform(x)
df['PCA1'] = new[:, 0]
df['PCA2'] = new[:, 1]

# sns.scatterplot(x=new[:,0],y=new[:,1],hue=function)
# plt.xlabel('PCA1')
# plt.ylabel('PCA2')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.tight_layout()
# plt.savefig('psa_cluster.pdf')
# plt.show()


# Set the color palette to a heatmap
# Creating a custom colormap for the heatmap
#custom_cmap = LinearSegmentedColormap.from_list("custom_heatmap", ["#FDF5F0", "#AB1E29"])
custom_cmap=ch.cubehelix2_16.mpl_colormap
# Create a scatter plot
plt.figure(figsize=(10, 8))
scatter = sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='negave', style='Group', palette=custom_cmap,s=150, edgecolor='black', linewidth=1)

# Adding title and labels
plt.xlabel('PCA1')
plt.ylabel('PCA2')

# Adding a color bar for the heatmap
norm = plt.Normalize(df['negave'].min(), df['negave'].max())
sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
sm.set_array([])
scatter.figure.colorbar(sm, label='Indel')
plt.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
plt.savefig('summary.pdf',format='pdf')

# Show the plot
plt.show()


# Create a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define the color palette
palette = sns.color_palette("bright", 5)
#plt.rc('font', family='Helvetica', size=7)

Plot each group with PCA components
for i, group in enumerate(df['Group'].unique()):
    group_data = df[df['Group'] == group]
    ax.scatter(group_data['PCA1'], group_data['PCA2'], group_data['average'], label=group, color=palette[i], linewidth=0.5)


# Removing the grid
ax.grid(False)

# Set the viewing angle for better perspective
ax.view_init(elev=20, azim=230)  # Adjust elevation and azimuthal angles

# Adding labels and title
ax.set_xlabel('ESM PCA1')
ax.set_ylabel('ESM PCA2')
ax.set_zlabel('Indel')
ax.set_title('3D Scatter Plot with PCA and Grouped Data')
ax.legend()
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))



