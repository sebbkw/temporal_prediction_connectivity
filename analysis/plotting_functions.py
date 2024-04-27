from matplotlib import pyplot as plt

def format_plot (ax=None, fontsize=12):
    if ax is None:
        ax = plt.gca()
        
    ax.figure.set_size_inches(6, 4)
    ax.figure.set_dpi(100)
        
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    
    ax.xaxis.get_label().set_fontsize(fontsize)
    ax.yaxis.get_label().set_fontsize(fontsize)
    
    try:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
    except:
        pass
    
    ax.legend(frameon=False, prop={'size': fontsize})
    
    
def save_plot(fig_n, file_name):
    plt.savefig(f'./figures/figure{fig_n}/{file_name}.pdf', bbox_inches='tight')
    