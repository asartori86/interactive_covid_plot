from bokeh.plotting import figure, output_notebook, show, curdoc
from bokeh.models import CDSView, GroupFilter, ColumnDataSource, CategoricalColorMapper
from bokeh.transform import factor_cmap
from bokeh.palettes import Paired, Spectral3, GnBu3, Turbo256, viridis, turbo, Category20,cividis,magma,plasma,Category10_4,Category10_9,Category10,Set1
from bokeh.models import HoverTool
from bokeh.models.widgets import TextInput
from bokeh.layouts import column, row
import itertools
import numpy as np
import pandas as pd

def loc_eval(x, b):
    loc_est = 0
    for i in enumerate(b): loc_est+=i[1]*(x**i[0])
    return(loc_est)


def loess(yvals, alpha=0.7, poly_degree=1):
    #all_data = sorted(zip(data[xvals].tolist(), data[yvals].tolist()), key=lambda x: x[0])
    yvals=yvals.to_numpy()
    xvals=np.arange(len(yvals))
    evalDF = pd.DataFrame(columns=['v','g'])
    n = len(xvals)
    m = n + 1
    q = int(np.floor(n * alpha) if alpha <= 1.0 else n)
    avg_interval = ((max(xvals)-min(xvals))/len(xvals))
    v_lb = min(xvals)-(.5*avg_interval)
    v_ub = (max(xvals)+(.5*avg_interval))
    v = enumerate(np.linspace(start=v_lb, stop=v_ub, num=m), start=1)
    xcols = [np.ones_like(xvals)]
    for j in range(1, (poly_degree + 1)):
        xcols.append([i ** j for i in xvals])
    X = np.vstack(xcols).T
    for i in v:
        iterpos = i[0]
        iterval = i[1]
        iterdists = sorted([(j, np.abs(j-iterval)) for j in xvals], key=lambda x: x[1])
        _, raw_dists = zip(*iterdists)
        scale_fact = raw_dists[q-1]
        scaled_dists = [(j[0],(j[1]/scale_fact)) for j in iterdists]
        weights = [(j[0],((1-np.abs(j[1]**3))**3 if j[1]<=1 else 0)) for j in scaled_dists]
        _, weights      = zip(*sorted(weights,     key=lambda x: x[0]))
        _, raw_dists    = zip(*sorted(iterdists,   key=lambda x: x[0]))
        _, scaled_dists = zip(*sorted(scaled_dists,key=lambda x: x[0]))
        W         = np.diag(weights)
        b         = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ yvals)
        local_est = loc_eval(iterval, b)
        iterDF2   = pd.DataFrame({
                       'v'  :[iterval],
                       'g'  :[local_est]
                       })
        evalDF = pd.concat([evalDF, iterDF2])
    evalDF = evalDF[['v','g']]
    return(evalDF)

def compute_new_cases(df,city):
    import numpy as np
    totale = df.loc[df.denominazione_provincia == city,'totale_casi'].to_numpy()
    new = np.zeros_like(totale)
    new[1:] = np.ediff1d(totale)
    df.loc[df.denominazione_provincia == city, 'new'] = new

df = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv',parse_dates=['data'])
for c in df.denominazione_provincia.unique():
    compute_new_cases(df,c)


tw = TextInput(title='List of cities (e.g., trieste treviso milano roma)', value='')

def make_tot_fig():
    

    hover_tool = HoverTool(tooltips=[
        ('City', '@denominazione_provincia'),
        ('Data','@data{%F}'),
        ('Totale casi', '@totale_casi',)],
        formatters={'@data': 'datetime'})
    # hover_tool.mode='vline'
    tot_fig = figure(x_axis_type='datetime', x_axis_label='data m/d', y_axis_label='totale casi',
           tools=[hover_tool,'crosshair'])


    return tot_fig

def make_plot_tot(tot_fig,df,cities):

    sub = df[df.denominazione_provincia.isin(cities)]
    cds = ColumnDataSource(data=sub)
    category_map = CategoricalColorMapper(factors=cities,palette=Category20[20])

    plot = tot_fig.circle(x='data', y='totale_casi', source=cds, 
                          color={'field':'denominazione_provincia', 'transform': category_map},
                          alpha=0.99, size=4,  legend_field='denominazione_provincia',
                          )
    

    tot_fig.legend.location='top_left'


##### nuovi casi ####

def make_new_cases():

    newc = figure(x_axis_type='datetime', x_axis_label='data m/d', y_axis_label='nuovi casi')

    return newc

def make_plot_newc(newc,df,cities):
    category_map = CategoricalColorMapper(factors=cities,palette=Category20[20])
    pn = newc.cross(x='data', y='new', source=df.loc[df.denominazione_provincia.isin(cities),:], 
                    color={'field':'denominazione_provincia', 'transform': category_map},
                    alpha=0.99, size=4,  legend_field='denominazione_provincia',
                    )
    newc.legend.location='top_left'

    for name, color in zip(cities, itertools.cycle(Category20[20])):
    
        evalDF = loess(df.loc[df.denominazione_provincia==name,'new'], alpha=0.9, poly_degree=1)
     
        newc.line(x=df.loc[df.denominazione_provincia==name,'data'].to_numpy(),y=evalDF['g'].to_numpy()[1:],  color=color, legend_label='Trend - '+name)





layout = column(tw,row(make_tot_fig(),make_new_cases()))



    

def callback (attr, old, new):
    l_cities = new.lower().split(' ')
    cities = [c.capitalize() for c in l_cities]
    sub = df[df.denominazione_provincia.isin(cities)]

    tf = make_tot_fig()
    nc = make_new_cases()
    make_plot_tot(tf,sub,cities)
    make_plot_newc(nc,sub,cities)

    layout.children[1].children[0:2] = [tf,nc]
    # layout.children[1].children[0] = tf
    # layout.children[1].children[1] = nc

    
tw.on_change('value',callback)

# curdoc().add_root(tw)
curdoc().add_root(layout)
# curdoc().add_root(pn)




