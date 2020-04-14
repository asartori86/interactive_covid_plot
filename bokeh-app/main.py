import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv',parse_dates=['data'])

from bokeh.plotting import figure, output_notebook, show, curdoc
from bokeh.models import CDSView, GroupFilter, ColumnDataSource, CategoricalColorMapper
from bokeh.transform import factor_cmap
from bokeh.palettes import Paired, Spectral3, GnBu3, Turbo256, viridis, turbo, Category20,cividis,magma,plasma,Category10_4,Category10_9,Category10,Set1
from bokeh.models import HoverTool
from bokeh.models.widgets import TextInput
from bokeh.layouts import column 

tw = TextInput(title='List of cities (e.g., trieste treviso milano roma)', value='')
hover_tool = HoverTool(tooltips=[
        ('City', '@denominazione_provincia'),
        ('Data','@data{%F}'),
        ('Totale casi', '@totale_casi',)],
                           formatters={'@data': 'datetime'})
p = figure(x_axis_type='datetime', x_axis_label='data m/d', y_axis_label='totale casi',
           tools=[hover_tool,'crosshair'])
cities = ['Trieste']
sub = df[df.denominazione_provincia.isin(cities)]
cds = ColumnDataSource(data=sub)
category_map = CategoricalColorMapper(factors=cities,palette=Category20[20])

plot = p.circle(x='data', y='totale_casi', source=cds, 
         color={'field':'denominazione_provincia', 'transform': category_map},
         alpha=0.99, size=4,  legend='denominazione_provincia',
         )
    

p.legend.location='top_left'

    
p_source = plot.data_source

    
def callback (attr, new, old):
    #print(tw.value)
    l_cities = old.lower().split(' ')
    
    cities = [c.capitalize() for c in l_cities]
    sub = df[df.denominazione_provincia.isin(cities)]
    plot.data_source.data = dict(ColumnDataSource(data=sub).data)
    category_map = CategoricalColorMapper(factors=cities,palette=Category20[20])
    plot.glyph.fill_color = {'field':'denominazione_provincia', 'transform': category_map}
    plot.glyph.line_color = {'field':'denominazione_provincia', 'transform': category_map}
    

    
tw.on_change('value',callback)

curdoc().add_root(column(tw,p))




