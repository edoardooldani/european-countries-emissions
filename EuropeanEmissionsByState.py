import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import itertools

import warnings
warnings.filterwarnings("ignore")

# Librerie "nuove" utilizzate
import folium
import webbrowser
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


# HEADER
# Country Code,Country Name,EEA Sector,EEA Sub-Sector,EEA Activity,NFR Code,NFR Name,
# Parent Sector Code,Pollutant,Year,Emissions,Unit,Notation Key,Format Name

# ---------------------- Map of the countries with their pollution ---------------------- #

def EmissionsDf(df):
  emissionDf = df.groupby('Country Name', as_index=False).sum()
  emissionDf.drop(columns= ['Year'], axis=1)

  return emissionDf


def emissionsMap(df):
  emissionDf = EmissionsDf(df)

  m = folium.Map(tiles= 'Stamen Terrain',location=[48, 0], zoom_start=4, min_zoom=2)

  country_shapes = 'european-union-countries.geojson'

  folium.Choropleth(
    geo_data=country_shapes,
    min_zoom=2,
    name='European countries emissions',
    data=emissionDf,
    columns= ['Country Name', 'Emissions'],
    key_on='feature.properties.name',
    fill_color='OrRd',
    nan_fill_color='black',
    line_opacity=1.3,
    legend_name='European countries emissions',
  ).add_to(m)

  folium.LayerControl().add_to(m)
  m.save("European countries emissions.html")
  webbrowser.open("European countries emissions.html")


# ---------------------- Trend of the 4 most polluting countries ---------------------- #

def emissionPerYear(state, ax):
  statedf = df.loc[df['Country Name'] == state]
  statedf = statedf.groupby('Year', as_index=False).sum()
      
  ax.plot(statedf['Year'].tolist(),statedf['Emissions'].tolist())

  return ax


def topCountriesPlot(df):

  topCountries = list(EmissionsDf(df).sort_values(by=['Emissions'], ascending = False)['Country Name'][:4])
  
  fig, axs = plt.subplots(2, 2)

  axs[0, 0] = emissionPerYear(topCountries[0], axs[0, 0])
  axs[0, 0].set_title(f'{topCountries[0]} emissions per year')

  axs[0, 1] = emissionPerYear(topCountries[1], axs[0, 1])
  axs[0, 1].set_title(f'{topCountries[1]} emissions per year')

  axs[1, 0] = emissionPerYear(topCountries[2], axs[1, 0])
  axs[1, 0].set_title(f'{topCountries[2]} emissions per year')

  axs[1, 1] = emissionPerYear(topCountries[3], axs[1, 1])
  axs[1, 1].set_title(f'{topCountries[3]} emissions per year')

  plt.show()


def totalEmissionPieChart(df):
  plt.figure(figsize=(8,6))
  plt.suptitle('Total emissions by countries in 1990 and in 2015')

  plt.subplot(1,2,1)
  plt.title('1990')

  tmpDf = df.loc[df['Year'] == 1990]
  tmpDf = tmpDf.groupby(by=['Country Name'], as_index=False).sum()

  emissionDict = dict(zip(tmpDf['Country Name'], tmpDf['Emissions']))
  emissionDict = dict(sorted(emissionDict.items(), key=lambda item: item[1], reverse=True))

  slicedDict = dict(itertools.islice(emissionDict.items(), 0, 8))

  otherDictTmp = dict(itertools.islice(emissionDict.items(), 8, None))
  otherDict = {'Other': sum(otherDictTmp.values())}

  slicedDict.update(otherDict)

  plt.pie(slicedDict.values())
  plt.legend(slicedDict.keys(), loc="upper left")

  

  plt.subplot(1,2,2)
  plt.title('2015')

  tmpDf = df.loc[df['Year'] == 2015]
  tmpDf = tmpDf.groupby(by=['Country Name'], as_index=False).sum()

  emissionDict = dict(zip(tmpDf['Country Name'], tmpDf['Emissions']))
  emissionDict = dict(sorted(emissionDict.items(), key=lambda item: item[1], reverse=True))

  slicedDict = dict(itertools.islice(emissionDict.items(),0, 8))

  otherDictTmp = dict(itertools.islice(emissionDict.items(), 8, None))
  otherDict = {'Other': sum(otherDictTmp.values())}

  
  slicedDict.update(otherDict)

  plt.pie(slicedDict.values())
  plt.legend(slicedDict.keys(), loc="upper left")


  plt.tight_layout()
  plt.show()



# ---------------------- Sectors in Europe ---------------------- #

def sectorPlotDiff(df):
  
  yearList = df['Year'].drop_duplicates().tolist()

  topCountries = list(EmissionsDf(df).sort_values(by=['Emissions'], ascending = False)['Country Name'][:8])
  df = df.loc[df['Country Name'].isin(topCountries)]

  yearListCut = sorted([x for x in yearList if x % 5 == 0])

  sectorDf = df.filter(items=['EEA Sector','Year','Emissions', 'Country Name'])
  sectorDf = sectorDf.loc[sectorDf['Year'].isin(yearListCut)]
  sectorDf = sectorDf.groupby(['EEA Sector','Year', 'Country Name'],as_index=False).sum()
  sectorDf = sectorDf.drop(sectorDf[sectorDf.Emissions <= 0].index)

  sectorDf = sectorDf.loc[sectorDf['Year'] > 1989]

  col_wrap = int(len(topCountries)/2)
  fig = px.scatter(sectorDf, y = 'EEA Sector', x ='Emissions',
                   color='Country Name', size='Emissions', facet_col='Country Name', facet_col_wrap = col_wrap,
                   hover_name="Country Name", log_x=True, size_max=40,
                   animation_frame="Year", animation_group="Emissions",
                   category_orders={'Year': yearListCut,
                                    'Country Name': topCountries}) 
    

  fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
  fig.for_each_trace(lambda t: t.update(name=t.name.split("=")[0]))

  fig.update_layout(
        title={
          'text': "Countries emissions per Sector"
        },
        font=dict(size=12),
        showlegend=False,
        sliders = [dict(font=dict(size= 10))]   
        )
  fig.show()
    

def subSectorHist(df, subsect):
  sectorDf = df.loc[df['EEA Sub-Sector'] == subsect]
  plt.figure(figsize=(10,6))
  plt.suptitle(f'{subsect} emissions by countries in 1990 and in 2015')

  plt.subplot(1,2,1)
  sectorDfTmp = sectorDf.loc[sectorDf['Year'] == 1990]
  sectorDfTmp = sectorDfTmp.groupby(by=['Country Name', 'EEA Sub-Sector'], as_index=False).sum()
  plt.bar(sectorDfTmp['Country Name'], sectorDfTmp['Emissions'])

  plt.xticks(rotation=90)

  plt.subplot(1,2,2)
  sectorDfTmp = sectorDf.loc[sectorDf['Year'] == 2015]
  sectorDfTmp = sectorDfTmp.groupby(by=['Country Name', 'EEA Sub-Sector'], as_index=False).sum()

  plt.bar(sectorDfTmp['Country Name'], sectorDfTmp['Emissions'])

  plt.xticks(rotation=90)

  plt.tight_layout()
  plt.show()
  

# ---------------------- Focus on Italy ---------------------- #

def sunBurstPlot(df, country='Italy'):
  
  countryDf = df.loc[df['Country Name'] == country]

  fig =px.sunburst(
      countryDf,
      path=['Country Name', 'EEA Sub-Sector', 'EEA Activity'],
      values='Emissions', color='EEA Sector'
  )
  fig.update_layout(showlegend=True)
  fig.show()


def percReducedEmission(df, country='Italy', sect='EEA Sector'):

  countryDf = df.loc[df['Country Name'] == country]
  countryDf = countryDf.dropna()

  
  yearList = countryDf['Year'].drop_duplicates().tolist()
  yearList = sorted([x for x in yearList if x % 5 == 0])

  sectorList = countryDf[sect].drop_duplicates().tolist()

  countryDf = countryDf.loc[countryDf['Year'].isin(yearList)]
  countryDf = countryDf.groupby(by=[sect, 'Year'], as_index=False).sum()

  bubbleChart = go.Figure()
  barChart = go.Figure()

  percRedDict = {}
  for sector in sectorList:

    
    yearList = countryDf['Year'].drop_duplicates().tolist()
    yearList = sorted([x for x in yearList if x % 5 == 0])

    sectorDf = countryDf.loc[countryDf[sect] == sector]
    sectorDf = sectorDf.sort_values(by=['Year'])

    perc = sectorDf['Emissions'].pct_change() * -1

    # Bar Chart
    barChart.add_trace(go.Bar(
        x=yearList,
        y= perc,
        base=0,
        name=sector
         ))

    # Bubble Chart
        
    s = perc * 100

    
    s.dropna(inplace=True)
    s = list(s)
    

    for index in range(len(s)):

      if s[index] < 0:
        s[index] = 0
        
    if not s:
      continue
    
    sizeref = 2.* max(s) /(60**2)

    percRedDict[sector] = list(zip(yearList, s))[1:]
    
    bubbleChart.add_trace(go.Scatter(
        x=yearList[1:], y=s,
        name=sector, marker_size=s
        ))

    bubbleChart.update_traces(mode='markers', marker=dict(sizemode='area', sizeref=sizeref, line_width=2))


  bubbleChart.update_layout(
    title="Italy percentage of emissions reduction by years",
    xaxis_title="Years",
    yaxis_title="Reduction percentage",
    showlegend=True,
    legend_title=sect,
    font=dict(
        size=18
    ),
    legend=dict(font=dict(size= 12))
  )

  barChart.update_layout(
    title="Italy percentage of emissions reduction by years",
    xaxis_title="Years",
    yaxis_title="Reduction percentage",
    legend_title=sect
  )

  barChart.show()
  bubbleChart.show()


def subSectDiff(df, country, subsect):
  countryDf = df.loc[(df['Country Name'] == country) & (df['EEA Sub-Sector'] == subsect)]

  yearList = countryDf['Year'].drop_duplicates().tolist()
  
  activityDf = countryDf.groupby(by=['EEA Activity', 'Year'], as_index=False).sum()
  
  sns.lineplot(data=activityDf, x='Year', y='Emissions', hue='EEA Activity')
  plt.title(f'{country} {subsect} emissions')
  plt.show()



# ---------------------- Prepare dataframe ---------------------- #

df = pd.read_csv('EuropeEmission.csv')
df.drop(df.loc[df['Country Name'] == 'EU28'].index, inplace = True)

#Alleggerisco il dataframe filtrando solo le colonne interessanti
df = df.filter(items=['Country Name','EEA Sector','EEA Sub-Sector','EEA Activity','Year','Emissions', 'Unit'])


#Aggiusto l'unità di misura ed elimino la colonna unità
adjustEmission = df.loc[df['Unit'] == 'Gg (1000 tonnes)'].filter(items=['Emissions', 'Unit'])
adjustEmission['Emissions'] = adjustEmission['Emissions']*1000
df['Emissions'] = adjustEmission['Emissions']
df = df.drop(columns= ['Unit'], axis=1)

#Aggiusto il nome di uno stato membro per farlo combaciare col nome presente nella libreria folium
df['Country Name'] = df['Country Name'].replace('Czechia', 'Czech Republic')

df = df[df.Emissions.apply(lambda x: str(x) != 'nan')]



# ---------------------- Function calls ---------------------- #

#Europe

emissionsMap(df)
topCountriesPlot(df)
totalEmissionPieChart(df)
sectorPlotDiff(df)
subSectorHist(df, 'Road-transport')

#Italy

sunBurstPlot(df, 'Italy')
percReducedEmission(df, 'Italy', 'EEA Sector')
percReducedEmission(df, 'Italy', 'EEA Sub-Sector')
subSectDiff(df, 'Italy', 'Road-transport')
subSectDiff(df, 'Italy', 'Heavy industry')
