import pandas as pd
import plotly.express as px
import dash
import dash_html_components as html
import dash_core_components as dcc

#read airline data into 
airline_data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/airline_data.csv', 
                            encoding = "ISO-8859-1",
                            dtype={'Div1Airport': str, 'Div1TailNum': str, 
                                   'Div2Airport': str, 'Div2TailNum': str})
#randomly sample 500 data points for the purpose of this lab
data = airline_data.sample(n = 500, random_state = 42)

#Pie chart creation
fig = px.pie(data, values = 'Flights',names = 'DistanceGroup',
             title = 'Distance group proportion by flights')

#create a dash application
app = dash.Dash(__name__)
#create an outer division using html.Div
#add title 

app.layout = html.Div(children = [html.H1('Airline Dashboard',
                                            style = {'textAlign':'center',
                                                    'color':'#503D35',
                                                    'font-size':40}),
                                html.P('Proprtion of distance group (250mile distance interval group_by flight.',
                                        style = {'textAlign':'center','color':'#F57241'}),
                                dcc.Graph(figure=fig),])

if __name__ == '__main__':
    app.run_server()