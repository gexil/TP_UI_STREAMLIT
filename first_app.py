

# https://docs.streamlit.io/
# https://plotly.com/python/plotly-express/

import streamlit as st
import pandas as pd
import numpy as np

import ptvsd
ptvsd.enable_attach(address=('localhost', 5678))
# Only include this line if you always wan't to attach the debugger
#ptvsd.wait_for_attach()


def sum(num1, num2):
    #num1 = num1 + 5665699
    return num1 + num2

def mul(num1, num2):
    return num1 * num2


if __name__ == '__main__':
    # Title
    st.title("My Streamlit first app")
    
    # Header
    st.header("My Test")
    # Subheader
    st.subheader("Go")
    # Text
    st.text("For a simple text")

    frame_form = st.form(key='sum form')
    frame_num1 = frame_form.number_input('Enter first number: ', value=1)
    frame_num2 = frame_form.number_input('Enter first number: ', value=2)
    frame_submit = frame_form.form_submit_button('Sum')

    
    if frame_submit:
        frame_result = sum(frame_num1, frame_num2)
        st.write(f'result : {frame_result}')

    #### SIDE Bar #####################################################################
    st.sidebar.header("Function")
    st.sidebar.text("Multiplication")

    sidebar_form = st.sidebar.form(key='Multiplication form')
    sidebar_num1 = sidebar_form.number_input('Enter first number: ', value=3)
    sidebar_num2 = sidebar_form.number_input('Enter first number: ', value=4)
    sidebar_submit = sidebar_form.form_submit_button('Mul')

    
    if sidebar_submit:
        sidebar_result = mul(sidebar_num1, sidebar_num2)
        st.sidebar.write(f'result : {sidebar_result}')

    # Input text
    end = int(st.sidebar.text_input('Input:', '10'))
    
    # Slider
    slider_output =  st.sidebar.slider('slider', 0, end, 0)
    st.sidebar.write(f'slider value : {slider_output}')

    ######################################################################### 

    
    # 2 Columns #############################################################
    left_column, right_column = st.columns(2)
    # Left
    with left_column:
         # Data table
        st.write("A data table:")
        st.write(pd.DataFrame({
            'first column': [1, 2, 3, 4],
            'second column': [10, 20, 30, 40]
        }))

    with right_column:
        chosen = st.radio(
            'Sorting hat',
            ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
        st.write(f"You are in {chosen} house!")
            
        dataframe = np.random.randn(5, 5)
        st.dataframe(dataframe)
    #########################################################################


    # Graphic Part #########################################################
    
    # Plotly
    
    import plotly.graph_objs as go
    import plotly.figure_factory as ff

    st.title("Plotly examples")

    st.header("Chart with two lines")

    trace0 = go.Scatter(x=[1, 2, 3, 4], y=[10, 15, 13, 17])
    trace1 = go.Scatter(x=[1, 2, 3, 4], y=[16, 5, 11, 9])
    data = [trace0, trace1]
    st.write(data)

    
    st.header("Fancy density plot")

    # Add histogram data
    x1 = np.random.randn(200) - 2
    x2 = np.random.randn(200)
    x3 = np.random.randn(200) + 2

    # Group data together
    hist_data = [x1, x2, x3]

    group_labels = ["Group 1", "Group 2", "Group 3"]

    # Create distplot with custom bin_size
    fig = ff.create_distplot(hist_data, group_labels, bin_size=[0.1, 0.25, 0.5])

    # Plot
    st.plotly_chart(fig)
    


    # Matplotlib
    st.header("Matplotlib chart in Plotly")

    import matplotlib.pyplot as plt

    f = plt.figure()
    arr = np.random.normal(1, 1, size=100)
    plt.hist(arr, bins=20)

    st.plotly_chart(f)

    ### 3D plot

    st.header("3D plot")

    x, y, z = np.random.multivariate_normal(np.array([0, 0, 0]), np.eye(3), 400).transpose()

    trace1 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            size=12,
            color=z,  # set color to an array/list of desired values
            colorscale="Viridis",  # choose a colorscale
            opacity=0.8,
        ),
    )

    data = [trace1]
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
    fig = go.Figure(data=data, layout=layout)

    st.write(fig)

    #########################################################################



    


   

    