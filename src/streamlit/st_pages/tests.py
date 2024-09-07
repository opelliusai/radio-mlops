import streamlit as st


def main(title):
    st.text('Fixed width text')
    st.markdown('_Markdown_')  # see #*
    st.caption('Balloons. Hundreds of them...')
    st.latex(r''' e^{i\pi} + 1 = 0 ''')
    st.write('Most objects')  # df, err, func, keras!
    st.write(['st', 'is <', 3])  # see *
    st.title('My title')
    st.header('My header')
    st.subheader('My sub')
    st.code('for i in range(8): foo()')

# * optional kwarg unsafe_allow_html = True
    # Display interactive
    st.button('Hit me')
    # st.data_editor('Edit data', data)
    st.checkbox('Check me out')
    st.radio('Pick one:', ['nose', 'ear'])
    st.selectbox('Select', [1, 2, 3])
    st.multiselect('Multiselect', [1, 2, 3])
    st.slider('Slide me', min_value=0, max_value=10)
    st.select_slider('Slide to select', options=[1, '2'])
    st.text_input('Enter some text')
    st.number_input('Enter a number')
    st.text_area('Area for textual entry')
    st.date_input('Date input')
    st.time_input('Time entry')
    st.file_uploader('File uploader')
    st.download_button('On the dl', data)
    st.camera_input("一二三,茄子!")
    st.color_picker('Pick a color')

    st.write("Display text")
    st.text('Fixed width text')
    st.markdown('_Markdown_')  # see #*
    st.caption('Balloons. Hundreds of them...')
    st.latex(r''' e^{i\pi} + 1 = 0 ''')
    st.write('Most objects')  # df, err, func, keras!
    st.write(['st', 'is <', 3])  # see *
    st.title('My title')
    st.header('My header')
    st.subheader('My sub')
    st.code('for i in range(8): foo()')

    # * optional kwarg unsafe_allow_html = True

    # st.dataframe(my_dataframe)
    # st.table(data.iloc[0:10])
    st.json({'foo': 'bar', 'fu': 'ba'})
    st.metric(label="Temp", value="273 K", delta="1.2 K")

    ##
    col1, col2 = st.columns(2)
    col1.write('Column 1')
    col2.write('Column 2')

    # Three columns with different widths
    col1, col2, col3 = st.columns([3, 1, 1])
    # col1 is wider

    # Using 'with' notation:
    with col1:
        st.write('This is column 1')

    # Insert containers separated into tabs:
    tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
    tab1.write("this is tab 1")
    tab2.write("this is tab 2")

    # control flow

    # Stop execution immediately:
    st.stop()
    # Rerun script immediately:
    st.experimental_rerun()

    # Display progress
    # Show a spinner during a process
    with st.spinner(text='In progress'):
        time.sleep(3)
        st.success('Done')

    # Show and update progress bar
    bar = st.progress(50)
    time.sleep(3)
    bar.progress(100)

    st.balloons()
    st.snow()
    st.toast('Mr Stay-Puft')
    st.error('Error message')
    st.warning('Warning message')
    st.info('Info message')
    st.success('Success message')
    st.exception(e)

    # Placeholders
    # Replace any single element.
    element = st.empty()
    element.line_chart(...)
    element.text_input(...)  # Replaces previous.

    # Insert out of order.
    elements = st.container()
    elements.line_chart(...)
    st.write("Hello")
    elements.text_input(...)  # Appears above "Hello".

    st.help(pandas.DataFrame)
    st.get_option(key)
    st.set_option(key, value)
    st.set_page_config(layout='wide')
    st.experimental_show(objects)
    st.experimental_get_query_params()
    st.experimental_set_query_params(**params)

    # Group multiple widgets:
    with st.form(key='my_form'):
        username = st.text_input('Username')
        password = st.text_input('Password')
        st.form_submit_button('Login')

    # You can also use "with" notation:
    with tab1:
        st.radio('Select one:', [1, 2])
