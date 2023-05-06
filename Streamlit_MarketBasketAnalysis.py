# Libraries used
import streamlit as st
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import math
from streamlit_option_menu import option_menu
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder



# Setting page icon, title and layout
st.set_page_config(
    page_title="Streamlit: Studying Customer Behavior Through Product Analytics",
    page_icon="📊",
    layout="wide",
)
st.markdown(f'<h1 style="color:#494949;font-size:50px;">{"📊   Studying Customer Behavior Through Product Analytics"}</h1>', unsafe_allow_html=True)



# naming pages to shuffle between
page = st.selectbox('Pages ⬇️',('Home Page','Data & Automatic Reporting','Visualization Dashboard', 'Market Basket Analysis','Quantity Prediction'))



# Upload CSV data sidebar
with st.sidebar.subheader('Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Please Upload Here", type=["csv"])



# Page 1: Home page where I included an image and brief introduction of Hallab 
if page == "Home Page":
    image = "https://www.hallab.com.lb/media/wysiwyg/hallab-slider-home/DAOU9590.jpg"
    st.image(image, use_column_width = True)
    st.header("Hallab 1881")
    st.subheader("Since 1881, and originating from the ancestral Lebanese city of Tripoli, Hallab 1881 has been a pioneer in the world of Lebanese sweets. Today, “Kasr El Helou” is considered one of Tripoli’s most renowned landmarks, visited by thousands of people from all parts of the globe.")
    st.subheader("Further to maintaining the tradition of preserving the authenticity of Lebanese sweets with “Hallab Traditional”, Hallab 1881 has also striven for diversification with a range of sugar-free sweets and a specialized division for occidental sweets that offers a rich and wide ranging selection of cakes and chocolates. The “1881 Hallab” restaurant in Kasr El Helou Tripoli offers a daily assortment of appetizing “Plats du jour” to choose from and features comprehensive banqueting and reception equipment and facilities.")
    st.subheader("Hallab 1881 currently operates 12 branches in various cities and Lebanese towns. The company has also expanded into the GCC with branches in Saudi Arabia and Kuwait, and will soon boast a strong duty free presence at main airports in the region. Hallab 1881 reinforces its in-house service with mail deliveries covering the entire Lebanese region and catering to international orders via DHL- delivering within 72 hours.")



# Spaces so that upon scrolling down nothing shows if the dataset is still not uploaded
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")



# Page 2: Dataset is shown along automatic report generated to give information on datset and statistics on every column in the dataset with final visuals and insights
if page == "Data & Automatic Reporting":
    if uploaded_file is not None:
        @st.cache
        def load_csv():
            csv = pd.read_csv(uploaded_file, encoding = 'ISO-8859-1')
            return csv
        df = load_csv()
        pr = ProfileReport(df, explorative=True, dark_mode=True)
        st.header('*Dataset Used for Market Basket Analysis:*')
        st.write(df)
        st.write('---')
        st.header('*Automatic Pandas Profiling Report*')
        st_profile_report(pr)
    else:
        st.info('Awaiting for CSV file to be uploaded!')



# Page 3: Visual shared from tableau public where I included the dashboard having insights 
if page == "Visualization Dashboard":
   
    def main():
        html_temp = """<div class='tableauPlaceholder' id='viz1683398772819' style='position: relative'><noscript><a href='#'><img alt='Main ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fi&#47;FinalVisualizationDashboard_16833958526660&#47;Main&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='FinalVisualizationDashboard_16833958526660&#47;Main' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fi&#47;FinalVisualizationDashboard_16833958526660&#47;Main&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1683398772819');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1200px';vizElement.style.height='677px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1200px';vizElement.style.height='677px';} else { vizElement.style.width='100%';vizElement.style.height='1127px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
        components.html(html_temp,width=1400, height=900)
    if __name__ == "__main__":
        main()


# Retreiving dataframe from the uploaded file
df = pd.read_csv(uploaded_file, encoding = 'ISO-8859-1')

#Cleaning Data
df = df.dropna(subset=['INVOICENUMBER'])
df = df[df['QTY'] >= 0]
df['QTY'] = df['QTY'].apply(math.ceil)



# Page 4: Market Basket Analysis
if page == "Market Basket Analysis":

    st.markdown("""<strong>What is Market Basket Analysis?</strong>""",unsafe_allow_html=True)

    # Writing an explanation of market basket analysis
    st.write("Market Basket Analysis is a technique used by retailers and businesses to understand how products are often purchased together by customers. It involves analyzing sales transaction data to identify which items are frequently bought together and to find patterns in customers' shopping habits.")
    st.write("This can help retailers improve their product offerings, optimize their store layouts and promotions, and increase sales. The insights gained from Market Basket Analysis can be used to identify opportunities for cross-selling, increase customer loyalty, and improve the overall shopping experience for customers.")

    st.write("   ")
    st.write("   ")

    st.markdown("""<strong>Market Basket Analysis by Service Type and Product Type</strong>""",unsafe_allow_html=True)

    service_type = st.selectbox('🕵 Select Service Type:',('Take Away','Delivery','Dine-In','ALL'))

    if service_type == "Take Away":
    
        # Corresponding New Dataframe
        df1 = df[df['MENUDESC'] == 'TAKE AWAY']

        #Grouping by INVOICENUMBER to find all the products each customer purchased
        df_mba1 = df1.groupby(['INVOICENUMBER','ITEM_DESCRIPTION'])['QTY'].sum().unstack().reset_index().fillna(0).set_index('INVOICENUMBER')

        #Performing One Hot Encoding: 1 if the product has been purchased, 0 if not
        def encode_units(x):
            if x <=0:
                return 0
            if x >= 1:
                return 1
        df_mba_encode1= df_mba1.applymap(encode_units)
    
        #Removing orders with less than 2 items
        df_mba_f1 = df_mba_encode1[(df_mba_encode1>0).sum(axis=1)>=2]

        #Creating a frequent items from the basket that have a support above 0.016
        frequent_itemsets_plus1 = apriori(df_mba_f1, min_support=0.016,use_colnames=True).sort_values('support',ascending=False).reset_index(drop=True)
        frequent_itemsets_plus1['length'] = frequent_itemsets_plus1['itemsets'].apply(lambda x: len(x))

        #Generate a dataframe containing the rules and their corresponding metrics.
        rules1 = association_rules(frequent_itemsets_plus1,metric='lift',min_threshold=1).sort_values('lift',ascending=False).reset_index(drop=True)

        #Sorting the dataframe by descending lift value
        rules1 = rules1.sort_values("lift",ascending=False).reset_index(drop= True)

        #Creating a selectbox for Selecting a Product and getting the corresponding recommendation
        buttontype = st.selectbox("🍫 Select Product Type", [set(x) for x in rules1.antecedents], 0)
        buttontype_return1 = set(rules1.loc[rules1.antecedents == buttontype]["consequents"].iloc[0])
        buttontype_return2 = f"{round(rules1.loc[rules1.antecedents == buttontype]['support'].iloc[0]*100, 2)}%"
        buttontype_return3 = f"{round(rules1.loc[rules1.antecedents == buttontype]['confidence'].iloc[0]*100, 2)}%"
        buttontype_return4 = round(rules1.loc[rules1.antecedents == buttontype]["lift"].iloc[0],2)

        st.write("   ")
        st.markdown("""<strong>🔎 Results:</strong>""",unsafe_allow_html=True)
        st.write("   ")

        st.write(f'Customer who buys {buttontype} is likely to buy: {buttontype_return1}')
        st.write(f'{buttontype_return2} of all the transactions under analysis showed that {buttontype} and {buttontype_return1} were purchased together')
        st.write(f'If a customer buys {buttontype} there is a {buttontype_return3} chance that they will buy {buttontype_return1} as well.')
        st.write(f'The purchase of {buttontype_return1} lifts the purchase of {buttontype} by {buttontype_return4} times. ')



    if service_type == "Delivery":

        # Corresponding New Dataframe
        df2 = df[df['MENUDESC'] == 'DELIVERY']



        #Grouping by INVOICENUMBER to find all the products each customer purchased
        df_mba2 = df2.groupby(['INVOICENUMBER','ITEM_DESCRIPTION'])['QTY'].sum().unstack().reset_index().fillna(0).set_index('INVOICENUMBER')

        #Performing One Hot Encoding: 1 if the product has been purchased, 0 if not
        def encode_units(x):
            if x <=0:
                return 0
            if x >= 1:
                return 1
        df_mba_encode2= df_mba2.applymap(encode_units)
    
        #Removing orders with less than 2 items
        df_mba_f2 = df_mba_encode2[(df_mba_encode2>0).sum(axis=1)>=2]

        #Creating a frequent items from the basket that have a support above 0.038
        frequent_itemsets_plus2 = apriori(df_mba_f2, min_support=0.038,use_colnames=True).sort_values('support',ascending=False).reset_index(drop=True)
        frequent_itemsets_plus2['length'] = frequent_itemsets_plus2['itemsets'].apply(lambda x: len(x))

        #Generate a dataframe containing the rules and their corresponding metrics.
        rules2 = association_rules(frequent_itemsets_plus2,metric='lift',min_threshold=1).sort_values('lift',ascending=False).reset_index(drop=True)

        #Sorting the dataframe by descending lift value
        rules2 = rules2.sort_values("lift",ascending=False).reset_index(drop= True)

        #Creating a selectbox for Selecting a Product and getting the corresponding recommendation
        buttontype = st.selectbox("🍫 Select Product Type", [set(x) for x in rules2.antecedents], 0)
        buttontype_return1 = set(rules2.loc[rules2.antecedents == buttontype]["consequents"].iloc[0])
        buttontype_return2 = f"{round(rules2.loc[rules2.antecedents == buttontype]['support'].iloc[0]*100, 2)}%"
        buttontype_return3 = f"{round(rules2.loc[rules2.antecedents == buttontype]['confidence'].iloc[0]*100, 2)}%"
        buttontype_return4 = round(rules2.loc[rules2.antecedents == buttontype]["lift"].iloc[0],2)

        st.write("   ")
        st.markdown("""<strong>🔎 Results:</strong>""",unsafe_allow_html=True)
        st.write("   ")

        st.write(f'Customer who buys {buttontype} is likely to buy: {buttontype_return1}')
        st.write(f'{buttontype_return2} of all the transactions under analysis showed that {buttontype} and {buttontype_return1} were purchased together')
        st.write(f'If a customer buys {buttontype} there is a {buttontype_return3} chance that they will buy {buttontype_return1} as well.')
        st.write(f'The purchase of {buttontype_return1} lifts the purchase of {buttontype} by {buttontype_return4} times. ')



    if service_type == "Dine-In":

        # Corresponding New Dataframe
        df3 = df[df['MENUDESC'] == 'TABLES']



        #Grouping by INVOICENUMBER to find all the products each customer purchased
        df_mba3 = df3.groupby(['INVOICENUMBER','ITEM_DESCRIPTION'])['QTY'].sum().unstack().reset_index().fillna(0).set_index('INVOICENUMBER')

        #Performing One Hot Encoding: 1 if the product has been purchased, 0 if not
        def encode_units(x):
            if x <=0:
                return 0
            if x >= 1:
                return 1
        df_mba_encode3= df_mba3.applymap(encode_units)
    
        #Removing orders with less than 2 items
        df_mba_f3 = df_mba_encode3[(df_mba_encode3>0).sum(axis=1)>=2]

        #Creating a frequent items from the basket that have a support above 0.037
        frequent_itemsets_plus3 = apriori(df_mba_f3, min_support=0.037,use_colnames=True).sort_values('support',ascending=False).reset_index(drop=True)
        frequent_itemsets_plus3['length'] = frequent_itemsets_plus3['itemsets'].apply(lambda x: len(x))

        #Generate a dataframe containing the rules and their corresponding metrics.
        rules3 = association_rules(frequent_itemsets_plus3,metric='lift',min_threshold=1).sort_values('lift',ascending=False).reset_index(drop=True)

        #Sorting the dataframe by descending lift value
        rules3 = rules3.sort_values("lift",ascending=False).reset_index(drop= True)

        #Creating a selectbox for Selecting a Product and getting the corresponding recommendation
        buttontype = st.selectbox("🍫 Select Product Type", [set(x) for x in rules3.antecedents], 0)
        buttontype_return1 = set(rules3.loc[rules3.antecedents == buttontype]["consequents"].iloc[0])
        buttontype_return2 = f"{round(rules3.loc[rules3.antecedents == buttontype]['support'].iloc[0]*100, 2)}%"
        buttontype_return3 = f"{round(rules3.loc[rules3.antecedents == buttontype]['confidence'].iloc[0]*100, 2)}%"
        buttontype_return4 = round(rules3.loc[rules3.antecedents == buttontype]["lift"].iloc[0],2)

        st.write("   ")
        st.markdown("""<strong>🔎 Results:</strong>""",unsafe_allow_html=True)
        st.write("   ")

        st.write(f'Customer who buys {buttontype} is likely to buy: {buttontype_return1}')
        st.write(f'{buttontype_return2} of all the transactions under analysis showed that {buttontype} and {buttontype_return1} were purchased together')
        st.write(f'If a customer buys {buttontype} there is a {buttontype_return3} chance that they will buy {buttontype_return1} as well.')
        st.write(f'The purchase of {buttontype_return1} lifts the purchase of {buttontype} by {buttontype_return4} times. ')

    if service_type == "ALL":

        #Grouping by INVOICENUMBER to find all the products each customer purchased
        df_mba = df.groupby(['INVOICENUMBER','ITEM_DESCRIPTION'])['QTY'].sum().unstack().reset_index().fillna(0).set_index('INVOICENUMBER')

        #Performing One Hot Encoding: 1 if the product has been purchased, 0 if not
        def encode_units(x):
            if x <=0:
                return 0
            if x >= 1:
                return 1
        df_mba_encode= df_mba.applymap(encode_units)
    
        #Removing orders with less than 2 items
        df_mba_f = df_mba_encode[(df_mba_encode>0).sum(axis=1)>=2]

        #Creating a frequent items from the basket that have a support above 0.037
        frequent_itemsets_plus = apriori(df_mba_f, min_support=0.037,use_colnames=True).sort_values('support',ascending=False).reset_index(drop=True)
        frequent_itemsets_plus['length'] = frequent_itemsets_plus['itemsets'].apply(lambda x: len(x))

        #Generate a dataframe containing the rules and their corresponding metrics.
        rules = association_rules(frequent_itemsets_plus,metric='lift',min_threshold=1).sort_values('lift',ascending=False).reset_index(drop=True)

        #Sorting the dataframe by descending lift value
        rules = rules.sort_values("lift",ascending=False).reset_index(drop= True)

        #Creating a selectbox for Selecting a Product and getting the corresponding recommendation
        buttontype = st.selectbox("🍫 Select Product Type", [set(x) for x in rules.antecedents], 0)
        buttontype_return1 = set(rules.loc[rules.antecedents == buttontype]["consequents"].iloc[0])
        buttontype_return2 = f"{round(rules.loc[rules.antecedents == buttontype]['support'].iloc[0]*100, 2)}%"
        buttontype_return3 = f"{round(rules.loc[rules.antecedents == buttontype]['confidence'].iloc[0]*100, 2)}%"
        buttontype_return4 = round(rules.loc[rules.antecedents == buttontype]["lift"].iloc[0],2)

        st.write("   ")
        st.markdown("""<strong>🔎 Results:</strong>""",unsafe_allow_html=True)
        st.write("   ")

        st.write(f'Customer who buys {buttontype} is likely to buy: {buttontype_return1}')
        st.write(f'{buttontype_return2} of all the transactions under analysis showed that {buttontype} and {buttontype_return1} were purchased together')
        st.write(f'If a customer buys {buttontype} there is a {buttontype_return3} chance that they will buy {buttontype_return1} as well.')
        st.write(f'The purchase of {buttontype_return1} lifts the purchase of {buttontype} by {buttontype_return4} times. ')



# Page 5: Quantity Prediction 
if page == "Quantity Prediction":

    st.markdown("""<strong>This is a machine learning model that predicts the quantity, based on Service and Product Types (Model Status: In-Progress Model!)</strong>""",unsafe_allow_html=True)

    # Writing an explanation of market basket analysis
    st.write("This code uses a machine learning model called Random Forest Regressor to predict the quantity of a certain item based on the service type and product type. The model is trained on a dataset that contains information about different items and their quantities. To make a prediction, the user enters the prodcut type and the service type, and the model uses this information to estimate the quantity. Theis is a test model to be improved and trained on more data in the future, this model's accuracy is evaluated using R^2 score, which measures how well the model fits the data and it is 23.03% accurate.")

    st.write("   ")

    # Preprocessing the data
    onehot_enc = OneHotEncoder()
    X = onehot_enc.fit_transform(df[['ITEM_DESCRIPTION', 'MENUDESC']])
    y = df['QTY']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating the Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Creating the Streamlit app
    def app():
        # Creating the input form for the user
        menudesc = st.text_input('🕵 Enter the Service Type:')
        item_description = st.text_input('🍫 Enter the Product Type:')
        submit_button = st.button('Predict Quantity')
    
        # Predicting the QTY on form submission
        if submit_button:
            input_data = onehot_enc.transform([[item_description, menudesc]])
            qty_prediction = model.predict(input_data)[0]

            st.write("   ")
            st.markdown("""<strong>🔎 Results:</strong>""",unsafe_allow_html=True)
            st.write(f'Upon {menudesc} the predicted order quantity of {item_description} is {qty_prediction:.0f} KG')

            # Evaluating the model accuracy
            score = model.score(X_test, y_test)
            st.write(f'The model accuracy on this test set is {score*100:.2f}%')

    if __name__ == '__main__':
        app()