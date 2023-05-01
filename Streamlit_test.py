#Libraries used
import streamlit as st
import pandas as pd
import numpy as np
import math
from streamlit_option_menu import option_menu
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori
import streamlit.components.v1 as components

#Setting page width to wide
st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center;'>Luanatic Witty Little Things</h1>", unsafe_allow_html=True)
st.write(" ")
st.write(" ")

# Create a menu
page = option_menu(
    menu_title=None,
    options=["Overview","Visualization Dashboard","Market Basket Analysis by Product Type","Market Basket Analysis by Product Theme"],
    orientation="horizontal"
    )

#First Page
if page == "Overview":

    #Column Split
    col1, col2 = st.columns([3, 1])
    with col1:
        #Title
        #st.image('Header.png', width=900)
        st.write("Luanatic have a variety of products that they sell such as mugs, espresso mugs, posters, keychains, bibs, tote bags, magnets, notebooks, stickers, pouches, bottles, and more...") 
        st.write("Moreover, each product they have, comes in different themes. For example, Mug Habibi, Bib Khalto Rocks, Poster Alo Beirut, Espresso Mug Shaffeh, Keychain Supermama, etc...")
        st.write("In this project, the main purpose was to analyze Luanatic customer purchase behavior on the website and create a recommendation system to optimize the cross-selling process and in turn increase profitability.")

    #with col2:
        #Image
        #st.image('luanaticstreamlit.png')

#Second Page
if page == "Visualization Dashboard":
   
    def main():
        html_temp = """<div class='tableauPlaceholder' id='viz1682867654710' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ca&#47;CapstoneViz_16827936456050&#47;FINALDASHBOARD&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='CapstoneViz_16827936456050&#47;FINALDASHBOARD' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ca&#47;CapstoneViz_16827936456050&#47;FINALDASHBOARD&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1682867654710');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1400px';vizElement.style.height='900px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1400px';vizElement.style.height='900px';} else { vizElement.style.width='100%';vizElement.style.height='900px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
        components.html(html_temp,width=1400, height=900)
    if __name__ == "__main__":
        main()

#Third Page
if page == "Market Basket Analysis by Product Type":

    data = pd.read_excel('Cleaned_MarketBasket_2023.xlsx')
    df = data 

    st.markdown("""<strong>Market Basket Analysis by Product Type</strong>""",unsafe_allow_html=True
    )
    #Writing an explanation of market basket analysis
    st.write("This market basket application analyzes consumer purchases by product type in order to recommend product B, if the customer purchases product A")
    st.write("   ")
    st.write("   ")

    #Cleaning Data
    df = df.dropna(subset=['INVOICENUMBER'])
    df = df[df['QTY'] >= 0]
    df['QTY'] = df['QTY'].apply(math.ceil)


#Market Basket Analysis Product Type

    #Grouping by Order ID to find all the products each customer purchased
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

    #Creating a frequent items from the basket that have a support above 0.04
    frequent_itemsets_plus = apriori(df_mba_f, min_support=0.017,use_colnames=True).sort_values('support',ascending=False).reset_index(drop=True)
    frequent_itemsets_plus['length'] = frequent_itemsets_plus['itemsets'].apply(lambda x: len(x))

    #Generate a dataframe containing the rules and their corresponding metrics.
    rules = association_rules(frequent_itemsets_plus,metric='lift',min_threshold=1).sort_values('lift',ascending=False).reset_index(drop=True)

    #Sorting the dataframe by descending lift value
    rules = rules.sort_values("lift",ascending=False).reset_index(drop= True)

    #Creating a selectbox for Selecting a Product and getting the corresponding recommendation
    buttontype = st.selectbox("Select Product Type", rules.antecedents, 0)
    buttontype_return1 = rules.loc[rules.antecedents == buttontype]["consequents"].iloc[0]
    buttontype_return2 = rules.loc[rules.antecedents == buttontype]["support"].iloc[0]
    buttontype_return3 = rules.loc[rules.antecedents == buttontype]["confidence"].iloc[0]
    buttontype_return4 = rules.loc[rules.antecedents == buttontype]["lift"].iloc[0]
    st.write(f'Customer who buys the selected product is likely to buy: {buttontype_return1}')
    st.write(f'with support of: {buttontype_return2}')
    st.write(f'and confidence of: {buttontype_return3}')
    st.write(f'and lift of: {buttontype_return4}')
