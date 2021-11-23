# #import streamlit as st

# import numpy as np
# import pandas as pd

# st.markdown("""# Crypto Predicto""")

# # Increase per day
# col1, col2, col3 = st.columns(3)
# col1.metric("SPDR S&P 500", "$437.8", "-$1.25")

# # Price  - hist and predction - line chart
# def get_line_chart_data():

#     return pd.DataFrame(
#             np.random.randn(20, 3),
#             columns=['a', 'b', 'c']
#         )

# df = get_line_chart_data()

# st.line_chart(df)


import streamlit as st
import pandas as pd
import plotly.express as px
from pytrends.request import TrendReq
pytrend = TrendReq()


st.set_page_config(layout = "wide")
# df = pd.DataFrame(px.data.gapminder())
st.header("Crypot Predicto")
## Countries
clist = ['Bitcoint','Ethereum','Dogecoin','Monero']
currency = st.selectbox("Select a country:",clist)
col1, col2 = st.columns(2)
pytrend.build_payload(kw_list=[currency])
# Interest by Region
df = pytrend.interest_over_time()


fig = px.line(df.reset_index(), x='date', y=currency) #px.line(df[df['country'] == currency], x = "year", y = "gdpPercap",title = "GDP per Capita")
col1.plotly_chart(fig,use_container_width = True)

col2.subheader("A narrow column with the data")
col2.table(df.sort_values(by=currency, ascending=False))
