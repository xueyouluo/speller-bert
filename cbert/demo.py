import streamlit as st
import requests

st.set_page_config(layout="wide")

st.title('纠错测试')

left,right = st.columns(2)

left.subheader('输入')
right.subheader('检测结果')
text = left.text_area('',height=400,value='本质品类描述了商品本质所属的最细类别，它聚合了一类商品，承载了用户最终的消费需求，如“搞钙牛奶”、“牛揉干”等。本质品类与类目也是有一定的区别，类目是若干品类的集合，它是抽象后的品类概念，不能够名缺到具体的某类商品品类上，如“乳制品”、“水果”等。')

if left.button('检测'):
    text = requests.post('http://localhost:8505/predict',json={"text":text}).json()
    right.markdown(text['text'],unsafe_allow_html=True)
