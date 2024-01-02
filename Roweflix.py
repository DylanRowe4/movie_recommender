import os, json
import streamlit as st
from streamlit_modal import Modal
import streamlit.components.v1 as components
from st_click_detector import click_detector
from movie_recommenders.MovieFinder import MovieRecommender

#instantiate our movie recommender class
recommender = MovieRecommender()

#page configuration
st.set_page_config(page_title='Roweflix', layout='wide')

#set style and css hacks
style = """
<style>
.stApp {
background-color: #90cea1;
background-image: linear-gradient(to bottom right, #90cea1, 50%, #01b4e4);
}
div.block-container {
padding-top:2rem;
}
.diagonal-gradient {
}
img {
height: 160px;
width: 200px;
border-radius: 5px;
border: 5px solid white;
}
div {
background-color: transparent;
}
p {
text-size: 110%;
color: white;
}
p.modal {
text-size: 120%;
color: black;
}
p.title {
font-size: 220%;
text-align: center;
color: white;
font-weight: bold;
}
[data-testid="stExpander"] {
height: 240px;
border-style: solid;
border-radius: 12px;
border-color: white;
overflow-x: scroll;
}
details {
overflow-x:scroll;
}
[data-testid="stExpanderDetails"] {
width: 1500px;
}
[data-testid="stExpanderToggleIcon"] {
display: none;
}
div[data-modal-container='true'] > div:first-child {
width: 80%;
position: absolute;
top: 1%;
left: 10%;
right: 10%;
}
div[data-modal-container='true'] > div:first-child > div:first-child > div:first-child {
width: 100%;
overflow-y: scroll;
max-height: 400px;
overflow-x: hidden;
}
footer {
visibility: hidden;
}
footer:after {
content:'This application uses TMDB and the TMDB APIs but is not endorsed, certified, or otherwise approved by TMDB.'; 
visibility: visible;
display: block;
position: relative;
#background-color: red;
padding: 5px;
top: 2px;
}
div.stButton > button:first-child {
background-color: transparent;
border-color: white;
}
.tooltip {
position: relative;
display: inline-block;
cursor: pointer;
}
.tooltip .tooltiptext {
visibility: hidden;
width: 400px;
height: 150px;
background-color: black;
color: white;
border-radius: 5px;
z-index: 10;
padding: 10px 10px;
left:95%;
line-height: 16px;
position: absolute;
overflow-y: scroll;
}
.tooltip:hover .tooltiptext {
visibility: visible;
opacity: 0.8;
}
</style>
"""
st.write(style, unsafe_allow_html=True)

st.write("<br>", unsafe_allow_html=True)
st.write("<p class='title'>Roweflix</p>", unsafe_allow_html=True)

#search text
search_text = st.text_input('Please input movie title or description for search.')

img_style = "style='height: 160px; width: 120px; border-radius: 5px; border: 5px solid white;'"
#actions to take when button clicked
if search_text:
    #search for all genre recommendations
    movies, genres = recommender.search_movies(search_text, 10)
    #store movie titles to remove from other searches
    main_recommendations = [movie['title'] for movie in movies]
    
    #show results from main search
    with st.expander("Top Recommendations", expanded=True):
        #create column for each movie
        rec_col = st.columns(len(movies))
        
        for i in range(len(movies)):
            
            #add each movie in new column as a button
            with rec_col[i]:
                #create a tooltip to display over each image
                tooltip_string = f"""
                <u><strong>Movie:</strong></u><br>{movies[i]['title']}
                <br><br>
                <u><strong>Genre:</u></strong><br>{', '.join(movies[i]['genres'])}
                <br><br>
                <u><strong>Rating:</u></strong><br>{movies[i]['vote_average']}
                ({movies[i]['vote_count']:,} reviews)
                <br><br>
                <u><strong>Overview:</u></strong><br>{movies[i]['overview']}<br><br>
                """
                html = f"""
                <div class="tooltip">
                <img {img_style} src='{movies[i]['poster_path']}'/>
                    <span class="tooltiptext">{tooltip_string}</span></div>"""
                #show tooltip and image in streamlit
                st.markdown(html, unsafe_allow_html=True)
    
    #create a new row for each genre of the original movie
    for genre in sorted(genres):
        genre_movies, genres_list = recommender.search_movies(search_text, 10, genres=genre,
                                                              remove_movie_names=main_recommendations)
        
        #show results from each genre
        with st.expander(genre, expanded=True):

            #create column for each movie
            genre_col = st.columns(len(movies))

            for i in range(len(genre_movies)):

                #add each movie in new column as a button
                with genre_col[i]:
                    #create a tooltip to display over each image
                    tooltip_string = f"""
                    <u><strong>Movie:</strong></u><br>{genre_movies[i]['title']}
                    <br><br>
                    <u><strong>Genre:</u></strong><br>{', '.join(genre_movies[i]['genres'])}
                    <br><br>
                    <u><strong>Rating:</u></strong><br>{genre_movies[i]['vote_average']}
                    ({genre_movies[i]['vote_count']:,} reviews)
                    <br><br>
                    <u><strong>Overview:</u></strong><br>{genre_movies[i]['overview']}<br>
                    """
                    html = f"""
                    <div class="tooltip">
                    <img {img_style} src='{genre_movies[i]['poster_path']}'/>
                        <span class="tooltiptext">{tooltip_string}</span></div>"""
                    #show tooltip and image in streamlit
                    st.markdown(html, unsafe_allow_html=True)