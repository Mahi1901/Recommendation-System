from django.contrib import auth, messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from numpy import savez_compressed
from numpy import load
from sklearn.feature_extraction.text import CountVectorizer
from django.contrib.auth.models import User
import speech_recognition as sr
import imdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import sqlite3

global a1
global df

def loginhandler(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(username=username, password=password)
        if user is not None:
            return home(request)
    return render(request, 'App/login.html')


def logout(request):
    logout(request)
    return render(request, 'App/signup.html')


def search(request):
    print('A1 in search : ', a1)
    return render(request, 'App/search.html', {"search": a1})

def getimageurl(dfn, title1):
    #access = imdb.IMDb()
    #search1 = access.search_movie(title)
    #movie_id = search1[0].movieID
    #movie = access.get_movie(movie_id)
    #url = movie['cover url']
    #print("Avaneesh")
    url = dfn[dfn.title == title1]["Url"].values[0]
    #print(url)
    return url

def register(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")
        confirmpass = request.POST.get("confirm_password")
        if confirmpass == password:
            user = User.objects.create_user(username, email, password)
            user.save()
            return render(request, 'App/login.html')

    return render(request, 'App/signup.html')

def home(request):
    print("in home")
    global df
    data = pd.read_csv('App/templates/App/Netflix.csv')
    #    'C:\\Users\\Admin\\PycharmProjects\\pythonProject4\\Recommendation-System\\Project\\App\\templates\\App\\Netflix.csv

    df = data.copy()

    search_languages = request.POST.getlist("languages")
    print(search_languages)
    if len(search_languages)>0:
        matches_regex = "|".join(search_languages)
        matches_bools = data.language.str.contains(matches_regex, regex=True)
        df = data[matches_bools]

    print('reading csv')
    title_list = ["title"]
    df1 = df.copy()
    df2 = df.copy()
    df3 = df.copy()
    global df4
    df4 = df.copy()
    global df5
    df5 = df.copy()
    a = preprocessing(df,df2,df3,df4)
    search_web = request.POST.get("search")
    if search_web:
        #print(search_web)
        a1 = recommendation(df5,search_web, a)
        save_history_title(df2, search_web)
        print('A1 : ', a1)
        return render(request, 'App/search.html', {"search": a1})
    lst = top_recommendation(df2)
    new = new_release(df3)
    comedies=get_genre_wise_list(df3,'Comedies')
    horror=get_genre_wise_list(df3,'Horror')
    country = get_country_wise_list(df2, 'India')
    shape = df.shape

    mydict = {
        "df": df.to_html(),
        "shape": shape,
        "new":new,
        "top":lst,
        "comedies":comedies,
        "horror":horror,
        "country":country
    }
    return render(request, 'App/home.html', mydict)


# Function for combining the values of these columns into a single string.
def combine_features(row):
    return row['title'] + ' ' + row['listed_in'] + ' ' + row['director'] + ' ' + row['cast']

def preprocessing(df,df2,df3,df4):
    df2.dropna(inplace=True)
    df3.dropna(inplace=True)
    df4.dropna(inplace=True)

    dup_bool = df.duplicated(
        ['show_id', 'type', 'title', 'director', 'cast', 'country', 'date_added', 'release_year', 'rating', 'duration',
         'listed_in', 'description', 'ratingdescription', 'user_rating_score', 'user_rating_size'])
    dups = sum(dup_bool)  # by considering all columns.
    features = ['listed_in', 'title', 'cast', 'director']
    df1 = df[['title']].copy()
    df1.dropna(inplace=True)
    df1.head()
    global li1
    li1 = []
    for i, row in df1.iterrows():
        li1.append(row['title'])

    df['title'] = df['title'].str.replace(" ", "")
    global li2
    li2 = []
    for i, row in df.iterrows():
        li2.append(row['title'])
    global li3
    li3 = []
    for i, row in df.iterrows():
        li3.append(row['title'].lower())

    #print("val :", li1)
    #print("key: ", li2)
    df['listed_in'] = df['listed_in'].str.replace(" ", "")
    df['director'] = df['director'].str.replace(" ", "")
    for feature in features:
        df[feature] = df[feature].fillna('')
    df['combined_features'] = df.apply(combine_features, axis=1)
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df['combined_features'])

    start = datetime.now()

    if os.path.isfile('cosine_sim2.npz'):
        print("It is already present in my local repository. Loading...\n\n")
        dict_data = load("cosine_sim2.npz")
        cosine_sim = dict_data['arr_0']
        print("DONE..")
    else:
        print("File is not present in my Local Repository..Creating....\n\n")
        cosine_sim = cosine_similarity(count_matrix)
        print('Saving it into my Local Repository....\n\n')
        savez_compressed("cosine_sim2.npz", cosine_sim)
        print("DONE..\n")

    print(datetime.now() - start)

    return cosine_sim


# Functions to get movie title from movie index.
def get_title_from_index(dfn,ind):
    return dfn[dfn.index == ind]["title"].values[0]


# Functions to get movie index from input movie title.
def get_index_from_title(title1):
    title1 = title1.lower()
    title = title1.replace(" ", "")
    i = 0
    for i in range(0, len(li3)):
        if li3[i] == title:
            break
    return i

def recommendation(dfn, movie, cosine_sim):
    m1 = movie.replace(" ", "")
    try:
        m1 = get_index_from_title(m1)
        similar_movies = list(enumerate(
            cosine_sim[
                m1]))  # We will sort the list similar_movies according to similarity scores in descending order.
        sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:]
        lst = []
        i = 0
        print("Top 10 similar movies to " + movie + " are:\n")
        try:
            url = getimageurl(dfn,movie)
            lst_entry = {
                'title': movie,
                'href': url
            }
            lst.append(lst_entry)
        except Exception as e:
            print(e)
            url = ""

        #lst.append(movie)
        for element in sorted_similar_movies:
            try:
                ind = li2.index(get_title_from_index(dfn,element[0]))
                url = getimageurl(dfn,li1[ind])
                lst_entry = {
                    'title': li1[ind],
                    'href': url
                }
                lst.append(lst_entry)
                i = i + 1
            except:
                url = ""
            if i > 10:
                break
    except:
        lst = None
    return lst

def top_recommendation(df2):
    df2.sort_values(by="user_rating_score", ascending=False)
    # len(df2)
    df2.head(15)
    a = df2['title']
    a = a.tolist()
    lst = []
    for i in range(0, 10):
        try:
            url = getimageurl(df2,a[i])
            lst_entry = {
                'title': a[i],
                'href': url
            }
            lst.append(lst_entry)
        except:
            url = ""
    return lst

def new_release(df3):
    df3.sort_values(by=['date_added'], ascending=False)
    # Recommending the movie_title on the top release_date.
    a = df3['title']
    a = a.tolist()
    lst = []
    for i in range(0, 15):
        try:
            url = getimageurl(df3,a[i])
            lst_entry = {
                'title': a[i],
                'href': url
            }
            lst.append(lst_entry)
        except:
            url=""
    return lst

# Function to get list of the movie_title for the given genre parameter.
def get_genre_wise_list(df3,genre):
    # global df3
    titles = []
    dict = {genre: '1'}
    df3['Value'] = df3['listed_in'].str.split().apply(lambda x: [dict[i] for i in x if i in dict.keys()])
    i = 0
    for ind in df3.index:
        if df3['Value'][ind]:
            if df3['Value'][ind][0] == '1':
                try:
                    #url = getimageurl(df3['title'][ind])
                    lst_entry = {
                        'title': df3['title'][ind],
                        'href': df3['Url'][ind] #url
                    }
                    titles.append(lst_entry)
                    i = i + 1
                except:
                    url=""
        if i > 15:
            break
    return titles

# Function to get list of the movie_title for given country.
def get_country_wise_list(df2, country):
  # df2 contains descending order wise sorted data on column user_rating_score
  titles = []
  dict = {country: '1'}
  # following statement searches all the records with country passed to function in listed_in column
  df2['Value'] = df2['country'].str.split().apply(lambda x: [dict[i] for i in x if i in dict.keys()])
  i=0
  for ind in df2.index:
    if df2['Value'][ind]:
        if df2['Value'][ind][0]=='1':
            try:
                #url = getimageurl(df2['title'][ind])
                lst_entry = {
                  'title': df2['title'][ind],
                  'href': df2['Url'][ind] #url
                }
                titles.append(lst_entry)
                i=i+1
            except:
                url=""
    if i>15:
      break
  return titles

def Movies(request):
    global df
    df4 = df.copy()
    df4.sort_values(by="user_rating_score", ascending=False)
    gk1 = df4.groupby('type')
    dfs = gk1.get_group('Movie')
    an = dfs['title']
    an1 = an.head(10)
    an1 = an1.tolist()
    print(an1)
    a = df4['title']
    a = a.tolist()
    lst_movie = []
    for i in range(0, 21):
        if len(a)>20:
            lst_entry = {
                'title': a[i],
                'href': df4[df4.title == a[i]]["Url"].values[0]
            }
        lst_movie.append(lst_entry)
    return render(request,'App/movies.html',{"movies":lst_movie})

def TvShows(request):
    df5.sort_values(by="user_rating_score", ascending=False)
    gk1 = df5.groupby('type')
    dfs = gk1.get_group('TV Show')
    ans = dfs['title']
    ans1 = ans.head(10)
    ans1 = ans1.tolist()
    a = dfs['title']
    a = a.tolist()
    lst_tv = []
    for i in range(0, 21):
        if len(a)>20:
            lst_entry = {
                'title': a[i],
                'href': df5[df5.title == a[i]]["Url"].values[0]
            }
            lst_tv.append(lst_entry)
    return render(request,'App/tv_shows.html',{'tv_shows':lst_tv})

def movie_recommend(dfn, original_title):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
    matrix = tf.fit_transform(df4['Genre'])
    cosine_similarities = linear_kernel(matrix,matrix)
    movie_title = dfn['Title']
    indices = pd.Series(dfn.index, index=dfn['Title'])
    try:
      idx = indices[original_title]
      sim_scores = list(enumerate(cosine_similarities[idx]))
      sim_scores = sorted(sim_scores, key=lambda x: x[0], reverse=True)
      sim_scores = sim_scores[1:31]
      movie_indices = [i[0] for i in sim_scores]
    except:
      return ""

    a = movie_title.iloc[movie_indices]
    a = a.tolist()
    lst = []
    for i in range(0, 15):
        try:
            #url = getimageurl(a[i])
            lst_entry = {
                'title': a[i],
                'href': dfn[dfn.title == a[i]]["Url"].values[0]
            }
            lst.append(lst_entry)
        except:
            url=""
    return lst

#Functions to create search histroy from input movie title into the sqlite database.
def save_history_title(dfn, title):
  a=dfn[dfn["title"] == title]["listed_in"]
  a=a.tolist()
  if len(a)<=0:
    return False

  genre = a[0]
  if title != "" and genre!=0:
      conn = sqlite3.connect("db.sqlite3")
      cursor = conn.cursor()
      cursor.execute("INSERT INTO `SearchHistory` (Title, Genre) VALUES(?, ?)", (str(title), str(genre)))
      conn.commit()
      cursor.close()
      conn.close()
  return True
