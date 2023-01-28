#!/usr/bin/env python
# coding: utf-8

# In[1]:


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import pandas as pd
import numpy as np
import time
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm.notebook import trange, tqdm
import re
import sqlite3
from datetime import date


# In[2]:


import nltk
from nltk import word_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
import pymorphy2
import torch, numpy
from transformers import BertTokenizer, BertModel, BertConfig


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import plotly.express as px


# # База данных

# In[158]:


class DB_helper():
    
    def __init__(self):
        self.conection = sqlite3.connect('food_database1.db')
        self.cursorObj = self.conection.cursor()
        
    def sql_table(self):  #создаем таблички

        self.cursorObj.execute("CREATE TABLE IF NOT EXISTS recipe(url_id integer PRIMARY KEY, title text, portions integer, cooking_time text, likes integer, dislikes integer,description text, calories integer,squirrels integer,fats integer, carbohydrates integer, process text,add_time date )")
        self.cursorObj.execute("CREATE TABLE IF NOT EXISTS ingridients(id integer PRIMARY KEY AUTOINCREMENT, product text)")
        self.cursorObj.execute("PRAGMA foreign_keys=on")
        self.cursorObj.execute("CREATE TABLE IF NOT EXISTS ingredient_index(recipe_id integer, ingridient_id integer,gramms text, FOREIGN KEY (recipe_id) REFERENCES recipe(url_id),FOREIGN KEY (ingridient_id) REFERENCES ingridients(id))")
        self.cursorObj.execute("CREATE TABLE IF NOT EXISTS tags(id integer PRIMARY KEY AUTOINCREMENT, tag_name text)")
        self.cursorObj.execute("CREATE TABLE IF NOT EXISTS tags_index(recipe_id integer, tag_id integer, FOREIGN KEY (recipe_id) REFERENCES recipe(url_id),FOREIGN KEY (tag_id) REFERENCES tags(id))")
        self.cursorObj.execute("CREATE TABLE IF NOT EXISTS comments(recipe_id integer, com_name text, FOREIGN KEY (recipe_id) REFERENCES recipe(url_id))")
        self.conection.commit()
    
    def __get_current_date(self):
        current_date = date.today()
        timee = str(current_date)
        return timee
    
    def insert_into_recipe(self, data):  #инфа в табличку рецепт

        entities = [v for v in data.values()][:12]
        date_ = self.__get_current_date()
        entities.append(date_)
        entities = tuple(entities)
        existed_urls_id = [x[0] for x in  self.cursorObj.execute("select url_id from recipe").fetchall()] # массив idшек
        if entities[0] in existed_urls_id:
            pass
        else:
            self.cursorObj.execute('INSERT INTO recipe(url_id, title, portions, cooking_time, likes, dislikes,description, calories,squirrels,fats, carbohydrates, process, add_time) VALUES(?, ?, ?, ?, ?, ?,?,?,?,?,?,?,?)', entities)
            self.conection.commit()

    def insert_into_ingridients(self, data):  #инфа в таблиoчку ингредиенты
        
        ingredients = [k for k in data['ingredients'].keys()]
        existed_ingrid = [x[0] for x in  self.cursorObj.execute("select product from ingridients").fetchall()] # массив ингредиентов

        for ing in ingredients:
            if ing in existed_ingrid:
                pass
            else:
                self.cursorObj.execute('INSERT INTO ingridients(product) VALUES(?)', (ing,))
                self.conection.commit() 
        
    def insert_into_ingredient_index(self, data):

        existed_ids = [x for x in  self.cursorObj.execute("select * from ingredient_index").fetchall()]
        id_rec = [row[0] for row in  self.cursorObj.execute('select url_id from recipe').fetchall()]

        for k,v in data['ingredients'].items():
            for row in  self.cursorObj.execute('select * from ingridients').fetchall():
                if row[1] == k and data['url'] in id_rec:
                    entities =(data['url'],row[0],v)
                    if entities in existed_ids:
                        pass
                    else:
                        self.cursorObj.execute('INSERT INTO ingredient_index(recipe_id,ingridient_id,gramms) VALUES(?,?,?)', entities)    
                        self.conection.commit()

    def insert_into_tags(self, data):

        existed_tags = [x[0] for x in self.cursorObj.execute("select tag_name from tags").fetchall()] 
        for tag in data['tags']:
            if tag in existed_tags:
                pass
            else:
                self.cursorObj.execute('INSERT INTO tags(tag_name) VALUES(?)', (tag,))
                self.conection.commit()

    def insert_into_tags_index(self,data):
       
        existed_tags = [x for x in self.cursorObj.execute("select * from tags_index").fetchall()]
        id_rec = [row[0] for row in self.cursorObj.execute('select url_id from recipe').fetchall()]

        for tag in data['tags']:
            for row in self.cursorObj.execute('select * from tags').fetchall():
                if row[1] == tag and data['url'] in id_rec:
                    entities = (data['url'],row[0])

                    if entities in existed_tags:
                        pass
                    else:
                        self.cursorObj.execute('INSERT INTO tags_index(recipe_id, tag_id ) VALUES(?,?)', entities)    
                        self.conection.commit()

    def insert_into_comments(self,data):
        existed_coms = [x for x in self.cursorObj.execute("select * from comments").fetchall()] 
        if len(data['comments']) == 0:
            self.cursorObj.execute('INSERT INTO comments(recipe_id, com_name) VALUES(?,?)', (data['url'], None))
            self.conection.commit() 
        else:
            for com in data['comments']:
                if (data['url'],com) in existed_coms:
                    pass
                else:
                    self.cursorObj.execute('INSERT INTO comments(recipe_id, com_name) VALUES(?,?)', (data['url'], com))
                    self.conection.commit()
                    
    def full_insert(self, data):
        self.insert_into_recipe(data)
        self.insert_into_ingridients(data)
        self.insert_into_ingredient_index(data)
        self.insert_into_tags(data)
        self.insert_into_tags_index(data)
        self.insert_into_comments(data)
        
    def existed_ids(self, link_id):
        id_rec = [row[0] for row in self.cursorObj.execute('select url_id from recipe').fetchall()]
        if link_id in id_rec:
            return True
        else:
            return False

    def get_rec_by_tagname(self, tag):
        return self.cursorObj.execute(f"""SELECT tg.tag_name, title, description, process FROM 'recipe' as rec 
        join tags_index as tin 
        on rec.url_id = tin.recipe_id 
        join tags as tg 
        on tg.id = tin.tag_id 
        where tg.tag_name = '{tag}' limit 2000 """).fetchall()
    
    def get_by_date(self, date_):
        return self.cursorObj.execute(f"""SELECT * FROM 'recipe' as rec  
        where rec.add_time = '{date_}' """).fetchall()
    
    def amount_of_rows(self):
        return len([row[0] for row in self.cursorObj.execute('select url_id from recipe').fetchall()])

    def del_recipe(self, recipe_id):
        self.cursorObj.execute(f'DELETE FROM  ingredient_index where recipe_id ={recipe_id}')
        self.conection.commit() 
        self.cursorObj.execute(f'DELETE FROM  tags_index where recipe_id ={recipe_id}')
        self.conection.commit() 
        self.cursorObj.execute(f'DELETE FROM comments where recipe_id ={recipe_id}')
        self.conection.commit() 
        self.cursorObj.execute(f'DELETE FROM  recipe where url_id ={recipe_id}')
        self.conection.commit() 
    
    def del_ingredient(self, ingr_id):
        self.cursorObj.execute(f'DELETE FROM  ingredient_index where ingridient_id ={ingr_id}')
        self.conection.commit() 
        self.cursorObj.execute(f'DELETE FROM ingridients where id ={ingr_id}')
        self.conection.commit()


# # Краулер

# In[3]:


class Parse():
    def __init__(self,link):
        self.all_links = []
        self.link = link
        self.driver = webdriver.Chrome()
        self.all_recipe_info = []
        
    def __get_links_on_page(self):
        links_on_page = self.driver.find_elements(By.CLASS_NAME,"emotion-1eugp2w") 
        links_on_page = [r.find_element(By.TAG_NAME,"a").get_attribute('href') for r in links_on_page]
        return links_on_page
    
    def __get_id(self,url):
        return int(re.findall('\d+', url)[0]) 
    
    def __get_description(self, soup):
        if soup.find('span', {'class': 'emotion-1x1q7i2'}) == None:
            description = None
        else:
            description = soup.find('span', {'class': 'emotion-1x1q7i2'}).get_text().replace('\xa0',' ').replace('\n',' ')
        return description

    def __get_ingredients(self, soup):
        ingred = {}
        ingredients = soup.find_all('span', {'itemprop': 'recipeIngredient'})
        ingredients = [i.get_text() for i in ingredients]
        grams = soup.find_all('span', {'class': 'emotion-15im4d2'})
        grams = [i.get_text() for i in grams]
        for k in range(len(grams)):
            ingred[ingredients[k]] = grams[k]

        return ingred

    def __get_recipe(self, soup):
        rec = ''
        recipe = soup.find_all('span', {'class': 'emotion-6kiu05'})
        recipe = [i.get_text().replace('\xa0',' ') for i in recipe]
        for i in recipe:
            rec += i
        return rec

    def __get_likes(self, soup):
        likes = soup.find_all('span', {'class': 'emotion-1w5q7lf'})
        likes = [int(i.get_text()) for i in likes[:2]]
        return likes

    def __get_comments(self, soup):
        text_comments = soup.find_all('span', {'class': 'emotion-1m8rn2'})
        text_comments = [i.get_text().replace('\xa0',' ') for i in text_comments]
        return text_comments

    def __get_f_values(self, soup):
        food_values = soup.find_all('div', {'class': 'emotion-8fp9e2'})
        food_values = [int(i.get_text()) for i in food_values]
        return food_values

    def __get_tags(self, soup):
        tags = soup.find_all('div', {'class': 'emotion-zwg7c9'})
        tags = [i.get_text().replace(' ','') for i in tags]
        return tags
    
    def __get_title(self, soup):
        return soup.find('div', {'class': 'emotion-1h7uuyv'}).get_text().replace('\xa0',' ') 
    
    def __get_portions(self, soup):
        return int(soup.find('span', {'itemprop': 'recipeYield'}).get_text()) 
    
    def __get_cooking_time(self, soup):
        return soup.find('div', {'class': 'emotion-my9yfq'}).get_text()
    
    def get_the_recipe_info(self, soup, url):
        recept = {}
        recept["url"] = self.__get_id(url)
        recept["title"] = self.__get_title(soup)
        recept['portions']= self.__get_portions(soup)
        recept['cooking_time'] = self.__get_cooking_time(soup)
        recept['likes'] = self.__get_likes(soup)[0]
        recept['dislikes'] = self.__get_likes(soup)[1]
        recept['description'] = self.__get_description(soup)
        recept['calories'] = self.__get_f_values(soup)[0]
        recept['squirrels'] = self.__get_f_values(soup)[1]
        recept['fats'] = self.__get_f_values(soup)[2]
        recept['carbohydrates'] = self.__get_f_values(soup)[3]
        recept['process'] = self.__get_recipe(soup)
        recept['ingredients'] = self.__get_ingredients(soup)
        recept['tags'] = self.__get_tags(soup)
        recept['comments'] = self.__get_comments(soup)

        return  recept
    
    def parse_main(self, page_start, page_end):
        
        for k in tqdm(range(page_start, page_end + 1)): 
            self.driver.get(self.link + f'?page={k}')
            self.driver.find_elements(By.CLASS_NAME,'emotion-43d4uw')[1].click()
            time.sleep(2)
            links = self.__get_links_on_page()
            for i in links:
                if  i in self.all_links:
                    pass
                else:
                    self.all_links.append(i)
                    
    def get_recipe_data(self):
        
        for i in self.all_links:
            self.driver.get(i)
            time.sleep(2)
            try:
                soup = BeautifulSoup(self.driver.page_source)
                self.all_recipe_info.append(self.get_the_recipe_info(soup,i))
                        
            except AttributeError:
                self.driver.refresh()
                soup = BeautifulSoup(self.driver.page_source)
                self.all_recipe_info.append(self.get_the_recipe_info(soup,i))  
                        
       
            


# In[6]:


# driver = webdriver.Chrome()
# driver.get('https://www.avito.ru/')
# input-input-Zpzc1


# In[120]:


class Main():
    
    def __init__(self, db, parse):
        self.db = db
        self.parse = parse
        
    def insert_recipes_into_db(self):
        for rec in tqdm(self.parse.all_recipe_info):
            if self.db.existed_ids(rec["url"])== True:
                    pass
            else:
                self.db.full_insert(rec)
        


# In[4]:


parse = Parse('https://eda.ru/recepty')
parse.parse_main(1,1)
# parse.get_recipe_data()


# In[164]:


db = DB_helper()
m = Main(db,parse)
m.insert_recipes_into_db()


# In[117]:


parse.all_recipe_info


# # Аналитика

# In[122]:


class Normalization():
    def __init__(self, recipes):
        self.morph = pymorphy2.MorphAnalyzer()
        self.stopwords = stopwords.words("russian")
        self.recipes = recipes
        self.normal_text = []
        
    def __remove_chars_from_text(self, text, chars):
        return "".join([ch for ch in text if ch not in chars])
    
    def __remove_stopwords(self, text, stopwords):
        return [ch for ch in text if ch not in stopwords]
    
    def __transform_dbtuple(self,recipes_from_db):
        recipe_prepared = []
        for rec in recipes_from_db:
            t = ''
            for i in rec:
                if i == None:
                    i = ''
                t = t + ' ' + i
            recipe_prepared.append(self.__remove_chars_from_text(t.replace('.', ' '), '0123456789():,\–.—«»xa!%').lower())
        return recipe_prepared
    
    def __full_transform_dbtuples(self):
        prepared = []
        for el in self.recipes:
            prepared.append(self.__transform_dbtuple(el))
        return prepared
            
    def __transform_to_words(self):
        self.stopwords.extend(['нами', 'многое','другой','затем','та','самая', 'которая'])
        mass_of_words = []
        for n in range(len(self.__full_transform_dbtuples())):
            for t in self.__full_transform_dbtuples()[n]:
                mass_of_words.append(self.__remove_stopwords(word_tokenize(t), self.stopwords))
        return mass_of_words
    
    def words_normalize(self):
        for word in self.__transform_to_words():
            normal_words = " ".join([self.morph.parse(w)[0].normal_form for w in word]).replace('драник ', 'завтраки ', 1).replace('борщ ', 'суп ', 1).replace('винегрет ', 'салаты ', 1).replace('вино', 'напитки ', 1)
            self.normal_text.append(normal_words)
    


# In[165]:


tags = ['Драники','Выпечка и десерты','Основные блюда','Борщ','Винегрет','Вина']
recipes = [db.get_rec_by_tagname(t) for t in tags]

norm = Normalization(recipes)
norm.words_normalize()
texts = norm.normal_text


# In[240]:


class Analitika():
    def __init__(self, texts,vectors):
        self.name_model ='DeepPavlov/rubert-base-cased'
        self.config = BertConfig.from_pretrained(self.name_model, output_hidden_states =True,
                                    output_attentions =True,
                                    return_dict=True)
        self.tokenizer = BertTokenizer.from_pretrained(self.name_model)
        self.model = BertModel.from_pretrained(self.name_model, config=self.config)
        self.texts = texts
        self.scaler = StandardScaler()
        self.classifier = KNeighborsClassifier(n_neighbors=27)
        self.vectors = vectors
        
    def __get_output(self, text):
        with torch.no_grad():
            input_ids = self.tokenizer(text, return_tensors='pt')
        output = self.model(**input_ids)
        return output

    def __get_CLS_vector(self, text):
        output = self.__get_output(text)
        with torch.no_grad():
            cls = output.last_hidden_state[0][0]
            return numpy.array(cls)

    def __get_MEANWORD_vector(self, text):
        output = self.__get_output(text)
        with torch.no_grad():
            return numpy.array(output.last_hidden_state.mean(1)[0])
        
    def create_vectors(self):
        for t in tqdm(self.texts):
             self.vectors.append(self.__get_CLS_vector(t[:512]))   
    
    def __create_tags(self):
        tags = []
        for i in self.texts:
            tags.append(re.findall('\w+\s', i)[0]) 
        return tags  
    
    def __create_df(self):
        df = pd.DataFrame(self.vectors)
        df.insert(0, "tag", self.__create_tags())
        return df
    
    def plot_2d(self):
        
        pca = PCA(n_components=2)
        pr = pca.fit_transform(self.__create_df().drop("tag", axis=1).values)
        pr = pd.DataFrame(data = pr
                     , columns = ['x', 'y' ])
        prf = pd.concat([pr, self.__create_df()[['tag']]], axis = 1)

        fig = px.scatter(prf, x='x', y='y',
                      color='tag')
        fig.show()
    
    def plot_3d(self):
        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(self.__create_df().drop("tag", axis=1).values)
        principalDf = pd.DataFrame(data = principalComponents
                     , columns = ['x', 'y','z' ])

        finalDf = pd.concat([principalDf, self.__create_df()[['tag']]], axis = 1)

        fig = px.scatter_3d(finalDf, x='x', y='y', z='z', color='tag')
        fig.show()
        
    def knn(self):
        
        X = self.__create_df().drop("tag", axis=1).values
        y = self.__create_df()["tag"].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        self.scaler.fit(X_train)

        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)
       
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
    
    def knn_predictor(self, text):
        v = self.__get_CLS_vector(text[:512])
        xd = pd.DataFrame(v).T
        xd = xd.values
        xtest = self.scaler.transform(xd)
        y_pred = self.classifier.predict(xtest)
        return y_pred
        


# In[242]:


texts1[0]


# In[241]:


aa = Analitika(texts, vectors)
aa.knn()
aa.knn_predictor(texts1[0])


# In[167]:


aa.plot_2d()


# In[168]:


aa.plot_3d()


# In[162]:


rrr = [db.get_by_date('2022-06-24','Вина')]
norm1 = Normalization(rrr)
norm1.words_normalize()
texts1 = norm1.normal_text


# In[13]:


import pickle 
with open('vectorss.pickle', 'rb') as f:
     vectors = pickle.load(f)


# In[63]:


import matplotlib.pyplot as plt
plt.plot(error_rates)

