import json
import pandas as pd
import PyPDF2
import re
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

folder_path = 'public/docs/'


def cleaning_document(documents_df):
    """
    here the documents_df is the dataframe that is taken for the cleaning
    """
    # documents_df['documents']= documents_df.content
    # removing special characters and stop words from the text
    stop_words_l=stopwords.words('english')
    documents_df.documents = documents_df['documents'].astype(str)
    documents_df['documents_cleaned']=documents_df.documents.apply(
        lambda x: " ".join(
        re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l))
    
    return documents_df

def clean_para(para):
    stop_words_l = stopwords.words('english')
    documents_para = str(para)
    processed_para = (" ".join(re.sub(r'[^a-zA-Z]', ' ', w).lower() for w in documents_para.split() if re.sub(
        r'[^a-zA-Z]', ' ', w).lower() not in stop_words_l))
    return processed_para


def extract_convert_text_from_pdf_to_json(files):
    final_arr = []

    for file_name in files:

        pdfFileObj = open(folder_path+file_name, 'rb')
        pdf_name_f = file_name.split('/')[-1]
        pdfReader = PyPDF2.PdfReader(pdfFileObj)
        pages = len(pdfReader.pages)

        res = {
            'filename': pdf_name_f,
            'content': []
        }

        # Loop for reading all the Pages
        for i in range(pages):
            print("Page {} of {}".format(str(i+1), str(pdf_name_f)))
            # Creating a page object
            pageObj = pdfReader.pages[i]
            lines = pageObj.extract_text().split("\n")
            # print(lines)
            temp = list([])
            for k in range(0, len(lines)+1, 5):
                temp.append(lines[k:k+5])
            for para_numb, paragraph in enumerate(temp, 1):
                pgh_text = ' '.join(paragraph)
                if len(pgh_text) > 5:
                    cleaned_para_text = clean_para(pgh_text)
                    res['content'].append({
                        'page_number': i + 1,
                        'para_number': para_numb,
                        'd': pgh_text.strip(),
                        'cleaned_d': cleaned_para_text
                    })

            # conversion of dictionary format to json format.
            json_data = json.dumps(res)
        final_arr.append(json_data)

        # return pd.DataFrame(res)
    return final_arr

def similarity_model(documents_df1):
    tfidfvectoriser=TfidfVectorizer()#(ngram_range=(2,5), max_features=10000)
    tfidfvectoriser.fit(documents_df1.documents_cleaned)
    tfidf_vectors=tfidfvectoriser.transform(documents_df1.documents_cleaned)
    return tfidfvectoriser, tfidf_vectors
    

def similarity(tfidfvectoriser, tfidf_vectors, user_phrase):
    d1 = []
    d1.append(user_phrase)
    test = {'documents':d1}
    test_df_old= pd.DataFrame(test)
    # test_df = test_df_old.copy()
    test_df = cleaning_document(test_df_old)
    tfidf_user_vectors = tfidfvectoriser.transform(test_df.documents_cleaned)
    pairwise_similarities=np.dot(tfidf_vectors,tfidf_user_vectors.T).toarray()
    
    return pairwise_similarities

# documents_df1 = "Already_defined which is the master database"
# user_phrase = "What do you mean?"
# tfidfvectoriser, tfidf_vectors = similarity_model(documents_df1)
# pairwise_similarities = similarity(tfidfvectoriser, tfidf_vectors, user_phrase)
# documents_df1['similarities'] = pairwise_similarities
