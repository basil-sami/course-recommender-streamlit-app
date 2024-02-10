import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
from surprise import NMF
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split


models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          )


def load_ratings():
    return pd.read_csv("ratings.csv")
def load_genres():
    return pd.read_csv("course_genre.csv")
def load_user_scores():
    return pd.read_csv("course_ratings.csv")
def load_course_sims():
    return pd.read_csv("sim.csv")
def load_courses():
    df = pd.read_csv("course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df
def load_bow():
    return pd.read_csv("courses_bows.csv")
def load_cluster():
    return pd.read_csv("clusters.csv")
def load_cluster_pca():
    return pd.read_csv("clusters_with_pca.csv")
def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("ratings.csv", index=False)
        return new_id
# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res


# Model training
def train(model_name, params):
    # TODO: Add model training code here

        # Course Similarity model

        if model_name == models[0]:
            idx_id_dict, id_idx_dict = get_doc_dicts()
            sim_matrix = load_course_sims().to_numpy()
            users = []
            courses = []
            scores = []
            res_dict = {}
        
        elif  model_name == models[3]:
            n_comp=params['component_no']
            cluster_no = params['cluster_no']
            pcadataframe,pca=make_pca(n_comp)
            km=KMeans(n_clusters=cluster_no)
            km.fit(pcadataframe.iloc[:,1:])
            course_genres_df=load_ratings()
            user_score_df=load_user_scores()
            get_user_labels_kmeans_with_pca(user_score_df, course_genres_df, km, pca)
        elif model_name == models[4]:

            reader = Reader(
                line_format='user item rating', sep=',', skip_lines=1, rating_scale=(2, 3))

            coruse_dataset = Dataset.load_from_file("ratings.csv", reader=reader)
            print(coruse_dataset)
            trainset, testset = train_test_split(coruse_dataset, test_size=.01)
            algo = NMF()
            algo.fit(trainset)
            filename = 'knn.sav'
            pickle.dump(algo, open(filename, 'wb'))










        pass


# Prediction
def predict(model_name, user_ids, params):
    sim_threshold = 0.6
    if "sim_threshold" in params:
        sim_threshold = params["sim_threshold"] / 100.0
    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    users = []
    courses = []
    scores = []
    res_dict = {}

    for user_id in user_ids:
        # Course Similarity model
        if model_name == models[0]:
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)
        elif model_name == models[1]:
            sim_threshold = params["sim_threshold"]
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            ratings_df = load_ratings()
            courses_df=load_genres()
            courses,scores,users=one_user_score_rec(user_id,enrolled_course_ids, courses_df, sim_threshold)
        elif model_name == models[2]:
            cluster_no=params['cluster_no']
            courses_df = load_genres()
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            km,res=train_k_cluster(cluster_no, enrolled_course_ids,courses_df)
            test_users_labelled=load_cluster()

            courses,scores,users=get_course(res, ratings_df, user_id, test_users_labelled)
        elif model_name == models[3]:
            cluster_no = params['cluster_no']
            component_no=params['component_no']
            courses_df = load_genres()
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            test_users_labelled = load_cluster_pca()
            pca = pickle.load(open('pca_model.sav', 'rb'))
            km=pickle.load(open('km_with_pca.sav','rb'))



            result=predict_k_cluster_kmeans(enrolled_course_ids, courses_df, km, pca)

            courses, scores, users = get_course(result, ratings_df, user_id, test_users_labelled)
        elif model_name == models[4]:
            users=[]
            scores=[]
            courses=[]

            courses_df = load_genres()
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            algo = NMF()


            # Read the course rating dataset with columns user item rating
            reader = Reader(
                line_format='user item rating', sep=',', skip_lines=1, rating_scale=(2, 3))

            coruse_dataset = Dataset.load_from_file("course_ratings.csv", reader=reader)
            trainset, testset = train_test_split(coruse_dataset, test_size=.0000001)
            algo.fit(trainset)
            all_courses = set(courses_df['COURSE_ID'].values)
            unknown_courses = all_courses.difference(enrolled_course_ids)

            for cor in unknown_courses:
                res = algo.predict(user_id,cor,)

                if res.est >2:
                    users.append(user_id)
                    courses.append(cor)
                    scores.append(res.est)
                

                

            # Read the course rating dataset with columns user item rating





            print(res)



















            # TODO: Add prediction model code here

    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    return res_df


def generate_recommendation_scores(test_users_df,result_df,course_df,score_threshold):
    users = []
    courses = []
    scores = []
    score_threshold = 10
    test_users = test_users_df.groupby(['user']).max().reset_index(drop=False)
    test_user_ids = test_users_df['user'].to_list()
    for user_id in test_user_ids:
        test_user_profile = result_df[result_df['user_id'] == user_id]
        # get user vector for the current user id
        if result_df[result_df['user_id'] == user_id].empty:
            continue
        test_user_vector = result_df[result_df['user_id'] == user_id].iloc[0, 1:].tolist()

        # get the unknown course ids for the current user id
        enrolled_courses = test_users_df[test_users_df['user'] == user_id]['item'].to_list()
        all_courses = set(course_df['COURSE_ID'].values)
        unknown_courses = all_courses.difference(enrolled_courses)
        unknown_course_df = course_df[course_df['COURSE_ID'].isin(unknown_courses)]
        unknown_course_ids = unknown_course_df['COURSE_ID'].tolist()

        unkown_course_array = course_df[course_df['COURSE_ID'].isin(unknown_course_ids)].iloc[:, 2:].to_numpy()
        # user np.dot() to get the recommendation scores for each course
        recommendation_scores = np.dot( unkown_course_array ,test_user_vector[1:])/np.dot(test_user_vector,[1])

        for i in range(0, len(unknown_course_ids)):
            score = recommendation_scores[i]
            # Only keep the courses with high recommendation score
            if score >= score_threshold:
                users.append(user_id)
                courses.append(unknown_course_ids[i])
                scores.append(recommendation_scores[i])

    return users, courses, scores


def get_test_user_df(user_df, coureses_df):
    res = []
    users = []
    user_score = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]

    test_users = user_df.groupby(['user']).max().reset_index(drop=False)
    test_user_ids = test_users['user'].to_list()
    for user_id in test_user_ids:
        user_score = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]
        enrolled_course_ids = user_df[(user_df['user'] == user_id)]['item'].tolist()
        enrolled_course_rating = user_df[(user_df['user'] == user_id)]['rating'].tolist()
        for course, rating in zip(enrolled_course_ids, enrolled_course_rating):

            user_score = user_score + coureses_df[(coureses_df['COURSE_ID'] == course)].iloc[0, 2:] *int( rating)
            users.append(user_id)
            res.append(user_score)
        res.append(user_score)

    result_df = pd.DataFrame(res)
    result_df.insert(0, "user_id", users, True)
    result_df.reset_index(inplace=True)
    result_df.drop('index', axis=1)
    result_df.to_csv("course_ratings.csv", index=False)
    print('domne')
    return res
def one_user_score( enrolled_course_ids, coureses_df):
    user_score = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]

    for course in enrolled_course_ids:
        user_score = user_score + coureses_df[(coureses_df['COURSE_ID'] == course)].iloc[0, 2:] * int(2)


    return user_score
def one_user_score_rec( user,enrolled_courses, course_df,score_threshold):
    courses=[]
    scores=[]
    users=[]
    score=one_user_score( enrolled_courses, course_df)
    print(score)

    all_courses = set(course_df['COURSE_ID'].values)
    unknown_courses = all_courses.difference(enrolled_courses)
    unknown_course_df = course_df[course_df['COURSE_ID'].isin(unknown_courses)]
    unknown_course_ids = unknown_course_df['COURSE_ID'].tolist()

    unkown_course_array = course_df[course_df['COURSE_ID'].isin(unknown_course_ids)].iloc[:, 2:].to_numpy()
    # user np.dot() to get the recommendation scores for each course
    recommendation_scores = np.dot(unkown_course_array, score)
    print(recommendation_scores)

    for i in range(0, len(unknown_course_ids)):
        score = recommendation_scores[i]
        # Only keep the courses with high recommendation score
        if score >= score_threshold:
            users.append(user)

            courses.append(unknown_course_ids[i])
            scores.append(recommendation_scores[i])
    return courses,scores,users
def train_k_cluster(cluster_no,enrolled,course_df):
    user_scores=load_user_scores()
    user_scores = user_scores.drop(['index', 'user_id'] , axis=1)
    km = KMeans(n_clusters=cluster_no)

    km.fit(user_scores[1:])

    user_score=one_user_score(enrolled,course_df)
    result=km.predict([user_score])

    return km,result


def get_course(cluster, test_users_df, user, test_users_labelled):
    score=[]
    users=[]
    print(cluster)
    courses = test_users_labelled[test_users_labelled['cluster'] == cluster[0]].groupby('item').count().sort_values(
        by='user', ascending=False)
    print(test_users_labelled[test_users_labelled['cluster'] == cluster[0]])
    courses.reset_index(inplace=True)
    top_courses = courses['item'].tolist()
    print(top_courses)
    ## - First get all courses belonging to the same cluster and figure out what are the popular ones (such as course enrollments beyond a threshold like 100)

    ## - Get the user's current enrolled courses
    enrolled_courses = test_users_df[test_users_df['user'] == user]['item']
    ## - Check if there are any courses on the popular course list which are new/unseen to the user.
    rec = list(set(top_courses) - set(enrolled_courses))
    rec=rec[:10]
    i=1
    for col in rec:

        score.append(i)
        i=i+1
        users.append(user)
    return rec, score,users
def get_user_labels_kmeans(test_users_df, course_genres_df, km):
    c = []
    df1 = pd.DataFrame()

    test_users_labelled = pd.DataFrame()
    for user in test_users_df['user'].unique():

        c = course_genres_df[
                course_genres_df['COURSE_ID'].isin(test_users_df[test_users_df['user'] == user]['item'])].iloc[:,
            2:].values
        user_score = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]

        for c in c:
            user_score = user_score + c * int(2)

        df = pd.DataFrame([user_score])

        df1 = pd.concat([df, df1])

    user_ids = test_users_df.loc[:, test_users_df.columns == 'user']['user'].unique()

    cluster_labels = km.predict(df1)
    user_ids = pd.DataFrame(user_ids)

    cluster_df = combine_cluster_labels(user_ids, cluster_labels)
    test_users_labelled = pd.merge(test_users_df, cluster_df, left_on='user', right_on='user')
    test_users_labelled.to_csv("clusters.csv", index=False)
    return test_users_labelled
def combine_cluster_labels(user_ids, labels):
    labels_df = pd.DataFrame(labels)
    cluster_df = pd.merge(user_ids, labels_df, left_index=True, right_index=True)
    cluster_df.columns = ['user', 'cluster']
    return cluster_df
def make_pca(n):
    user_profile_df=load_user_scores()
    features = user_profile_df.loc[:, (user_profile_df.columns != 'user_id')& (user_profile_df.columns != 'index')]
    user_ids = user_profile_df.loc[:, user_profile_df.columns == 'user_id']
    PCAmod = PCA(n_components=n)
    PCAmod.fit(features)
    ids=PCAmod.transform(features)
    col_names = ['_'.join(['pc', str(x)]) for x in range(n)]
    pca_dataframe = pd.DataFrame(ids,
                        columns=col_names)
    pca_dataframe = pd.concat([user_ids, pca_dataframe], axis=1)
    filename = 'pca_model.sav'
    pickle.dump(PCAmod, open(filename, 'wb'))
    return pca_dataframe,PCAmod
def make_pca_model(cluster_no,enrolled,course_df,pca_dataframe,PCAmod):
    user_scores = pca_dataframe
    user_scores = user_scores.drop(['user_id'], axis=1)
    km = KMeans(n_clusters=cluster_no)

    km.fit(user_scores[:])

    result = km.predict(user_scores)


    return km, result
def get_user_labels_kmeans_with_pca(test_users_df, course_ratings_df, km,pca):
    c = []
    df1 = test_users_df
    user_ids = course_ratings_df.loc[:, course_ratings_df.columns == 'user']['user'].unique()
    cluster_labels=pca.transform(df1.iloc[:,2:])

    cluster_labels = km.predict(cluster_labels)
    user_ids = pd.DataFrame(user_ids)

    cluster_df = combine_cluster_labels(user_ids, cluster_labels)
    print(test_users_df)
    print(cluster_df)
    test_users_labelled = pd.merge(course_ratings_df, cluster_df, left_on='user', right_on='user')
    test_users_labelled.to_csv("clusters_with_pca.csv", index=False)
    filename = 'km_with_pca.sav'
    pickle.dump(km, open(filename, 'wb'))
    return test_users_labelled


def predict_k_cluster_kmeans( enrolled, course_df, km,pca):
    user_score = one_user_score(enrolled, course_df)
    user_score=pca.transform([user_score])
    result = km.predict(user_score)
    return  result

