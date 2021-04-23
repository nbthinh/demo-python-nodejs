"""
Nếu đã biết thông tin người dùng(người dùng đã đăng nhập), thì ta sẽ dùng lọc cộng tác trên thông tin người dùng đó để lấy rating và trả về sản phẩm gợi ý
Nếu chưa biết thông tin người dùng, ta sẽ dựa trên cookie để lấy email người dùng và gợi ý sản phẩm dựa trên lọc cộng tác
Nếu trên cookie không có email người dùng thì ta sinh ngẫu nhiên 1 email trong luot_xem
"""
from scipy import sparse
from mysql.connector import errorcode
import mysql.connector
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
# from colaborative import CF

class Colaborative_Filtering:
    def __init__(self, Y_data):
        self.Y_data = Y_data
        self.k = 3 # Ta sẽ dự đoán tỉ lệ đánh giá của người dùng chưa biết dựa trên tỉ lệ đánh giá của 2 người dùng gần gũi nhất đã đánh giá cho item i
        self.dist_func = cosine_similarity
        self.Ybar_data = None
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1

    def normalizeY(self):
        users = self.Y_data[:, 0] # lấy tất cả id người dùng
        self.Ybar_data = self.Y_data.copy() # copy Y_data để tiện xử lý sau này
        self.mu = np.zeros((self.n_users, )) # Tạo một ma trận lưu trữ giá trị trung trung bình của rating từng người dùng
        for n in range(self.n_users): # Truy xuất qua từng người dùng
            ids = np.where(users == n)[0].astype(np.int32) # Với mỗi người dùng, ta sẽ lấy index của người dùng đó
            item_ids = self.Y_data[ids, 1]
            rating = self.Y_data[ids, 2] # Load các giá trị đánh giá của người dùng đó
            mean_val = np.mean(rating) # Tính giá trị trung bình của các đánh giá người dùng đó
            if np.isnan(mean_val): mean_val = 0
            self.Ybar_data[ids, 2] = rating - mean_val 
            self.mu[n] = mean_val
            # Ma trận tỉ lệ đánh giá của chúng ta hầu như đêu là các phần tử 0, vì vậy để tiết kiệm không gian lưu trữ, ta chỉ lưu trữ những giá trị đánh giá khác 0 thông qua chỉ số của chúng
            self.Ybar = sparse.coo_matrix(   ( self.Ybar_data[:, 2], (  self.Ybar_data[:, 1], self.Ybar_data[:, 0]  ) ),    (self.n_items, self.n_users)   )
            self.Ybar = self.Ybar.tocsr()


    def caculate_similarity(self):
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)
    
    def fit(self):
        self.normalizeY()
        self.caculate_similarity()
    
    def predict(self, user, item):
        ids = np.where(self.Y_data[:, 1] == item)[0].astype(np.int32)
        users_rated_for_item = self.Y_data[ids, 0].astype(np.int32)
        sim = self.S[user, users_rated_for_item]
        a = np.argsort(sim)[-self.k:]
        nearest_sim = sim[a]
        rate = self.Ybar[item, users_rated_for_item[a]]
        return (rate*nearest_sim)[0]/(np.abs(nearest_sim).sum() + 1e-8)

    def get_recommend(self, user):
        ids = np.where(self.Y_data[:, 0] == user)[0]
        items_rated_by_user = self.Y_data[ids, 1].tolist()
        recommend_item = []
        for i in range(self.n_items):
            if i not in items_rated_by_user: # Những item mà người dùng user chưa đánh giá
                predict_rate = self.predict(user, i)
                if predict_rate > 0:
                    recommend_item.append(i)
        return recommend_item

    def print_recommendation(self):
        print('Recommendation: ')
        for u in range(self.n_users):
            recommended_items = self.get_recommend(u)
            print('    Recommend item(s):', recommended_items, 'to user', u)

if __name__ == "__main__":
    # data file
    mydb = mysql.connector.connect(user='root', password='', host='localhost', database='recommendation_system')
    if mydb:
        print("Connected successfully")
    else:
        print("Error in connecting to MySQL")
    mycursor = mydb.cursor()
    sql = "SELECT * FROM luot_xem"
    mycursor.execute(sql)
    view_list = mycursor.fetchall()
    """
    ('thinh.nguyen0908117036@hcmut.edu.vn', 1, 5)
    ('thinh.nguyen0908117036@hcmut.edu.vn', 3, 5)
    ('hqviet0302@yahoo.com', 6, 5)
    ('hqviet0302@yahoo.com', 1, 3)
    ('thinh.nguyen0908117036@hcmut.edu.vn', 6, 1)
    ('thiennq@gmail.com', 3, 2)
    ('thiennq@gmail.com', 6, 1)
    """
    # Pre-process
    viewed_user = []
    viewed_item = []
    for each_view in view_list:
        if each_view[0] not in viewed_user:
            viewed_user.append(each_view[0])
        if each_view[1] not in viewed_item:
            viewed_item.append(each_view[1])
    user_dict = {} # Dùng để tra lại người dùng
    user_dict_ = {}
    item_dict = {} # Dùng để tra lại sản phẩm
    item_dict_ = {}
    for n in range(len(viewed_user)):
        user_dict[n] = viewed_user[n]
        user_dict_[viewed_user[n]] = n
    for n in range(len(viewed_item)):
        item_dict[n] = viewed_item[n]
        item_dict_[viewed_item[n]] = n

    data_list = []
    for each_row in view_list:
        data_list.append([user_dict_[each_row[0]], item_dict_[each_row[1]], float(each_row[2])])
    data = np.array(data_list)

    # r_cols = ['user_id', 'item_id', 'rating']
    # ratings = pd.read_csv('ex.dat', sep = ' ', names = r_cols, encoding='latin-1')
    # Y_data = ratings.to_numpy()

    recommend = Colaborative_Filtering(data)
    recommend.fit()
    print(user_dict)
    for each in user_dict:
        get = recommend.get_recommend(each)
        print(get)
    

    # rs = CF(data, k = 2, uuCF = 1)
    # rs.fit()
    # rs.print_recommendation()
