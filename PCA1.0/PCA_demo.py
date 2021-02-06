from PCA_FaultDection import *
from scipy.io import loadmat


path_train = './data/d00.mat'
path_test= './data/d01te.mat'

X_Train = loadmat(path_train)['d00']
X_test = loadmat(path_test)['d01te']

#初始化PCA模型参数
model = PCA_FaultDection(cumper=0.85, signifi=0.95)

#数据标准化（若是标准化过后的数据则无需这一步）
[X_Train,X_test] = model.normalize(X_Train, X_test)


#训练模型
model.train(X_Train)

#代入测试数据
testresult = model.test(X_test)

#检测结果可视化
model.visualization(model,testresult)

# 单样本 贡献图条状图可视化
single_con_result=model.single_sample_con(X_test[6:7]) # 实际取得是第七个样本，必须写成切片索引的方式 : idx:idx+1
model.con_bar_vis(single_con_result)

#单样本重构贡献图条状图可视化
single_recon_result=model.single_sample_recon(X_test[6:7])# 实际取得是第七个样本，必须写成切片索引的方式 : idx:idx+1
model.recon_bar_vis(single_recon_result)

#多样本贡献图热力图可视化
multi_con_result=model.multi_sample_con(X_test[150:200])
model.con_vis_headmap(multi_con_result)

#多样本重构贡献图热力图可视化
multi_recon_result=model.multi_sample_recon(X_test[150:200])
model.recon_vis_headmap(multi_recon_result)

