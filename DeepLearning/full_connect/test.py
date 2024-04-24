import pandas as pd
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from scipy.stats import pearsonr
import time

from torch.utils.data import TensorDataset
from sklearn.metrics import r2_score
from model import FullyConnectedNetwork
from data_loader import StockDataset
from torch.utils.data import DataLoader


def grouped_r2_score(output,target,labels):
    l = len(labels)
    pred = []
    true = []
    r2_sum = 0.0
    cnt = 0
    pred.append(output[0])
    true.append(target[0])
    for i in range(1,l):
        if labels[i] != labels[i - 1] :
            cnt += 1
            r2_sum += r2_score(pred,true)
            pred.clear()
            true.clear()
        pred.append(output[i])
        true.append(target[i])
    r2 = r2_sum / cnt
    return r2



# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
device = 'cuda'

# train_pth = '/mnt/tmpdata/lvhanglong/data_20240312/df_facs_train_sample.h5'
# test_pth = '/mnt/tmpdata/lvhanglong/data_20240312/df_facs_test.h5'
# val_pth = '/mnt/tmpdata/lvhanglong/data_20240312/df_facs_val.h5'

train_pth = '/mnt/tmpdata/lvhanglong/data_20240312/df_facs_val.h5'
test_pth = '/mnt/tmpdata/lvhanglong/data_20240312/df_facs_val.h5'
val_pth = '/mnt/tmpdata/lvhanglong/data_20240312/df_facs_val.h5'

# train_pth = '../first_500_rows.csv'
# test_pth = '../first_500_rows.csv'
# val_pth = '../first_500_rows.csv'


feature_columns = ['factor_0', 'factor_1', 'factor_2', 'factor_3', 'factor_4', 'factor_5', 'factor_6',
                'factor_7', 'factor_8', 'factor_9', 'factor_10', 'factor_11', 'factor_12', 'factor_13',
                'factor_14', 'factor_15', 'factor_16', 'factor_17', 'factor_18', 'factor_19', 'factor_20', 
                'factor_21', 'factor_22', 'factor_23', 'factor_24', 'factor_25', 'factor_26', 'factor_27', 
                'factor_28', 'factor_29', 'factor_30', 'factor_31', 'factor_32', 'factor_33', 'factor_34', 
                'factor_35', 'factor_36', 'factor_37', 'factor_38', 'factor_39', 'factor_40', 'factor_41', 
                'factor_42', 'factor_43', 'factor_44', 'factor_45', 'factor_46', 'factor_47', 'factor_48', 
                'factor_49', 'factor_50', 'factor_51', 'factor_52', 'factor_53', 'factor_54', 'factor_55', 
                'factor_56', 'factor_57', 'factor_58', 'factor_59', 'factor_60', 'factor_61', 'factor_62', 
                'factor_63', 'factor_64', 'factor_65', 'factor_66', 'factor_67', 'factor_68', 'factor_69', 
                'factor_70', 'factor_71', 'factor_72', 'factor_73', 'factor_74', 'factor_75', 'factor_76', 
                'factor_77', 'factor_78', 'factor_79', 'factor_80', 'factor_81', 'factor_82', 'factor_83', 
                'factor_84', 'factor_85', 'factor_86', 'factor_87', 'factor_88', 'factor_89', 'factor_90', 
                'factor_91', 'factor_92', 'factor_93', 'factor_94', 'factor_95', 'factor_96', 'factor_97', 
                'factor_98', 'factor_99', 'factor_100', 'factor_101', 'factor_102', 'factor_103', 'factor_104', 
                'factor_105', 'factor_106', 'factor_107', 'factor_108', 'factor_109', 'factor_110', 'factor_111', 
                'factor_112', 'factor_113', 'factor_114', 'factor_115', 'factor_116', 'factor_117', 'factor_118', 
                'factor_119', 'factor_120', 'factor_121', 'factor_122', 'factor_123', 'factor_124', 'factor_125', 
                'factor_126', 'factor_127', 'factor_128', 'factor_129', 'factor_130', 'factor_131', 'factor_132', 
                'factor_133', 'factor_134', 'factor_135', 'factor_136', 'factor_137', 'factor_138', 'factor_139', 
                'factor_140', 'factor_141', 'factor_142', 'factor_143', 'factor_144', 'factor_145', 'factor_146', 
                'factor_147', 'factor_148', 'factor_149', 'factor_150', 'factor_151', 'factor_152', 'factor_153', 
                'factor_154', 'factor_155', 'factor_156', 'factor_157', 'factor_158', 'factor_159', 'factor_160', 
                'factor_161', 'factor_162', 'factor_163', 'factor_164', 'factor_165', 'factor_166', 'factor_167', 
                'factor_168', 'factor_169', 'factor_170', 'factor_171', 'factor_172', 'factor_173', 'factor_174', 
                'factor_175', 'factor_176', 'factor_177', 'factor_178', 'factor_179', 'factor_180', 'factor_181', 
                'factor_182', 'factor_183', 'factor_184', 'factor_185', 'factor_186', 'factor_187', 'factor_188', 
                'factor_189', 'factor_190', 'factor_191', 'factor_192', 'factor_193', 'factor_194', 'factor_195', 
                'factor_196', 'factor_197', 'factor_198', 'factor_199', 'factor_200', 'factor_201', 'factor_202', 
                'factor_203', 'factor_204', 'factor_205', 'factor_206', 'factor_207', 'factor_208', 'factor_209', 
                'factor_210', 'factor_211', 'factor_212', 'factor_213', 'factor_214', 'factor_215', 'factor_216', 
                'factor_217', 'factor_218', 'factor_219', 'factor_220', 'factor_221', 'factor_222', 'factor_223', 
                'factor_224', 'factor_225', 'factor_226', 'factor_227', 'factor_228', 'factor_229', 'factor_230', 
                'factor_231', 'factor_232', 'factor_233', 'factor_234', 'factor_235', 'factor_236', 'factor_237', 
                'factor_238', 'factor_239', 'factor_240', 'factor_241', 'factor_242', 'factor_243', 'factor_244', 
                'factor_245', 'factor_246', 'factor_247', 'factor_248', 'factor_249', 'factor_250', 'factor_251', 
                'factor_252', 'factor_253', 'factor_254', 'factor_255', 'factor_256', 'factor_257', 'factor_258', 
                'factor_259', 'factor_260', 'factor_261', 'factor_262', 'factor_263', 'factor_264', 'factor_265', 
                'factor_266', 'factor_267', 'factor_268', 'factor_269', 'factor_270', 'factor_271', 'factor_272', 
                'factor_273', 'factor_274', 'factor_275', 'factor_276', 'factor_277', 'factor_278', 'factor_279', 
                'factor_280', 'factor_281', 'factor_282', 'factor_283', 'factor_284', 'factor_285', 'factor_286', 
                'factor_287', 'factor_288', 'factor_289', 'factor_290', 'factor_291', 'factor_292', 'factor_293', 
                'factor_294', 'factor_295', 'factor_296', 'factor_297', 'factor_298', 'factor_299', 'factor_300', 
                'factor_301', 'factor_302', 'factor_303', 'factor_304', 'factor_305', 'factor_306', 'factor_307', 
                'factor_308', 'factor_309', 'factor_310', 'factor_311', 'factor_312', 'factor_313', 'factor_314', 
                'factor_315', 'factor_316', 'factor_317', 'factor_318', 'factor_319', 'factor_320', 'factor_321', 
                'factor_322', 'factor_323', 'factor_324', 'factor_325', 'factor_326', 'factor_327', 'factor_328', 
                'factor_329', 'factor_330', 'factor_331', 'factor_332', 'factor_333', 'factor_334', 'factor_335', 
                'factor_336', 'factor_337', 'factor_338', 'factor_339', 'factor_340', 'factor_341', 'factor_342', 
                'factor_343', 'factor_344', 'factor_345', 'factor_346', 'factor_347', 'factor_348', 'factor_349', 
                'factor_350', 'factor_351', 'factor_352', 'factor_353', 'factor_354', 'factor_355', 'factor_356', 
                'factor_357', 'factor_358', 'factor_359', 'factor_360', 'factor_361', 'factor_362', 'factor_363', 
                'factor_364', 'factor_365', 'factor_366', 'factor_367', 'factor_368', 'factor_369', 'factor_370', 
                'factor_371', 'factor_372', 'factor_373', 'factor_374', 'factor_375', 'factor_376', 'factor_377', 
                'factor_378', 'factor_379', 'factor_380', 'factor_381', 'factor_382', 'factor_383', 'factor_384', 
                'factor_385', 'factor_386', 'factor_387', 'factor_388', 'factor_389', 'factor_390', 'factor_391', 
                'factor_392', 'factor_393', 'factor_394', 'factor_395', 'factor_396', 'factor_397', 'factor_398', 
                'factor_399', 'factor_400', 'factor_401', 'factor_402', 'factor_403', 'factor_404', 'factor_405', 
                'factor_406', 'factor_407', 'factor_408', 'factor_409', 'factor_410', 'factor_411', 'factor_412', 
                'factor_413', 'factor_414', 'factor_415', 'factor_416', 'factor_417', 'factor_418', 'factor_419', 
                'factor_420', 'factor_421', 'factor_422', 'factor_423', 'factor_424', 'factor_425', 'factor_426', 
                'factor_427', 'factor_428', 'factor_429', 'factor_430', 'factor_431', 'factor_432', 'factor_433', 
                'factor_434', 'factor_435', 'factor_436', 'factor_437', 'factor_438', 'factor_439', 'factor_440', 
                'factor_441', 'factor_442', 'factor_443', 'factor_444', 'factor_445', 'factor_446', 'factor_447', 
                'factor_448', 'factor_449', 'factor_450', 'factor_451', 'factor_452', 'factor_453', 'factor_454', 
                'factor_455', 'factor_456', 'factor_457', 'factor_458', 'factor_459', 'factor_460', 'factor_461', 
                'factor_462', 'factor_463', 'factor_464', 'factor_465', 'factor_466', 'factor_467', 'factor_468', 
                'factor_469', 'factor_470', 'factor_471', 'factor_472', 'factor_473', 'factor_474', 'factor_475', 
                'factor_476', 'factor_477']  # 特征列名
target_column = 'Ret5MinPrice'  # 目标列名

t = time.time()
print(f'start creating dataset:{t}')

# 创建数据集
train_dataset = StockDataset(filepath = train_pth, features = feature_columns, target = target_column, is_train = 1)

print(f'train dataset complete:{t - time.time()}')

t = time.time()

val_dataset = StockDataset(filepath = test_pth, features = feature_columns, target = target_column,scaler=train_dataset.scaler)

print(f'val dataset complete:{t - time.time()}')
t = time.time()

test_dataset = StockDataset(filepath = val_pth, features = feature_columns, target = target_column,scaler=train_dataset.scaler)

print(f'test dataset complete:{t - time.time()}')
t = time.time()

#创建数据集并存储

# train_dataset.save_data("/mnt/tmpdata/lvhanglong/train")

# print(f'train dataset save complete:{t - time.time()}')
# t = time.time()

# val_dataset.save_data("/mnt/tmpdata/lvhanglong/val")

# print(f'val dataset complete:{t - time.time()}')
# t = time.time()

# test_dataset.save_data("/mnt/tmpdata/lvhanglong/test")

# print(f'test dataset complete:{t - time.time()}')
# t = time.time()

# 创建数据加载器
train_loader = DataLoader(train_dataset.get_dataset(), batch_size=50000, shuffle=True,num_workers=4)

print(f'train dataloader complete:{t - time.time()}')
t = time.time()

val_loader = DataLoader(val_dataset.get_dataset(), batch_size=50000, shuffle=False,num_workers=4)

print(f'val dataloader complete:{t - time.time()}')
t = time.time()

test_loader = DataLoader(test_dataset.get_dataset(), batch_size=50000, shuffle=False,num_workers=4)

print(f'test dataloader complete:{t - time.time()}')
t = time.time()

print(len(val_loader.dataset))

# 实例化模型
model = FullyConnectedNetwork(input_dim = 478, output_dim = 1)
if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        # 包装模型以在多个GPU上并行处理
        model = nn.DataParallel(model)
model.to('cuda')

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

best_val_loss = float('inf')
best_val_r2 = float('-inf')
best_model_path = None


print(f'training preparation complete:{t - time.time()}')
t = time.time()

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        print(f'batch train complete:{t - time.time()}')
        t = time.time()

    # 在每个epoch后进行验证
    model.eval()
    val_loss = 0.0
    r2 = 0.0
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for val_data, val_target in val_loader:
            val_data, val_target = val_data.to(device), val_target.to(device)
            val_output = model(val_data)
            val_loss += criterion(val_output, val_target).item()
            all_targets.extend(val_target.view(-1).tolist())  # 收集真实值
            all_outputs.extend(val_output.view(-1).tolist())  # 收集预测值
            r2 += r2_score(val_output, val_target)
    val_loss /= len(val_loader.dataset)
    r2 /= len(val_loader)
    print(f'Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Train Loss: {loss.item()}, Val Loss: {val_loss}, Val R^2: {r2}')
    
    print(f'batch val complete:{t - time.time()}')
    t = time.time()
    # 检查是否有更好的验证损失
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_r2 = r2
        # 保存模型参数
        best_model_path = f'model_best_{epoch}_{batch_idx}_r2_{best_val_r2}.pth'
        # torch.save(model.state_dict(), best_model_path)
        print(f'Best model saved to {best_model_path} with Val Loss: {best_val_loss}, Val R^2: {best_val_r2}')



# 测试模型
model.eval()
test_loss = 0.0
r2 = 0.0
all_targets = []
all_outputs = []
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        r2 += r2_score(output, target)
        all_targets.extend(target.view(-1).tolist())  # 收集真实值
        all_outputs.extend(output.view(-1).tolist())  # 收集预测值
test_loss /= len(test_loader.dataset)
r2 /= len(val_loader)
print(f'Test Loss: {test_loss}, R2:{r2}')
# 计算皮尔逊相关系数
pearson_corr, _ = pearsonr(all_targets, all_outputs)
print(f'Test Loss: {test_loss}, R^2:{r2},Test Pearson Correlation: {pearson_corr}')

print(f'test complete:{t - time.time()}')
t = time.time()
