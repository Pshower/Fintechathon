1. 下载和解压ADD数据集到本地
2. 使用0 move file将数据集文件按照标签分别移动到01_data_train/genuine和01_data_train/fake；测试集同理
3. 使用1 Data Transform将所有原始wav文件转化为MFCC后，整合为一个JSON文件
4. 使用2 Model Train * 训练和测试模型，分别有LSTM， Transformer, ViT三个模型；模型参数保存在02_model文件夹下
5. 使用3 KYC Integrate App打开最终的项目文件，运行后启动软件，可以执行导入音频文件、录音、播放音频、检测基本功能。


其他：关于使用环境配置：
